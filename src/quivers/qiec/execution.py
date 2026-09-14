"""Named, checked execution for whole QIEC modules.

The evaluator deliberately operates on core terms and process-local
attachments.  This module supplies the user-facing boundary above it: it
selects a named computation, specializes its static telescope, validates host
arguments and results, creates a new attachment table for every invocation,
and reports failures with stable diagnostic codes.

Runtime providers are explicit.  A configuration names registered provider
factories (or carries provider objects when used from Python); no module import
or ambient attachment registry is consulted by ``run_named``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from importlib.metadata import entry_points
import json
from pathlib import Path
from types import MappingProxyType
from typing import Literal, NoReturn, Protocol, cast

from quivers.qiec.evaluator import (
    EvaluationError,
    Evaluator,
    FuelExhaustedError,
    RuntimeAttachments,
    RuntimeConstructor,
    RuntimeClause,
    RuntimeHandler,
    RuntimeRequest,
    RuntimeTypeMismatch,
    RuntimeValidator,
    Resumption,
)
from quivers.qiec.builtins import BUILTIN_EFFECTS, draw_handler, score_handler
from quivers.qiec.canonical import (
    LOG_WEIGHT,
    SAMPLEABLE_CONSTRUCTOR,
    SITE_CONSTRUCTOR,
    tensor_shape,
)
from quivers.qiec.effects import HandlerDef
from quivers.qiec.evidence import BranchGiven, Reflexivity
from quivers.qiec.identifiers import OperationId, SourceOrigin
from quivers.qiec.kinds import (
    EffectBinder,
    IndexBinder,
    NatSort,
    ShapeSort,
    TypeBinder,
    UserIndexSort,
)
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.substitution import (
    StaticSubstitution,
    instantiate_telescope,
    substitute_computation,
    substitute_type,
)
from quivers.qiec.terms import (
    Computation,
    Local,
)
from quivers.qiec.types import (
    BOOL,
    INT,
    REAL,
    STRING,
    UNIT,
    EqualityType,
    FunctionType,
    IndexConstructor,
    IndexLiteral,
    IndexTerm,
    ShapeIndex,
    StaticArgument,
    TypeApplication,
    TypeConstructorRef,
    TypeExpr,
    TypeVariable,
)


type TraceObserver = Callable[["ExecutionTraceEvent"], None]


@dataclass(frozen=True, slots=True)
class ExecutionTraceEvent:
    """One stable event emitted by a named QIEC invocation.

    Parameters
    ----------
    sequence
        The event's position in the invocation's trace, from zero.
    event
        The stable event name, such as ``"run.started"``.
    computation
        The name of the computation being run.
    detail
        JSON-compatible event data; frozen on construction.
    """

    sequence: int
    event: str
    computation: str
    detail: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze ``detail`` so a recorded event cannot change after emission."""
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))

    def to_data(self) -> dict[str, object]:
        """Render the event as JSON-compatible data.

        Returns
        -------
        dict[str, object]
            The sequence number, event name, computation name, and detail
            mapping, with every detail value made JSON-compatible.
        """
        return {
            "sequence": self.sequence,
            "event": self.event,
            "computation": self.computation,
            "detail": _json_value(dict(self.detail)),
        }


@dataclass(slots=True)
class TraceRecorder:
    """Collect events while also satisfying the :class:`TraceObserver` API.

    Parameters
    ----------
    events
        The recorded events, in order; empty to start.
    """

    events: list[ExecutionTraceEvent] = field(default_factory=list)

    def __call__(self, event: ExecutionTraceEvent) -> None:
        """Record one event.

        Parameters
        ----------
        event
            The event the invocation just emitted.
        """
        self.events.append(event)


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostic:
    """A stable failure at the source-to-host execution boundary.

    Parameters
    ----------
    code
        The stable diagnostic code, such as ``"qiec-run-arity"``.
    message
        The human-readable explanation.
    computation
        The named computation involved, if any.
    origin
        The source location involved, if any.
    severity
        ``"error"``, ``"warning"``, or ``"note"``.
    """

    code: str
    message: str
    computation: str | None = None
    origin: SourceOrigin | None = None
    severity: Literal["error", "warning", "note"] = "error"

    def to_data(self) -> dict[str, object]:
        """Render the diagnostic as JSON-compatible data.

        Returns
        -------
        dict[str, object]
            The code, severity, message, computation name, and origin (or
            ``None`` when the failure has no source location).
        """
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "computation": self.computation,
            "origin": self.origin.to_data() if self.origin is not None else None,
        }


class ExecutionFailure(EvaluationError):
    """Raised with a serializable diagnostic when named execution cannot run.

    Parameters
    ----------
    diagnostic
        The failure, kept on the error as ``diagnostic``; its message
        becomes the error's message.
    """

    def __init__(self, diagnostic: ExecutionDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class RuntimeProvider(Protocol):
    """An explicit source of process-local handlers and type validators."""

    @property
    def name(self) -> str:
        """The provider's stable name, reported in traces and result labels."""
        ...

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Bind this provider's handlers into a fresh attachment table.

        Parameters
        ----------
        module
            The validated module whose handlers and effects are being served.
        attachments
            The per-invocation table receiving runtime handlers.
        """
        ...

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        """Supply a host validator for a runtime type, if this provider has one.

        Parameters
        ----------
        type_
            The closed QIEC type an argument or result must inhabit.

        Returns
        -------
        RuntimeValidator | None
            A predicate over host values, or ``None`` when this provider does
            not cover the type.
        """
        ...


@dataclass(frozen=True, slots=True)
class CoreRuntimeProvider:
    """Primitive validators plus explicitly configured structural handlers.

    ``options.handlers`` maps authored handler names to either
    ``{"kind": "passthrough"}``, ``{"kind": "state", "initial": ...}``,
    or ``{"kind": "scripted", "responses": {"op": [...]}}``.  The source
    handler still controls types, coverage, and resumption grades; options
    supply only process-local clause behavior and state.

    Parameters
    ----------
    options
        JSON-compatible configuration; frozen on construction.
    name
        The provider's name, reported in traces and result labels.
    """

    options: Mapping[str, object] = field(default_factory=dict)
    name: str = "core"

    def __post_init__(self) -> None:
        """Freeze ``options`` so a provider's configuration cannot drift."""
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
        """Bind one structural handler per configured authored handler.

        Parameters
        ----------
        module
            The validated module whose handlers the configuration names.
        attachments
            The per-invocation table receiving the constructed handlers.

        Raises
        ------
        ValueError
            If ``options.handlers`` is not an object, names a handler the module
            does not declare, holds a non-object configuration, or requests a
            kind other than ``passthrough``, ``scripted``, or ``state``.
        """
        configured = self.options.get("handlers", {})
        if not isinstance(configured, Mapping):
            raise ValueError("core runtime option 'handlers' must be an object")
        definitions = {handler.name: handler for handler in module.handlers}
        unknown = set(configured) - set(definitions)
        if unknown:
            raise ValueError(
                f"core runtime names unknown handlers: {', '.join(sorted(unknown))}"
            )
        operation_names = {
            operation.id: operation.name
            for effect in module.effects
            for operation in effect.operations
        }
        for handler_name, raw_options in configured.items():
            if not isinstance(handler_name, str) or not isinstance(
                raw_options, Mapping
            ):
                raise ValueError("each core handler configuration must be an object")
            definition = definitions[handler_name]
            kind = raw_options.get("kind")
            if kind == "passthrough":
                runtime = self._passthrough(definition, raw_options, operation_names)
            elif kind == "scripted":
                runtime = self._scripted(definition, raw_options, operation_names)
            elif kind == "state":
                runtime = self._state(definition, raw_options, operation_names)
            elif kind == "draw":
                runtime = self._prelude(definition, draw_handler, "Random")
            elif kind == "score":
                runtime = self._prelude(
                    definition,
                    lambda **_: score_handler(
                        weight_validator=_require_core_handler_validator(
                            LOG_WEIGHT, definition.name, "weight"
                        ),
                        answer_validator=_require_core_handler_validator(
                            definition.input_type, definition.name, "input"
                        ),
                        expose_total=definition.output_type != definition.input_type,
                    )[0],
                    "Score",
                )
            else:
                raise ValueError(
                    f"core handler {handler_name!r} has unknown kind {kind!r}"
                )
            attachments.bind_handler(runtime)

    def _prelude(
        self,
        definition: HandlerDef,
        factory: Callable[..., RuntimeHandler],
        interface: str,
    ) -> RuntimeHandler:
        """Serve a module's foreign handler with a prelude implementation.

        Parameters
        ----------
        definition
            The module's handler declaration, which must handle the
            prelude interface named and declare the prelude handler's
            clauses and grades.
        factory
            The prelude factory building the implementation; called with
            ``result_validator`` when it accepts one.
        interface
            The prelude interface the implementation serves.

        Returns
        -------
        RuntimeHandler
            The prelude implementation re-keyed under the module's
            declaration, so the manifest check passes and the module's
            types and grades govern dispatch.

        Raises
        ------
        ValueError
            If the declaration handles another interface, or its clauses
            or grades differ from the prelude implementation's.
        """
        if definition.effect.name != interface or definition.effect.id != next(
            effect.ref.id for effect in BUILTIN_EFFECTS if effect.ref.name == interface
        ):
            raise ValueError(
                f"core handler {definition.name!r} must handle the prelude "
                f"interface {interface!r} to use this implementation"
            )
        if interface == "Random":
            built = factory(
                result_validator=lambda value: True,
                answer_type=definition.input_type,
            )
        else:
            built = factory()
        expected = {
            (clause.operation, clause.grade) for clause in built.definition.clauses
        }
        declared = {(clause.operation, clause.grade) for clause in definition.clauses}
        if expected != declared:
            raise ValueError(
                f"core handler {definition.name!r} declares clauses that differ "
                f"from the prelude implementation for {interface!r}"
            )
        rekeyed = replace(built, definition=definition)
        if built.context_factory is not None:
            prototype_factory = built.context_factory

            def context_factory() -> RuntimeHandler:
                """Build a fresh installation under the module's declaration.

                Returns
                -------
                RuntimeHandler
                    The prelude installation re-keyed to the declaration.
                """
                return replace(prototype_factory(), definition=definition)

            rekeyed.context_factory = context_factory
        return rekeyed

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        """Supply the primitive validator for a closed runtime type.

        Parameters
        ----------
        type_
            The closed QIEC type an argument or result must inhabit.

        Returns
        -------
        RuntimeValidator | None
            The core validator, or ``None`` for types the core runtime cannot
            check on its own.
        """
        return _core_validator(type_)

    def _passthrough(
        self,
        definition,
        options: Mapping[str, object],
        operation_names: Mapping[OperationId, str],
    ) -> RuntimeHandler:
        """Build a handler that answers each operation with its own argument.

        Parameters
        ----------
        definition
            The authored handler declaration supplying types and clauses.
        options
            The handler's configuration; ``responses`` maps operation names to
            the values used when an operation carries no runtime argument.
        operation_names
            Operation identities mapped to their declared names.

        Returns
        -------
        RuntimeHandler
            A duplicable-context handler over the declaration's clauses.

        Raises
        ------
        ValueError
            If ``responses`` is not an object, or if the core runtime cannot
            validate the handler's input or output type.
        """
        input_validator = _require_core_handler_validator(
            definition.input_type, definition.name, "input"
        )
        output_validator = _require_core_handler_validator(
            definition.output_type, definition.name, "output"
        )
        defaults = options.get("responses", {})
        if not isinstance(defaults, Mapping):
            raise ValueError("passthrough 'responses' must be an object")

        def clause(operation):  # type: ignore[no-untyped-def]
            """Build the runtime clause for one declared operation.

            Parameters
            ----------
            operation
                The handler clause declaration being implemented.

            Returns
            -------
            RuntimeClause
                A clause resuming with the request's argument or configured response.
            """
            operation_name = operation_names.get(
                operation.operation, str(operation.operation)
            )

            def invoke(request, resume, _context):  # type: ignore[no-untyped-def]
                """Answer a request by echoing its argument or the configured response.

                Parameters
                ----------
                request
                    The performed operation and its runtime arguments.
                resume
                    The continuation into the handled computation.
                _context
                    Runtime services, unused by this clause.

                Returns
                -------
                object
                    What the resumed computation produced.

                Raises
                ------
                ValueError
                    If the request carries no argument and no response is configured
                    for the operation.
                """
                if request.arguments:
                    response = request.arguments[0]
                elif operation_name in defaults:
                    response = _tuplify(defaults[operation_name])
                else:
                    raise ValueError(
                        f"passthrough handler needs a response for {operation_name!r}"
                    )
                return _checked_resume(request, resume, response)

            return RuntimeClause(invoke, lambda _value: True)

        return RuntimeHandler(
            definition,
            {item.operation: clause(item) for item in definition.clauses},
            input_validator=input_validator,
            output_validator=output_validator,
            duplicable_context=True,
        )

    def _scripted(
        self,
        definition,
        options: Mapping[str, object],
        operation_names: Mapping[OperationId, str],
    ) -> RuntimeHandler:
        """Build a handler that answers operations from scripted queues.

        Parameters
        ----------
        definition
            The authored handler declaration supplying types and clauses.
        options
            The handler's configuration; ``responses`` maps operation names to
            a value or list of values consumed in order.
        operation_names
            Operation identities mapped to their declared names.

        Returns
        -------
        RuntimeHandler
            A mutable-context handler whose ``context_factory`` yields a fresh
            copy of the queues for every installation.

        Raises
        ------
        ValueError
            If ``responses`` is missing or not an object, or if the core runtime
            cannot validate the handler's input or output type.
        """
        responses = options.get("responses")
        if not isinstance(responses, Mapping):
            raise ValueError("scripted handler requires a 'responses' object")

        def make() -> RuntimeHandler:
            """Build a handler installation with its own copy of the queues.

            Returns
            -------
            RuntimeHandler
                A handler whose queues no other installation shares.
            """
            queues = {
                str(name): list(value if isinstance(value, list) else [value])
                for name, value in responses.items()
            }

            def clause(operation):  # type: ignore[no-untyped-def]
                """Build the runtime clause for one declared operation.

                Parameters
                ----------
                operation
                    The handler clause declaration being implemented.

                Returns
                -------
                RuntimeClause
                    A clause resuming with the next scripted response.
                """
                operation_name = operation_names.get(
                    operation.operation, str(operation.operation)
                )

                def invoke(_request, resume, _context):  # type: ignore[no-untyped-def]
                    """Answer a request with the next queued response for its operation.

                    Parameters
                    ----------
                    _request
                        The performed operation, used only to validate the response
                        type.
                    resume
                        The continuation into the handled computation.
                    _context
                        Runtime services, unused by this clause.

                    Returns
                    -------
                    object
                        What the resumed computation produced.

                    Raises
                    ------
                    ValueError
                        If the operation's queue is exhausted.
                    """
                    queue = queues.get(operation_name, [])
                    if not queue:
                        raise ValueError(
                            f"scripted responses exhausted for {operation_name!r}"
                        )
                    return _checked_resume(_request, resume, _tuplify(queue.pop(0)))

                return RuntimeClause(invoke, lambda _value: True)

            return RuntimeHandler(
                definition,
                {item.operation: clause(item) for item in definition.clauses},
                input_validator=_require_core_handler_validator(
                    definition.input_type, definition.name, "input"
                ),
                output_validator=_require_core_handler_validator(
                    definition.output_type, definition.name, "output"
                ),
                mutable_context=True,
            )

        prototype = make()
        prototype.context_factory = make
        return prototype

    def _state(
        self,
        definition,
        options: Mapping[str, object],
        operation_names: Mapping[OperationId, str],
    ) -> RuntimeHandler:
        """Build a handler serving ``get`` and ``put`` over one state cell.

        Parameters
        ----------
        definition
            The authored handler declaration supplying types and clauses.
        options
            The handler's configuration; ``initial`` is the starting state.
        operation_names
            Operation identities mapped to their declared names.

        Returns
        -------
        RuntimeHandler
            A mutable-context handler whose ``context_factory`` yields a fresh
            cell holding ``initial`` for every installation.

        Raises
        ------
        ValueError
            If ``initial`` is absent, or if the core runtime cannot validate the
            handler's input or output type.
        """
        if "initial" not in options:
            raise ValueError("state handler requires an 'initial' value")
        initial = _tuplify(options["initial"])

        def make() -> RuntimeHandler:
            """Build a handler installation with its own state cell.

            Returns
            -------
            RuntimeHandler
                A handler whose state no other installation shares.
            """
            state = [initial]

            def clause(operation):  # type: ignore[no-untyped-def]
                """Build the runtime clause for one declared operation.

                Parameters
                ----------
                operation
                    The handler clause declaration being implemented.

                Returns
                -------
                RuntimeClause
                    A clause reading or replacing the cell, by operation name.
                """
                operation_name = operation_names.get(
                    operation.operation, str(operation.operation)
                )

                def invoke(request, resume, _context):  # type: ignore[no-untyped-def]
                    """Answer ``get`` with the cell's value or ``put`` by replacing it.

                    Parameters
                    ----------
                    request
                        The performed operation and its runtime arguments.
                    resume
                        The continuation into the handled computation.
                    _context
                        Runtime services, unused by this clause.

                    Returns
                    -------
                    object
                        What the resumed computation produced.

                    Raises
                    ------
                    ValueError
                        If ``put`` is given anything other than one argument, or if the
                        operation is neither ``get`` nor ``put``.
                    """
                    if operation_name == "get":
                        return _checked_resume(request, resume, state[0])
                    if operation_name == "put":
                        if len(request.arguments) != 1:
                            raise ValueError("state.put expects one runtime argument")
                        state[0] = request.arguments[0]
                        return _checked_resume(request, resume, None)
                    raise ValueError(
                        f"state provider does not implement operation {operation_name!r}"
                    )

                return RuntimeClause(invoke, lambda _value: True)

            return RuntimeHandler(
                definition,
                {item.operation: clause(item) for item in definition.clauses},
                input_validator=_require_core_handler_validator(
                    definition.input_type, definition.name, "input"
                ),
                output_validator=_require_core_handler_validator(
                    definition.output_type, definition.name, "output"
                ),
                mutable_context=True,
            )

        prototype = make()
        prototype.context_factory = make
        return prototype


type RuntimeProviderFactory = Callable[[Mapping[str, object]], RuntimeProvider]


_PROVIDER_FACTORIES: dict[str, RuntimeProviderFactory] = {
    "core": lambda options: CoreRuntimeProvider(options),
}


def available_runtime_providers() -> tuple[str, ...]:
    """Return built-in and installed ``quivers.qiec_runtime`` entry points.

    Returns
    -------
    tuple[str, ...]
        Every selectable provider name, sorted.
    """

    installed = {
        point.name for point in entry_points().select(group="quivers.qiec_runtime")
    }
    return tuple(sorted({*_PROVIDER_FACTORIES, *installed}))


def _provider_factory(name: str) -> RuntimeProviderFactory:
    """Resolve a provider name to its factory.

    Parameters
    ----------
    name
        A registered name or a ``quivers.qiec_runtime`` entry-point name.

    Returns
    -------
    RuntimeProviderFactory
        The callable building a provider from JSON-compatible options.

    Raises
    ------
    KeyError
        If no factory is registered or installed under ``name``.
    RuntimeError
        If more than one entry point claims ``name``.
    TypeError
        If the entry point loads to something that is not callable.
    """
    local = _PROVIDER_FACTORIES.get(name)
    if local is not None:
        return local
    matches = tuple(entry_points().select(group="quivers.qiec_runtime", name=name))
    if not matches:
        raise KeyError(name)
    if len(matches) != 1:
        raise RuntimeError(f"multiple runtime-provider entry points named {name!r}")
    loaded = matches[0].load()
    if not callable(loaded):
        raise TypeError(f"runtime-provider entry point {name!r} is not callable")
    return cast(RuntimeProviderFactory, loaded)


def register_runtime_provider(name: str, factory: RuntimeProviderFactory) -> None:
    """Register a provider factory for explicit configuration by name.

    Registration makes a provider selectable; it never activates that provider
    for an invocation.  Replacing an existing name is rejected so plugin load
    order cannot silently change execution semantics.

    Parameters
    ----------
    name
        The name a configuration uses to select the provider.
    factory
        The callable building a provider from JSON-compatible options.

    Raises
    ------
    ValueError
        If ``name`` is empty or already registered.
    """

    if not name or name in _PROVIDER_FACTORIES:
        raise ValueError(f"runtime provider {name!r} is already registered or invalid")
    _PROVIDER_FACTORIES[name] = factory


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    """One named provider and its JSON-compatible construction options.

    Parameters
    ----------
    provider
        The registered or entry-point name of the provider; must be
        non-empty.
    options
        The options passed to the provider's factory; frozen on
        construction.
    """

    provider: str
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject empty provider names and freeze the options.

        Raises
        ------
        ValueError
            If ``provider`` is empty.
        """
        if not self.provider:
            raise ValueError("runtime provider name cannot be empty")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    """Complete, explicit runtime selection for one or more invocations.

    Parameters
    ----------
    selections
        Providers to build by name for every invocation; the core provider
        alone by default.
    providers
        Already-built provider objects, appended after the selections when
        used from Python.
    """

    selections: tuple[RuntimeSelection, ...] = (RuntimeSelection("core"),)
    providers: tuple[RuntimeProvider, ...] = ()

    @property
    def label(self) -> str:
        """A ``+``-joined label naming every selected and supplied provider.

        Returns
        -------
        str
            The provider names in order, or ``"detached"`` when there are none.
        """
        names = [selection.provider for selection in self.selections]
        names.extend(provider.name for provider in self.providers)
        return "+".join(names) if names else "detached"

    def instantiate(self) -> tuple[RuntimeProvider, ...]:
        """Build every named provider and append the programmatic ones.

        Returns
        -------
        tuple[RuntimeProvider, ...]
            The providers in selection order, followed by ``providers``.

        Raises
        ------
        ExecutionFailure
            With code ``qiec-run-provider`` if a selection names an unknown
            provider, its factory cannot be loaded, or the factory rejects the
            selection's options.
        """
        providers: list[RuntimeProvider] = []
        for selection in self.selections:
            try:
                factory = _provider_factory(selection.provider)
            except KeyError as error:
                available = ", ".join(available_runtime_providers()) or "(none)"
                raise ExecutionFailure(
                    ExecutionDiagnostic(
                        "qiec-run-provider",
                        f"unknown runtime provider {selection.provider!r}; "
                        f"available providers: {available}",
                    )
                ) from error
            except Exception as error:
                raise ExecutionFailure(
                    ExecutionDiagnostic(
                        "qiec-run-provider",
                        f"could not load runtime provider {selection.provider!r}: {error}",
                    )
                ) from error
            try:
                providers.append(factory(selection.options))
            except Exception as error:
                raise ExecutionFailure(
                    ExecutionDiagnostic(
                        "qiec-run-provider",
                        f"runtime provider {selection.provider!r} rejected its "
                        f"configuration: {error}",
                    )
                ) from error
        providers.extend(self.providers)
        return tuple(providers)

    def to_data(self) -> dict[str, object]:
        """Render the configuration as JSON-compatible data.

        Returns
        -------
        dict[str, object]
            The named selections with their options, and the names of any
            programmatic providers.
        """
        return {
            "providers": [
                {"name": selection.provider, "options": dict(selection.options)}
                for selection in self.selections
            ],
            "programmatic_providers": [provider.name for provider in self.providers],
        }


def runtime_configuration_from_data(data: object) -> RuntimeConfiguration:
    """Decode the non-executable JSON runtime configuration format.

    Parameters
    ----------
    data
        The decoded JSON document; ``providers`` lists names or
        ``{"name": ..., "options": {...}}`` objects and defaults to ``core``.

    Returns
    -------
    RuntimeConfiguration
        The selections in document order, with no programmatic providers.

    Raises
    ------
    ValueError
        If ``data`` is not an object, ``providers`` is not a list, an entry
        is neither a name nor an object with a ``name``, or an entry's
        ``options`` is not an object.
    """

    if not isinstance(data, dict):
        raise ValueError("runtime configuration must be a JSON object")
    raw = data.get("providers", [{"name": "core"}])
    if not isinstance(raw, list):
        raise ValueError("runtime configuration 'providers' must be a list")
    selections: list[RuntimeSelection] = []
    for item in raw:
        if isinstance(item, str):
            selections.append(RuntimeSelection(item))
            continue
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(
                "each runtime provider must be a name or object with 'name'"
            )
        options = item.get("options", {})
        if not isinstance(options, dict):
            raise ValueError("runtime provider 'options' must be an object")
        selections.append(RuntimeSelection(item["name"], options))
    return RuntimeConfiguration(tuple(selections))


def load_runtime_configuration(path: str | Path) -> RuntimeConfiguration:
    """Load an explicit runtime selection without importing executable code.

    Parameters
    ----------
    path
        A JSON file in the format :func:`runtime_configuration_from_data`
        accepts.

    Returns
    -------
    RuntimeConfiguration
        The decoded selection.

    Raises
    ------
    ValueError
        If the file cannot be read, is not valid JSON, or does not decode
        as a runtime configuration.
    """

    source = Path(path)
    try:
        data = json.loads(source.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not load runtime configuration {source}: {error}"
        ) from error
    return runtime_configuration_from_data(data)


def parse_static_arguments(
    module: QiecModule,
    computation: str,
    assignments: tuple[str, ...],
) -> tuple[StaticArgument, ...]:
    """Resolve CLI ``NAME=TERM`` specializations in telescope order.

    This intentionally accepts only closed, nominal terms available in the
    module: primitive or declared types, closed index constructors/literals,
    and declared effect applications.  Static variables and inferred holes are
    not runtime configuration values.

    Parameters
    ----------
    module
        The module declaring ``computation`` and the names its terms may use.
    computation
        The named computation whose telescope is being instantiated.
    assignments
        ``NAME=TERM`` strings, one per telescope binder, in any order.

    Returns
    -------
    tuple[StaticArgument, ...]
        The parsed arguments in the telescope's order.

    Raises
    ------
    ExecutionFailure
        With code ``qiec-run-computation`` if the computation is unknown, or
        ``qiec-run-static`` if an assignment is malformed, repeated, names a
        binder the telescope lacks, omits one it has, or does not parse at
        the binder's sort.
    """

    selected = next(
        (
            candidate
            for candidate in module.computations
            if candidate.name == computation
        ),
        None,
    )
    if selected is None:
        _fail("qiec-run-computation", f"unknown QIEC computation {computation!r}")
    assert selected is not None
    values: dict[str, str] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not name or not value:
            _fail(
                "qiec-run-static",
                f"static argument {assignment!r} must have the form NAME=TERM",
                computation,
                selected.origin,
            )
        if name in values:
            _fail(
                "qiec-run-static",
                f"static argument {name!r} was supplied more than once",
                computation,
                selected.origin,
            )
        values[name] = value
    expected = {binder.name for binder in selected.telescope}
    unknown = set(values) - expected
    missing = expected - set(values)
    if unknown or missing:
        detail: list[str] = []
        if missing:
            detail.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            detail.append(f"unknown {', '.join(sorted(unknown))}")
        _fail(
            "qiec-run-static",
            "invalid static specialization: " + "; ".join(detail),
            computation,
            selected.origin,
        )
    try:
        return tuple(
            _parse_static(module, binder, values[binder.name])
            for binder in selected.telescope
        )
    except (TypeError, ValueError) as error:
        _fail(
            "qiec-run-static",
            f"invalid static specialization for {computation!r}: {error}",
            computation,
            selected.origin,
        )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """The checked value and trace produced by one named invocation.

    Parameters
    ----------
    computation
        The name of the computation run.
    value
        The validated host result.
    result_type
        The computation's result type under its specialization.
    runtime
        The label of the runtime configuration used.
    trace
        Every event the invocation emitted, in order.
    """

    computation: str
    value: object
    result_type: TypeExpr
    runtime: str
    trace: tuple[ExecutionTraceEvent, ...]

    def to_data(self) -> dict[str, object]:
        """Render the result as JSON-compatible data.

        Returns
        -------
        dict[str, object]
            ``ok`` set to ``True``, the computation name, the JSON-compatible
            value, the rendered result type, the runtime label, and the trace.
        """
        return {
            "ok": True,
            "computation": self.computation,
            "value": _json_value(self.value),
            "result_type": _render_type(self.result_type),
            "runtime": self.runtime,
            "trace": [event.to_data() for event in self.trace],
        }


def run_named(
    module: QiecModule,
    computation: str,
    arguments: tuple[object, ...] = (),
    *,
    static_arguments: tuple[StaticArgument, ...] = (),
    runtime: RuntimeConfiguration | None = None,
    observer: TraceObserver | None = None,
    fuel: int | None = None,
) -> ExecutionResult:
    """Validate and execute one named computation from ``module``.

    The attachment table and every configured provider instance are allocated
    inside this call.  Mutable handler state can thus never leak from one named
    invocation to the next.

    Parameters
    ----------
    module
        The module declaring the computation; it is validated before the
        body is evaluated.
    computation
        The name of the computation to run.
    arguments
        Host values for the computation's value parameters, in order.
    static_arguments
        Closed terms instantiating the computation's telescope, in order;
        required whenever the telescope is non-empty.
    runtime
        The providers to attach; defaults to the core provider alone.
    observer
        A callback receiving every trace event as it is emitted.
    fuel
        The number of machine steps the run may take, or ``None`` for no
        bound. General recursion may diverge, and a bound is the only way
        a caller can be sure a run ends.

    Returns
    -------
    ExecutionResult
        The validated result value, its specialized type, the runtime label,
        and the full trace.

    Raises
    ------
    ExecutionFailure
        With code ``qiec-run-computation`` for an unknown name,
        ``qiec-run-static`` for a bad specialization, ``qiec-run-arity`` for
        a wrong argument count, ``qiec-run-provider`` when a provider cannot
        be built or attached, ``qiec-run-validator`` when no provider covers
        an argument or result type, ``qiec-run-argument`` or
        ``qiec-run-result`` when a value fails validation,
        ``qiec-run-module`` when the module fails validation,
        ``qiec-run-fuel`` when the step budget runs out, and
        ``qiec-run-evaluation`` when evaluation or a runtime clause fails.
    """

    selected = next(
        (
            candidate
            for candidate in module.computations
            if candidate.name == computation
        ),
        None,
    )
    if selected is None:
        available = ", ".join(item.name for item in module.computations) or "(none)"
        _fail(
            "qiec-run-computation",
            f"unknown QIEC computation {computation!r}; available: {available}",
            computation,
        )
    assert selected is not None
    config = runtime or RuntimeConfiguration()
    recorder = TraceRecorder()
    sequence = 0

    def emit(event: str, detail: Mapping[str, object] | None = None) -> None:
        """Record one trace event and forward it to the observer.

        Parameters
        ----------
        event
            The stable event name.
        detail
            JSON-compatible event data; ``None`` records an empty mapping.
        """
        nonlocal sequence
        trace_event = ExecutionTraceEvent(sequence, event, selected.name, detail or {})
        sequence += 1
        recorder(trace_event)
        if observer is not None:
            observer(trace_event)

    emit("run.started", {"runtime": config.label})
    substitution = _specialize(selected, static_arguments)
    if static_arguments:
        emit(
            "run.specialized",
            {"arguments": [_render_static(argument) for argument in static_arguments]},
        )
    specialized_parameters, specialized_body = _specialize_computation(
        selected, substitution
    )
    specialized_result = substitute_type(selected.type.result, substitution)
    if len(arguments) != len(selected.parameters):
        _fail(
            "qiec-run-arity",
            f"computation {selected.name!r} expects {len(selected.parameters)} "
            f"value arguments, got {len(arguments)}",
            selected.name,
            selected.origin,
        )

    providers = config.instantiate()
    attachments = RuntimeAttachments()
    try:
        for provider in providers:
            provider.attach(module, attachments)
            emit("runtime.attached", {"provider": provider.name})
    except ExecutionFailure:
        raise
    except Exception as error:
        _fail(
            "qiec-run-provider",
            f"runtime provider failed while attaching: {error}",
            selected.name,
            selected.origin,
        )

    for position, (parameter, specialized_parameter, value) in enumerate(
        zip(selected.parameters, specialized_parameters, arguments, strict=True)
    ):
        expected = specialized_parameter.type
        validator = _validator_for(expected, providers)
        if validator is None:
            _fail(
                "qiec-run-validator",
                f"no configured runtime provider can validate argument "
                f"{parameter.name!r} at type {_render_type(expected)}",
                selected.name,
                selected.origin,
            )
        try:
            accepted = validator(value)
        except Exception as error:
            _fail(
                "qiec-run-argument",
                f"validator raised for argument {parameter.name!r}: {error}",
                selected.name,
                selected.origin,
            )
        if accepted is False:
            _fail(
                "qiec-run-argument",
                f"argument {position} ({parameter.name}) does not inhabit "
                f"{_render_type(expected)}: {value!r}",
                selected.name,
                selected.origin,
            )
        emit("argument.validated", {"name": parameter.name, "position": position})

    try:
        registry = validate_module(module)
    except Exception as error:
        _fail(
            "qiec-run-module",
            f"QIEC module failed validation before execution: {error}",
            selected.name,
            selected.origin,
        )
    environment = dict(zip(specialized_parameters, arguments, strict=True))

    def evaluator_trace(event: str, detail: Mapping[str, object]) -> None:
        """Forward an evaluator trace event into the invocation's trace.

        Parameters
        ----------
        event
            The stable event name the evaluator emitted.
        detail
            The event's data.
        """
        emit(event, detail)

    evaluator = Evaluator(
        attachments,
        trace_hook=evaluator_trace,
        module=module,
        fuel=fuel,
        type_validator=lambda type_: _validator_for(type_, providers),
    )
    try:
        value = evaluator.evaluate_checked(specialized_body, registry, environment)
    except ExecutionFailure:
        raise
    except FuelExhaustedError as error:
        emit("run.failed", {"error": type(error).__name__, "steps": error.steps})
        _fail(
            "qiec-run-fuel",
            f"computation {selected.name!r} did not finish within {error.steps} steps",
            selected.name,
            selected.origin,
        )
    except EvaluationError as error:
        emit("run.failed", {"error": type(error).__name__})
        _fail(
            "qiec-run-evaluation",
            f"computation {selected.name!r} failed: {error}",
            selected.name,
            selected.origin,
        )
    except Exception as error:
        emit("run.failed", {"error": type(error).__name__})
        _fail(
            "qiec-run-evaluation",
            f"runtime clause for computation {selected.name!r} failed: {error}",
            selected.name,
            selected.origin,
        )
    validator = _validator_for(specialized_result, providers)
    if validator is None:
        _fail(
            "qiec-run-validator",
            f"no configured runtime provider can validate result type "
            f"{_render_type(specialized_result)}",
            selected.name,
            selected.origin,
        )
    try:
        accepted = validator(value)
    except Exception as error:
        _fail(
            "qiec-run-result",
            f"result validator raised: {error}",
            selected.name,
            selected.origin,
        )
    if accepted is False:
        _fail(
            "qiec-run-result",
            f"result does not inhabit {_render_type(specialized_result)}: {value!r}",
            selected.name,
            selected.origin,
        )
    emit("run.completed", {"result_type": _render_type(specialized_result)})
    return ExecutionResult(
        selected.name,
        value,
        specialized_result,
        config.label,
        tuple(recorder.events),
    )


def _specialize(
    computation: NamedComputation,
    arguments: tuple[StaticArgument, ...],
) -> StaticSubstitution:
    """Build the substitution instantiating a computation's telescope.

    Parameters
    ----------
    computation
        The named computation being specialized.
    arguments
        Closed static terms, one per telescope binder, in order.

    Returns
    -------
    StaticSubstitution
        The binder-to-argument mapping.

    Raises
    ------
    ExecutionFailure
        With code ``qiec-run-static`` if the telescope is non-empty but no
        arguments were supplied, or if the arguments do not instantiate it.
    """
    if computation.telescope and not arguments:
        expected = ", ".join(
            f"{binder.name}:{_binder_sort(binder)}" for binder in computation.telescope
        )
        _fail(
            "qiec-run-static",
            f"computation {computation.name!r} is polymorphic; supply all static "
            f"arguments in telescope order ({expected})",
            computation.name,
            computation.origin,
        )
    try:
        return instantiate_telescope(computation.telescope, arguments)
    except (TypeError, ValueError) as error:
        _fail(
            "qiec-run-static",
            f"invalid static specialization for {computation.name!r}: {error}",
            computation.name,
            computation.origin,
        )


def _parse_static(
    module: QiecModule,
    binder: TypeBinder | IndexBinder | EffectBinder,
    text: str,
) -> StaticArgument:
    """Parse one closed static term at a telescope binder's sort.

    The spellings are the source language's own: a type is ``Int``,
    ``Vec[Int](S(Z))``, or ``Pair[Int, Bool]``; an index is ``3``,
    ``S(S(Z))``, or the shape ``[2, 3]``; an effect is ``State[Int]``.

    Parameters
    ----------
    module
        The module whose families and effects the term may name.
    binder
        The binder fixing whether a type, index, or effect is expected.
    text
        The term's source.

    Returns
    -------
    StaticArgument
        The parsed type application, index term, or effect application.

    Raises
    ------
    ValueError
        If the head names nothing the binder's sort admits, the static or
        index argument count does not match the head's telescope, a family
        requiring arguments is given none, a sort without named
        constructors is applied, or an index term cannot be read at the
        binder's sort.
    """
    head, static_parts, index_parts = _static_application(text)
    if isinstance(binder, TypeBinder):
        if index_parts and head == "":
            raise ValueError(f"a shape {text!r} is not a type")
        family = next((item for item in module.families if item.name == head), None)
        if family is not None:
            if len(family.parameters) != len(static_parts):
                raise ValueError(
                    f"family {head!r} expects {len(family.parameters)} static "
                    f"arguments, got {len(static_parts)}"
                )
            if len(family.indices) != len(index_parts):
                raise ValueError(
                    f"family {head!r} expects {len(family.indices)} indices, "
                    f"got {len(index_parts)}"
                )
            return TypeApplication(
                family.type_constructor,
                (
                    *(
                        _parse_static(module, expected, part)
                        for expected, part in zip(
                            family.parameters, static_parts, strict=True
                        )
                    ),
                    *(
                        _parse_static(module, expected, part)
                        for expected, part in zip(
                            family.indices, index_parts, strict=True
                        )
                    ),
                ),
            )
        if static_parts or index_parts:
            raise ValueError(f"unknown type constructor {head!r}")
        if head not in {"Unit", "Bool", "Int", "Real", "String"}:
            raise ValueError(f"unknown closed runtime type {head!r}")
        return TypeApplication(TypeConstructorRef.builtin(head))
    if isinstance(binder, EffectBinder):
        if index_parts:
            raise ValueError(f"an effect application {text!r} takes no indices")
        effect = next((item for item in module.effects if item.ref.name == head), None)
        if effect is None:
            raise ValueError(f"unknown effect interface {head!r}")
        if len(effect.telescope) != len(static_parts):
            raise ValueError(
                f"effect {head!r} expects {len(effect.telescope)} arguments, "
                f"got {len(static_parts)}"
            )
        return effect.apply(
            tuple(
                _parse_static(module, expected, part)
                for expected, part in zip(effect.telescope, static_parts, strict=True)
            )
        )
    if static_parts:
        raise ValueError(f"index term {text!r} cannot take static arguments")
    if head == "":
        if not isinstance(binder.sort, ShapeSort):
            raise ValueError(f"a shape {text!r} is not an index of {binder.sort!r}")
        dimensions = tuple(
            _parse_static(module, IndexBinder("_", NatSort()), part)
            for part in index_parts
        )
        return ShapeIndex(dimensions)  # type: ignore[arg-type]
    if index_parts:
        if not isinstance(binder.sort, UserIndexSort):
            raise ValueError(f"index sort {binder.sort!r} has no named constructors")
        arguments = tuple(
            _parse_static(module, IndexBinder("_", binder.sort), part)
            for part in index_parts
        )
        return IndexConstructor(head, arguments, binder.sort)  # type: ignore[arg-type]
    if isinstance(binder.sort, NatSort) and head.isdigit():
        return IndexLiteral(int(head), binder.sort)
    if isinstance(binder.sort, UserIndexSort):
        return IndexConstructor(head, (), binder.sort)
    raise ValueError(f"cannot parse closed index term {text!r} for {binder.sort!r}")


def _static_application(
    text: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Split ``HEAD[STATIC, ...](INDEX, ...)`` into its three parts.

    Either argument group may be absent. A bare ``[...]`` with no head is
    a shape literal and is returned as index parts under an empty head.

    Parameters
    ----------
    text
        The term's source; brackets and parentheses nest, and commas split
        only at depth zero.

    Returns
    -------
    tuple[str, tuple[str, ...], tuple[str, ...]]
        The head, the bracketed static arguments, and the parenthesized
        index arguments, each stripped.

    Raises
    ------
    ValueError
        If the text is empty, a group is unbalanced or out of order, the
        head is missing where one is required, or an argument is empty.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("static terms cannot be empty")
    head_end = next(
        (index for index, character in enumerate(stripped) if character in "[("),
        len(stripped),
    )
    head = stripped[:head_end].strip()
    rest = stripped[head_end:]
    static_parts: tuple[str, ...] = ()
    index_parts: tuple[str, ...] = ()
    if rest.startswith("["):
        group, rest = _split_group(rest, "[", "]", text)
        static_parts = group
    if rest.startswith("("):
        group, rest = _split_group(rest, "(", ")", text)
        index_parts = group
    if rest.strip():
        raise ValueError(f"malformed static application {text!r}")
    if head == "" and (index_parts or not static_parts):
        raise ValueError(f"malformed static application {text!r}")
    if head == "":
        return "", (), static_parts
    return head, static_parts, index_parts


def _split_group(
    text: str,
    opener: str,
    closer: str,
    subject: str,
) -> tuple[tuple[str, ...], str]:
    """Split one leading bracketed group into its top-level arguments.

    Parameters
    ----------
    text
        Source starting with ``opener``.
    opener
        The group's opening character.
    closer
        The group's closing character.
    subject
        The whole term, named in any diagnostic.

    Returns
    -------
    tuple[tuple[str, ...], str]
        The group's stripped arguments, and the source after the group.

    Raises
    ------
    ValueError
        If the group is unbalanced or an argument is empty.
    """
    depth = 0
    parts: list[str] = []
    start = 1
    for index, character in enumerate(text):
        if character in "[(":
            depth += 1
        elif character in "])":
            depth -= 1
            if depth < 0:
                raise ValueError(f"malformed static application {subject!r}")
            if depth == 0:
                if character != closer:
                    raise ValueError(f"malformed static application {subject!r}")
                parts.append(text[start:index].strip())
                if any(not part for part in parts):
                    raise ValueError(f"malformed static application {subject!r}")
                return tuple(parts), text[index + 1 :]
        elif character == "," and depth == 1:
            parts.append(text[start:index].strip())
            start = index + 1
    raise ValueError(f"malformed static application {subject!r}")


def _binder_sort(binder: TypeBinder | IndexBinder | EffectBinder) -> str:
    """Name a telescope binder's sort for diagnostics.

    Parameters
    ----------
    binder
        The telescope binder.

    Returns
    -------
    str
        ``Type`` or ``Effect`` for kind binders, or the index sort's name.
    """
    if isinstance(binder, TypeBinder):
        return "Type" if getattr(binder.kind, "tag", "") == "type" else "Effect"
    if isinstance(binder, IndexBinder):
        return getattr(binder.sort, "name", getattr(binder.sort, "tag", "index"))
    return "Effect"


def _specialize_computation(
    computation: NamedComputation,
    substitution: StaticSubstitution,
) -> tuple[tuple[Local, ...], Computation]:
    """Apply a named telescope capture-freely to every runtime-relevant term.

    Parameters
    ----------
    computation
        The named computation whose parameters and body are specialized.
    substitution
        The telescope instantiation to apply.

    Returns
    -------
    tuple[tuple[Local, ...], Computation]
        The parameter locals at their specialized types, and the body
        with every static position substituted. A body's references to a
        parameter are rewritten to the same specialized local, so an
        environment keyed by the returned parameters binds them.
    """
    parameters = tuple(
        Local(parameter.name, substitute_type(parameter.type, substitution))
        for parameter in computation.parameters
    )
    return parameters, substitute_computation(computation.body, substitution)


def _validator_for(
    type_: TypeExpr,
    providers: tuple[RuntimeProvider, ...],
) -> RuntimeValidator | None:
    """Find the validator for a type, preferring later providers.

    Parameters
    ----------
    type_
        The closed QIEC type to validate.
    providers
        The attached providers in configuration order.

    Returns
    -------
    RuntimeValidator | None
        The first validator found scanning from the last provider, or
        ``None`` when no provider covers the type.
    """
    for provider in reversed(providers):
        validator = provider.validator_for(type_)
        if validator is not None:
            return validator
    return None


def _require_core_handler_validator(
    type_: TypeExpr,
    handler: str,
    role: str,
) -> RuntimeValidator:
    """Fetch the core validator for a handler's input or output type.

    Parameters
    ----------
    type_
        The handler's input or output type.
    handler
        The handler's name, for the diagnostic.
    role
        ``"input"`` or ``"output"``, for the diagnostic.

    Returns
    -------
    RuntimeValidator
        The core validator for ``type_``.

    Raises
    ------
    ValueError
        If the core runtime cannot validate ``type_``.
    """
    validator = _core_validator(type_)
    if validator is None:
        raise ValueError(
            f"core runtime cannot validate {role} of open handler {handler!r}; "
            "use a typed provider plugin"
        )
    return validator


def _checked_resume(
    request: RuntimeRequest,
    resume: Resumption,
    value: object,
) -> object:
    """Resume with a configured response after checking its type.

    Parameters
    ----------
    request
        The performed operation, whose declared result type the response
        must inhabit.
    resume
        The continuation into the handled computation.
    value
        The host value to resume with.

    Returns
    -------
    object
        What the resumed computation produced.

    Raises
    ------
    RuntimeTypeMismatch
        If the core runtime cannot validate the operation's result type, the
        validator raises, or the value does not inhabit the type.
    """
    validator = _core_validator(request.core.result_type)
    if validator is None:
        raise RuntimeTypeMismatch(
            "core runtime cannot validate operation result type "
            f"{_render_type(request.core.result_type)}; use a typed provider plugin"
        )
    try:
        accepted = validator(value)
    except Exception as error:
        raise RuntimeTypeMismatch(
            f"operation result validator raised for {request.operation}: {error}"
        ) from error
    if accepted is False:
        raise RuntimeTypeMismatch(
            f"configured response for operation {request.operation} does not inhabit "
            f"{_render_type(request.core.result_type)}: {value!r}"
        )
    return resume(value)


def _tuplify(value: object) -> object:
    """Turn JSON lists into tuples recursively, leaving other values alone.

    Parameters
    ----------
    value
        A decoded JSON value.

    Returns
    -------
    object
        The value with every list, at any depth, replaced by a tuple.
    """
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuplify(item) for key, item in value.items()}
    return value


def _core_validator(type_: TypeExpr) -> RuntimeValidator | None:
    """Build the core runtime's validator for a closed type.

    Parameters
    ----------
    type_
        The closed QIEC type to validate.

    Returns
    -------
    RuntimeValidator | None
        A predicate for primitives, products of validatable components,
        declared constructors, equality evidence, and functions; ``None``
        for type variables and products with an unvalidatable component.
    """
    if type_ == UNIT:
        return lambda value: value is None
    if type_ == BOOL:
        return lambda value: isinstance(value, bool)
    if type_ == INT:
        return lambda value: isinstance(value, int) and not isinstance(value, bool)
    if type_ == REAL:
        return lambda value: (
            isinstance(value, int | float) and not isinstance(value, bool)
        )
    if type_ == STRING:
        return lambda value: isinstance(value, str)
    if isinstance(type_, TypeApplication):
        if type_ == LOG_WEIGHT:
            return lambda value: (
                isinstance(value, int | float) and not isinstance(value, bool)
            )
        if type_.constructor == SAMPLEABLE_CONSTRUCTOR:
            return lambda value: (
                callable(getattr(value, "log_prob", None))
                and (
                    callable(getattr(value, "sample", None))
                    or callable(getattr(value, "rsample", None))
                )
            )
        if type_.constructor == SITE_CONSTRUCTOR:
            return lambda value: isinstance(value, str) and bool(value)
        shape = tensor_shape(type_)
        if shape is not None:
            element_validator = _core_validator(shape[0])
            if element_validator is None:
                return None
            return _tensor_validator(element_validator, shape[1])
        if type_.constructor.name.startswith("Product"):
            type_nodes = (TypeVariable, TypeApplication, FunctionType, EqualityType)
            if not all(isinstance(item, type_nodes) for item in type_.arguments):
                return None
            validators = tuple(
                _core_validator(cast(TypeExpr, item)) for item in type_.arguments
            )
            if any(validator is None for validator in validators):
                return None

            def product(value: object) -> bool:
                """Check a tuple component-wise against the product's validators.

                Parameters
                ----------
                value
                    The host value to check.

                Returns
                -------
                bool
                    Whether ``value`` is a tuple of the right length whose every
                    component passes its validator.
                """
                return (
                    isinstance(value, tuple)
                    and len(value) == len(validators)
                    and all(
                        validator is not None and validator(item) is not False
                        for validator, item in zip(validators, value, strict=True)
                    )
                )

            return product

        def constructor(value: object) -> bool:
            """Check that a value is a runtime constructor of this type.

            Parameters
            ----------
            value
                The host value to check.

            Returns
            -------
            bool
                Whether ``value`` is a :class:`RuntimeConstructor` whose result type
                is exactly ``type_``.
            """
            return isinstance(value, RuntimeConstructor) and value.result_type == type_

        return constructor
    if isinstance(type_, EqualityType):
        return lambda value: isinstance(value, Reflexivity | BranchGiven)
    if isinstance(type_, FunctionType):
        return callable
    if isinstance(type_, TypeVariable):
        return None
    return None


def _tensor_validator(
    element: RuntimeValidator,
    dimensions: tuple[IndexTerm, ...],
) -> RuntimeValidator:
    """Build the validator for a tensor of a given shape.

    Parameters
    ----------
    element : RuntimeValidator
        The validator for one element.
    dimensions : tuple[IndexTerm, ...]
        The static shape; a literal dimension fixes a length, any other
        index term leaves it open.

    Returns
    -------
    RuntimeValidator
        A predicate accepting nested tuples of the shape's rank whose
        lengths match the literal dimensions and whose elements pass.
    """

    def validate(value: object, depth: int = 0) -> bool:
        """Check one level of nesting against the shape.

        Parameters
        ----------
        value
            The value at this level.
        depth
            How many dimensions have been descended.

        Returns
        -------
        bool
            Whether the value fits the remaining dimensions.
        """
        if depth == len(dimensions):
            return element(value) is not False
        if not isinstance(value, tuple):
            return False
        dimension = dimensions[depth]
        if isinstance(dimension, IndexLiteral) and isinstance(dimension.value, int):
            if len(value) != dimension.value:
                return False
        return all(validate(item, depth + 1) for item in value)

    return validate


def _fail(
    code: str,
    message: str,
    computation: str | None = None,
    origin: SourceOrigin | None = None,
) -> NoReturn:
    """Raise an :class:`ExecutionFailure` carrying a stable diagnostic.

    Parameters
    ----------
    code
        The stable diagnostic code.
    message
        The human-readable explanation.
    computation
        The named computation involved, if any.
    origin
        The source location involved, if any.

    Raises
    ------
    ExecutionFailure
        Always.
    """
    raise ExecutionFailure(ExecutionDiagnostic(code, message, computation, origin))


def _render_type(type_: TypeExpr) -> str:
    """Render a type for diagnostics and result data.

    Parameters
    ----------
    type_
        The type to render.

    Returns
    -------
    str
        Source-like text: variable names, ``Head[args]`` applications,
        ``A -> B`` functions, and ``Eq[l, r]`` equalities.
    """
    if isinstance(type_, TypeVariable):
        return type_.name
    if isinstance(type_, TypeApplication):
        if not type_.arguments:
            return type_.constructor.name
        arguments = ", ".join(_render_static(item) for item in type_.arguments)
        return f"{type_.constructor.name}[{arguments}]"
    if isinstance(type_, FunctionType):
        return f"{_render_type(type_.parameter)} -> {_render_type(type_.result)}"
    if isinstance(type_, EqualityType):
        return f"Eq[{_render_static(type_.left)}, {_render_static(type_.right)}]"
    return repr(type_)


def _render_static(argument: StaticArgument) -> str:
    """Render a static argument for diagnostics and trace data.

    Parameters
    ----------
    argument
        The type, index term, effect application, or shape to render.

    Returns
    -------
    str
        Source-like text for the argument.
    """
    if isinstance(
        argument, TypeVariable | TypeApplication | FunctionType | EqualityType
    ):
        return _render_type(argument)
    if isinstance(argument, IndexConstructor):
        if not argument.arguments:
            return argument.name
        return f"{argument.name}({', '.join(_render_static(item) for item in argument.arguments)})"
    if isinstance(argument, IndexLiteral):
        return str(argument.value)
    name = getattr(argument, "name", None)
    arguments = getattr(argument, "arguments", ())
    if isinstance(name, str):
        if arguments:
            return f"{name}[{', '.join(_render_static(item) for item in arguments)}]"
        return name
    dimensions = getattr(argument, "dimensions", None)
    if dimensions is not None:
        return f"Shape[{', '.join(_render_static(item) for item in dimensions)}]"
    return repr(argument)


def _json_value(value: object) -> object:
    """Convert a host value into JSON-compatible data.

    Parameters
    ----------
    value
        Any host value an invocation produced or traced.

    Returns
    -------
    object
        JSON scalars unchanged; bytes as a hex object; tuples and lists as
        lists; mappings with string keys; runtime constructors as objects
        naming their constructor, static arguments, fields, and result type;
        and anything else as its Python type name and ``repr``.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, RuntimeConstructor):
        return {
            "constructor": str(value.constructor),
            "static_arguments": [
                _render_static(cast(StaticArgument, argument))
                for argument in value.static_arguments
            ],
            "fields": [_json_value(field) for field in value.fields],
            "result_type": _render_type(value.result_type),
        }
    return {"python_type": type(value).__name__, "repr": repr(value)}


__all__ = [
    "CoreRuntimeProvider",
    "ExecutionDiagnostic",
    "ExecutionFailure",
    "ExecutionResult",
    "ExecutionTraceEvent",
    "RuntimeConfiguration",
    "RuntimeProvider",
    "RuntimeSelection",
    "TraceObserver",
    "TraceRecorder",
    "available_runtime_providers",
    "load_runtime_configuration",
    "parse_static_arguments",
    "register_runtime_provider",
    "run_named",
    "runtime_configuration_from_data",
]
