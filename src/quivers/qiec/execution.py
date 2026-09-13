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
    RuntimeAttachments,
    RuntimeConstructor,
    RuntimeClause,
    RuntimeHandler,
    RuntimeRequest,
    RuntimeTypeMismatch,
    RuntimeValidator,
    Resumption,
)
from quivers.qiec.identifiers import OperationId, SourceOrigin
from quivers.qiec.kinds import (
    EffectBinder,
    IndexBinder,
    NatSort,
    TypeBinder,
    UserIndexSort,
)
from quivers.qiec.module import NamedComputation, QiecModule, validate_module
from quivers.qiec.substitution import (
    StaticSubstitution,
    instantiate_telescope,
    substitute_effect,
    substitute_static,
    substitute_type,
)
from quivers.qiec.terms import (
    AttachmentRef,
    Bind,
    Case,
    CaseBranch,
    CaseMotive,
    Computation,
    ConstructorValue,
    EvidenceValue,
    Handle,
    LiteralValue,
    Local,
    Perform,
    Return,
    TransportValue,
    Value,
    Var,
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
    StaticArgument,
    TypeApplication,
    TypeConstructorRef,
    TypeExpr,
    TypeVariable,
)


type TraceObserver = Callable[["ExecutionTraceEvent"], None]


@dataclass(frozen=True, slots=True)
class ExecutionTraceEvent:
    """One stable event emitted by a named QIEC invocation."""

    sequence: int
    event: str
    computation: str
    detail: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))

    def to_data(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "event": self.event,
            "computation": self.computation,
            "detail": _json_value(dict(self.detail)),
        }


@dataclass(slots=True)
class TraceRecorder:
    """Collect events while also satisfying the :class:`TraceObserver` API."""

    events: list[ExecutionTraceEvent] = field(default_factory=list)

    def __call__(self, event: ExecutionTraceEvent) -> None:
        self.events.append(event)


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostic:
    """A stable failure at the source-to-host execution boundary."""

    code: str
    message: str
    computation: str | None = None
    origin: SourceOrigin | None = None
    severity: Literal["error", "warning", "note"] = "error"

    def to_data(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "computation": self.computation,
            "origin": self.origin.to_data() if self.origin is not None else None,
        }


class ExecutionFailure(EvaluationError):
    """Raised with a serializable diagnostic when named execution cannot run."""

    def __init__(self, diagnostic: ExecutionDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class RuntimeProvider(Protocol):
    """An explicit source of process-local handlers and type validators."""

    @property
    def name(self) -> str: ...

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None: ...

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None: ...


@dataclass(frozen=True, slots=True)
class CoreRuntimeProvider:
    """Primitive validators plus explicitly configured structural handlers.

    ``options.handlers`` maps authored handler names to either
    ``{"kind": "passthrough"}``, ``{"kind": "state", "initial": ...}``,
    or ``{"kind": "scripted", "responses": {"op": [...]}}``.  The source
    handler still controls types, coverage, and resumption grades; options
    supply only process-local clause behavior and state.
    """

    options: Mapping[str, object] = field(default_factory=dict)
    name: str = "core"

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))

    def attach(self, module: QiecModule, attachments: RuntimeAttachments) -> None:
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
            else:
                raise ValueError(
                    f"core handler {handler_name!r} has unknown kind {kind!r}"
                )
            attachments.bind_handler(runtime)

    def validator_for(self, type_: TypeExpr) -> RuntimeValidator | None:
        return _core_validator(type_)

    def _passthrough(
        self,
        definition,
        options: Mapping[str, object],
        operation_names: Mapping[OperationId, str],
    ) -> RuntimeHandler:
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
            operation_name = operation_names.get(
                operation.operation, str(operation.operation)
            )

            def invoke(request, resume, _context):  # type: ignore[no-untyped-def]
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
        responses = options.get("responses")
        if not isinstance(responses, Mapping):
            raise ValueError("scripted handler requires a 'responses' object")

        def make() -> RuntimeHandler:
            queues = {
                str(name): list(value if isinstance(value, list) else [value])
                for name, value in responses.items()
            }

            def clause(operation):  # type: ignore[no-untyped-def]
                operation_name = operation_names.get(
                    operation.operation, str(operation.operation)
                )

                def invoke(_request, resume, _context):  # type: ignore[no-untyped-def]
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
        if "initial" not in options:
            raise ValueError("state handler requires an 'initial' value")
        initial = _tuplify(options["initial"])

        def make() -> RuntimeHandler:
            state = [initial]

            def clause(operation):  # type: ignore[no-untyped-def]
                operation_name = operation_names.get(
                    operation.operation, str(operation.operation)
                )

                def invoke(request, resume, _context):  # type: ignore[no-untyped-def]
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
    """Return built-in and installed ``quivers.qiec_runtime`` entry points."""

    installed = {
        point.name for point in entry_points().select(group="quivers.qiec_runtime")
    }
    return tuple(sorted({*_PROVIDER_FACTORIES, *installed}))


def _provider_factory(name: str) -> RuntimeProviderFactory:
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
    """

    if not name or name in _PROVIDER_FACTORIES:
        raise ValueError(f"runtime provider {name!r} is already registered or invalid")
    _PROVIDER_FACTORIES[name] = factory


@dataclass(frozen=True, slots=True)
class RuntimeSelection:
    """One named provider and its JSON-compatible construction options."""

    provider: str
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("runtime provider name cannot be empty")
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    """Complete, explicit runtime selection for one or more invocations."""

    selections: tuple[RuntimeSelection, ...] = (RuntimeSelection("core"),)
    providers: tuple[RuntimeProvider, ...] = ()

    @property
    def label(self) -> str:
        names = [selection.provider for selection in self.selections]
        names.extend(provider.name for provider in self.providers)
        return "+".join(names) if names else "detached"

    def instantiate(self) -> tuple[RuntimeProvider, ...]:
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
        return {
            "providers": [
                {"name": selection.provider, "options": dict(selection.options)}
                for selection in self.selections
            ],
            "programmatic_providers": [provider.name for provider in self.providers],
        }


def runtime_configuration_from_data(data: object) -> RuntimeConfiguration:
    """Decode the non-executable JSON runtime configuration format."""

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
    """Load an explicit runtime selection without importing executable code."""

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
    """The checked value and trace produced by one named invocation."""

    computation: str
    value: object
    result_type: TypeExpr
    runtime: str
    trace: tuple[ExecutionTraceEvent, ...]

    def to_data(self) -> dict[str, object]:
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
) -> ExecutionResult:
    """Validate and execute one named computation from ``module``.

    The attachment table and every configured provider instance are allocated
    inside this call.  Mutable handler state can thus never leak from one named
    invocation to the next.
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
        emit(event, detail)

    evaluator = Evaluator(
        attachments,
        trace_hook=evaluator_trace,
    )
    try:
        value = evaluator.evaluate_checked(specialized_body, registry, environment)
    except ExecutionFailure:
        raise
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
    head, parts = _static_application(text)
    if isinstance(binder, TypeBinder):
        if parts:
            family = next((item for item in module.families if item.name == head), None)
            if family is None:
                raise ValueError(f"unknown type constructor {head!r}")
            telescope = (*family.parameters, *family.indices)
            if len(telescope) != len(parts):
                raise ValueError(
                    f"type constructor {head!r} expects {len(telescope)} arguments, "
                    f"got {len(parts)}"
                )
            return TypeApplication(
                family.type_constructor,
                tuple(
                    _parse_static(module, expected, part)
                    for expected, part in zip(telescope, parts, strict=True)
                ),
            )
        family = next((item for item in module.families if item.name == head), None)
        if family is not None:
            if family.parameters or family.indices:
                raise ValueError(f"type family {head!r} requires static arguments")
            return TypeApplication(family.type_constructor)
        if head not in {"Unit", "Bool", "Int", "Real", "String"}:
            raise ValueError(f"unknown closed runtime type {head!r}")
        return TypeApplication(TypeConstructorRef.builtin(head))
    if isinstance(binder, EffectBinder):
        effect = next((item for item in module.effects if item.ref.name == head), None)
        if effect is None:
            raise ValueError(f"unknown effect interface {head!r}")
        if len(effect.telescope) != len(parts):
            raise ValueError(
                f"effect {head!r} expects {len(effect.telescope)} arguments, "
                f"got {len(parts)}"
            )
        return effect.apply(
            tuple(
                _parse_static(module, expected, part)
                for expected, part in zip(effect.telescope, parts, strict=True)
            )
        )
    if parts:
        if not isinstance(binder.sort, UserIndexSort):
            raise ValueError(f"index sort {binder.sort!r} has no named constructors")
        arguments = tuple(
            _parse_static(module, IndexBinder("_", binder.sort), part) for part in parts
        )
        return IndexConstructor(head, arguments, binder.sort)  # type: ignore[arg-type]
    if isinstance(binder.sort, NatSort) and text.isdigit():
        return IndexLiteral(int(text), binder.sort)
    if isinstance(binder.sort, UserIndexSort):
        return IndexConstructor(head, (), binder.sort)
    raise ValueError(f"cannot parse closed index term {text!r} for {binder.sort!r}")


def _static_application(text: str) -> tuple[str, tuple[str, ...]]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("static terms cannot be empty")
    if "[" not in stripped:
        return stripped, ()
    if not stripped.endswith("]"):
        raise ValueError(f"malformed static application {text!r}")
    head, body = stripped.split("[", 1)
    body = body[:-1]
    parts: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(body):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth < 0:
                raise ValueError(f"malformed static application {text!r}")
        elif character == "," and depth == 0:
            parts.append(body[start:index].strip())
            start = index + 1
    if depth:
        raise ValueError(f"malformed static application {text!r}")
    parts.append(body[start:].strip())
    if not head or any(not part for part in parts):
        raise ValueError(f"malformed static application {text!r}")
    return head.strip(), tuple(parts)


def _binder_sort(binder: TypeBinder | IndexBinder | EffectBinder) -> str:
    if isinstance(binder, TypeBinder):
        return "Type" if getattr(binder.kind, "tag", "") == "type" else "Effect"
    if isinstance(binder, IndexBinder):
        return getattr(binder.sort, "name", getattr(binder.sort, "tag", "index"))
    return "Effect"


def _specialize_computation(
    computation: NamedComputation,
    substitution: StaticSubstitution,
) -> tuple[tuple[Local, ...], Computation]:
    """Apply a named telescope capture-freely to every runtime-relevant term."""

    local_map: dict[Local, Local] = {}
    parameters: list[Local] = []
    for parameter in computation.parameters:
        specialized = Local(
            parameter.name, substitute_type(parameter.type, substitution)
        )
        local_map[parameter] = specialized
        parameters.append(specialized)

    def evidence(item):  # type: ignore[no-untyped-def]
        return replace(
            item,
            equality=replace(
                item.equality,
                left=substitute_static(item.equality.left, substitution),
                right=substitute_static(item.equality.right, substitution),
            ),
        )

    def value(item: Value, locals_: Mapping[Local, Local]) -> Value:
        if isinstance(item, Var):
            return Var(locals_.get(item.local, item.local))
        if isinstance(item, LiteralValue):
            return LiteralValue(item.value, substitute_type(item.type, substitution))
        if isinstance(item, ConstructorValue):
            return ConstructorValue(
                item.constructor,
                tuple(
                    substitute_static(argument, substitution)
                    for argument in item.static_arguments
                ),
                tuple(value(field, locals_) for field in item.fields),
                substitute_type(item.result_type, substitution),
            )
        if isinstance(item, EvidenceValue):
            return EvidenceValue(evidence(item.evidence))
        if isinstance(item, AttachmentRef):
            return AttachmentRef(
                item.attachment, substitute_type(item.type, substitution)
            )
        if isinstance(item, TransportValue):
            return TransportValue(
                evidence(item.evidence),
                value(item.value, locals_),
                substitute_type(item.target_type, substitution),
            )
        raise TypeError(f"unsupported QIEC value {type(item).__name__}")

    def body(item: Computation, locals_: Mapping[Local, Local]) -> Computation:
        if isinstance(item, Return):
            return Return(value(item.value, locals_))
        if isinstance(item, Bind):
            binder = Local(
                item.binder.name,
                substitute_type(item.binder.type, substitution),
            )
            nested = dict(locals_)
            nested[item.binder] = binder
            return Bind(
                binder,
                body(item.first, locals_),
                body(item.then, nested),
            )
        if isinstance(item, Perform):
            request = item.request
            return Perform(
                replace(
                    request,
                    effect=substitute_effect(request.effect, substitution),
                    static_arguments=tuple(
                        substitute_static(argument, substitution)
                        for argument in request.static_arguments
                    ),
                    arguments=tuple(
                        value(argument, locals_) for argument in request.arguments
                    ),
                    result_type=substitute_type(request.result_type, substitution),
                )
            )
        if isinstance(item, Handle):
            return Handle(
                item.instance,
                item.handler,
                body(item.computation, locals_),
                tuple(
                    substitute_static(argument, substitution)
                    for argument in item.static_arguments
                ),
            )
        if isinstance(item, Case):
            branches: list[CaseBranch] = []
            for branch in item.branches:
                nested = dict(locals_)
                fields: list[Local] = []
                for field in branch.fields:
                    specialized = Local(
                        field.name,
                        substitute_type(field.type, substitution),
                    )
                    nested[field] = specialized
                    fields.append(specialized)
                branches.append(
                    CaseBranch(
                        branch.constructor,
                        tuple(
                            substitute_static(argument, substitution)
                            for argument in branch.static_arguments
                        ),
                        tuple(fields),
                        body(branch.body, nested),
                        branch.scope,
                    )
                )
            return Case(
                value(item.scrutinee, locals_),
                CaseMotive(
                    item.motive.indices,
                    substitute_type(item.motive.result_type, substitution),
                ),
                tuple(branches),
            )
        raise TypeError(f"unsupported QIEC computation {type(item).__name__}")

    return tuple(parameters), body(computation.body, local_map)


def _validator_for(
    type_: TypeExpr,
    providers: tuple[RuntimeProvider, ...],
) -> RuntimeValidator | None:
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
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuplify(item) for key, item in value.items()}
    return value


def _core_validator(type_: TypeExpr) -> RuntimeValidator | None:
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
            return isinstance(value, RuntimeConstructor) and value.result_type == type_

        return constructor
    if isinstance(type_, EqualityType):
        from quivers.qiec.evidence import BranchGiven, Reflexivity

        return lambda value: isinstance(value, Reflexivity | BranchGiven)
    if isinstance(type_, FunctionType):
        return callable
    if isinstance(type_, TypeVariable):
        return None
    return None


def _fail(
    code: str,
    message: str,
    computation: str | None = None,
    origin: SourceOrigin | None = None,
) -> NoReturn:
    raise ExecutionFailure(ExecutionDiagnostic(code, message, computation, origin))


def _render_type(type_: TypeExpr) -> str:
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
