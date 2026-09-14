"""Small-step reference evaluator for the Quivers Indexed Effect Core.

The evaluator is deliberately an interpreter, not an optimized execution
engine.  It gives ``Return``, ``Bind``, ``Perform``, and ``Handle`` one
operational meaning against which lowerings and backend runtimes can be tested.
Core terms remain serializable: host values and executable handler clauses live
only in :class:`RuntimeAttachments` and are addressed by stable QIEC IDs.

Handlers are lexical and deep.  A resumption reinstalls the matched handler and
every inner continuation frame, whereas a handler clause is evaluated only in
the outer context.  This distinction is easiest to make explicit with the CEK
machine below: handling an operation splits the continuation at its matching
handler boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING
from quivers.qiec.effects import (
    EffectRequest,
    HandlerDef,
    ResumptionGrade,
)
from quivers.qiec.identifiers import (
    AttachmentId,
    EffectInstanceId,
    HandlerId,
    OperationId,
)
from quivers.qiec.terms import (
    AttachmentRef,
    Bind,
    Case,
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
from quivers.qiec.substitution import (
    instantiate_telescope,
    substitute_effect,
    substitute_row,
    substitute_type,
)
from quivers.qiec.types import StaticArgument, TypeExpr

if TYPE_CHECKING:
    from quivers.qiec.checking import KernelRegistry


type RuntimeValidator = Callable[[object], bool | None]
type OperationClause = Callable[
    ["RuntimeRequest", "Resumption", "ClauseContext"],
    object,
]
type ReturnClause = Callable[[object, "ClauseContext"], object]
type ResponseHook = Callable[[object], object]
type EvaluationTraceHook = Callable[[str, Mapping[str, object]], None]


class EvaluationError(RuntimeError):
    """Base class for failures at the typed-core/runtime boundary."""


class MissingAttachmentError(EvaluationError):
    """A stable runtime reference has no value or handler attachment."""


class RuntimeTypeMismatch(EvaluationError):
    """A host result failed the validator for its QIEC result type."""


class UnhandledEffectError(EvaluationError):
    """No enclosing lexical handler accepts an effect request.

    Parameters
    ----------
    request
        The request nothing handled; kept on the error as ``request``.
    """

    def __init__(self, request: EffectRequest) -> None:
        super().__init__(
            "unhandled QIEC operation "
            f"{request.operation} on instance {request.instance}"
        )
        self.request = request


class InvalidHandlerError(EvaluationError):
    """A runtime handler contradicts its checked structural signature."""


@dataclass(frozen=True, slots=True)
class HandlerManifest:
    """Immutable checked handler definitions accepted by an evaluator.

    Construct this from a :class:`KernelRegistry` after its declarations have
    passed kernel registration.  Runtime clause objects are then accepted only
    when their complete structural definitions equal the manifest entry, not
    merely when their stable IDs collide.

    Parameters
    ----------
    definitions
        The checked handler definitions by identity.
    """

    definitions: Mapping[HandlerId, HandlerDef]

    def __post_init__(self) -> None:
        """Freeze the definitions behind a read-only view.

        The manifest is what a runtime resolves a handler through, so it
        is copied and wrapped rather than aliased: a caller mutating the
        mapping it passed in would otherwise change which handler a
        running computation reaches.
        """
        object.__setattr__(
            self,
            "definitions",
            MappingProxyType(dict(self.definitions)),
        )

    @classmethod
    def from_registry(cls, registry: KernelRegistry) -> HandlerManifest:
        """Build a manifest from a checked registry.

        Parameters
        ----------
        registry : KernelRegistry
            A registry whose handlers have already been checked.

        Returns
        -------
        HandlerManifest
            A manifest over those handlers.
        """
        return cls(registry.handlers)


class ResumptionUsageError(EvaluationError):
    """A handler clause used its resumption outside the declared grade."""


class NonDuplicableContinuationError(ResumptionUsageError):
    """An unrestricted clause captured state that cannot safely be copied."""


@dataclass(frozen=True, slots=True)
class RuntimeConstructor:
    """Erased runtime representation of a checked constructor value.

    Parameters
    ----------
    constructor
        The constructor's stable identity.
    static_arguments
        The constructor's telescope instantiation, as evaluated static data.
    fields
        The evaluated field values, in declaration order.
    result_type
        The family type the value inhabits.
    """

    constructor: object
    static_arguments: tuple[object, ...]
    fields: tuple[object, ...]
    result_type: TypeExpr


@dataclass(frozen=True, slots=True)
class AttachmentBinding:
    """One host value attached to an otherwise serializable core term.

    ``duplicable`` is a semantic assertion about the host value, not merely a
    promise that ``copy.copy`` happens to work.  Multi-shot resumptions reject
    captured attachment values unless this bit was supplied explicitly.

    Parameters
    ----------
    value
        The host value.
    type
        The type the value inhabits.
    validator
        The predicate that checks a host value against ``type``.
    duplicable
        Whether an unrestricted resumption may capture the value.
    """

    value: object
    type: TypeExpr
    validator: RuntimeValidator
    duplicable: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeClause:
    """Executable operation clause and validator for the resumed value.

    Parameters
    ----------
    invoke
        The clause body, called with the request, its resumption, and a
        clause context.
    result_validator
        The predicate every value resumed through this clause must pass.
    """

    invoke: OperationClause
    result_validator: RuntimeValidator


def _identity_return(value: object, _context: ClauseContext) -> object:
    """The default return clause: hand the value back unchanged.

    Parameters
    ----------
    value : object
        What the handled computation returned.
    _context : ClauseContext
        The clause context, unused by the identity.

    Returns
    -------
    object
        The value, unchanged. A handler that does not answer returns
        specially behaves as though it were not there for them.
    """
    return value


def _accept(_value: object) -> bool:
    """The default validator: accept anything.

    Parameters
    ----------
    _value : object
        The value to validate.

    Returns
    -------
    bool
        Always True. A clause that declares no runtime validator is
        checked statically and needs no further gate.
    """
    return True


@dataclass(slots=True)
class RuntimeHandler:
    """Executable attachment for a structural :class:`HandlerDef`.

    The checked definition contains coverage, grades, input/output types, and
    introduced effects.  This attachment contains only process-local code and
    state.  It is never serialized as QIEC.

    Parameters
    ----------
    definition
        The checked declaration this attachment implements.
    clauses
        One runtime clause per covered operation, by identity.
    return_clause
        How the handler answers the computation's return; the identity
        return by default.
    input_validator
        The predicate the handled computation's result must pass.
    output_validator
        The predicate the handler's answer must pass.
    duplicable_context
        Whether an unrestricted resumption may capture an installation of
        this handler.
    mutable_context
        Whether an installation carries state; such a handler needs a
        ``context_factory`` so installations do not share it.
    fork_context
        Builds an independent copy of an installation for another shot;
        required when the context is both mutable and duplicable.
    context_factory
        Builds a fresh installation each time the handler is installed.
    on_enter
        Called when an installation is entered.
    on_exit
        Called once when the handled computation completes.
    on_drop
        Called once when the handled computation is abandoned instead;
        exclusive with ``on_exit``.
    """

    definition: HandlerDef
    clauses: Mapping[OperationId, RuntimeClause]
    return_clause: ReturnClause = _identity_return
    input_validator: RuntimeValidator = _accept
    output_validator: RuntimeValidator = _accept
    duplicable_context: bool = False
    mutable_context: bool = False
    fork_context: Callable[["RuntimeHandler"], "RuntimeHandler"] | None = None
    context_factory: Callable[[], "RuntimeHandler"] | None = None
    on_enter: Callable[[], None] | None = None
    on_exit: Callable[[], None] | None = None
    on_drop: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        """Check the executable clauses against the checked declaration.

        Raises
        ------
        InvalidHandlerError
            If the runtime implements an operation the declaration does
            not cover, or leaves a declared one unimplemented. Both
            directions matter: an extra clause would never be dispatched
            to, and a missing one would fail only when that operation was
            first performed, perhaps long after the handler was
            installed.
        """
        structural = {clause.operation for clause in self.definition.clauses}
        executable = set(self.clauses)
        unknown = executable - structural
        if unknown:
            rendered = ", ".join(str(operation) for operation in sorted(unknown))
            raise InvalidHandlerError(
                f"runtime handler {self.definition.name!r} implements undeclared "
                f"clauses: {rendered}"
            )
        missing = structural - executable
        if missing:
            rendered = ", ".join(str(operation) for operation in sorted(missing))
            raise InvalidHandlerError(
                f"runtime handler {self.definition.name!r} has no attachment for "
                f"declared clauses: {rendered}"
            )
        if self.fork_context is not None and not self.duplicable_context:
            raise InvalidHandlerError(
                "a handler with fork_context must declare a duplicable context"
            )
        if (
            self.mutable_context
            and self.duplicable_context
            and self.fork_context is None
        ):
            raise InvalidHandlerError("a mutable duplicable handler needs fork_context")


@dataclass(slots=True)
class RuntimeAttachments:
    """Process-local values and handlers keyed by stable core identifiers.

    Parameters
    ----------
    values
        Attached host values by attachment identity.
    handlers
        Runtime handlers by handler identity.
    """

    values: dict[AttachmentId, AttachmentBinding] = field(default_factory=dict)
    handlers: dict[HandlerId, RuntimeHandler] = field(default_factory=dict)

    def bind_value(
        self,
        attachment: AttachmentId,
        value: object,
        type: TypeExpr,
        *,
        validator: RuntimeValidator = _accept,
        duplicable: bool = False,
    ) -> AttachmentRef:
        """Attach a host value and return its stable core reference.

        Parameters
        ----------
        attachment : AttachmentId
            The identity the core term refers to the value by.
        value : object
            The host value.
        type : TypeExpr
            The type the core ascribes to it.
        validator : RuntimeValidator
            Checks the value really inhabits that type. The default
            accepts anything, for a value the static check already
            settles.
        duplicable : bool
            Whether the value may be copied. This is an assertion about
            the value, not a guess: a multi-shot resumption capturing a
            non-duplicable attachment is refused, because copying it
            would give two continuations a shared mutable thing.

        Returns
        -------
        AttachmentRef
            The core reference to use in a term.

        Raises
        ------
        InvalidHandlerError
            If the identity is already bound to a different value, type,
            validator, or duplicability. Rebinding is refused rather than
            overwriting, since a term already built would silently start
            meaning something else.
        RuntimeValidationError
            If the validator rejects the value.
        """
        _validate(value, validator, type, f"attachment {attachment}")
        binding = AttachmentBinding(value, type, validator, duplicable)
        previous = self.values.get(attachment)
        if previous is not None:
            same_binding = (
                previous.value is value
                and previous.type == type
                and previous.validator is validator
                and previous.duplicable == duplicable
            )
            if not same_binding:
                raise InvalidHandlerError(
                    f"runtime attachment {attachment} is already bound differently"
                )
        self.values[attachment] = binding
        return AttachmentRef(attachment, type)

    def bind_handler(self, handler: RuntimeHandler) -> HandlerId:
        """Attach executable clauses to their structural handler identity.

        Parameters
        ----------
        handler : RuntimeHandler
            The executable handler, already checked against its
            declaration.

        Returns
        -------
        HandlerId
            The identity it was bound under.

        Raises
        ------
        InvalidHandlerError
            If a different handler is already bound to that identity.
        """
        id = handler.definition.id
        previous = self.handlers.get(id)
        if previous is not None and previous is not handler:
            raise InvalidHandlerError(f"runtime handler {id} is already bound")
        self.handlers[id] = handler
        return id


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    """An evaluated request supplied to a process-local handler clause.

    Parameters
    ----------
    core
        The stable request as written, before evaluation.
    arguments
        The evaluated value arguments, in order.
    resumption_path
        The ordinal of each resumption taken to reach this request.
    """

    core: EffectRequest
    arguments: tuple[object, ...]
    resumption_path: tuple[int, ...]

    @property
    def instance(self) -> EffectInstanceId:
        """The lexical instance this request was performed on.

        Returns
        -------
        EffectInstanceId
            The instance, which decides which handler answers.
        """
        return self.core.instance

    @property
    def operation(self) -> OperationId:
        """The operation requested.

        Returns
        -------
        OperationId
            The operation, which decides which clause answers.
        """
        return self.core.operation

    @property
    def address(
        self,
    ) -> tuple[
        str,
        tuple[tuple[str, str | int], ...],
        tuple[int, ...],
    ]:
        """Return the static, dynamic, and multi-shot address components.

        Returns
        -------
        tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]
            The source site, the dynamic address frames, and the
            resumption path. One request site reached twice, under a loop
            or under a resumed continuation, yields two addresses, which
            is what lets a trace name each occurrence separately.
        """
        static, dynamic, declared_path = self.core.origin.dynamic_key()
        return (static, dynamic, (*declared_path, *self.resumption_path))


@dataclass(frozen=True, slots=True)
class ClauseComputation:
    """Ask the evaluator to run a structural clause body outside the handler.

    Host values introduced by a handler must first be installed through
    :meth:`ClauseContext.attach`; arbitrary callback closures do not enter the
    computation tree.

    Parameters
    ----------
    computation
        The stable computation to run.
    bindings
        Locals the computation refers to, each with its evaluated value.
    """

    computation: Computation
    bindings: tuple[tuple[Local, object], ...] = ()


@dataclass(frozen=True, slots=True)
class Forward:
    """Forward a request to an outer handler, optionally observing its reply.

    Parameters
    ----------
    on_response
        Called with the outer handler's answer before it reaches the
        resumed computation, or ``None`` to forward blindly.
    """

    on_response: ResponseHook | None = None


@dataclass(frozen=True, slots=True)
class _BindFrame:
    """A continuation frame awaiting the first half of a bind.

    Parameters
    ----------
    binder
        The local the awaited value is bound to.
    then
        The computation run once the value arrives.
    environment
        The bindings in scope for ``then``.
    """

    binder: Local
    then: Computation
    environment: Mapping[Local, object]


@dataclass(frozen=True, slots=True)
class _HandlerLifecycle:
    """One installation's finalization state, shared by every frame for it.

    Parameters
    ----------
    handler
        The installed handler.
    finalized
        A one-element list holding whether ``exit`` or ``drop`` has fired;
        a list so frozen frames can share and update it.
    """

    handler: RuntimeHandler
    finalized: list[bool] = field(default_factory=lambda: [False])

    def exit(self) -> None:
        """Finalize the handler because its computation completed.

        Idempotent, and exclusive with `drop`: a handler is finalized
        once, and which hook fires records whether its computation ran to
        completion or was abandoned. A handler holding a resource needs
        that distinction to know whether to commit or discard.
        """
        if self.finalized[0]:
            return
        self.finalized[0] = True
        if self.handler.on_exit is not None:
            self.handler.on_exit()

    def drop(self) -> None:
        """Finalize the handler because its computation was abandoned.

        Idempotent, and exclusive with `exit`. Abandonment happens when
        an enclosing clause does not resume, so the handled computation
        never reaches its end.
        """
        if self.finalized[0]:
            return
        self.finalized[0] = True
        if self.handler.on_drop is not None:
            self.handler.on_drop()


@dataclass(frozen=True, slots=True)
class _HandlerFrame:
    """A stack frame marking an installed handler.

    Parameters
    ----------
    instance
        The effect instance the handler intercepts.
    lifecycle
        The installation's handler and finalization state.
    definition
        The checked declaration installed.
    environment
        The bindings in scope where the handler was installed.
    """

    instance: EffectInstanceId
    lifecycle: _HandlerLifecycle
    definition: HandlerDef
    environment: Mapping[Local, object]

    @property
    def handler(self) -> RuntimeHandler:
        """The executable handler this frame installed.

        Returns
        -------
        RuntimeHandler
            The handler, reached through its lifecycle so the two cannot
            drift apart.
        """
        return self.lifecycle.handler


@dataclass(frozen=True, slots=True)
class _ResponseHookFrame:
    """A stack frame observing the answer to a forwarded request.

    Parameters
    ----------
    hook
        Called with the answer as it passes back through this frame.
    """

    hook: ResponseHook


@dataclass(frozen=True, slots=True)
class _DelimiterFrame:
    """Stop a nested clause evaluation before consuming its outer context."""


type _Frame = _BindFrame | _HandlerFrame | _ResponseHookFrame | _DelimiterFrame


class ClauseContext:
    """Disciplined runtime services available to an attached handler clause.

    Parameters
    ----------
    evaluator
        The evaluator running the clause.
    environment
        The bindings in scope at the handler.
    outer_stack
        The continuation frames outside the handler.
    resumption_path
        The ordinal of each resumption taken to reach the clause.
    request
        The request the clause answers, or ``None`` in a return clause.
    """

    __slots__ = (
        "_environment",
        "_evaluator",
        "_outer_stack",
        "_request",
        "_resumption_path",
    )

    def __init__(
        self,
        evaluator: Evaluator,
        environment: Mapping[Local, object],
        outer_stack: tuple[_Frame, ...],
        resumption_path: tuple[int, ...],
        request: RuntimeRequest | None,
    ) -> None:
        self._evaluator = evaluator
        self._environment = environment
        self._outer_stack = outer_stack
        self._resumption_path = resumption_path
        self._request = request

    @property
    def request(self) -> RuntimeRequest | None:
        """The request this clause is answering.

        Returns
        -------
        RuntimeRequest or None
            The request, or None in a return clause, which answers a
            completed computation rather than an operation.
        """
        return self._request

    @property
    def attachments(self) -> RuntimeAttachments:
        """The host values available to this clause.

        Returns
        -------
        RuntimeAttachments
            The evaluator's attachment table.
        """
        return self._evaluator.attachments

    def evaluate(self, computation: Computation) -> object:
        """Evaluate a clause body in the outer handler context.

        A delimiter prevents the clause body from consuming the continuation
        outside the handled expression.  Effects performed here can reach only
        handlers outside the matched handler, which is the chosen deep-handler
        semantics.

        Parameters
        ----------
        computation : Computation
            The clause body to run.

        Returns
        -------
        object
            What the body produced.
        """
        stack = [*self._outer_stack, _DelimiterFrame()]
        return self._evaluator._evaluate_delimited(
            computation,
            dict(self._environment),
            stack,
            self._outer_stack,
            self._resumption_path,
        )

    def attach(
        self,
        value: object,
        type: TypeExpr,
        *,
        role: str,
        validator: RuntimeValidator = _accept,
        duplicable: bool = False,
    ) -> AttachmentRef:
        """Install a generated host value under a deterministic runtime ID.

        Parameters
        ----------
        value : object
            The host value to attach.
        type : TypeExpr
            The type the core ascribes to it.
        role : str
            What the value is for, which enters its derived identity.
        validator : RuntimeValidator
            Checks the value inhabits the type.
        duplicable : bool
            Whether the value may be copied by a multi-shot resumption.

        Returns
        -------
        AttachmentRef
            A reference usable in a core term. The identity derives from
            the run, the request, and the role rather than being
            generated, so a trace can name the same attachment twice.
        """
        if self._request is None:
            seed: tuple[object, ...] = (
                self._evaluator._run_serial,
                "return-clause",
                self._resumption_path,
                role,
            )
        else:
            seed = (
                self._evaluator._run_serial,
                self._request.address,
                str(self._request.operation),
                role,
            )
        id = AttachmentId.derive("runtime-generated", *seed)
        return self.attachments.bind_value(
            id,
            value,
            type,
            validator=validator,
            duplicable=duplicable,
        )


class Resumption:
    """Dynamically checked, delimited continuation supplied to one clause.

    Parameters
    ----------
    evaluator
        The evaluator that captured the continuation.
    request
        The request whose result the continuation awaits.
    grade
        How often the clause may invoke the continuation.
    captured_environment
        The bindings in scope at the request.
    captured_stack
        The continuation frames between the request and the handler.
    outer_stack
        The continuation frames outside the handler.
    validator
        The predicate every resumed value must pass.
    path
        The ordinal of each resumption taken to reach the clause.
    """

    __slots__ = (
        "_captured_environment",
        "_captured_stack",
        "_calls",
        "_closed",
        "_evaluator",
        "_grade",
        "_outer_stack",
        "_path",
        "_request",
        "_seed_stack",
        "_owned_start",
        "_validator",
    )

    def __init__(
        self,
        evaluator: Evaluator,
        request: RuntimeRequest,
        grade: ResumptionGrade,
        captured_environment: Mapping[Local, object],
        captured_stack: tuple[_Frame, ...],
        outer_stack: tuple[_Frame, ...],
        validator: RuntimeValidator,
        path: tuple[int, ...],
    ) -> None:
        self._evaluator = evaluator
        self._request = request
        self._grade = grade
        self._captured_environment = captured_environment
        self._captured_stack = captured_stack
        self._outer_stack = outer_stack
        self._validator = validator
        self._path = path
        self._calls = 0
        self._closed = False
        delimiters = tuple(
            index
            for index, frame in enumerate(captured_stack)
            if isinstance(frame, _DelimiterFrame)
        )
        self._owned_start = delimiters[-1] + 1 if delimiters else 0
        if grade is ResumptionGrade.UNRESTRICTED:
            evaluator._validate_duplicable_capture(
                captured_environment,
                captured_stack[self._owned_start :],
            )
            # Snapshot before the clause body runs. Every shot, including the
            # first, forks from this untouched seed rather than a previous
            # branch or the live handler context.
            self._seed_stack = self._fork_owned(captured_stack)
        else:
            self._seed_stack = captured_stack

    @property
    def grade(self) -> ResumptionGrade:
        """The grade this resumption is held to.

        Returns
        -------
        ResumptionGrade
            The declared grade, checked statically and again here.
        """
        return self._grade

    @property
    def calls(self) -> int:
        """How many times this resumption has been invoked so far.

        Returns
        -------
        int
            The count, which the grade bounds.
        """
        return self._calls

    def __call__(self, value: object) -> object:
        """Resume the suspended computation with a value.

        The static analysis already rejects a clause that can exceed its
        grade. This guard stays because it protects against a foreign
        handler, whose body the analysis never saw, and against a
        compiler bug: a grade is what lets a runtime discard a
        continuation, so exceeding it is not recoverable.

        Parameters
        ----------
        value : object
            What the resumed operation supplies.

        Returns
        -------
        object
            What the resumed computation produced.

        Raises
        ------
        ResumptionUsageError
            If the grade is zero, or an affine or linear resumption is
            invoked more than once.
        NonDuplicableContinuationError
            If an unrestricted resumption would have to copy a captured
            value that is not duplicable.
        RuntimeValidationError
            If the value does not inhabit the operation's result type.
        """
        next_call = self._calls + 1
        if self._grade is ResumptionGrade.ZERO:
            raise ResumptionUsageError("a grade-0 clause cannot resume")
        if (
            self._grade in (ResumptionGrade.AFFINE, ResumptionGrade.LINEAR)
            and next_call > 1
        ):
            raise ResumptionUsageError(
                f"a {self._grade.value!r} resumption was invoked more than once"
            )
        _validate(
            value,
            self._validator,
            self._request.core.result_type,
            f"result of operation {self._request.operation}",
        )
        self._calls = next_call
        self._evaluator._emit_trace(
            "resumption.invoked",
            {
                "operation": str(self._request.operation),
                "instance": str(self._request.instance),
                "grade": self._grade.value,
                "shot": next_call - 1,
            },
        )
        captured = (
            list(self._fork_owned(self._seed_stack))
            if self._grade is ResumptionGrade.UNRESTRICTED
            else list(self._captured_stack)
        )
        # Outer handlers remain visible to effects performed by the resumed
        # continuation, but the delimiter prevents normal completion from
        # consuming the continuation outside the handled expression.
        stack = [*self._outer_stack, _DelimiterFrame(), *captured]
        protected = (
            *self._outer_stack,
            *captured[: self._owned_start],
        )
        try:
            return self._evaluator._drive_value(
                value,
                dict(self._captured_environment),
                stack,
                (*self._path, next_call - 1),
            )
        finally:
            # A normally returned shot has already exited every copied frame.
            # Exceptional or delimited exits leave some frames live.  Lifecycle
            # guards make this final sweep exact-once in either case.
            self._evaluator._drop_local_handlers(
                (*captured[self._owned_start :], *stack),
                protected,
            )

    def _check_completed(self) -> None:
        """Confirm a linear resumption was actually used.

        Raises
        ------
        ResumptionUsageError
            If a linear clause finished without resuming exactly once.
            Checked at completion rather than at the call, because
            failing to resume can only be observed once the clause is
            over.
        """
        if self._grade is ResumptionGrade.LINEAR and self._calls != 1:
            raise ResumptionUsageError(
                "a grade-'1' resumption must be invoked exactly once"
            )

    def _close_handled_capture(self) -> None:
        """Finalize continuations consumed by this handler clause.

        Idempotent. Every owned frame is dropped even if an earlier drop
        raised, and the first failure is re-raised afterwards, so one
        handler's failing finalizer cannot leave the rest of the captured
        stack un-finalized.

        Raises
        ------
        BaseException
            The first failure raised by any finalizer, after the rest
            have run.
        """
        if self._closed:
            return
        self._closed = True
        pending: list[BaseException] = []
        try:
            self._evaluator._drop_stack(self._captured_stack[self._owned_start :])
        except BaseException as error:
            pending.append(error)
        if self._grade is ResumptionGrade.UNRESTRICTED:
            try:
                self._evaluator._drop_stack(self._seed_stack[self._owned_start :])
            except BaseException as error:
                pending.append(error)
        if pending:
            raise pending[0]

    def _release_forward_seed(self) -> None:
        """Discard only an unrestricted snapshot when forwarding the live stack."""
        if self._grade is ResumptionGrade.UNRESTRICTED:
            self._evaluator._drop_stack(self._seed_stack[self._owned_start :])

    def _fork_owned(self, stack: tuple[_Frame, ...]) -> tuple[_Frame, ...]:
        """Copy the frames this clause owns, sharing the rest.

        A multi-shot resumption needs its own copy of what it captured,
        or two shots would share one handler's state. Frames below the
        owned region belong to an enclosing context and are shared rather
        than copied, since the clause does not control their lifetime.

        Parameters
        ----------
        stack : tuple[_Frame, ...]
            The stack to fork.

        Returns
        -------
        tuple[_Frame, ...]
            The shared prefix followed by forked copies of the owned
            frames.
        """
        prefix = stack[: self._owned_start]
        owned = stack[self._owned_start :]
        return (*prefix, *self._evaluator._fork_stack(owned, fork=True))


def _validate(
    value: object,
    validator: RuntimeValidator,
    expected: TypeExpr,
    subject: str,
) -> None:
    """Check a host value against the type the core ascribes to it.

    Parameters
    ----------
    value : object
        The host value.
    validator : RuntimeValidator
        The predicate to apply.
    expected : TypeExpr
        The type the core believes the value has, named in any
        diagnostic.
    subject : str
        What is being validated.

    Raises
    ------
    RuntimeTypeMismatch
        If the validator returns False, or raises. A raising validator is
        reported as a mismatch rather than propagating, so a caller need
        handle only one failure class at the boundary.
    """
    try:
        accepted = validator(value)
    except Exception as error:
        raise RuntimeTypeMismatch(
            f"validator raised for {subject} at type {expected!r}: {error}"
        ) from error
    if accepted is False:
        raise RuntimeTypeMismatch(
            f"{subject} does not inhabit the checked type {expected!r}: {value!r}"
        )


class Evaluator:
    """Execute QIEC computations with explicit runtime attachments.

    Parameters
    ----------
    attachments
        The host values and handlers available; empty by default.
    handler_manifest
        Checked handler definitions every attached handler must match
        exactly on each :meth:`evaluate`, or ``None`` to skip that check
        there; :meth:`evaluate_checked` always derives a manifest from
        its registry.
    trace_hook
        Called with each execution event's name and payload, or ``None``
        for no tracing.
    """

    def __init__(
        self,
        attachments: RuntimeAttachments | None = None,
        *,
        handler_manifest: HandlerManifest | None = None,
        trace_hook: EvaluationTraceHook | None = None,
    ) -> None:
        self.attachments = attachments or RuntimeAttachments()
        self.handler_manifest = handler_manifest
        self.trace_hook = trace_hook
        self._run_serial = 0

    def _emit_trace(self, event: str, detail: Mapping[str, object]) -> None:
        """Report one execution event, when a hook is installed.

        Parameters
        ----------
        event : str
            The event name.
        detail : Mapping[str, object]
            Its payload.
        """
        if self.trace_hook is not None:
            self.trace_hook(event, detail)

    def evaluate(
        self,
        computation: Computation,
        environment: Mapping[Local, object] | None = None,
    ) -> object:
        """Evaluate a closed or explicitly supplied computation.

        Parameters
        ----------
        computation : Computation
            The computation to run.
        environment : Mapping[Local, object] or None
            Values for the computation's free locals. None means it is
            closed.

        Returns
        -------
        object
            What the computation produced.

        Raises
        ------
        EvaluationError
            If a request reaches no handler, a handler is missing from
            the manifest, a resumption exceeds its grade, or a host value
            fails validation.
        """
        if self.handler_manifest is not None:
            self._validate_handler_manifest(self.handler_manifest)
        self._run_serial += 1
        stack: list[_Frame] = []
        try:
            return self._drive_computation(
                computation,
                dict(environment or {}),
                stack,
                (),
            )
        except BaseException:
            self._drop_stack(tuple(stack))
            raise

    def evaluate_checked(
        self,
        computation: Computation,
        registry: KernelRegistry,
        environment: Mapping[Local, object] | None = None,
    ) -> object:
        """Kernel-check a computation and fail closed on handler attachments.
        Parameters
        ----------
        computation : Computation
            The computation to run.
        registry : KernelRegistry
            Declarations to check it against.
        environment : Mapping[Local, object] or None
            Values for its free locals.

        Returns
        -------
        object
            What the computation produced.

        Raises
        ------
        KernelError
            If the computation does not check.
        EvaluationError
            If execution fails, as in `evaluate`.
        """
        from quivers.qiec.checking import CheckContext, infer_computation

        runtime_environment = dict(environment or {})
        context = CheckContext()
        for local in runtime_environment:
            context = context.extend(local)
        infer_computation(
            computation,
            registry,
            context,
        )
        manifest = HandlerManifest.from_registry(registry)
        self._validate_handler_manifest(manifest)
        self._run_serial += 1
        stack: list[_Frame] = []
        try:
            return self._drive_computation(
                computation,
                runtime_environment,
                stack,
                (),
            )
        except BaseException:
            self._drop_stack(tuple(stack))
            raise

    def evaluate_value(
        self,
        value: Value,
        environment: Mapping[Local, object] | None = None,
    ) -> object:
        """Erase and evaluate a checked QIEC value.
        Parameters
        ----------
        value : Value
            The value to evaluate.
        environment : Mapping[Local, object] or None
            Values for its free locals.

        Returns
        -------
        object
            The host representation of the value.
        """
        return self._value(value, environment or {})

    def _value(self, value: Value, environment: Mapping[Local, object]) -> object:
        """Reduce a core value to its host representation.

        Parameters
        ----------
        value : Value
            The core value.
        environment : Mapping[Local, object]
            Values for its free locals.

        Returns
        -------
        object
            The host value.

        Raises
        ------
        MissingAttachmentError
            If an attachment reference names nothing.
        RuntimeTypeMismatch
            If an attachment's bound type disagrees with the reference, or a host value
            fails its validator.
        EvaluationError
            If a local is unbound.
        """
        if isinstance(value, Var):
            try:
                return environment[value.local]
            except KeyError as error:
                raise EvaluationError(
                    f"unbound QIEC local {value.local.name!r}"
                ) from error
        if isinstance(value, LiteralValue):
            return value.value
        if isinstance(value, ConstructorValue):
            return RuntimeConstructor(
                value.constructor,
                value.static_arguments,
                tuple(self._value(field, environment) for field in value.fields),
                value.result_type,
            )
        if isinstance(value, EvidenceValue):
            return value.evidence
        if isinstance(value, AttachmentRef):
            try:
                binding = self.attachments.values[value.attachment]
            except KeyError as error:
                raise MissingAttachmentError(
                    f"no host value is attached at {value.attachment}"
                ) from error
            if binding.type != value.type:
                raise RuntimeTypeMismatch(
                    f"attachment {value.attachment} has type {binding.type!r}, "
                    f"but the core reference expects {value.type!r}"
                )
            _validate(
                binding.value,
                binding.validator,
                value.type,
                f"attachment {value.attachment}",
            )
            return binding.value
        if isinstance(value, TransportValue):
            # The kernel has already checked the evidence.  Transport is erased
            # dynamically but remains explicit in serialized QIEC.
            return self._value(value.value, environment)
        raise TypeError(f"unsupported QIEC value {type(value).__name__}")

    def _validate_handler_manifest(self, manifest: HandlerManifest) -> None:
        """Require every attached handler to match its declared signature.

        Parameters
        ----------
        manifest : HandlerManifest
            The manifest to check.

        Raises
        ------
        InvalidHandlerError
            If an attached handler's definition differs from the manifest's. A provider
            that supplied a handler for one signature must not answer for another,
            however alike the two look.
        """
        for id, runtime in self.attachments.handlers.items():
            expected = manifest.definitions.get(id)
            if expected is None:
                raise InvalidHandlerError(
                    f"runtime handler {id} is absent from the checked manifest"
                )
            if runtime.definition != expected:
                raise InvalidHandlerError(
                    f"runtime handler {id} does not match its checked definition"
                )

    def _drive_computation(
        self,
        computation: Computation,
        environment: dict[Local, object],
        stack: list[_Frame],
        resumption_path: tuple[int, ...],
    ) -> object:
        """Run the machine until the computation and its stack are finished.

        Explicitly stacked rather than recursive, so a deeply nested or
        recursive computation is bounded by memory rather than by the
        host's call depth.

        Parameters
        ----------
        computation : Computation
            The term to run.
        environment : Mapping[Local, object]
            Values in scope.
        stack : list[_Frame]
            The machine's frame stack, mutated in place.
        resumption_path : tuple[int, ...]
            Which shot of which resumption is running, for addressing.

        Returns
        -------
        object
            The final value.

        Raises
        ------
        EvaluationError
            If a request reaches no handler, or a runtime rule is broken.
        """
        current = computation
        env = environment
        while True:
            if isinstance(current, Return):
                return self._drive_value(
                    self._value(current.value, env),
                    env,
                    stack,
                    resumption_path,
                )
            if isinstance(current, Bind):
                stack.append(_BindFrame(current.binder, current.then, dict(env)))
                current = current.first
                continue
            if isinstance(current, Handle):
                try:
                    handler = self.attachments.handlers[current.handler]
                except KeyError as error:
                    raise MissingAttachmentError(
                        f"no runtime clauses are attached for handler {current.handler}"
                    ) from error
                definition = self._instantiate_handler_definition(
                    handler.definition,
                    current.static_arguments,
                )
                lifecycle = self._install_handler(handler)
                self._emit_trace(
                    "handler.entered",
                    {
                        "handler": definition.name,
                        "handler_id": str(current.handler),
                        "instance": str(current.instance),
                    },
                )
                stack.append(
                    _HandlerFrame(
                        current.instance,
                        lifecycle,
                        definition,
                        dict(env),
                    )
                )
                current = current.computation
                continue
            if isinstance(current, Case):
                scrutinee = self._value(current.scrutinee, env)
                if not isinstance(scrutinee, RuntimeConstructor):
                    raise EvaluationError(
                        "QIEC case scrutinee did not evaluate to a constructor"
                    )
                branch = next(
                    (
                        candidate
                        for candidate in current.branches
                        if candidate.constructor == scrutinee.constructor
                    ),
                    None,
                )
                if branch is None:
                    raise EvaluationError(
                        f"no reachable case branch for constructor {scrutinee.constructor}"
                    )
                if len(branch.fields) != len(scrutinee.fields):
                    raise EvaluationError(
                        "checked case branch/runtime constructor arity mismatch"
                    )
                env = dict(env)
                env.update(zip(branch.fields, scrutinee.fields, strict=True))
                current = branch.body
                continue
            if isinstance(current, Perform):
                runtime_request = RuntimeRequest(
                    current.request,
                    tuple(
                        self._value(argument, env)
                        for argument in current.request.arguments
                    ),
                    resumption_path,
                )
                self._emit_trace(
                    "operation.requested",
                    {
                        "operation": str(runtime_request.operation),
                        "instance": str(runtime_request.instance),
                        "address": runtime_request.address,
                    },
                )
                return self._dispatch(runtime_request, env, stack, len(stack) - 1)
            raise TypeError(f"unsupported QIEC computation {type(current).__name__}")

    def _drive_value(
        self,
        value: object,
        environment: dict[Local, object],
        stack: list[_Frame],
        resumption_path: tuple[int, ...],
    ) -> object:
        """Unwind frames with a returned value until the stack is empty.

        Parameters
        ----------
        value : object
            The value being returned.
        stack : list[_Frame]
            The frame stack, mutated in place.
        resumption_path : tuple[int, ...]
            The current resumption address.

        Returns
        -------
        object
            The value that survives to the bottom, after every frame has answered.

        Raises
        ------
        EvaluationError
            If a frame's finalizer or return clause fails.
        """
        current = value
        env = environment
        while stack:
            frame = stack.pop()
            if isinstance(frame, _DelimiterFrame):
                return current
            if isinstance(frame, _ResponseHookFrame):
                current = frame.hook(current)
                continue
            if isinstance(frame, _BindFrame):
                env = dict(frame.environment)
                env[frame.binder] = current
                return self._drive_computation(
                    frame.then,
                    env,
                    stack,
                    resumption_path,
                )
            if isinstance(frame, _HandlerFrame):
                handler = frame.handler
                try:
                    _validate(
                        current,
                        handler.input_validator,
                        frame.definition.input_type,
                        f"return input of handler {frame.definition.name!r}",
                    )
                    context = ClauseContext(
                        self,
                        frame.environment,
                        tuple(stack),
                        resumption_path,
                        None,
                    )
                    answer = handler.return_clause(current, context)
                    answer = self._resolve_clause_answer(answer, context)
                    _validate(
                        answer,
                        handler.output_validator,
                        frame.definition.output_type,
                        f"return result of handler {frame.definition.name!r}",
                    )
                except BaseException:
                    frame.lifecycle.drop()
                    raise
                frame.lifecycle.exit()
                self._emit_trace(
                    "handler.returned",
                    {
                        "handler": frame.definition.name,
                        "instance": str(frame.instance),
                    },
                )
                current = answer
                env = dict(frame.environment)
                continue
            raise AssertionError(f"unknown evaluator frame {frame!r}")
        return current

    def _dispatch(
        self,
        request: RuntimeRequest,
        environment: dict[Local, object],
        stack: list[_Frame],
        search_index: int,
    ) -> object:
        """Find the handler for a request and run its clause.

        The search runs outward from the request, so the innermost
        handler of the named instance answers, and a partial handler that
        does not cover the operation forwards to the next one out.

        Parameters
        ----------
        request : EffectRequest
            The request to answer.
        environment : Mapping[Local, object]
            Values in scope at the request.
        stack : list[_Frame]
            The frame stack, searched outward for a handler.
        search_index : int
            Where in the stack to begin looking, so a forwarded request
            resumes the search outside the handler that forwarded it
            rather than finding that handler again.

        Returns
        -------
        object
            What the clause produced.

        Raises
        ------
        UnhandledEffectError
            If no handler at or below `search_index` covers the request.
        ResumptionUsageError
            If the clause misuses its resumption.
        """
        index = search_index
        while index >= 0:
            frame = stack[index]
            if (
                not isinstance(frame, _HandlerFrame)
                or frame.instance != request.instance
            ):
                index -= 1
                continue
            handler = frame.handler
            definition = frame.definition
            if definition.effect != request.core.effect:
                raise InvalidHandlerError(
                    f"handler {definition.name!r} and request disagree about "
                    f"the interface of lexical instance {request.instance}"
                )
            structural_clause = definition.clause(request.operation)
            if structural_clause is None:
                if definition.total:
                    raise InvalidHandlerError(
                        f"total handler {definition.name!r} has no clause for "
                        f"operation {request.operation}"
                    )
                index -= 1
                continue

            runtime_clause = handler.clauses[request.operation]
            self._emit_trace(
                "operation.handled",
                {
                    "operation": str(request.operation),
                    "instance": str(request.instance),
                    "handler": definition.name,
                    "grade": structural_clause.grade.value,
                },
            )
            outer_stack = tuple(stack[:index])
            captured_stack = tuple(stack[index:])
            resumption = Resumption(
                self,
                request,
                structural_clause.grade,
                dict(environment),
                captured_stack,
                outer_stack,
                runtime_clause.result_validator,
                request.resumption_path,
            )
            context = ClauseContext(
                self,
                frame.environment,
                outer_stack,
                request.resumption_path,
                request,
            )
            try:
                answer = runtime_clause.invoke(request, resumption, context)
            except BaseException:
                resumption._close_handled_capture()
                raise
            if isinstance(answer, Forward):
                if resumption.calls:
                    resumption._close_handled_capture()
                    raise ResumptionUsageError(
                        "a forwarding clause cannot also invoke its resumption"
                    )
                if definition.total:
                    resumption._close_handled_capture()
                    raise InvalidHandlerError(
                        f"total handler {definition.name!r} attempted to forward"
                    )
                # Forwarding preserves the live captured continuation for the
                # outer handler, but an omega-only pristine snapshot is dead.
                resumption._release_forward_seed()
                if answer.on_response is not None:
                    stack.append(_ResponseHookFrame(answer.on_response))
                self._emit_trace(
                    "operation.forwarded",
                    {
                        "operation": str(request.operation),
                        "instance": str(request.instance),
                        "handler": definition.name,
                    },
                )
                index -= 1
                continue

            try:
                resumption._check_completed()
                answer = self._resolve_clause_answer(answer, context)
                _validate(
                    answer,
                    handler.output_validator,
                    definition.output_type,
                    f"operation result of handler {definition.name!r}",
                )
            finally:
                resumption._close_handled_capture()
            return self._drive_value(
                answer,
                dict(frame.environment),
                list(outer_stack),
                request.resumption_path,
            )
        self._emit_trace(
            "operation.unhandled",
            {
                "operation": str(request.operation),
                "instance": str(request.instance),
                "address": request.address,
            },
        )
        raise UnhandledEffectError(request.core)

    def _resolve_clause_answer(
        self,
        answer: object,
        context: ClauseContext,
    ) -> object:
        """Interpret a clause's answer, forwarding it when asked.

        Parameters
        ----------
        answer : object
            What the clause returned, which may be a forwarding request.
        context : ClauseContext
            The clause's runtime context, carrying the request it was
            answering and the stack outside the handler.

        Returns
        -------
        object
            The value the clause ultimately produced.

        Raises
        ------
        UnhandledEffectError
            If a forward finds no outer handler.
        """
        if not isinstance(answer, ClauseComputation):
            return answer
        env = dict(context._environment)
        env.update(answer.bindings)
        stack = [*context._outer_stack, _DelimiterFrame()]
        return self._evaluate_delimited(
            answer.computation,
            env,
            stack,
            context._outer_stack,
            context._resumption_path,
        )

    def _evaluate_delimited(
        self,
        computation: Computation,
        environment: dict[Local, object],
        stack: list[_Frame],
        protected: tuple[_Frame, ...],
        resumption_path: tuple[int, ...],
    ) -> object:
        """Evaluate a local computation and finalize only frames it installs.
        Parameters
        ----------
        computation : Computation
            The body to run.
        environment : Mapping[Local, object]
            Values in scope.
        stack : list[_Frame]
            The stack to run against, ending in a delimiter.
        protected : tuple[_Frame, ...]
            Frames the body borrows rather than owns, which must survive
            when it finishes.
        resumption_path : tuple[int, ...]
            The current resumption address.

        Returns
        -------
        object
            What the body produced. The delimiter stops normal completion from consuming
            the continuation outside the handled expression, which is what makes these
            deep handlers rather than shallow ones.
        """
        try:
            return self._drive_computation(
                computation,
                environment,
                stack,
                resumption_path,
            )
        finally:
            self._drop_local_handlers(tuple(stack), protected)

    def _validate_duplicable_capture(
        self,
        environment: Mapping[Local, object],
        stack: tuple[_Frame, ...],
    ) -> None:
        """Refuse a multi-shot capture of a value that cannot be copied.

        Parameters
        ----------
        environment : Mapping[Local, object]
            Values the continuation closed over.
        stack : tuple[_Frame, ...]
            The frames a multi-shot resumption would copy.

        Raises
        ------
        NonDuplicableContinuationError
            If a captured attachment is not marked duplicable. Copying it would give two
            shots a shared mutable value, which is a data race rather than two
            independent continuations.
        """
        for local, value in environment.items():
            if not self._duplicable_value(value):
                raise NonDuplicableContinuationError(
                    f"unrestricted resumption captures nonduplicable local {local.name!r}"
                )
        for frame in stack:
            if (
                isinstance(frame, _HandlerFrame)
                and not frame.handler.duplicable_context
            ):
                raise NonDuplicableContinuationError(
                    "unrestricted resumption captures nonduplicable handler "
                    f"{frame.handler.definition.name!r}"
                )

    def _duplicable_value(self, value: object) -> bool:
        """Whether one captured value may be copied for another shot.

        Parameters
        ----------
        value : object
            The captured value.

        Returns
        -------
        bool
            True when the value is a scalar, or an attachment explicitly marked
            duplicable. Duplicability is asserted rather than inferred, because
            `copy.copy` succeeding says nothing about whether copying is sound.
        """
        if value is None or isinstance(value, (bool, int, float, str, bytes)):
            return True
        if isinstance(value, tuple):
            return all(self._duplicable_value(item) for item in value)
        if isinstance(value, frozenset):
            return all(self._duplicable_value(item) for item in value)
        if isinstance(value, RuntimeConstructor):
            return all(self._duplicable_value(item) for item in value.fields)
        return any(
            binding.value is value and binding.duplicable
            for binding in self.attachments.values.values()
        )

    def _fork_stack(self, stack: tuple[_Frame, ...], *, fork: bool) -> list[_Frame]:
        """Copy a run of frames for an independent continuation.

        Parameters
        ----------
        stack : tuple[_Frame, ...]
            The frames to fork.
        fork : bool
            Whether to copy handler state, or share it.

        Returns
        -------
        tuple[_Frame, ...]
            The forked frames.

        Raises
        ------
        NonDuplicableContinuationError
            If a frame holds a value that cannot be copied.
        """
        if not fork:
            return list(stack)
        result: list[_Frame] = []
        try:
            for frame in stack:
                if not isinstance(frame, _HandlerFrame):
                    result.append(frame)
                    continue
                clone = frame.handler
                if frame.handler.fork_context is not None:
                    clone = frame.handler.fork_context(frame.handler)
                    if clone is frame.handler and frame.handler.mutable_context:
                        raise InvalidHandlerError(
                            f"fork_context for {frame.handler.definition.name!r} "
                            "returned the live mutable handler"
                        )
                    clone_lifecycle = _HandlerLifecycle(clone)
                    try:
                        if clone.definition != frame.handler.definition:
                            raise InvalidHandlerError(
                                f"fork_context for "
                                f"{frame.handler.definition.name!r} changed its "
                                "checked definition"
                            )
                        if clone.mutable_context and clone.fork_context is None:
                            raise InvalidHandlerError(
                                f"forked handler {clone.definition.name!r} "
                                "cannot be forked again"
                            )
                    except BaseException:
                        clone_lifecycle.drop()
                        raise
                else:
                    clone_lifecycle = _HandlerLifecycle(clone)
                result.append(
                    _HandlerFrame(
                        frame.instance,
                        clone_lifecycle,
                        frame.definition,
                        frame.environment,
                    )
                )
        except BaseException:
            self._drop_stack(tuple(result))
            raise
        return result

    @staticmethod
    def _install_handler(prototype: RuntimeHandler) -> _HandlerLifecycle:
        """Allocate one runtime context for a dynamic Handle installation.

        A handler with a context factory gets a fresh instance per
        installation, so two `handle` expressions over one declaration do
        not share state. One without is installed as it stands.

        Parameters
        ----------
        prototype : RuntimeHandler
            The attached handler, or the factory that makes one.

        Returns
        -------
        _HandlerLifecycle
            The installed handler paired with its finalization state, so
            it is exited or dropped exactly once.

        Raises
        ------
        InvalidHandlerError
            If a context factory returned a handler for a different
            checked definition, or if a handler holding mutable state was
            installed without a factory. The second is the one that
            matters: sharing one mutable handler between two
            installations would let them see each other's state.

        Notes
        -----
        A failure here drops the lifecycle before propagating, so a
        handler whose entry hook raised is still finalized rather than
        left half-installed.
        """
        handler = (
            prototype.context_factory()
            if prototype.context_factory is not None
            else prototype
        )
        lifecycle = _HandlerLifecycle(handler)
        try:
            if handler.definition != prototype.definition:
                raise InvalidHandlerError(
                    f"context factory for {prototype.definition.name!r} "
                    "changed its checked definition"
                )
            if handler is prototype and prototype.mutable_context:
                raise InvalidHandlerError(
                    f"mutable handler {prototype.definition.name!r} needs a "
                    "per-installation context_factory"
                )
            if handler.on_enter is not None:
                handler.on_enter()
        except BaseException:
            lifecycle.drop()
            raise
        return lifecycle

    @staticmethod
    def _instantiate_handler_definition(
        definition: HandlerDef,
        arguments: tuple[StaticArgument, ...],
    ) -> HandlerDef:
        """Apply one structural handler telescope at its dynamic installation.

        Parameters
        ----------
        definition : HandlerDef
            The declared handler.
        arguments : tuple[StaticArgument, ...]
            Static arguments supplied where it is installed.

        Returns
        -------
        HandlerDef
            The handler with its interface, input, output, and introduced
            row substituted, so the clause bodies are checked and run at
            the types this installation uses.

        Raises
        ------
        InvalidHandlerError
            If the arguments do not saturate the telescope, or are
            ill-kinded.
        """
        try:
            substitution = instantiate_telescope(
                definition.telescope,
                arguments,
            )
        except (TypeError, ValueError) as error:
            raise InvalidHandlerError(
                f"invalid static arguments for handler {definition.name!r}: {error}"
            ) from error
        return replace(
            definition,
            effect=substitute_effect(definition.effect, substitution),
            input_type=substitute_type(definition.input_type, substitution),
            output_type=substitute_type(definition.output_type, substitution),
            introduced=substitute_row(definition.introduced, substitution),
            telescope=(),
        )

    @staticmethod
    def _drop_local_handlers(
        candidates: tuple[_Frame, ...],
        protected: tuple[_Frame, ...],
    ) -> None:
        """Drop candidate handler frames except explicitly borrowed lifecycles.

        A forwarded or resumed continuation borrows frames it did not
        install, and those must outlive this unwind. Comparing lifecycle
        identity rather than frame equality is what distinguishes a
        borrowed frame from a structurally identical one this clause owns.

        Parameters
        ----------
        candidates : tuple[_Frame, ...]
            Frames being unwound.
        protected : tuple[_Frame, ...]
            Frames whose lifecycles are borrowed and must not be dropped.
        """
        protected_lifecycles = {
            id(frame.lifecycle)
            for frame in protected
            if isinstance(frame, _HandlerFrame)
        }
        Evaluator._drop_stack(
            tuple(
                frame
                for frame in candidates
                if not isinstance(frame, _HandlerFrame)
                or id(frame.lifecycle) not in protected_lifecycles
            )
        )

    @staticmethod
    def _drop_stack(stack: tuple[_Frame, ...]) -> None:
        """Finalize each captured handler context once, inner-first.

        Inner-first because a handler installed later may depend on one
        installed earlier, so it has to finish first. Every frame is
        attempted even if an earlier finalizer raised, and the first
        failure is re-raised afterwards, so one failing handler cannot
        leave the rest un-finalized.

        Parameters
        ----------
        stack : tuple[_Frame, ...]
            The frames to finalize.

        Raises
        ------
        BaseException
            The first failure any finalizer raised, after all have run.
        """
        pending: list[BaseException] = []
        for frame in reversed(stack):
            if not isinstance(frame, _HandlerFrame):
                continue
            try:
                frame.lifecycle.drop()
            except BaseException as error:
                pending.append(error)
        if pending:
            raise pending[0]


__all__ = [
    "AttachmentBinding",
    "ClauseComputation",
    "ClauseContext",
    "EvaluationError",
    "EvaluationTraceHook",
    "Evaluator",
    "Forward",
    "HandlerManifest",
    "InvalidHandlerError",
    "MissingAttachmentError",
    "NonDuplicableContinuationError",
    "OperationClause",
    "Resumption",
    "ResumptionUsageError",
    "ReturnClause",
    "RuntimeAttachments",
    "RuntimeClause",
    "RuntimeConstructor",
    "RuntimeHandler",
    "RuntimeRequest",
    "RuntimeTypeMismatch",
    "RuntimeValidator",
    "UnhandledEffectError",
]
