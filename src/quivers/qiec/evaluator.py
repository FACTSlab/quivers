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


class EvaluationError(RuntimeError):
    """Base class for failures at the typed-core/runtime boundary."""


class MissingAttachmentError(EvaluationError):
    """A stable runtime reference has no value or handler attachment."""


class RuntimeTypeMismatch(EvaluationError):
    """A host result failed the validator for its QIEC result type."""


class UnhandledEffectError(EvaluationError):
    """No enclosing lexical handler accepts an effect request."""

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
    """

    definitions: Mapping[HandlerId, HandlerDef]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "definitions",
            MappingProxyType(dict(self.definitions)),
        )

    @classmethod
    def from_registry(cls, registry: KernelRegistry) -> HandlerManifest:
        return cls(registry.handlers)


class ResumptionUsageError(EvaluationError):
    """A handler clause used its resumption outside the declared grade."""


class NonDuplicableContinuationError(ResumptionUsageError):
    """An unrestricted clause captured state that cannot safely be copied."""


@dataclass(frozen=True, slots=True)
class RuntimeConstructor:
    """Erased runtime representation of a checked constructor value."""

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
    """

    value: object
    type: TypeExpr
    validator: RuntimeValidator
    duplicable: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeClause:
    """Executable operation clause and validator for the resumed value."""

    invoke: OperationClause
    result_validator: RuntimeValidator


def _identity_return(value: object, _context: ClauseContext) -> object:
    return value


def _accept(_value: object) -> bool:
    return True


@dataclass(slots=True)
class RuntimeHandler:
    """Executable attachment for a structural :class:`HandlerDef`.

    The checked definition contains coverage, grades, input/output types, and
    introduced effects.  This attachment contains only process-local code and
    state.  It is never serialized as QIEC.
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
    """Process-local values and handlers keyed by stable core identifiers."""

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
        """Attach a host value and return its stable core reference."""
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
        """Attach executable clauses to their structural handler identity."""
        id = handler.definition.id
        previous = self.handlers.get(id)
        if previous is not None and previous is not handler:
            raise InvalidHandlerError(f"runtime handler {id} is already bound")
        self.handlers[id] = handler
        return id


@dataclass(frozen=True, slots=True)
class RuntimeRequest:
    """An evaluated request supplied to a process-local handler clause."""

    core: EffectRequest
    arguments: tuple[object, ...]
    resumption_path: tuple[int, ...]

    @property
    def instance(self) -> EffectInstanceId:
        return self.core.instance

    @property
    def operation(self) -> OperationId:
        return self.core.operation

    @property
    def address(
        self,
    ) -> tuple[
        str,
        tuple[tuple[str, str | int], ...],
        tuple[int, ...],
    ]:
        """Return the static, dynamic, and multi-shot address components."""
        static, dynamic, declared_path = self.core.origin.dynamic_key()
        return (static, dynamic, (*declared_path, *self.resumption_path))


@dataclass(frozen=True, slots=True)
class ClauseComputation:
    """Ask the evaluator to run a structural clause body outside the handler.

    Host values introduced by a handler must first be installed through
    :meth:`ClauseContext.attach`; arbitrary callback closures do not enter the
    computation tree.
    """

    computation: Computation
    bindings: tuple[tuple[Local, object], ...] = ()


@dataclass(frozen=True, slots=True)
class Forward:
    """Forward a request to an outer handler, optionally observing its reply."""

    on_response: ResponseHook | None = None


@dataclass(frozen=True, slots=True)
class _BindFrame:
    binder: Local
    then: Computation
    environment: Mapping[Local, object]


@dataclass(frozen=True, slots=True)
class _HandlerLifecycle:
    handler: RuntimeHandler
    finalized: list[bool] = field(default_factory=lambda: [False])

    def exit(self) -> None:
        if self.finalized[0]:
            return
        self.finalized[0] = True
        if self.handler.on_exit is not None:
            self.handler.on_exit()

    def drop(self) -> None:
        if self.finalized[0]:
            return
        self.finalized[0] = True
        if self.handler.on_drop is not None:
            self.handler.on_drop()


@dataclass(frozen=True, slots=True)
class _HandlerFrame:
    instance: EffectInstanceId
    lifecycle: _HandlerLifecycle
    definition: HandlerDef
    environment: Mapping[Local, object]

    @property
    def handler(self) -> RuntimeHandler:
        return self.lifecycle.handler


@dataclass(frozen=True, slots=True)
class _ResponseHookFrame:
    hook: ResponseHook


@dataclass(frozen=True, slots=True)
class _DelimiterFrame:
    """Stop a nested clause evaluation before consuming its outer context."""


type _Frame = _BindFrame | _HandlerFrame | _ResponseHookFrame | _DelimiterFrame


class ClauseContext:
    """Disciplined runtime services available to an attached handler clause."""

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
        return self._request

    @property
    def attachments(self) -> RuntimeAttachments:
        return self._evaluator.attachments

    def evaluate(self, computation: Computation) -> object:
        """Evaluate a clause body in the outer handler context.

        A delimiter prevents the clause body from consuming the continuation
        outside the handled expression.  Effects performed here can reach only
        handlers outside the matched handler, which is the chosen deep-handler
        semantics.
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
        """Install a generated host value under a deterministic runtime ID."""
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
    """Dynamically checked, delimited continuation supplied to one clause."""

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
        return self._grade

    @property
    def calls(self) -> int:
        return self._calls

    def __call__(self, value: object) -> object:
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
        if self._grade is ResumptionGrade.LINEAR and self._calls != 1:
            raise ResumptionUsageError(
                "a grade-'1' resumption must be invoked exactly once"
            )

    def _close_handled_capture(self) -> None:
        """Finalize continuations consumed by this handler clause."""
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
        prefix = stack[: self._owned_start]
        owned = stack[self._owned_start :]
        return (*prefix, *self._evaluator._fork_stack(owned, fork=True))


def _validate(
    value: object,
    validator: RuntimeValidator,
    expected: TypeExpr,
    subject: str,
) -> None:
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
    """Execute QIEC computations with explicit runtime attachments."""

    def __init__(
        self,
        attachments: RuntimeAttachments | None = None,
        *,
        handler_manifest: HandlerManifest | None = None,
    ) -> None:
        self.attachments = attachments or RuntimeAttachments()
        self.handler_manifest = handler_manifest
        self._run_serial = 0

    def evaluate(
        self,
        computation: Computation,
        environment: Mapping[Local, object] | None = None,
    ) -> object:
        """Evaluate a closed or explicitly supplied computation."""
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
        """Kernel-check a computation and fail closed on handler attachments."""
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
        """Erase and evaluate a checked QIEC value."""
        return self._value(value, environment or {})

    def _value(self, value: Value, environment: Mapping[Local, object]) -> object:
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
                return self._dispatch(runtime_request, env, stack, len(stack) - 1)
            raise TypeError(f"unsupported QIEC computation {type(current).__name__}")

    def _drive_value(
        self,
        value: object,
        environment: dict[Local, object],
        stack: list[_Frame],
        resumption_path: tuple[int, ...],
    ) -> object:
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
        raise UnhandledEffectError(request.core)

    def _resolve_clause_answer(
        self,
        answer: object,
        context: ClauseContext,
    ) -> object:
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
        """Evaluate a local computation and finalize only frames it installs."""
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
        """Allocate one runtime context for a dynamic Handle installation."""
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
        """Apply one structural handler telescope at its dynamic installation."""
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
        """Drop candidate handler frames except explicitly borrowed lifecycles."""
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
        """Finalize each captured handler context once, inner-first."""
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
