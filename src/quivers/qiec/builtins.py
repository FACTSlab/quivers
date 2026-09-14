"""Canonical QIEC prelude effects and their small reference handlers.

The declarations in this module are stable structural data.  The factory
functions produce process-local :class:`~quivers.qiec.evaluator.RuntimeHandler`
attachments; samplers, distributions, semirings, and observation tables
consequently never become serializable QIEC fields.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import operator
from typing import cast

from quivers.qiec.effects import (
    ArgumentDef,
    EffectDef,
    EffectRequest,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    OperationDef,
    ResumptionGrade,
    RowEntry,
)
from quivers.qiec.evaluator import (
    ClauseContext,
    Forward,
    InvalidHandlerError,
    Resumption,
    RuntimeClause,
    RuntimeHandler,
    RuntimeRequest,
    RuntimeTypeMismatch,
    RuntimeValidator,
)
from quivers.qiec.identifiers import (
    EffectId,
    EffectInstanceId,
    HandlerId,
    OperationId,
    SourceOrigin,
    TypeId,
)
from quivers.qiec.kinds import TypeBinder
from quivers.qiec.terms import Perform
from quivers.qiec.types import (
    UNIT,
    EffectRef,
    TypeApplication,
    TypeConstructorRef,
    TypeExpr,
    TypeVariable,
)


def _constructor(name: str, *binders: TypeBinder) -> TypeConstructorRef:
    """A type constructor in the prelude's own namespace.

    Parameters
    ----------
    name : str
        The constructor's name.
    *binders : TypeBinder
        Its kinding telescope.

    Returns
    -------
    TypeConstructorRef
        The constructor. Its identity derives from the ``builtin``
        namespace, so `Site` means the same type in every module without
        being declared in any of them.
    """
    return TypeConstructorRef(
        TypeId.derive("builtin", name),
        name,
        tuple(binders),
    )


_A_BINDER = TypeBinder("a")
_K_BINDER = TypeBinder("k")
_P_BINDER = TypeBinder("payload")
_S_BINDER = TypeBinder("state")
_ANSWER_BINDER = TypeBinder("answer")
A = TypeVariable("a")
K = TypeVariable("k")
PAYLOAD = TypeVariable("payload")
STATE = TypeVariable("state")
ANSWER = TypeVariable("answer")

SITE_CONSTRUCTOR = _constructor("Site", _A_BINDER)
SAMPLEABLE_CONSTRUCTOR = _constructor("Sampleable", _A_BINDER)
CHOICES_CONSTRUCTOR = _constructor("Choices", _A_BINDER)
CHOICE_RESULTS_CONSTRUCTOR = _constructor("ChoiceResults", _A_BINDER)
STATE_RESULT_CONSTRUCTOR = _constructor("StateResult", _A_BINDER, _S_BINDER)
WEIGHTED_RESULT_CONSTRUCTOR = _constructor("WeightedResult", _A_BINDER, _K_BINDER)

SITE_A = TypeApplication(SITE_CONSTRUCTOR, (A,))
SAMPLEABLE_A = TypeApplication(SAMPLEABLE_CONSTRUCTOR, (A,))
CHOICES_A = TypeApplication(CHOICES_CONSTRUCTOR, (A,))
CHOICE_RESULTS_A = TypeApplication(CHOICE_RESULTS_CONSTRUCTOR, (A,))
STATE_RESULT = TypeApplication(STATE_RESULT_CONSTRUCTOR, (ANSWER, STATE))
WEIGHTED_RESULT = TypeApplication(WEIGHTED_RESULT_CONSTRUCTOR, (ANSWER, K))
LOG_WEIGHT = TypeApplication(_constructor("LogWeight"))


def _effect_ref(name: str) -> EffectRef:
    """An unsaturated reference to a prelude effect interface.

    Parameters
    ----------
    name : str
        The interface's name.

    Returns
    -------
    EffectRef
        The declaration reference, in the ``prelude`` namespace so it
        names the same interface in every module.
    """
    return EffectRef(EffectId.derive("prelude", name), name)


def _operation_id(effect: EffectRef, name: str) -> OperationId:
    """The identity of one operation on a prelude interface.

    Parameters
    ----------
    effect : EffectRef
        The interface that declares it.
    name : str
        The operation's name.

    Returns
    -------
    OperationId
        The identity, derived from the interface and the name, so two
        interfaces may each declare `get` without collision.
    """
    return OperationId.derive(str(effect.id), name)


RANDOM = _effect_ref("Random")
RANDOM_SAMPLE = _operation_id(RANDOM, "sample")
RANDOM_EFFECT = EffectDef(
    RANDOM,
    (),
    (
        OperationDef(
            RANDOM_SAMPLE,
            "sample",
            (_A_BINDER,),
            (ArgumentDef("site", SITE_A), ArgumentDef("sampleable", SAMPLEABLE_A)),
            A,
        ),
    ),
)

SCORE = _effect_ref("Score")
SCORE_ADD = _operation_id(SCORE, "add")
SCORE_EFFECT = EffectDef(
    SCORE,
    (),
    (OperationDef(SCORE_ADD, "add", (), (ArgumentDef("weight", LOG_WEIGHT),), UNIT),),
)

STATE_EFFECT_REF = _effect_ref("State")
STATE_GET = _operation_id(STATE_EFFECT_REF, "get")
STATE_PUT = _operation_id(STATE_EFFECT_REF, "put")
STATE_EFFECT = EffectDef(
    STATE_EFFECT_REF,
    (_S_BINDER,),
    (
        OperationDef(STATE_GET, "get", (), (), STATE),
        OperationDef(
            STATE_PUT,
            "put",
            (),
            (ArgumentDef("value", STATE),),
            UNIT,
        ),
    ),
)

ABORT = _effect_ref("Abort")
ABORT_ABORT = _operation_id(ABORT, "abort")
ABORT_EFFECT = EffectDef(
    ABORT,
    (_P_BINDER,),
    (
        OperationDef(
            ABORT_ABORT,
            "abort",
            (_A_BINDER,),
            (ArgumentDef("error", PAYLOAD),),
            A,
        ),
    ),
)

CHOOSE = _effect_ref("Choose")
CHOOSE_CHOOSE = _operation_id(CHOOSE, "choose")
CHOOSE_EFFECT = EffectDef(
    CHOOSE,
    (),
    (
        OperationDef(
            CHOOSE_CHOOSE,
            "choose",
            (_A_BINDER,),
            (ArgumentDef("alternatives", CHOICES_A),),
            A,
        ),
    ),
)

WEIGHT = _effect_ref("Weight")
WEIGHT_ADD = _operation_id(WEIGHT, "add")
WEIGHT_EFFECT = EffectDef(
    WEIGHT,
    (_K_BINDER,),
    (
        OperationDef(
            WEIGHT_ADD,
            "add",
            (),
            (ArgumentDef("weight", K),),
            UNIT,
        ),
    ),
)

BUILTIN_EFFECTS = (
    RANDOM_EFFECT,
    SCORE_EFFECT,
    STATE_EFFECT,
    ABORT_EFFECT,
    CHOOSE_EFFECT,
    WEIGHT_EFFECT,
)

_RUNTIME_ADD = cast(Callable[[object, object], object], operator.add)


def state_effect(state_type: TypeExpr) -> EffectRef:
    """Construct the concrete ``State[state_type]`` interface.

    Parameters
    ----------
    state_type : TypeExpr
        The type to instantiate the interface's binder at.

    Returns
    -------
    EffectRef
        The saturated application.

    Raises
    ------
    TypeError
        If the argument is not a type.
    ValueError
        If it is ill-kinded.
    """
    return STATE_EFFECT.apply((state_type,))


def abort_effect(error_type: TypeExpr) -> EffectRef:
    """Construct the concrete ``Abort[error_type]`` interface.

    Parameters
    ----------
    error_type : TypeExpr
        The type to instantiate the interface's binder at.

    Returns
    -------
    EffectRef
        The saturated application.

    Raises
    ------
    TypeError
        If the argument is not a type.
    ValueError
        If it is ill-kinded.
    """
    return ABORT_EFFECT.apply((error_type,))


def weight_effect(weight_type: TypeExpr) -> EffectRef:
    """Construct the concrete ``Weight[weight_type]`` interface.

    Parameters
    ----------
    weight_type : TypeExpr
        The type to instantiate the interface's binder at.

    Returns
    -------
    EffectRef
        The saturated application.

    Raises
    ------
    TypeError
        If the argument is not a type.
    ValueError
        If it is ill-kinded.
    """
    return WEIGHT_EFFECT.apply((weight_type,))


def _unit(value: object) -> bool:
    """Whether a host value inhabits `Unit`.

    Parameters
    ----------
    value : object
        The value to test.

    Returns
    -------
    bool
        True only for None, which is `Unit`'s single inhabitant.
    """
    return value is None


def _accepts(validator: RuntimeValidator, value: object) -> bool:
    """Whether a validator admits a value.

    Parameters
    ----------
    validator : RuntimeValidator
        The predicate to apply.
    value : object
        The value to test.

    Returns
    -------
    bool
        True unless the validator returned False exactly. A validator
        returning None reads as acceptance, so one written for its side
        effect does not reject everything.
    """
    return validator(value) is not False


def _pair_of(
    first: RuntimeValidator,
    second: RuntimeValidator,
) -> RuntimeValidator:
    def validate(value: object) -> bool:
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and _accepts(first, value[0])
            and _accepts(second, value[1])
        )

    return validate


def _tuple_of(item: RuntimeValidator) -> RuntimeValidator:
    def validate(value: object) -> bool:
        return isinstance(value, tuple) and all(_accepts(item, part) for part in value)

    return validate


def _handler_def(
    name: str,
    effect: EffectRef,
    clauses: tuple[tuple[OperationId, ResumptionGrade], ...],
    *,
    key: str,
    input_type: TypeExpr = ANSWER,
    output_type: TypeExpr = ANSWER,
    introduced: EffectRow = EffectRow(),
    total: bool = True,
    telescope: tuple[TypeBinder, ...] = (),
) -> HandlerDef:
    return HandlerDef(
        HandlerId.derive("prelude", name, key),
        name,
        effect,
        tuple(HandlerClauseDef(operation, grade) for operation, grade in clauses),
        input_type,
        output_type,
        introduced,
        total=total,
        telescope=telescope,
    )


def _generic_binders(
    *,
    answer_type: TypeExpr | None = None,
    state_type: TypeExpr | None = None,
    error_type: TypeExpr | None = None,
    weight_type: TypeExpr | None = None,
) -> tuple[TypeBinder, ...]:
    """Declare exactly the free prelude parameters used by a handler."""
    result: list[TypeBinder] = []
    if answer_type == ANSWER:
        result.append(_ANSWER_BINDER)
    if state_type == STATE:
        result.append(_S_BINDER)
    if error_type == PAYLOAD:
        result.append(_P_BINDER)
    if weight_type == K:
        result.append(_K_BINDER)
    return tuple(result)


def _expect_arguments(
    request: RuntimeRequest, count: int, handler: str
) -> tuple[object, ...]:
    if len(request.arguments) != count:
        raise InvalidHandlerError(
            f"{handler} expected {count} operation arguments, got "
            f"{len(request.arguments)}"
        )
    return request.arguments


def _require(
    value: object,
    validator: RuntimeValidator,
    type: TypeExpr,
    subject: str,
) -> None:
    try:
        accepted = validator(value)
    except Exception as error:
        raise RuntimeTypeMismatch(f"validator raised for {subject}: {error}") from error
    if accepted is False:
        raise RuntimeTypeMismatch(f"{subject} does not inhabit {type!r}: {value!r}")


def _default_draw(sampleable: object) -> object:
    rsample = getattr(sampleable, "rsample", None)
    if callable(rsample):
        return rsample()
    sample = getattr(sampleable, "sample", None)
    if callable(sample):
        return sample()
    if callable(sampleable):
        return sampleable()
    raise InvalidHandlerError(
        "Random.sample received a nonsampleable runtime attachment"
    )


def draw_handler(
    result_validator: RuntimeValidator,
    *,
    draw: Callable[[object, object, RuntimeRequest], object] | None = None,
    answer_type: TypeExpr = ANSWER,
    duplicable_context: bool = False,
    key: str = "draw",
) -> RuntimeHandler:
    """Interpret ``Random.sample`` by drawing exactly once."""
    definition = _handler_def(
        "Random.draw",
        RANDOM,
        ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def sample(
        request: RuntimeRequest,
        resume: Resumption,
        _context: ClauseContext,
    ) -> object:
        site, sampleable = _expect_arguments(request, 2, definition.name)
        value = (
            draw(site, sampleable, request)
            if draw is not None
            else _default_draw(sampleable)
        )
        return resume(value)

    return RuntimeHandler(
        definition,
        {RANDOM_SAMPLE: RuntimeClause(sample, result_validator)},
        duplicable_context=duplicable_context,
    )


@dataclass(slots=True)
class ScoreAccumulator:
    """Mutable result owned by one installed Score handler."""

    total: object
    contributions: list[object] = field(default_factory=list)


def score_handler(
    *,
    identity: object = 0.0,
    combine: Callable[[object, object], object] = _RUNTIME_ADD,
    weight_validator: RuntimeValidator = lambda _value: True,
    answer_validator: RuntimeValidator = lambda _value: True,
    total_validator: RuntimeValidator | None = None,
    expose_total: bool = False,
    answer_type: TypeExpr = ANSWER,
    key: str = "score",
) -> tuple[RuntimeHandler, ScoreAccumulator]:
    """Accumulate ``Score.add`` contributions in their evaluation order."""
    accumulator = ScoreAccumulator(identity)
    checked_total = total_validator or weight_validator
    output_type = (
        TypeApplication(WEIGHTED_RESULT_CONSTRUCTOR, (answer_type, LOG_WEIGHT))
        if expose_total
        else answer_type
    )
    definition = _handler_def(
        "Score.accumulate",
        SCORE,
        ((SCORE_ADD, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=output_type,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def make_context() -> RuntimeHandler:
        local = ScoreAccumulator(identity)

        def add(
            request: RuntimeRequest,
            resume: Resumption,
            _context: ClauseContext,
        ) -> object:
            (weight,) = _expect_arguments(request, 1, definition.name)
            _require(weight, weight_validator, LOG_WEIGHT, "score contribution")
            local.contributions.append(weight)
            local.total = combine(local.total, weight)
            return resume(None)

        def finish(value: object, _context: ClauseContext) -> object:
            return (value, local.total) if expose_total else value

        def publish() -> None:
            accumulator.total = local.total
            accumulator.contributions[:] = local.contributions

        return RuntimeHandler(
            definition,
            {SCORE_ADD: RuntimeClause(add, _unit)},
            return_clause=finish,
            input_validator=answer_validator,
            output_validator=(
                _pair_of(answer_validator, checked_total)
                if expose_total
                else answer_validator
            ),
            mutable_context=True,
            on_exit=publish,
        )

    prototype = make_context()
    prototype.context_factory = make_context
    return prototype, accumulator


class MissingValuePolicy(str, Enum):
    FORWARD = "forward"
    ERROR = "error"


class ExtraValuePolicy(str, Enum):
    IGNORE = "ignore"
    ERROR = "error"


def _site_and_sampleable(
    request: RuntimeRequest, handler: str
) -> tuple[object, object]:
    site, sampleable = _expect_arguments(request, 2, handler)
    try:
        hash(site)
    except TypeError as error:
        raise InvalidHandlerError("probabilistic site keys must be hashable") from error
    return site, sampleable


def _log_density(sampleable: object, value: object) -> object:
    log_prob = getattr(sampleable, "log_prob", None)
    if not callable(log_prob):
        raise InvalidHandlerError(
            "a scored observation/replay needs a runtime sampleable with log_prob"
        )
    return log_prob(value)


def _emit_score(
    context: ClauseContext,
    request: RuntimeRequest,
    score_instance: EffectInstanceId,
    weight: object,
    weight_validator: RuntimeValidator,
    *,
    role: str,
) -> None:
    weight_ref = context.attach(
        weight,
        LOG_WEIGHT,
        role=role,
        validator=weight_validator,
    )
    source = request.core.origin.origin
    generated_origin = request.core.origin.__class__(
        SourceOrigin(
            source.module,
            (*source.structural_path, "generated", role),
            role,
            source.source_protocol,
            source.file,
            source.line,
            source.column,
        ),
        request.core.origin.dynamic_path,
        request.core.origin.resumption_path,
        "duplicate",
        (request.core.origin.static_site,),
    )
    context.evaluate(
        Perform(
            EffectRequest(
                score_instance,
                SCORE,
                SCORE_ADD,
                (),
                (weight_ref,),
                UNIT,
                generated_origin,
            )
        )
    )


def condition_handler(
    observations: Mapping[object, object],
    *,
    score_instance: EffectInstanceId,
    result_validator: RuntimeValidator,
    weight_validator: RuntimeValidator = lambda _value: True,
    missing: MissingValuePolicy = MissingValuePolicy.FORWARD,
    extra: ExtraValuePolicy = ExtraValuePolicy.ERROR,
    answer_type: TypeExpr = ANSWER,
    key: str = "condition",
) -> RuntimeHandler:
    """Supply likelihood observations and emit their densities as Score.

    Unobserved sites forward to the next ``Random`` handler.  The explicit
    introduced Score row distinguishes likelihood observation from intervention
    and unscored replay.
    """
    definition = _handler_def(
        "Random.condition",
        RANDOM,
        ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        introduced=EffectRow((RowEntry(score_instance, SCORE),)),
        total=False,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def make_context() -> RuntimeHandler:
        seen: set[object] = set()

        def sample(
            request: RuntimeRequest,
            resume: Resumption,
            context: ClauseContext,
        ) -> object:
            site, sampleable = _site_and_sampleable(request, definition.name)
            if site not in observations:
                if missing is MissingValuePolicy.ERROR:
                    raise InvalidHandlerError(f"missing observation for site {site!r}")
                return Forward()
            value = observations[site]
            _require(
                value,
                result_validator,
                request.core.result_type,
                f"observation {site!r}",
            )
            seen.add(site)
            weight = _log_density(sampleable, value)
            _emit_score(
                context,
                request,
                score_instance,
                weight,
                weight_validator,
                role="condition-score",
            )
            return resume(value)

        def finish(value: object, _context: ClauseContext) -> object:
            if extra is ExtraValuePolicy.ERROR:
                extras = set(observations) - seen
                if extras:
                    rendered = ", ".join(
                        repr(item) for item in sorted(extras, key=repr)
                    )
                    raise InvalidHandlerError(f"unused observations: {rendered}")
            return value

        return RuntimeHandler(
            definition,
            {RANDOM_SAMPLE: RuntimeClause(sample, result_validator)},
            return_clause=finish,
            mutable_context=True,
        )

    prototype = make_context()
    prototype.context_factory = make_context
    return prototype


class ReplayPolicy(str, Enum):
    CLAMP_AND_SCORE = "clamp-and-score"
    CLAMP_NO_SCORE = "clamp-no-score"
    INTERVENE = "intervene"


def replay_handler(
    values: Mapping[object, object],
    *,
    result_validator: RuntimeValidator,
    policy: ReplayPolicy = ReplayPolicy.CLAMP_NO_SCORE,
    score_instance: EffectInstanceId | None = None,
    weight_validator: RuntimeValidator = lambda _value: True,
    extra: ExtraValuePolicy = ExtraValuePolicy.ERROR,
    answer_type: TypeExpr = ANSWER,
    key: str = "replay",
) -> RuntimeHandler:
    """Replay selected sites under an explicit scoring/intervention policy."""
    if policy is ReplayPolicy.CLAMP_AND_SCORE and score_instance is None:
        raise ValueError("scored replay requires a Score effect instance")
    introduced = (
        EffectRow((RowEntry(score_instance, SCORE),))
        if score_instance is not None and policy is ReplayPolicy.CLAMP_AND_SCORE
        else EffectRow()
    )
    definition = _handler_def(
        f"Random.replay.{policy.value}",
        RANDOM,
        ((RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        introduced=introduced,
        total=False,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def make_context() -> RuntimeHandler:
        seen: set[object] = set()

        def sample(
            request: RuntimeRequest,
            resume: Resumption,
            context: ClauseContext,
        ) -> object:
            site, sampleable = _site_and_sampleable(request, definition.name)
            if site not in values:
                return Forward()
            value = values[site]
            _require(
                value,
                result_validator,
                request.core.result_type,
                f"replay {site!r}",
            )
            seen.add(site)
            if policy is ReplayPolicy.CLAMP_AND_SCORE:
                assert score_instance is not None
                _emit_score(
                    context,
                    request,
                    score_instance,
                    _log_density(sampleable, value),
                    weight_validator,
                    role="replay-score",
                )
            return resume(value)

        def finish(value: object, _context: ClauseContext) -> object:
            if extra is ExtraValuePolicy.ERROR:
                extras = set(values) - seen
                if extras:
                    rendered = ", ".join(
                        repr(item) for item in sorted(extras, key=repr)
                    )
                    raise InvalidHandlerError(f"unused replay values: {rendered}")
            return value

        return RuntimeHandler(
            definition,
            {RANDOM_SAMPLE: RuntimeClause(sample, result_validator)},
            return_clause=finish,
            mutable_context=True,
        )

    prototype = make_context()
    prototype.context_factory = make_context
    return prototype


@dataclass(frozen=True, slots=True)
class TraceEvent:
    address: tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]
    instance: EffectInstanceId
    operation: OperationId
    arguments: tuple[object, ...]
    result: object
    mode: str


@dataclass(slots=True)
class TraceRecorder:
    """Ordered trace; repeated dynamic addresses remain observable as events."""

    events: list[TraceEvent] = field(default_factory=list)

    def record(self, request: RuntimeRequest, result: object, *, mode: str) -> object:
        self.events.append(
            TraceEvent(
                request.address,
                request.instance,
                request.operation,
                request.arguments,
                result,
                mode,
            )
        )
        return result


def trace_handler(
    effect: EffectRef,
    operations: Iterable[OperationId],
    recorder: TraceRecorder,
    *,
    validators: Mapping[OperationId, RuntimeValidator],
    answer_type: TypeExpr = ANSWER,
    key: str = "trace",
) -> RuntimeHandler:
    """Observe forwarded operation replies without interpreting the effect."""
    operation_tuple = tuple(operations)
    definition = _handler_def(
        f"{effect.name}.trace",
        effect,
        tuple((operation, ResumptionGrade.AFFINE) for operation in operation_tuple),
        key=key,
        input_type=answer_type,
        output_type=answer_type,
        total=False,
        telescope=_generic_binders(answer_type=answer_type),
    )

    def make_clause(operation: OperationId):
        def clause(
            request: RuntimeRequest,
            _resume: Resumption,
            _context: ClauseContext,
        ) -> object:
            return Forward(
                lambda result: recorder.record(request, result, mode="forwarded")
            )

        try:
            validator = validators[operation]
        except KeyError as error:
            raise ValueError(
                f"trace handler needs a validator for {operation}"
            ) from error

        def checked_clause(
            request: RuntimeRequest,
            resume: Resumption,
            context: ClauseContext,
        ) -> object:
            answer = clause(request, resume, context)
            assert isinstance(answer, Forward)
            hook = answer.on_response

            def validate_and_record(result: object) -> object:
                _require(
                    result,
                    validator,
                    request.core.result_type,
                    f"traced result of {request.operation}",
                )
                assert hook is not None
                return hook(result)

            return Forward(validate_and_record)

        return RuntimeClause(checked_clause, validator)

    return RuntimeHandler(
        definition,
        {operation: make_clause(operation) for operation in operation_tuple},
    )


@dataclass(slots=True)
class StateCell:
    value: object


def state_handler(
    initial: object,
    state_validator: RuntimeValidator,
    *,
    state_type: TypeExpr = STATE,
    expose_final: bool = False,
    answer_type: TypeExpr = ANSWER,
    answer_validator: RuntimeValidator = lambda _value: True,
    key: str = "state",
) -> tuple[RuntimeHandler, StateCell]:
    """Interpret lexical State.get/put with a nonduplicable mutable cell."""
    _require(initial, state_validator, state_type, "initial state")
    cell = StateCell(initial)
    effect = state_effect(state_type)
    definition = _handler_def(
        "State.run",
        effect,
        (
            (STATE_GET, ResumptionGrade.LINEAR),
            (STATE_PUT, ResumptionGrade.LINEAR),
        ),
        key=key,
        input_type=answer_type,
        output_type=(
            TypeApplication(STATE_RESULT_CONSTRUCTOR, (answer_type, state_type))
            if expose_final
            else answer_type
        ),
        telescope=_generic_binders(
            answer_type=answer_type,
            state_type=state_type,
        ),
    )

    def make_context() -> RuntimeHandler:
        local = StateCell(initial)

        def get(
            request: RuntimeRequest,
            resume: Resumption,
            _context: ClauseContext,
        ) -> object:
            _expect_arguments(request, 0, definition.name)
            return resume(local.value)

        def put(
            request: RuntimeRequest,
            resume: Resumption,
            _context: ClauseContext,
        ) -> object:
            (new_value,) = _expect_arguments(request, 1, definition.name)
            _require(new_value, state_validator, state_type, "state update")
            local.value = new_value
            return resume(None)

        def finish(value: object, _context: ClauseContext) -> object:
            return (value, local.value) if expose_final else value

        def publish() -> None:
            cell.value = local.value

        return RuntimeHandler(
            definition,
            {
                STATE_GET: RuntimeClause(get, state_validator),
                STATE_PUT: RuntimeClause(put, _unit),
            },
            return_clause=finish,
            input_validator=answer_validator,
            output_validator=(
                _pair_of(answer_validator, state_validator)
                if expose_final
                else answer_validator
            ),
            mutable_context=True,
            on_exit=publish,
        )

    prototype = make_context()
    prototype.context_factory = make_context
    return prototype, cell


@dataclass(frozen=True, slots=True)
class Aborted:
    error: object


def abort_handler(
    *,
    on_abort: Callable[[object], object] = Aborted,
    result_validator: RuntimeValidator = lambda _value: True,
    error_type: TypeExpr = PAYLOAD,
    input_type: TypeExpr = ANSWER,
    output_type: TypeExpr = ANSWER,
    output_validator: RuntimeValidator = lambda _value: True,
    duplicable_context: bool = False,
    key: str = "abort",
) -> RuntimeHandler:
    """Handle Abort.abort without resuming the discarded continuation."""
    effect = abort_effect(error_type)
    definition = _handler_def(
        "Abort.run",
        effect,
        ((ABORT_ABORT, ResumptionGrade.ZERO),),
        key=key,
        input_type=input_type,
        output_type=output_type,
        telescope=tuple(
            dict.fromkeys(
                (
                    *_generic_binders(answer_type=input_type),
                    *_generic_binders(answer_type=output_type),
                    *_generic_binders(error_type=error_type),
                )
            )
        ),
    )

    def abort(
        request: RuntimeRequest,
        _resume: Resumption,
        _context: ClauseContext,
    ) -> object:
        (error,) = _expect_arguments(request, 1, definition.name)
        return on_abort(error)

    return RuntimeHandler(
        definition,
        {ABORT_ABORT: RuntimeClause(abort, result_validator)},
        output_validator=output_validator,
        duplicable_context=duplicable_context,
    )


def choose_handler(
    choice_validator: RuntimeValidator,
    *,
    combine: Callable[[list[object]], object] | None = None,
    combine_validator: RuntimeValidator | None = None,
    answer_type: TypeExpr = ANSWER,
    answer_validator: RuntimeValidator = lambda _value: True,
    key: str = "choose",
) -> RuntimeHandler:
    """Enumerate finite choices using an unrestricted deep resumption."""
    if combine is not None and combine_validator is None:
        raise ValueError("a custom Choose combine needs an output validator")
    definition = _handler_def(
        "Choose.enumerate",
        CHOOSE,
        ((CHOOSE_CHOOSE, ResumptionGrade.UNRESTRICTED),),
        key=key,
        input_type=answer_type,
        output_type=TypeApplication(CHOICE_RESULTS_CONSTRUCTOR, (answer_type,)),
        telescope=_generic_binders(answer_type=answer_type),
    )

    def choose(
        request: RuntimeRequest,
        resume: Resumption,
        _context: ClauseContext,
    ) -> object:
        (alternatives,) = _expect_arguments(request, 1, definition.name)
        if isinstance(alternatives, (str, bytes)):
            raise InvalidHandlerError("Choose alternatives must be a finite collection")
        try:
            options = tuple(alternatives)  # type: ignore[arg-type]
        except TypeError as error:
            raise InvalidHandlerError("Choose alternatives must be iterable") from error
        results = [resume(option) for option in options]
        if combine is not None:
            return combine(results)
        flattened: list[object] = []
        for result in results:
            if not isinstance(result, tuple):
                raise InvalidHandlerError(
                    "the default Choose handler expects its deep return clause "
                    "to produce tuples"
                )
            flattened.extend(result)
        return tuple(flattened)

    return RuntimeHandler(
        definition,
        {CHOOSE_CHOOSE: RuntimeClause(choose, choice_validator)},
        return_clause=lambda value, _context: (value,),
        input_validator=answer_validator,
        output_validator=(
            combine_validator
            if combine_validator is not None
            else _tuple_of(answer_validator)
        ),
        duplicable_context=True,
    )


@dataclass(slots=True)
class WeightAccumulator:
    total: object
    contributions: list[object] = field(default_factory=list)


def weight_handler(
    identity: object,
    combine: Callable[[object, object], object],
    weight_validator: RuntimeValidator,
    *,
    weight_type: TypeExpr = K,
    expose_total: bool = True,
    answer_type: TypeExpr = ANSWER,
    answer_validator: RuntimeValidator = lambda _value: True,
    total_validator: RuntimeValidator | None = None,
    key: str = "weight",
) -> tuple[RuntimeHandler, WeightAccumulator]:
    """Accumulate values in a caller-supplied semiring multiplication."""
    accumulator = WeightAccumulator(identity)
    checked_total = total_validator or weight_validator
    effect = weight_effect(weight_type)
    definition = _handler_def(
        "Weight.accumulate",
        effect,
        ((WEIGHT_ADD, ResumptionGrade.LINEAR),),
        key=key,
        input_type=answer_type,
        output_type=(
            TypeApplication(WEIGHTED_RESULT_CONSTRUCTOR, (answer_type, weight_type))
            if expose_total
            else answer_type
        ),
        telescope=_generic_binders(
            answer_type=answer_type,
            weight_type=weight_type,
        ),
    )

    def make_context() -> RuntimeHandler:
        local = WeightAccumulator(identity)

        def add(
            request: RuntimeRequest,
            resume: Resumption,
            _context: ClauseContext,
        ) -> object:
            (weight,) = _expect_arguments(request, 1, definition.name)
            _require(weight, weight_validator, weight_type, "semiring weight")
            local.contributions.append(weight)
            local.total = combine(local.total, weight)
            return resume(None)

        def finish(value: object, _context: ClauseContext) -> object:
            return (value, local.total) if expose_total else value

        def publish() -> None:
            accumulator.total = local.total
            accumulator.contributions[:] = local.contributions

        return RuntimeHandler(
            definition,
            {WEIGHT_ADD: RuntimeClause(add, _unit)},
            return_clause=finish,
            input_validator=answer_validator,
            output_validator=(
                _pair_of(answer_validator, checked_total)
                if expose_total
                else answer_validator
            ),
            mutable_context=True,
            on_exit=publish,
        )

    prototype = make_context()
    prototype.context_factory = make_context
    return prototype, accumulator


__all__ = [
    "A",
    "ABORT",
    "ABORT_ABORT",
    "ABORT_EFFECT",
    "ANSWER",
    "Aborted",
    "BUILTIN_EFFECTS",
    "CHOOSE",
    "CHOOSE_CHOOSE",
    "CHOOSE_EFFECT",
    "CHOICES_A",
    "CHOICE_RESULTS_A",
    "ExtraValuePolicy",
    "K",
    "LOG_WEIGHT",
    "MissingValuePolicy",
    "PAYLOAD",
    "RANDOM",
    "RANDOM_EFFECT",
    "RANDOM_SAMPLE",
    "ReplayPolicy",
    "SAMPLEABLE_A",
    "SCORE",
    "SCORE_ADD",
    "SCORE_EFFECT",
    "SITE_A",
    "STATE",
    "STATE_EFFECT",
    "STATE_EFFECT_REF",
    "STATE_GET",
    "STATE_PUT",
    "ScoreAccumulator",
    "StateCell",
    "TraceEvent",
    "TraceRecorder",
    "WEIGHT",
    "WEIGHT_ADD",
    "WEIGHT_EFFECT",
    "WeightAccumulator",
    "abort_handler",
    "abort_effect",
    "choose_handler",
    "condition_handler",
    "draw_handler",
    "replay_handler",
    "score_handler",
    "state_handler",
    "state_effect",
    "trace_handler",
    "weight_handler",
    "weight_effect",
]
