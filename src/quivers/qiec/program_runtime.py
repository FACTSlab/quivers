"""Running program entry points on the reference machine.

An elaborated program is a named computation whose row names the module's
canonical ``random`` and ``score`` instances. Running it means choosing
handlers for the two: this module wraps the entry point in a scoring
replay of ``Random`` and an accumulating ``Score``, so a program applied to
data and to a value for each of its sites yields the value it returns and
the log joint density the sites and observations accumulate. The wrapper is
an ordinary computation added to a copy of the module, so the kernel checks
it like any other before it runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from quivers.qiec.builtins import RANDOM_SAMPLE, SCORE_ADD
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.effects import (
    ComputationType,
    EffectRow,
    HandlerClauseDef,
    HandlerDef,
    ResumptionGrade,
)
from quivers.qiec.execution import (
    ExecutionResult,
    RuntimeConfiguration,
    RuntimeSelection,
    run_named,
)
from quivers.qiec.identifiers import ComputationId, HandlerId, SourceOrigin
from quivers.qiec.module import NamedComputation, QiecModule
from quivers.qiec.programs import ProgramEntry
from quivers.qiec.kinds import IndexBinder
from quivers.qiec.terms import Call, Handle, Local, Var
from quivers.qiec.types import (
    IndexLiteral,
    IndexVariable,
    ShapeIndex,
    StaticArgument,
    TypeApplication,
    TypeExpr,
    product_type,
)

#: The names of the wrapper's handlers, which no source may declare.
REPLAY_HANDLER = "__program_replay"
ACCUMULATE_HANDLER = "__program_accumulate"


@dataclass(frozen=True, slots=True)
class ProgramRun:
    """What running a program on the reference machine produced.

    Parameters
    ----------
    value
        The value the program returned.
    log_joint
        The log joint density: every replayed site's log density under
        its distribution plus every observation's and score's weight.
    result
        The underlying execution result, with its trace.
    """

    value: object
    log_joint: float
    result: ExecutionResult


def program_entry(module: QiecModule, name: str) -> ProgramEntry:
    """Find a program's entry point in a module.

    Parameters
    ----------
    module : QiecModule
        The module.
    name : str
        The program's name.

    Returns
    -------
    ProgramEntry
        The entry.

    Raises
    ------
    KeyError
        If the module has no program of that name.
    """
    for entry in module.entries:
        if entry.name == name:
            return entry
    raise KeyError(f"module {module.module!r} has no program {name!r}")


def _computation(module: QiecModule, identity: ComputationId) -> NamedComputation:
    """Find a computation by identity.

    Parameters
    ----------
    module : QiecModule
        The module.
    identity : ComputationId
        The computation's identity.

    Returns
    -------
    NamedComputation
        The computation.

    Raises
    ------
    KeyError
        If no computation has the identity.
    """
    for computation in module.computations:
        if computation.id == identity:
            return computation
    raise KeyError(f"module {module.module!r} has no computation {identity!r}")


def _marginal_handler(name: str) -> tuple[bool, str] | None:
    """Read a marginalization handler's kind off its name.

    The elaborator names its enumeration handlers ``enumerate_marginal``
    and ``enumerate_grouped_marginal``, with ``_sum`` or ``_mean``
    appended for a reduction other than ``logsumexp``.

    Parameters
    ----------
    name : str
        The handler's name.

    Returns
    -------
    tuple[bool, str] | None
        Whether the handler answers per group position, and its
        reduction; ``None`` for a handler of another kind.
    """
    for reduction in ("sum", "mean"):
        suffix = f"_{reduction}"
        if name.endswith(suffix):
            base = _marginal_handler(name[: -len(suffix)])
            return None if base is None else (base[0], reduction)
    if name.endswith("enumerate_grouped_marginal"):
        return (True, "logsumexp")
    if name.endswith("enumerate_marginal"):
        return (False, "logsumexp")
    return None


def joint_module(module: QiecModule, name: str) -> tuple[QiecModule, str]:
    """Extend a module with a computation scoring one program's log joint.

    Parameters
    ----------
    module : QiecModule
        The module holding the program.
    name : str
        The program's name.

    Returns
    -------
    tuple[QiecModule, str]
        The extended module and the wrapper computation's name. The
        wrapper takes the program's parameters, handles its ``score``
        instance with an accumulating handler and its ``random``
        instance with a replaying one, and returns the program's value
        paired with the accumulated log weight.

    Raises
    ------
    KeyError
        If the module has no such program.
    """
    entry = program_entry(module, name)
    computation = _computation(module, entry.computation)
    answer: TypeExpr = computation.type.result
    statics: tuple[StaticArgument, ...] = tuple(
        IndexVariable(binder.name, binder.sort)
        for binder in computation.telescope
        if isinstance(binder, IndexBinder)
    )
    random = next(
        instance
        for instance in module.instances
        if instance.entry.instance == entry.random_instance
    )
    score = next(
        instance
        for instance in module.instances
        if instance.entry.instance == entry.score_instance
    )
    replay = HandlerDef(
        HandlerId.derive(module.module, "handler", REPLAY_HANDLER),
        REPLAY_HANDLER,
        random.entry.effect,
        (HandlerClauseDef(RANDOM_SAMPLE, ResumptionGrade.LINEAR),),
        answer,
        answer,
        EffectRow((score.entry,)),
        total=True,
        implementation="foreign",
        telescope=computation.telescope,
    )
    accumulate = HandlerDef(
        HandlerId.derive(module.module, "handler", ACCUMULATE_HANDLER),
        ACCUMULATE_HANDLER,
        score.entry.effect,
        (HandlerClauseDef(SCORE_ADD, ResumptionGrade.LINEAR),),
        answer,
        product_type(answer, LOG_WEIGHT),
        EffectRow(),
        total=True,
        implementation="foreign",
        telescope=computation.telescope,
    )
    result_type = answer
    parameters = tuple(
        Local(parameter.name, parameter.type) for parameter in computation.parameters
    )
    call = Call(
        computation.id,
        computation.name,
        statics,
        tuple(Var(parameter) for parameter in parameters),
        result_type,
        computation.type.effects,
        SourceOrigin(
            module.module,
            ("programs", name, "joint", "call"),
            "call",
            module.source_protocol,
        ),
    )
    body = Handle(
        score.entry.instance,
        accumulate.id,
        Handle(random.entry.instance, replay.id, call, statics),
        statics,
    )
    wrapper_name = f"__joint_{name}"
    wrapper = NamedComputation(
        ComputationId.derive(module.module, "computation", wrapper_name),
        wrapper_name,
        computation.telescope,
        parameters,
        body,
        ComputationType(EffectRow(), product_type(result_type, LOG_WEIGHT)),
        SourceOrigin(
            module.module,
            ("programs", name, "joint"),
            "computation",
            module.source_protocol,
        ),
    )
    extended = replace(
        module,
        handlers=(*module.handlers, replay, accumulate),
        computations=(*module.computations, wrapper),
    )
    return extended, wrapper_name


def _extent_of(value: object, depth: int) -> int:
    """Read the extent of a nested host tensor at one axis.

    Parameters
    ----------
    value : object
        A host tensor: nested tuples or lists, or an object with a
        ``shape``.
    depth : int
        The axis, outermost first.

    Returns
    -------
    int
        The extent.

    Raises
    ------
    KeyError
        If the value has no such axis.
    """
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return int(shape[depth])
        except IndexError as error:
            raise KeyError(f"host value has no axis {depth}") from error
    current = value
    for _ in range(depth):
        if not isinstance(current, (tuple, list)) or not current:
            raise KeyError(f"host value has no axis {depth}")
        current = current[0]
    if not isinstance(current, (tuple, list)):
        raise KeyError(f"host value has no axis {depth}")
    return len(current)


def static_arguments_for(
    entry: ProgramEntry, data: Mapping[str, object]
) -> tuple[StaticArgument, ...]:
    """Read a program's static extents off the data it is applied to.

    Parameters
    ----------
    entry : ProgramEntry
        The program's entry point, whose telescope names the extents.
    data : Mapping[str, object]
        The host value of each parameter, by name.

    Returns
    -------
    tuple[StaticArgument, ...]
        One index literal per telescope binder, in telescope order: the
        length of the axis of the first parameter whose tensor type is
        shaped by the binder.

    Raises
    ------
    KeyError
        If a binder shapes no parameter, or the parameter is not given
        or has too few axes.
    """
    arguments: list[StaticArgument] = []
    for binder in entry.telescope:
        found: int | None = None
        for parameter in entry.parameters:
            type_ = parameter.type
            if not isinstance(type_, TypeApplication) or len(type_.arguments) != 2:
                continue
            shape = type_.arguments[1]
            if not isinstance(shape, ShapeIndex):
                continue
            for depth, dimension in enumerate(shape.dimensions):
                if (
                    isinstance(dimension, IndexVariable)
                    and dimension.name == binder.name
                ):
                    try:
                        found = _extent_of(data[parameter.name], depth)
                    except KeyError as error:
                        raise KeyError(
                            f"parameter {parameter.name!r} of program "
                            f"{entry.name!r} does not fix its extent "
                            f"{binder.name!r}: {error.args[0]}"
                        ) from error
                    break
            if found is not None:
                break
        if found is None:
            raise KeyError(
                f"program {entry.name!r} has an extent {binder.name!r} that no "
                "parameter fixes"
            )
        arguments.append(IndexLiteral(found, binder.sort))  # type: ignore[arg-type]
    return tuple(arguments)


def run_program(
    module: QiecModule,
    name: str,
    *,
    data: Mapping[str, object],
    sites: Mapping[str, object],
    fuel: int | None = None,
) -> ProgramRun:
    """Run a program with every site replayed and score its log joint.

    Parameters
    ----------
    module : QiecModule
        The module holding the program.
    name : str
        The program's name.
    data : Mapping[str, object]
        A host value for each of the program's parameters, by name:
        its data, observations, fibrations, and map numbers.
    sites : Mapping[str, object]
        A host value for each sampled site, by label. Every site the
        program reaches must be given, and every given site reached.
    fuel : int | None
        A step budget for the run.

    Returns
    -------
    ProgramRun
        The returned value and the log joint.

    Raises
    ------
    KeyError
        If the module has no such program, a parameter is not given, or
        an open extent cannot be read off the data.
    ExecutionFailure
        If the run fails, a site is missing, or a value does not inhabit
        its type.
    """
    entry = program_entry(module, name)
    extended, wrapper_name = joint_module(module, name)
    arguments = tuple(data[parameter.name] for parameter in entry.parameters)
    statics = static_arguments_for(entry, data)
    configured: dict[str, object] = {
        REPLAY_HANDLER: {
            "kind": "replay",
            "values": dict(sites),
            "score_instance": entry.score_instance,
        },
        ACCUMULATE_HANDLER: {"kind": "score"},
    }
    for handler in module.handlers:
        if handler.implementation != "foreign":
            continue
        marginal = _marginal_handler(handler.name)
        if handler.effect.name == "Random" and marginal is not None:
            grouped, reduction = marginal
            configured[handler.name] = {
                "kind": "enumerate",
                "grouped": grouped,
                "reduction": reduction,
            }
        elif handler.effect.name == "Weight" and handler.name.endswith(
            "collect_marginal"
        ):
            configured[handler.name] = {"kind": "collect"}
    runtime = RuntimeConfiguration(
        (RuntimeSelection("core", {"handlers": configured}),)
    )
    result = run_named(
        extended,
        wrapper_name,
        arguments,
        static_arguments=statics,
        runtime=runtime,
        fuel=fuel,
    )
    value, weight = result.value  # type: ignore[misc]
    return ProgramRun(value, float(weight), result)  # type: ignore[arg-type]


__all__ = [
    "ACCUMULATE_HANDLER",
    "REPLAY_HANDLER",
    "ProgramRun",
    "joint_module",
    "program_entry",
    "run_program",
    "static_arguments_for",
]
