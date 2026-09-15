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
from quivers.qiec.terms import Call, Handle, Local, Var
from quivers.qiec.types import TypeExpr, product_type

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
    )
    result_type = answer
    parameters = tuple(
        Local(parameter.name, parameter.type) for parameter in computation.parameters
    )
    call = Call(
        computation.id,
        computation.name,
        (),
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
        Handle(random.entry.instance, replay.id, call, ()),
        (),
    )
    wrapper_name = f"__joint_{name}"
    wrapper = NamedComputation(
        ComputationId.derive(module.module, "computation", wrapper_name),
        wrapper_name,
        (),
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
        If the module has no such program, or a parameter is not given.
    ExecutionFailure
        If the run fails, a site is missing, or a value does not inhabit
        its type.
    """
    entry = program_entry(module, name)
    extended, wrapper_name = joint_module(module, name)
    arguments = tuple(data[parameter.name] for parameter in entry.parameters)
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
        if handler.effect.name == "Random" and handler.name.endswith(
            "enumerate_marginal"
        ):
            configured[handler.name] = {"kind": "enumerate"}
        elif handler.effect.name == "Weight" and handler.name.endswith(
            "collect_marginal"
        ):
            configured[handler.name] = {"kind": "collect"}
    runtime = RuntimeConfiguration(
        (RuntimeSelection("core", {"handlers": configured}),)
    )
    result = run_named(extended, wrapper_name, arguments, runtime=runtime, fuel=fuel)
    value, weight = result.value  # type: ignore[misc]
    return ProgramRun(value, float(weight), result)  # type: ignore[arg-type]


__all__ = [
    "ACCUMULATE_HANDLER",
    "REPLAY_HANDLER",
    "ProgramRun",
    "joint_module",
    "program_entry",
    "run_program",
]
