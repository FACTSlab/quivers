"""Programs elaborate to computations that score as the runtime does.

A ``program`` becomes a named computation over the module's canonical
``random`` and ``score`` instances, with typed data, observation, and
fibration parameters, and one helper computation per marginalization
block. Run on the reference machine with every site replayed, its log
joint is the same number the torch runtime's trace accumulates.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributions as td

from quivers.dsl import Compiler, parse
from quivers.dsl.compiler import CompileError
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_lowering import QiecDiagnosticError, lower_qvr_to_qiec
from quivers.inference.trace import trace
from quivers.qiec import (
    INT,
    REAL,
    Bind,
    Call,
    DistributionValue,
    Handle,
    NewInstance,
    Perform,
    QiecModule,
    SegmentSum,
    dumps,
    loads,
    tensor_type,
)
from quivers.qiec.program_runtime import program_entry, run_program
from quivers.qiec.types import IndexLiteral, render_static
from quivers.qiec.kinds import NAT

BETA_BERNOULLI = """\
object Obs : FinSet 4
program prog : Obs -> Obs
    sample theta <- Beta(2.0, 2.0)
    observe y <- Bernoulli(theta)
    return theta
export prog
"""

LET_STEP = """\
object Obs : FinSet 4
program prog : Obs -> Obs
    sample a <- Normal(0.0, 1.0)
    sample b <- Normal(0.0, 1.0)
    let mu = a + b
    observe y <- Normal(mu, 0.5)
    return mu
export prog
"""

IID_PLATE = """\
object N : FinSet 3
object Obs : FinSet 8
program prog : Obs -> Obs
    sample xs : N <- Normal(0.0, 1.0)
    observe y : N <- Normal(xs, 0.5)
    return xs
export prog
"""

GROUPED = """\
object Component : FinSet 3
object Item : FinSet 2
object Resp : FinSet 4
program prog(concentration : Real) : Resp -> Resp
    sample probs <- Dirichlet(concentration) [over=Component]
    sample mu : Component <- Normal(0.0, 5.0)
    sample sigma : Component <- HalfNormal(1.0)
    marginalize cls : Component <- Categorical(probs) [over=Item, reduction=logsumexp]
        observe r : Resp <- Normal(mu[cls], sigma[cls]) [via=idx]
    return probs
export prog
"""

KERNEL = """\
object Obs : Real 2
morphism kernel : Obs -> Obs [role=kernel] ~ Normal
program prog : Obs -> Obs
    sample x <- kernel
    return x
export prog
"""

SCORED = """\
object Obs : FinSet 4
program prog : Obs -> Obs
    sample x <- Normal(0.0, 1.0)
    score penalty = x * x
    return x
export prog
"""


def _module(source: str) -> QiecModule:
    """Lower a program module.

    Parameters
    ----------
    source : str
        QVR text.

    Returns
    -------
    QiecModule
        The checked module.
    """
    return lower_qvr_to_qiec(parse(source), file_path="program.qvr")


def _tensor(value: object) -> torch.Tensor:
    """A one-dimensional tensor for a scalar or a sequence.

    Parameters
    ----------
    value : object
        A number, a Boolean, or a sequence of numbers.

    Returns
    -------
    torch.Tensor
        The values as floats.
    """
    if isinstance(value, bool | int | float):
        return torch.tensor([float(value)])
    return torch.tensor([float(item) for item in value])  # type: ignore[union-attr]


def _classic_log_joint(
    source: str,
    clamps: dict[str, object],
    *,
    scalars: dict[str, float] | None = None,
) -> float:
    """The torch runtime's log joint at clamped sites and observations.

    Parameters
    ----------
    source : str
        QVR text.
    clamps : dict[str, object]
        Every site and observation value, by name.
    scalars : dict[str, float] | None
        The program's scalar parameters, when it declares any.

    Returns
    -------
    float
        The trace's accumulated log joint.
    """
    program = Compiler(parse(source)).compile()
    if scalars:
        monadic = getattr(program, "prog")(**scalars).morphism
    else:
        monadic = program._morphism
    torch.manual_seed(0)
    observations = {name: _tensor(value) for name, value in clamps.items()}
    log_joint = trace(monadic, torch.zeros(1, 1), observations=observations).log_joint
    assert log_joint is not None
    return float(log_joint.sum())


def test_a_program_becomes_a_computation_over_canonical_instances() -> None:
    module = _module(BETA_BERNOULLI)
    entry = program_entry(module, "prog")
    computation = next(
        item for item in module.computations if item.id == entry.computation
    )
    assert [parameter.name for parameter in computation.parameters] == ["y"]
    assert [(parameter.name, parameter.role) for parameter in entry.parameters] == [
        ("y", "observation")
    ]
    assert [(site.name, site.kind, site.family) for site in entry.sites] == [
        ("theta", "sample", "Beta"),
        ("y", "observe", "Bernoulli"),
    ]
    instances = {instance.name: instance for instance in module.instances}
    assert instances["random"].entry.instance == entry.random_instance
    assert instances["score"].entry.instance == entry.score_instance
    assert {entry.instance for entry in computation.type.effects.entries} == {
        entry.random_instance,
        entry.score_instance,
    }
    assert isinstance(computation.body, Bind)
    assert isinstance(computation.body.first, Perform)
    assert loads(dumps(module)) == module


def test_plates_type_the_sampled_tensors() -> None:
    module = _module(IID_PLATE)
    computation = next(item for item in module.computations if item.name == "prog")
    assert computation.parameters[0].type == tensor_type(REAL, (IndexLiteral(3, NAT),))
    assert computation.type.result == tensor_type(REAL, (IndexLiteral(3, NAT),))
    entry = program_entry(module, "prog")
    assert [axis.name for axis in entry.sites[0].batch] == ["N"]


def test_a_marginalization_block_becomes_an_enumerated_helper() -> None:
    module = _module(GROUPED)
    names = [item.name for item in module.computations]
    assert names == ["prog", "prog__cls_marginal"]
    helper = module.computations[1]
    assert [parameter.name for parameter in helper.parameters] == [
        "probs",
        "mu",
        "sigma",
        "idx",
        "r",
    ]
    assert helper.parameters[3].type == tensor_type(INT, (IndexLiteral(4, NAT),))
    assert isinstance(helper.body, NewInstance)
    assert isinstance(helper.body.body, Handle)
    handler_names = {handler.name for handler in module.handlers}
    assert {"enumerate_marginal", "collect_marginal"} <= handler_names
    program = module.computations[0]
    calls = []
    node = program.body
    while isinstance(node, Bind):
        if isinstance(node.first, Call):
            calls.append(node.first.name)
        node = node.then
    assert calls == ["prog__cls_marginal"]
    inner = helper.body.body.computation
    assert isinstance(inner, Bind)
    latent = inner.first
    assert isinstance(latent, Perform)
    distribution = latent.request.arguments[1]
    assert isinstance(distribution, DistributionValue)
    assert [axis.name for axis in distribution.plate.batch] == ["Item"]
    weights = inner.then.body.computation.first.request.arguments[0]  # type: ignore[union-attr]
    assert isinstance(weights, SegmentSum)


@pytest.mark.parametrize(
    ("source", "clamps", "sites"),
    [
        (BETA_BERNOULLI, {"y": True}, {"theta": 0.3}),
        (LET_STEP, {"y": 0.7}, {"a": 0.2, "b": -0.4}),
        (IID_PLATE, {"y": (0.1, 0.2, 0.3)}, {"xs": (0.5, -0.5, 1.0)}),
        (SCORED, {}, {"x": 0.8}),
    ],
)
def test_reference_runs_agree_with_the_torch_runtime(
    source: str, clamps: dict[str, object], sites: dict[str, object]
) -> None:
    module = _module(source)
    run = run_program(module, "prog", data=clamps, sites=sites)
    expected = _classic_log_joint(source, {**clamps, **sites})
    assert run.log_joint == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_grouped_marginalization_agrees_with_the_torch_runtime_and_closed_form() -> (
    None
):
    module = _module(GROUPED)
    data = {
        "concentration": 1.5,
        "r": (0.1, 0.5, -0.3, 2.0),
        "idx": (0, 0, 1, 1),
    }
    sites = {"probs": (0.2, 0.3, 0.5), "mu": (0.0, 1.0, -1.0), "sigma": (1.0, 0.5, 2.0)}
    run = run_program(module, "prog", data=data, sites=sites)
    clamps = {key: value for key, value in data.items() if key != "concentration"}
    classic = _classic_log_joint(
        GROUPED, {**clamps, **sites}, scalars={"concentration": 1.5}
    )
    probs, mu, sigma = (torch.tensor(sites[name]) for name in ("probs", "mu", "sigma"))
    responses, index = torch.tensor(data["r"]), torch.tensor(data["idx"])
    closed = (
        td.Dirichlet(torch.full((3,), 1.5)).log_prob(probs)
        + td.Normal(0.0, 5.0).log_prob(mu).sum()
        + td.HalfNormal(1.0).log_prob(sigma).sum()
    )
    per_row = td.Normal(mu.unsqueeze(0), sigma.unsqueeze(0)).log_prob(
        responses.unsqueeze(-1)
    )
    for item in range(2):
        closed = closed + torch.logsumexp(
            probs.log() + per_row[index == item].sum(0), dim=0
        )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)
    assert run.log_joint == pytest.approx(classic, rel=1e-5)
    assert run.value == sites["probs"]


def test_a_kernel_morphism_reads_its_parameter_map() -> None:
    torch.manual_seed(1)
    program = Compiler(parse(KERNEL)).compile()
    monadic = program._morphism
    parameters = dict(monadic.named_parameters())
    weight = next(
        value for name, value in parameters.items() if name.endswith("weight")
    )
    bias = next(value for name, value in parameters.items() if name.endswith("bias"))
    torch.manual_seed(0)
    classic = trace(
        monadic,
        torch.tensor([[0.3, -0.2]]),
        observations={"x": torch.tensor([0.5, 0.1])},
    ).log_joint
    assert classic is not None
    module = _module(KERNEL)
    entry = program_entry(module, "prog")
    assert [(parameter.name, parameter.role) for parameter in entry.parameters] == [
        ("obs", "domain"),
        ("kernel_param_weight", "weight"),
        ("kernel_param_bias", "bias"),
    ]
    run = run_program(
        module,
        "prog",
        data={
            "obs": (0.3, -0.2),
            "kernel_param_weight": tuple(
                tuple(float(v) for v in row) for row in weight.detach()
            ),
            "kernel_param_bias": tuple(float(v) for v in bias.detach()),
        },
        sites={"x": (0.5, 0.1)},
    )
    assert run.log_joint == pytest.approx(float(classic.sum()), rel=1e-5)


def test_scores_add_as_weights() -> None:
    module = _module(SCORED)
    run = run_program(module, "prog", data={}, sites={"x": 0.8})
    assert run.log_joint == pytest.approx(
        -0.5 * 0.8**2 - 0.5 * math.log(2 * math.pi) + 0.64
    )


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("    sample (a, b) <- Normal(0.0, 1.0)\n    return a\n", "binds one name"),
        (
            "    observe y <- Normal(0.0, 1.0) [via=idx]\n    return y\n",
            "outside a grouped",
        ),
        (
            "    marginalize z <- Normal(0.0, 1.0) [reduction=mean]\n"
            "        observe y <- Normal(z, 1.0)\n    return z\n",
            "reduction",
        ),
    ],
)
def test_forms_outside_the_elaboration_are_reported_at_the_step(
    body: str, fragment: str
) -> None:
    source = (
        "object Obs : FinSet 4\nprogram prog : Obs -> Obs\n" + body + "export prog\n"
    )
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert captured.value.code == "qiec-program"
    assert fragment in captured.value.message
    assert captured.value.line == 3


def test_an_input_without_a_plate_gets_a_static_extent() -> None:
    """A vector input nothing shapes becomes an index parameter of the
    program, which a run reads off the data and the log joint agrees
    with torch on."""
    module = _module(
        "object Obs : FinSet 3\nprogram prog : Obs -> Obs\n"
        "    sample x <- Dirichlet(alpha)\n    return x\nexport prog\n"
    )
    computation = next(item for item in module.computations if item.name == "prog")
    assert [binder.name for binder in computation.telescope] == ["alpha_extent"]
    entry = program_entry(module, "prog")
    assert entry.telescope == computation.telescope
    (alpha,) = computation.parameters
    assert render_static(alpha.type) == "Tensor[Real]([alpha_extent])"
    assert render_static(computation.type.result) == "Tensor[Real]([alpha_extent])"
    run = run_program(
        module,
        "prog",
        data={"alpha": (1.0, 2.0, 3.0)},
        sites={"x": (0.2, 0.3, 0.5)},
    )
    expected = td.Dirichlet(torch.tensor([1.0, 2.0, 3.0])).log_prob(
        torch.tensor([0.2, 0.3, 0.5])
    )
    assert run.value == (0.2, 0.3, 0.5)
    assert run.log_joint == pytest.approx(expected.item(), rel=1e-6)
    with pytest.raises(KeyError, match="alpha"):
        run_program(module, "prog", data={"alpha": 1.0}, sites={"x": (1.0,)})


CALLS = """\
object Obs : FinSet 4

define shift(x : Real, by : Real) : Real !{} =
    return x + by

define noisy(x : Real) : Real !{random} =
    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))
    return y

instance random : Random

program prog : Obs -> Obs
    sample a <- Normal(0.0, 1.0)
    let b <- shift(a, 2.0)
    let c <- noisy(b)
    observe y <- Normal(c, 0.5)
    return c
export prog

define twice(y : Real) : Real !{random, score} =
    let first <- prog(y)
    let second <- prog(y)
    return first + second
"""


def test_programs_and_computations_call_each_other() -> None:
    module = _module(CALLS)
    program = next(item for item in module.computations if item.name == "prog")
    calls = []
    node = program.body
    while isinstance(node, Bind):
        if isinstance(node.first, Call):
            calls.append(node.first.name)
        node = node.then
    assert calls == ["shift", "noisy"]
    twice = next(item for item in module.computations if item.name == "twice")
    assert isinstance(twice.body, Bind)
    assert isinstance(twice.body.first, Call)
    assert twice.body.first.name == "prog"
    entry = program_entry(module, "prog")
    assert [site.name for site in entry.sites] == ["a", "y"]
    run = run_program(module, "prog", data={"y": 2.4}, sites={"a": 0.3, "noise": 2.35})
    expected = (
        -0.5 * 0.3**2
        - 0.5 * (0.05 / 0.1) ** 2
        - math.log(0.1)
        - 0.5 * (0.05 / 0.5) ** 2
        - math.log(0.5)
        - 1.5 * math.log(2 * math.pi)
    )
    assert run.value == 2.35
    assert run.log_joint == pytest.approx(expected)


def test_a_call_step_round_trips_through_the_emitter_and_is_qiec_only() -> None:
    module = parse(CALLS)
    source = module_to_source(module)
    assert "    let b <- shift(a, 2.0)\n" in source
    assert module_to_source(parse(source)) == source
    with pytest.raises(CompileError, match="only the QIEC route runs"):
        Compiler(module).compile()


def test_open_extents_reach_marginal_helpers_and_callers() -> None:
    """A captured input with an open extent keeps its shape inside the
    marginalization helper, a program passing the input on takes the
    extent as its own, and an authored computation names the extent
    with an index literal."""
    module = _module(
        "object Obs : FinSet 3\n"
        "program prog : Obs -> Obs\n"
        "    sample probs <- Dirichlet(alpha)\n"
        "    marginalize z <- Categorical(probs)\n"
        "        observe y <- Normal(mu[z], 1.0)\n"
        "    return probs\n"
        "export prog\n"
        "program outer : Obs -> Obs\n"
        "    let inner <- prog(alpha, mu, y)\n"
        "    return inner\n"
        "export outer\n"
        "define fixed(alpha : Tensor[Real]([3]), mu : Tensor[Real]([3]), y : Real)"
        " : Tensor[Real]([3]) !{random, score} =\n"
        "    let inner <- prog[3](alpha, mu, y)\n"
        "    return inner\n"
    )
    helper = next(item for item in module.computations if "marginal" in item.name)
    assert [binder.name for binder in helper.telescope] == ["alpha_extent"]
    outer = next(item for item in module.computations if item.name == "outer")
    assert [binder.name for binder in outer.telescope] == ["alpha_extent"]
    assert isinstance(outer.body, Bind)
    assert isinstance(outer.body.first, Call)
    assert [render_static(a) for a in outer.body.first.static_arguments] == [
        "alpha_extent"
    ]
    fixed = next(item for item in module.computations if item.name == "fixed")
    assert fixed.telescope == ()
    assert isinstance(fixed.body, Bind)
    assert isinstance(fixed.body.first, Call)
    assert [render_static(a) for a in fixed.body.first.static_arguments] == ["3"]
    data = {"alpha": (1.0, 2.0, 3.0), "y": 1.4, "mu": (0.0, 1.0, 2.0)}
    sites = {"probs": (0.2, 0.3, 0.5)}
    expected = td.Dirichlet(torch.tensor([1.0, 2.0, 3.0])).log_prob(
        torch.tensor([0.2, 0.3, 0.5])
    ).item() + math.log(
        sum(
            weight * math.exp(td.Normal(float(k), 1.0).log_prob(torch.tensor(1.4)))
            for k, weight in enumerate((0.2, 0.3, 0.5))
        )
    )
    for name in ("prog", "outer"):
        run = run_program(module, name, data=data, sites=sites)
        assert run.log_joint == pytest.approx(expected, rel=1e-6)
