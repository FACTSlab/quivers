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
    SiteValue,
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


REDUCED = """\
object Component : FinSet 3
object Item : FinSet 2
object Resp : FinSet 4
program prog : Resp -> Resp
    sample probs <- Dirichlet(1.5) [over=Component]
    sample mu : Component <- Normal(0.0, 5.0)
    sample sigma : Component <- HalfNormal(1.0)
    marginalize cls : Component <- Categorical(probs) [over=Item, reduction=REDUCTION]
        observe r : Resp <- Normal(mu[cls], sigma[cls]) [via=idx]
    return probs
export prog
"""

PRODUCT_GROUP = """\
object Component : FinSet 2
object Item : FinSet 2
object Subj : FinSet 3
object Resp : FinSet 6
program prog : Resp -> Resp
    sample probs <- Dirichlet(1.0) [over=Component]
    sample mu : Component <- Normal(0.0, 5.0)
    sample sigma : Component <- HalfNormal(1.0)
    marginalize cls : Component <- Categorical(probs) [over=[Item, Subj]]
        observe r : Resp <- Normal(mu[cls], sigma[cls]) [via=[item_idx, subj_idx]]
    return probs
export prog
"""

NESTED_SAME_GROUP = """\
object Outer : FinSet 2
object Inner : FinSet 2
object Item : FinSet 2
object Resp : FinSet 4
program prog : Resp -> Resp
    sample probs_outer <- Dirichlet(1.0) [over=Outer]
    sample probs_inner : Outer <- Dirichlet(1.0) [over=Inner]
    sample mu : Inner <- Normal(0.0, 5.0)
    sample sigma : Inner <- HalfNormal(1.0)
    marginalize z : Outer <- Categorical(probs_outer) [over=Item]
        marginalize s : Inner <- Categorical(probs_inner[z]) [over=Item]
            observe r : Resp <- Normal(mu[s], sigma[s]) [via=idx]
    return probs_outer
export prog
"""

NESTED_PROJECTED = """\
object Outer : FinSet 2
object Inner : FinSet 2
object Item : FinSet 2
object Subj : FinSet 2
object Resp : FinSet 4
program prog : Resp -> Resp
    sample probs_outer <- Dirichlet(1.0) [over=Outer]
    sample probs_inner : Outer <- Dirichlet(1.0) [over=Inner]
    sample mu : Inner <- Normal(0.0, 5.0)
    sample sigma : Inner <- HalfNormal(1.0)
    marginalize z : Outer <- Categorical(probs_outer) [over=Item]
        marginalize s : Inner <- Categorical(probs_inner[z]) [over=[Item, Subj]]
            observe r : Resp <- Normal(mu[s], sigma[s]) [via=[item_idx, subj_idx]]
    return probs_outer
export prog
"""

HOISTED_DRAW = """\
object Component : FinSet 2
object Resp : FinSet 4
program prog : Resp -> Resp
    sample probs <- Dirichlet(1.0) [over=Component]
    marginalize cls : Component <- Categorical(probs)
        sample mu : Component <- Normal(0.0, 5.0)
        observe r : Resp <- Normal(mu[cls], 1.0)
    return probs
export prog
"""


def _sampled_types(module: QiecModule, name: str) -> dict[str, str]:
    """The sampled type of every site a program's body draws at.

    Parameters
    ----------
    module : QiecModule
        The checked module.
    name : str
        The program's name.

    Returns
    -------
    dict[str, str]
        By site label, the rendered static argument of its
        ``Random.sample`` request.
    """
    computation = next(item for item in module.computations if item.name == name)
    found: dict[str, str] = {}
    node = computation.body
    while isinstance(node, Bind):
        first = node.first
        if isinstance(first, Perform) and first.request.arguments:
            label = first.request.arguments[0]
            if isinstance(label, SiteValue):
                found[label.label] = render_static(first.request.static_arguments[0])
        node = node.then
    return found


def _mixture_rows(mu: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
    """Per-row, per-class log-likelihoods of unit-scale normal responses.

    Parameters
    ----------
    mu : torch.Tensor
        One location per class.
    responses : torch.Tensor
        The responses.

    Returns
    -------
    torch.Tensor
        A ``(rows, classes)`` matrix of log densities.
    """
    return td.Normal(mu.unsqueeze(0), 1.0).log_prob(responses.unsqueeze(-1))


def _priors(
    probs: torch.Tensor, mu: torch.Tensor, concentration: float
) -> torch.Tensor:
    """The prior density of a mixture's class probabilities and components.

    Parameters
    ----------
    probs : torch.Tensor
        The class probabilities, under a symmetric Dirichlet.
    mu : torch.Tensor
        The component locations, under ``Normal(0, 5)``, whose scales
        are clamped at one under ``HalfNormal(1)``.
    concentration : float
        The Dirichlet's symmetric concentration.

    Returns
    -------
    torch.Tensor
        The summed prior log density.
    """
    classes = probs.shape[-1]
    return (
        td.Dirichlet(torch.full((classes,), concentration)).log_prob(probs).sum()
        + td.Normal(0.0, 5.0).log_prob(mu).sum()
        + td.HalfNormal(1.0).log_prob(torch.ones_like(mu)).sum()
    )


@pytest.mark.parametrize("reduction", ["logsumexp", "sum", "mean"])
def test_marginalization_reductions_aggregate_the_grouped_shots(
    reduction: str,
) -> None:
    source = REDUCED.replace("REDUCTION", reduction)
    module = _module(source)
    data = {"r": (0.1, 0.5, -0.3, 2.0), "idx": (0, 0, 1, 1)}
    sites = {
        "probs": (0.2, 0.3, 0.5),
        "mu": (0.0, 1.0, -1.0),
        "sigma": (1.0, 1.0, 1.0),
    }
    run = run_program(module, "prog", data=data, sites=sites)
    probs, mu = torch.tensor(sites["probs"]), torch.tensor(sites["mu"])
    responses, index = torch.tensor(data["r"]), torch.tensor(data["idx"])
    per_row = _mixture_rows(mu, responses)
    closed = _priors(probs, mu, 1.5)
    for item in range(2):
        weighted = probs.log() + per_row[index == item].sum(0)
        if reduction == "logsumexp":
            closed = closed + torch.logsumexp(weighted, dim=0)
        elif reduction == "sum":
            closed = closed + weighted.sum()
        else:
            closed = closed + weighted.mean()
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)
    classic = _classic_log_joint(source, {**data, **sites})
    assert run.log_joint == pytest.approx(classic, rel=1e-5)


def test_a_reduction_on_a_continuous_latent_is_refused() -> None:
    source = (
        "object Obs : FinSet 4\nprogram prog : Obs -> Obs\n"
        "    marginalize z <- Normal(0.0, 1.0) [reduction=logsumexp]\n"
        "        observe y <- Normal(z, 1.0)\n    return z\nexport prog\n"
    )
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert "finite support" in captured.value.message


def test_a_product_group_flattens_its_fibrations_row_major() -> None:
    module = _module(PRODUCT_GROUP)
    data = {
        "r": (0.1, 0.5, -0.3, 2.0, 1.0, -1.5),
        "item_idx": (0, 0, 1, 1, 0, 1),
        "subj_idx": (0, 1, 2, 0, 2, 1),
    }
    sites = {"probs": (0.4, 0.6), "mu": (0.0, 1.0), "sigma": (1.0, 1.0)}
    run = run_program(module, "prog", data=data, sites=sites)
    probs, mu = torch.tensor(sites["probs"]), torch.tensor(sites["mu"])
    responses = torch.tensor(data["r"])
    flat = torch.tensor(data["item_idx"]) * 3 + torch.tensor(data["subj_idx"])
    per_row = _mixture_rows(mu, responses)
    closed = _priors(probs, mu, 1.0)
    for group in range(6):
        closed = closed + torch.logsumexp(
            probs.log() + per_row[flat == group].sum(0), dim=0
        )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)
    classic = _classic_log_joint(PRODUCT_GROUP, {**data, **sites})
    assert run.log_joint == pytest.approx(classic, rel=1e-5)


def test_a_product_fibration_must_match_the_group_arity() -> None:
    source = PRODUCT_GROUP.replace("[via=[item_idx, subj_idx]]", "[via=item_idx]")
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert "names 1 factor(s) but the group 'ItemxSubj' has 2" in captured.value.message


def _nested_closed_form(
    probs_outer: torch.Tensor,
    probs_inner: torch.Tensor,
    mu: torch.Tensor,
    responses: torch.Tensor,
    inner_index: torch.Tensor,
    inner_groups: int,
    projection: torch.Tensor,
) -> torch.Tensor:
    """The joint of a nested grouped mixture by explicit enumeration.

    Parameters
    ----------
    probs_outer : torch.Tensor
        The outer class prior.
    probs_inner : torch.Tensor
        One inner class prior per outer class.
    mu : torch.Tensor
        One location per inner class.
    responses : torch.Tensor
        The responses.
    inner_index : torch.Tensor
        Each response's inner group.
    inner_groups : int
        The number of inner groups.
    projection : torch.Tensor
        Each inner group's outer group.

    Returns
    -------
    torch.Tensor
        The log joint.
    """
    per_row = _mixture_rows(mu, responses)
    total = td.Dirichlet(torch.ones(2)).log_prob(probs_outer) + _priors(
        probs_inner, mu, 1.0
    )
    for group in range(int(projection.max()) + 1):
        shots = []
        for z in range(2):
            inner_total = torch.zeros(())
            for inner_group in range(inner_groups):
                if int(projection[inner_group]) != group:
                    continue
                rows = per_row[inner_index == inner_group].sum(0)
                inner_total = inner_total + torch.logsumexp(
                    probs_inner[z].log() + rows, dim=0
                )
            shots.append(probs_outer[z].log() + inner_total)
        total = total + torch.logsumexp(torch.stack(shots), dim=0)
    return total


def test_a_nested_grouped_marginalization_adds_per_group_marginals() -> None:
    module = _module(NESTED_SAME_GROUP)
    data = {"r": (0.1, 0.5, -0.3, 2.0), "idx": (0, 0, 1, 1)}
    sites = {
        "probs_outer": (0.3, 0.7),
        "probs_inner": ((0.5, 0.5), (0.1, 0.9)),
        "mu": (0.0, 1.0),
        "sigma": (1.0, 1.0),
    }
    run = run_program(module, "prog", data=data, sites=sites)
    closed = _nested_closed_form(
        torch.tensor(sites["probs_outer"]),
        torch.tensor(sites["probs_inner"]),
        torch.tensor(sites["mu"]),
        torch.tensor(data["r"]),
        torch.tensor(data["idx"]),
        2,
        torch.arange(2),
    )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


def test_a_nested_product_group_projects_onto_the_outer_group() -> None:
    module = _module(NESTED_PROJECTED)
    data = {
        "r": (0.1, 0.5, -0.3, 2.0),
        "item_idx": (0, 0, 1, 1),
        "subj_idx": (0, 1, 0, 1),
    }
    sites = {
        "probs_outer": (0.3, 0.7),
        "probs_inner": ((0.5, 0.5), (0.1, 0.9)),
        "mu": (0.0, 1.0),
        "sigma": (1.0, 1.0),
    }
    run = run_program(module, "prog", data=data, sites=sites)
    flat = torch.tensor(data["item_idx"]) * 2 + torch.tensor(data["subj_idx"])
    closed = _nested_closed_form(
        torch.tensor(sites["probs_outer"]),
        torch.tensor(sites["probs_inner"]),
        torch.tensor(sites["mu"]),
        torch.tensor(data["r"]),
        flat,
        4,
        torch.tensor([0, 0, 1, 1]),
    )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


def test_a_nested_group_unrelated_to_the_outer_is_refused() -> None:
    source = (
        NESTED_PROJECTED.replace("[over=[Item, Subj]]", "[over=Subj]")
        .replace("[via=[item_idx, subj_idx]]", "[via=subj_idx]")
        .replace("object Subj : FinSet 2", "object Subj : FinSet 3")
        .replace("Categorical(probs_inner[z])", "Categorical(probs_outer)")
    )
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert "projected onto it" in captured.value.message


def test_a_draw_inside_a_block_is_drawn_once_before_it() -> None:
    module = _module(HOISTED_DRAW)
    entry = program_entry(module, "prog")
    assert [site.name for site in entry.sites] == ["probs", "mu", "cls", "r"]
    data = {"r": (0.1, 0.5, -0.3, 2.0)}
    sites = {"probs": (0.4, 0.6), "mu": (0.0, 1.0)}
    run = run_program(module, "prog", data=data, sites=sites)
    probs, mu = torch.tensor(sites["probs"]), torch.tensor(sites["mu"])
    per_row = _mixture_rows(mu, torch.tensor(data["r"]))
    closed = (
        td.Dirichlet(torch.ones(2)).log_prob(probs)
        + td.Normal(0.0, 5.0).log_prob(mu).sum()
        + torch.logsumexp(probs.log() + per_row.sum(0), dim=0)
    )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


def test_a_draw_reading_the_latent_inside_a_block_is_refused() -> None:
    source = HOISTED_DRAW.replace(
        "sample mu : Component <- Normal(0.0, 5.0)", "sample mu <- Normal(cls, 5.0)"
    ).replace("Normal(mu[cls], 1.0)", "Normal(mu, 1.0)")
    with pytest.raises(QiecDiagnosticError) as captured:
        _module(source)
    assert "draw per value of the latent" in captured.value.message


@pytest.mark.parametrize(
    ("draw", "shape"),
    [
        ("sample pc <- Dirichlet(1.0)", (3,)),
        ("sample pc : Cat <- Dirichlet(1.0)", (3,)),
        ("sample pc : Item <- Dirichlet(1.0, 2.0)", (5, 2)),
        ("sample pc : Item <- Dirichlet(1.0) [over=Cat]", (5, 3)),
        ("sample pc <- Dirichlet(1.0, 2.0, 3.0, 4.0)", (4,)),
    ],
)
def test_a_vector_family_gathers_its_spread_literals(
    draw: str, shape: tuple[int, ...]
) -> None:
    source = (
        "object Cat : FinSet 3\nobject Item : FinSet 5\n"
        f"program prog : Cat -> Cat\n    {draw}\n    return pc\nexport prog\n"
    )
    module = _module(source)
    sampled = _sampled_types(module, "prog")["pc"]
    assert sampled == f"Tensor[Real]([{', '.join(map(str, shape))}])"


def test_an_integer_atom_beside_an_integer_family_shares_its_element() -> None:
    source = (
        "object Obs : FinSet 4\nprogram prog : Obs -> Obs\n"
        "    sample rate <- Gamma(2.0, 1.0)\n"
        "    sample y <- Mixture([0.3, 0.7], [PointMass(0.0), Poisson(rate)])\n"
        "    return rate\nexport prog\n"
    )
    module = _module(source)
    assert _sampled_types(module, "prog")["y"] == "Int"
    run = run_program(module, "prog", data={}, sites={"rate": 1.5, "y": 0})
    closed = td.Gamma(2.0, 1.0).log_prob(torch.tensor(1.5)) + math.log(
        0.3 + 0.7 * math.exp(-1.5)
    )
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)


FINITE_KERNEL = """\
object Cls : FinSet 3
object Obs : Real 2
morphism kernel : Cls -> Obs [role=kernel] ~ Normal
program prog : Cls -> Obs
    sample x <- kernel
    return x
export prog
"""


def test_a_kernel_over_a_finite_domain_reads_its_table() -> None:
    torch.manual_seed(1)
    program = Compiler(parse(FINITE_KERNEL)).compile()
    monadic = program._morphism
    table = next(
        value
        for name, value in dict(monadic.named_parameters()).items()
        if name.endswith("table")
    )
    torch.manual_seed(0)
    classic = trace(
        monadic, torch.tensor([2]), observations={"x": torch.tensor([0.5, 0.1])}
    ).log_joint
    assert classic is not None
    module = _module(FINITE_KERNEL)
    assert loads(dumps(module)) == module
    entry = program_entry(module, "prog")
    assert [(parameter.name, parameter.role) for parameter in entry.parameters] == [
        ("cls", "domain"),
        ("kernel_param_table", "table"),
    ]
    rows = tuple(tuple(float(v) for v in row) for row in table.detach())
    run = run_program(
        module,
        "prog",
        data={"cls": 2, "kernel_param_table": rows},
        sites={"x": (0.5, 0.1)},
    )
    assert run.log_joint == pytest.approx(float(classic.sum()), rel=1e-5)
    loc = torch.tensor(rows[2][:2])
    scale = torch.tensor(rows[2][2:]).exp()
    closed = td.Normal(loc, scale).log_prob(torch.tensor([0.5, 0.1])).sum()
    assert run.log_joint == pytest.approx(float(closed), rel=1e-5)
