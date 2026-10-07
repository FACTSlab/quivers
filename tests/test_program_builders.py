"""Programs assembled from the exported builders, without the DSL.

The step records, inline distributions, and plate steps are the objects
the compiler emits, and these tests hold them to that contract: a
program built from them scores exactly as the compiled program does,
parameters keep their dtype, densities stay exact in the tails, and a
mis-declared parameter width is an error.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributions as D

from quivers.continuous import (
    CholeskyFactor,
    Draw,
    Euclidean,
    Indexed,
    LKJCorrelationFactor,
    Let,
    MixedInlineDistribution,
    MonadicProgram,
    Observe,
    OrderedLogistic,
    PlateDraw,
    ProductSpace,
    Score,
    VectorisedObserve,
    make_fixed_halfnormal,
    make_fixed_lkj_cholesky,
    make_fixed_normal,
    reading,
)
from quivers.core import FinSet, Unit
from quivers.dsl import loads

_HIERARCHICAL = """
object Group : FinSet 3
object Row : FinSet 6
object Val : Real 1

program model : Row -> Val
    sample mu : Group <- Normal(loc=0.0, scale=1.0)
    sample sigma <- HalfNormal(scale=1.0)
    let m = mu[group]
    observe y : Row <- Normal(loc=m, scale=sigma)
    return y

export model
"""


def _normal_likelihood(value: Euclidean) -> MixedInlineDistribution:
    return MixedInlineDistribution(
        ProductSpace(components=(value, value)),
        value,
        param_spec=[("var", 1), ("var", 1)],
        dist_builder=lambda params: D.Normal(params[0], params[1]),
    )


def _hierarchical_values() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "mu": torch.randn(3, 1, generator=generator),
        "sigma": torch.rand(1, 1, generator=generator) + 0.5,
        "y": torch.randn(6, generator=generator),
        "group": torch.tensor([0, 0, 1, 1, 2, 2]),
    }


@pytest.mark.parametrize("gather", ["indexed", "let"])
def test_built_program_scores_as_the_compiled_one(gather: str) -> None:
    compiled = loads(_HIERARCHICAL).morphism
    assert isinstance(compiled, MonadicProgram)
    value = Euclidean(name="Val", dim=1)
    means = PlateDraw(3, make_fixed_normal(0.0, 1.0, value), domain=Unit)
    observe = VectorisedObserve(_normal_likelihood(value), torch.zeros(6))
    if gather == "indexed":
        steps = [
            Draw(names=("mu",), morphism=means),
            Draw(names=("sigma",), morphism=make_fixed_halfnormal(1.0, value)),
            Observe(
                names=("y",),
                morphism=observe,
                args=(Indexed(name="mu", indices=("group",)), "sigma"),
            ),
        ]
    else:
        steps = [
            Draw(names=("mu",), morphism=means),
            Draw(names=("sigma",), morphism=make_fixed_halfnormal(1.0, value)),
            Let(
                name="m",
                value=reading(lambda env: env["mu"][env["group"]], ("mu", "group")),
            ),
            Observe(names=("y",), morphism=observe, args=("m", "sigma")),
        ]
    built = MonadicProgram(
        FinSet(name="Row", cardinality=6), value, steps, return_vars=("y",)
    )
    x = torch.arange(6)
    values = _hierarchical_values()
    torch.testing.assert_close(
        built.log_joint(x, values), compiled.log_joint(x, values)
    )
    # The joint is the sum of the factors, computed independently.
    group_means = values["mu"][values["group"]].squeeze(-1)
    expected = (
        D.Normal(0.0, 1.0).log_prob(values["mu"]).sum()
        + D.HalfNormal(1.0).log_prob(values["sigma"]).sum()
        + D.Normal(group_means, values["sigma"].reshape(())).log_prob(values["y"]).sum()
    )
    torch.testing.assert_close(built.log_joint(x, values)[0], expected)


def test_compiled_programs_expose_their_step_records() -> None:
    compiled = loads(_HIERARCHICAL).morphism
    assert isinstance(compiled, MonadicProgram)
    kinds = [step.kind for step in compiled.steps]
    assert kinds == ["draw", "draw", "let", "observe"]
    first = compiled.steps[0]
    assert isinstance(first, Draw)
    assert isinstance(compiled.step_module(first), PlateDraw)


def test_step_records_reject_malformed_steps() -> None:
    value = Euclidean(name="Val", dim=1)
    prior = make_fixed_normal(0.0, 1.0, value)
    with pytest.raises(ValueError, match="two draw steps bind 'z'"):
        MonadicProgram(
            Unit,
            value,
            [Draw(names=("z",), morphism=prior), Draw(names=("z",), morphism=prior)],
            return_vars=("z",),
        )
    with pytest.raises(ValueError, match="binds no name"):
        MonadicProgram(Unit, value, [Draw(names=(), morphism=prior)], ("z",))
    with pytest.raises(Exception, match="at least one index"):
        Indexed(name="mu", indices=())


def test_a_score_step_joins_the_joint() -> None:
    value = Euclidean(name="Val", dim=1)
    program = MonadicProgram(
        Unit,
        value,
        [
            Draw(names=("z",), morphism=make_fixed_normal(0.0, 1.0, value)),
            Score(
                name="bonus",
                score=reading(lambda env: env["z"].sum(dim=-1), ("z",)),
            ),
        ],
        return_vars=("z",),
    )
    z = torch.tensor([[0.5], [-1.0]])
    joint = program.log_joint(torch.zeros(2, dtype=torch.long), {"z": z})
    torch.testing.assert_close(
        joint, D.Normal(0.0, 1.0).log_prob(z).squeeze(-1) + z.squeeze(-1)
    )


def test_stacked_parameters_keep_float64() -> None:
    """A float64 ordered observation agrees with a float64 reference."""
    value = FinSet(name="Level", cardinality=4)
    predictor = Euclidean(name="Eta", dim=1)
    cutpoints = Euclidean(name="Cut", dim=3)
    likelihood = MixedInlineDistribution(
        ProductSpace(components=(predictor, cutpoints)),
        value,
        param_spec=[("var", 1), ("var", 3)],
        dist_builder=lambda params: OrderedLogistic(params[0], params[1]),
        discrete=True,
        param_event_ranks=(0, 1),
    )
    program = MonadicProgram(
        Euclidean(name="In", dim=1),
        value,
        [
            Observe(
                names=("y",),
                morphism=VectorisedObserve(likelihood, torch.zeros(5)),
                args=("eta", "cuts"),
            )
        ],
        return_vars=("y",),
    )
    eta = torch.tensor([0.1, -0.3, 1.7, 2.2, -1.1], dtype=torch.float64) * (1 + 1e-9)
    cuts = torch.tensor([-1.0, 0.25, 1.5], dtype=torch.float64) + 1e-10
    y = torch.tensor([0, 1, 2, 3, 1])
    joint = program.log_joint(
        torch.zeros(5, 1, dtype=torch.float64), {"eta": eta, "cuts": cuts, "y": y}
    )
    reference = OrderedLogistic(eta, cuts).log_prob(y).sum()
    assert joint.dtype == torch.float64
    # An observed plate's density is one total, broadcast over the rows.
    torch.testing.assert_close(joint, reference.expand(5), rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("eta", [800.0, -800.0, 40.0])
def test_ordered_logistic_is_exact_in_the_tails(eta: float) -> None:
    cuts = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
    dist = OrderedLogistic(torch.tensor(eta, dtype=torch.float64), cuts)
    log_probs = torch.stack([dist.log_prob(torch.tensor(k)) for k in range(4)])
    assert torch.isfinite(log_probs).all()
    # Exact forms: the extreme categories are log-sigmoids; an interior
    # category is log(sigmoid(b) - sigmoid(a)), whose tail behaviour is
    # the sum of the two log-sigmoids and log1p(-exp(a - b)).
    offsets = cuts - eta
    first = torch.nn.functional.logsigmoid(offsets[0])
    last = torch.nn.functional.logsigmoid(-offsets[-1])
    a, b = offsets[0], offsets[1]
    second = (
        torch.nn.functional.logsigmoid(b)
        + torch.nn.functional.logsigmoid(-a)
        + torch.log1p(-torch.exp(a - b))
    )
    torch.testing.assert_close(log_probs[0], first)
    torch.testing.assert_close(log_probs[1], second)
    torch.testing.assert_close(log_probs[3], last)
    assert log_probs.exp().sum().item() == pytest.approx(1.0)
    if eta == 800.0:
        # Below the smallest normal float's log, where a floored
        # probability would have clamped.
        assert log_probs[0].item() < -708.5


def test_mixed_inline_distribution_checks_declared_widths() -> None:
    domain = Euclidean(name="In", dim=4)
    value = Euclidean(name="Out", dim=1)

    def builder(params: list[torch.Tensor]) -> D.Distribution:
        return D.Normal(params[0], params[1].sum(dim=-1, keepdim=True).exp())

    declared = MixedInlineDistribution(
        domain,
        value,
        param_spec=[("var", 1), ("var", 2)],
        dist_builder=builder,
        param_event_ranks=(0, 1),
    )
    with pytest.raises(ValueError, match="declared widths sum to 3"):
        declared.log_prob(torch.zeros(2, 4), torch.zeros(2, 1))
    unresolved = MixedInlineDistribution(
        domain,
        value,
        param_spec=[("var", 1), ("var", None)],
        dist_builder=builder,
        param_event_ranks=(0, 1),
    )
    assert unresolved.log_prob(torch.zeros(2, 4), torch.zeros(2, 1)).shape[0] == 2
    ambiguous = MixedInlineDistribution(
        domain,
        value,
        param_spec=[("var", None), ("var", None)],
        dist_builder=builder,
    )
    with pytest.raises(ValueError, match="cannot be divided among 2"):
        ambiguous.log_prob(torch.zeros(2, 3), torch.zeros(2, 1))
    per_coordinate = MixedInlineDistribution(
        Euclidean(name="Pair", dim=8),
        Euclidean(name="Out", dim=4),
        param_spec=[("var", None), ("var", None)],
        dist_builder=lambda params: D.Normal(params[0], params[1].exp()),
    )
    lp = per_coordinate.log_prob(torch.zeros(2, 8), torch.zeros(2, 4))
    # Four per-coordinate densities per row, summed to one.
    torch.testing.assert_close(
        lp, D.Normal(0.0, 1.0).log_prob(torch.zeros(2, 4)).sum(dim=-1)
    )


def test_inline_lkj_cholesky_takes_a_literal_concentration() -> None:
    source = """
object Dim : FinSet 3
object Factor : Real 3

program prior : Dim -> Factor
    sample chol : Dim <- LKJCholesky(concentration=2.0)
    return chol

export prior
"""
    program = loads(source).morphism
    draw = program.rsample(torch.zeros(2, dtype=torch.long))
    assert draw.shape == (2, 3, 3)
    correlation = draw @ draw.transpose(-1, -2)
    torch.testing.assert_close(correlation.diagonal(dim1=-2, dim2=-1), torch.ones(2, 3))


@pytest.mark.parametrize("kind", ["inline", "factor"])
def test_plate_of_lkj_priors_draws_one_factor_per_group(kind: str) -> None:
    family = (
        make_fixed_lkj_cholesky(2.0, CholeskyFactor(name="L", dim=3))
        if kind == "inline"
        else LKJCorrelationFactor(3, 2.0, Unit)
    )
    plate = PlateDraw(4, family, domain=Unit)
    draw = plate.rsample(torch.zeros(7, dtype=torch.long))
    assert draw.shape == (4, 9)
    reference = D.LKJCholesky(3, torch.tensor(2.0)).log_prob(draw.reshape(4, 3, 3))
    torch.testing.assert_close(
        plate.log_prob(torch.zeros(1), draw).squeeze(), reference.sum()
    )


def test_lkj_factor_honours_sample_shape_and_normalizes() -> None:
    factor = LKJCorrelationFactor(3, 1.5, Unit)
    x = torch.zeros(5, dtype=torch.long)
    draw = factor.rsample(x, torch.Size([2]))
    assert draw.shape == (2, 5, 9)
    reference = D.LKJCholesky(3, torch.tensor(1.5)).log_prob(draw.reshape(2, 5, 3, 3))
    torch.testing.assert_close(factor.log_prob(x, draw), reference)


def test_categorical_logits_are_per_row_vectors() -> None:
    """Categorical logits stack as one vector per row, not one column."""
    rows = 4
    logits_space = Euclidean(name="Logits", dim=3)
    likelihood = MixedInlineDistribution(
        logits_space,
        FinSet(name="Class", cardinality=3),
        param_spec=[("var", 3)],
        dist_builder=lambda params: D.Categorical(logits=params[0]),
        discrete=True,
        param_event_ranks=(1,),
    )
    logits = torch.randn(rows, 3, generator=torch.Generator().manual_seed(1))
    y = torch.tensor([0, 2, 1, 2])
    expected = torch.log_softmax(logits, dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(likelihood.log_prob(logits, y), expected)
