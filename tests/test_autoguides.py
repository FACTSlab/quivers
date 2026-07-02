"""AutoGuide family conformance tests.

Covers the full `AutoGuide` ladder from mean-field Normal to
IAF and the structured / list combinators. Every guide is
exercised against the same `Guide` ABC contract:

* `Guide.rsample` returns a dict keyed by every latent site
  name with tensors on the correct constrained support.
* `Guide.log_prob` returns a finite `(batch,)`-shaped tensor.
* One SVI step reduces the objective on a Normal-Normal model.

The `AutoGuideList` and `AutoStructured` classes get extra
tests exercising their compositional surface (block partitioning,
per-site conditional dispatch, dependency ordering).
"""

from __future__ import annotations

import pytest
import torch

from quivers.dsl import loads
from quivers.inference.guides.auto_guide_list import AutoGuideList
from quivers.inference.guides.auto_structured import AutoStructured
from quivers.inference.guides.base import Guide
from quivers.inference.guides.delta import AutoDeltaGuide as AutoDelta
from quivers.inference.guides.flow import AutoIAFGuide as AutoIAFNormal
from quivers.inference.guides.laplace import (
    AutoLaplaceApproximation as AutoLaplace,
)
from quivers.inference.guides.multivariate_normal import (
    AutoLowRankMultivariateNormalGuide as AutoLowRankMVN,
    AutoMultivariateNormalGuide as AutoMultivariateNormal,
)
from quivers.inference.guides.normal import AutoNormalGuide as AutoNormal
from quivers.inference.objectives import ELBO
from quivers.inference.svi import SVI


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _normal_normal_model():
    """y_i ~ Normal(mu, 1) with mu ~ Normal(0, 1) for a
    conjugate one-parameter model."""
    return loads(
        "object Obs : FinSet 10\n"
        "program p : Obs -> Obs\n"
        "    sample mu <- Normal(0.0, 1.0)\n"
        "    observe y : Obs <- Normal(mu, 1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def _two_latent_model():
    """Two independent Normal latents, one observation depending on
    both. Useful for testing MVN correlations and structured
    dependencies."""
    return loads(
        "object Obs : FinSet 8\n"
        "program p : Obs -> Obs\n"
        "    sample a <- Normal(0.0, 1.0)\n"
        "    sample b <- Normal(0.0, 1.0)\n"
        "    let mu = sigmoid(a + b)\n"
        "    observe r : Obs <- Bernoulli(mu)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def _hierarchical_model():
    """Plate + scalar latents for coverage of the plate-shape path."""
    return loads(
        "object Subj : FinSet 4\n"
        "object Resp : FinSet 12\n"
        "program p : Resp -> Resp\n"
        "    sample sigma <- HalfNormal(1.0)\n"
        "    sample by_subj : Subj <- Normal(0.0, sigma)\n"
        "    let mu = sigmoid(by_subj[subj_idx])\n"
        "    observe r : Resp <- Bernoulli(mu)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def _normal_obs():
    return {"y": torch.zeros(10)}


def _two_latent_obs():
    return {"r": torch.ones(8)}


def _hierarchical_obs():
    return {
        "subj_idx": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
        "r": torch.zeros(12),
    }


# The set of every "single" autoguide the tests exercise uniformly.
# `AutoLaplace` is exercised separately because its two-phase design
# has a distinct sample/log-prob contract.

SINGLE_GUIDE_FACTORIES: list[tuple[str, object]] = [
    ("AutoNormal", lambda m, o: AutoNormal(m, observed_names=o)),
    (
        "AutoMultivariateNormal",
        lambda m, o: AutoMultivariateNormal(m, observed_names=o),
    ),
    (
        "AutoLowRankMVN",
        lambda m, o: AutoLowRankMVN(m, observed_names=o, rank=1),
    ),
    ("AutoDelta", lambda m, o: AutoDelta(m, observed_names=o)),
]


# ---------------------------------------------------------------------------
# 1. Uniform Guide-contract conformance for every autoguide.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,factory", SINGLE_GUIDE_FACTORIES)
def test_single_guide_rsample_shape_and_keys(name: str, factory) -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    guide = factory(model, obs_names)
    assert isinstance(guide, Guide)
    x = torch.zeros(1, 1)
    sample = guide.rsample(x)
    latent_set = set(guide.latent_names)
    assert set(sample.keys()) == latent_set, (
        f"{name}: rsample keys {set(sample.keys())!r} != latent set {latent_set!r}"
    )
    # HalfNormal should have positive samples.
    assert (sample["sigma"] > 0).all(), (
        f"{name}: sigma sample violates HalfNormal support"
    )


@pytest.mark.parametrize("name,factory", SINGLE_GUIDE_FACTORIES)
def test_single_guide_log_prob_finite(name: str, factory) -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    guide = factory(model, obs_names)
    x = torch.zeros(1, 1)
    sample = guide.rsample(x)
    log_q = guide.log_prob(x, sample)
    assert log_q.shape == (1,), f"{name}: log_prob shape {log_q.shape!r} != (1,)"
    assert torch.isfinite(log_q).all(), (
        f"{name}: log_prob returned non-finite value {log_q!r}"
    )


def test_iaf_rsample_and_log_prob() -> None:
    """AutoIAFNormal needs `>=2` latent dims, so we exercise it on the
    two-latent model."""
    torch.manual_seed(0)
    model = _two_latent_model()
    obs_names = {"r"}
    guide = AutoIAFNormal(
        model,
        observed_names=obs_names,
        num_flows=1,
        hidden_dim=8,
        num_hidden_layers=1,
    )
    assert isinstance(guide, Guide)
    x = torch.zeros(1, 1)
    sample = guide.rsample(x)
    assert set(sample.keys()) == set(guide.latent_names)
    log_q = guide.log_prob(x, sample)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()


# ---------------------------------------------------------------------------
# 2. SVI reduces ELBO loss with every guide.
# ---------------------------------------------------------------------------


def _run_svi_and_check_decrease(
    guide: Guide,
    model,
    x: torch.Tensor,
    obs: dict[str, torch.Tensor],
    n_steps: int = 40,
    lr: float = 0.05,
) -> tuple[float, float]:
    optim = torch.optim.Adam(guide.parameters(), lr=lr)
    svi = SVI(model=model, guide=guide, optim=optim, objective=ELBO())
    initial = svi.step(x, obs)
    for _ in range(n_steps - 1):
        svi.step(x, obs)
    final = svi.step(x, obs)
    return initial, final


@pytest.mark.parametrize("name,factory", SINGLE_GUIDE_FACTORIES)
def test_svi_step_reduces_loss(name: str, factory) -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    obs = _normal_obs()
    guide = factory(model, {"y"})
    x = torch.zeros(1, 1)
    initial, final = _run_svi_and_check_decrease(guide, model, x, obs)
    assert final < initial, (
        f"{name}: SVI failed to reduce loss ({initial:.4f} -> {final:.4f})"
    )


def test_iaf_svi_step_reduces_loss() -> None:
    torch.manual_seed(0)
    model = _two_latent_model()
    obs = _two_latent_obs()
    guide = AutoIAFNormal(
        model,
        observed_names={"r"},
        num_flows=1,
        hidden_dim=8,
        num_hidden_layers=1,
    )
    x = torch.zeros(1, 1)
    initial, final = _run_svi_and_check_decrease(
        guide, model, x, obs, n_steps=30, lr=0.02
    )
    assert final < initial, (
        f"AutoIAFNormal: SVI failed to reduce loss ({initial:.4f} -> {final:.4f})"
    )


# ---------------------------------------------------------------------------
# 3. Backward-compat aliases.
# ---------------------------------------------------------------------------


def test_short_name_aliases_are_the_old_classes() -> None:
    """The Pyro-flavored short names are the same classes as the
    canonical quivers ``*Guide`` classes."""
    from quivers.inference.guides.delta import AutoDeltaGuide
    from quivers.inference.guides.flow import AutoIAFGuide
    from quivers.inference.guides.laplace import AutoLaplaceApproximation
    from quivers.inference.guides.multivariate_normal import (
        AutoLowRankMultivariateNormalGuide,
        AutoMultivariateNormalGuide,
    )
    from quivers.inference.guides.normal import AutoNormalGuide

    assert AutoNormal is AutoNormalGuide
    assert AutoDelta is AutoDeltaGuide
    assert AutoLaplace is AutoLaplaceApproximation
    assert AutoMultivariateNormal is AutoMultivariateNormalGuide
    assert AutoLowRankMVN is AutoLowRankMultivariateNormalGuide
    assert AutoIAFNormal is AutoIAFGuide


# ---------------------------------------------------------------------------
# 4. AutoLaplace two-phase contract.
# ---------------------------------------------------------------------------


def test_laplace_map_phase_and_hessian_phase() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    obs = _normal_obs()
    guide = AutoLaplace(model, observed_names={"y"})
    x = torch.zeros(1, 1)
    # MAP phase: log_prob is zero everywhere.
    sample = guide.rsample(x)
    assert (guide.log_prob(x, sample) == 0).all()
    # Optimize the MAP a few steps.
    optim = torch.optim.Adam(guide.parameters(), lr=0.05)
    svi = SVI(model=model, guide=guide, optim=optim, objective=ELBO())
    for _ in range(10):
        svi.step(x, obs)
    # Fit the Hessian and check the guide now has a real density.
    guide.fit_hessian(model, x, obs)
    assert guide.hessian_fitted
    sample = guide.rsample(x)
    log_q = guide.log_prob(x, sample)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()
    # Now the log-prob should not be identically zero.
    assert (log_q != 0).any()


# ---------------------------------------------------------------------------
# 5. AutoGuideList block composition.
# ---------------------------------------------------------------------------


def test_auto_guide_list_concatenates_disjoint_parts() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    # Split sigma onto AutoDelta and by_subj onto AutoNormal by
    # manually building single-site sub-registries via a shared
    # observed-set trick: each part sees the other latent as
    # "observed", so its registry does not include it.
    delta_part = AutoDelta(model, observed_names=obs_names | {"by_subj"})
    normal_part = AutoNormal(model, observed_names=obs_names | {"sigma"})
    assert delta_part.latent_names == ["sigma"]
    assert normal_part.latent_names == ["by_subj"]
    guide = AutoGuideList({"global": delta_part, "local": normal_part})
    x = torch.zeros(1, 1)
    sample = guide.rsample(x)
    assert set(sample.keys()) == {"sigma", "by_subj"}
    log_q = guide.log_prob(x, sample)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()
    assert guide.part_labels == ("global", "local")
    assert guide.part("global") is delta_part


def test_auto_guide_list_rejects_overlapping_parts() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    a = AutoNormal(model, observed_names=obs_names | {"by_subj"})
    b = AutoNormal(model, observed_names=obs_names | {"by_subj"})
    with pytest.raises(ValueError, match="disjoint"):
        AutoGuideList({"a": a, "b": b})


def test_auto_guide_list_svi_reduces_loss() -> None:
    """AutoGuideList composed of AutoNormal + AutoDelta still
    reduces the ELBO on the two-latent model."""
    torch.manual_seed(0)
    model = _two_latent_model()
    obs = _two_latent_obs()
    a_guide = AutoNormal(model, observed_names={"r", "b"})
    b_guide = AutoNormal(model, observed_names={"r", "a"})
    guide = AutoGuideList({"a": a_guide, "b": b_guide})
    x = torch.zeros(1, 1)
    initial, final = _run_svi_and_check_decrease(guide, model, x, obs, n_steps=30)
    assert final < initial, (
        f"AutoGuideList SVI failed to reduce loss ({initial:.4f} -> {final:.4f})"
    )


# ---------------------------------------------------------------------------
# 6. AutoStructured — per-site conditionals + per-edge dependencies.
# ---------------------------------------------------------------------------


def test_auto_structured_uniform_conditional() -> None:
    """conditionals='normal' recovers a mean-field-Normal-shaped guide."""
    torch.manual_seed(0)
    model = _two_latent_model()
    obs_names = {"r"}
    guide = AutoStructured(model, observed_names=obs_names, conditionals="normal")
    assert guide.conditionals == {"a": "normal", "b": "normal"}
    x = torch.zeros(1, 1)
    sample = guide.rsample(x)
    assert set(sample.keys()) == {"a", "b"}
    log_q = guide.log_prob(x, sample)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()


def test_auto_structured_per_site_conditional_dispatch() -> None:
    """Per-site dict of conditionals — one site delta, one normal."""
    torch.manual_seed(0)
    model = _two_latent_model()
    obs_names = {"r"}
    guide = AutoStructured(
        model,
        observed_names=obs_names,
        conditionals={"a": "delta", "b": "normal"},
    )
    assert guide.conditionals == {"a": "delta", "b": "normal"}
    # The delta site "a" has no scale parameter; the normal site
    # "b" does.
    param_names = {name for name, _ in guide.named_parameters()}
    assert "loc_a" in param_names
    assert "loc_b" in param_names
    assert "log_scale_a" not in param_names
    assert "log_scale_b" in param_names
    x = torch.zeros(1, 1)
    sample = guide.rsample(x)
    log_q = guide.log_prob(x, sample)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()


def test_auto_structured_mvn_block() -> None:
    """Two 'mvn' sites share one Cholesky block."""
    torch.manual_seed(0)
    model = _two_latent_model()
    obs_names = {"r"}
    guide = AutoStructured(
        model,
        observed_names=obs_names,
        conditionals={"a": "mvn", "b": "mvn"},
    )
    # Shared MVN scale_tril parameters exist.
    param_names = {name for name, _ in guide.named_parameters()}
    assert "mvn_scale_diag_raw" in param_names
    assert "mvn_scale_offdiag" in param_names
    x = torch.zeros(1, 1)
    sample = guide.rsample(x)
    log_q = guide.log_prob(x, sample)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()


def test_auto_structured_linear_dependency() -> None:
    """'linear' dependency edge from a -> b registers a learnable
    affine module."""
    torch.manual_seed(0)
    model = _two_latent_model()
    obs_names = {"r"}
    guide = AutoStructured(
        model,
        observed_names=obs_names,
        conditionals="normal",
        dependencies={"b": {"a": "linear"}},
    )
    assert guide.dependencies == {"b": {"a": "linear"}}
    param_names = {name for name, _ in guide.named_parameters()}
    assert "deps.b.a.weight" in param_names
    x = torch.zeros(1, 1)
    sample = guide.rsample(x)
    log_q = guide.log_prob(x, sample)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()


def test_auto_structured_callable_dependency() -> None:
    """A user-supplied callable dependency composes with rsample and
    log_prob."""
    torch.manual_seed(0)
    model = _two_latent_model()
    obs_names = {"r"}

    def shift_fn(z_a: torch.Tensor) -> torch.Tensor:
        return 0.5 * z_a

    guide = AutoStructured(
        model,
        observed_names=obs_names,
        conditionals="normal",
        dependencies={"b": {"a": shift_fn}},
    )
    x = torch.zeros(1, 1)
    sample = guide.rsample(x)
    log_q = guide.log_prob(x, sample)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()


def test_auto_structured_rejects_downstream_dependency() -> None:
    """Dependencies must point strictly upstream in declaration order."""
    torch.manual_seed(0)
    model = _two_latent_model()
    obs_names = {"r"}
    with pytest.raises(ValueError, match="strictly upstream"):
        AutoStructured(
            model,
            observed_names=obs_names,
            conditionals="normal",
            dependencies={"a": {"b": "linear"}},
        )


def test_auto_structured_rejects_unknown_conditional() -> None:
    torch.manual_seed(0)
    model = _two_latent_model()
    obs_names = {"r"}
    with pytest.raises(ValueError, match="not in"):
        AutoStructured(model, observed_names=obs_names, conditionals="banana")


def test_auto_structured_svi_step_reduces_loss() -> None:
    """AutoStructured with linear dependency actually trains."""
    torch.manual_seed(0)
    model = _two_latent_model()
    obs = _two_latent_obs()
    guide = AutoStructured(
        model,
        observed_names={"r"},
        conditionals="normal",
        dependencies={"b": {"a": "linear"}},
    )
    x = torch.zeros(1, 1)
    initial, final = _run_svi_and_check_decrease(guide, model, x, obs, n_steps=30)
    assert final < initial, (
        f"AutoStructured SVI failed to reduce loss ({initial:.4f} -> {final:.4f})"
    )
