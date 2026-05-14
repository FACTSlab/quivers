"""Tests for :class:`Predictive` driving from an :class:`MCMCResult`."""

from __future__ import annotations

import torch

from quivers.dsl import loads
from quivers.inference import (
    AutoNormalGuide,
    HMCKernel,
    MCMC,
    Predictive,
)


def _normal_normal_model():
    return loads(
        "object Obs : 12\n"
        "program p : Obs -> Obs\n"
        "    mu <- Normal(0.0, 1.0)\n"
        "    observe y : Obs <- Normal(mu, 1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def test_predictive_with_mcmc_result_iterates_posterior_draws() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(12) + 1.0
    kernel = HMCKernel(step_size=0.1, num_steps=10, mass_matrix="identity")
    driver = MCMC(
        kernel=kernel,
        num_warmup=50,
        num_samples=30,
        num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})

    pred = Predictive(model, result)
    assert pred.num_samples == 2 * 30  # all draws by default

    draws = pred(torch.zeros(1, 1), observations={"y": y})
    assert "mu" in draws
    # First axis is the predictive-sample axis.
    assert draws["mu"].shape[0] == pred.num_samples


def test_predictive_caps_num_samples_at_mcmc_draw_count() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(12) + 1.0
    kernel = HMCKernel(step_size=0.1, num_steps=5, mass_matrix="identity")
    driver = MCMC(
        kernel=kernel,
        num_warmup=10,
        num_samples=15,
        num_chains=1,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    pred = Predictive(model, result, num_samples=1000)
    assert pred.num_samples == 15  # capped at available


def test_predictive_rejects_invalid_posterior_type() -> None:
    import pytest

    model = _normal_normal_model()
    with pytest.raises(TypeError, match="must be Guide or MCMCResult"):
        Predictive(model, "not a posterior")  # type: ignore[arg-type]


def test_predictive_with_guide_still_works() -> None:
    """The Layer 3 behavior — Predictive driven by a Guide —
    must continue to work after the MCMC overload."""
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(12) + 1.0
    guide = AutoNormalGuide(model, observed_names={"y"})
    pred = Predictive(model, guide, num_samples=5)
    draws = pred(torch.zeros(1, 1), observations={"y": y})
    assert "mu" in draws
    assert draws["mu"].shape[0] == 5


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_predictive_rejects_zero_num_samples() -> None:
    import pytest

    model = _normal_normal_model()
    guide = AutoNormalGuide(model, observed_names={"y"})
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        Predictive(model, guide, num_samples=0)


def test_predictive_negative_num_samples_rejected() -> None:
    import pytest

    model = _normal_normal_model()
    guide = AutoNormalGuide(model, observed_names={"y"})
    with pytest.raises(ValueError, match="num_samples must be >= 1"):
        Predictive(model, guide, num_samples=-5)


def test_predictive_with_guide_default_num_samples() -> None:
    """When num_samples is omitted, the guide path defaults to 100."""
    model = _normal_normal_model()
    guide = AutoNormalGuide(model, observed_names={"y"})
    pred = Predictive(model, guide)
    assert pred.num_samples == 100
