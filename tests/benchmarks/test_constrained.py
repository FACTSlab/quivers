"""Tier-6 benchmarks: constrained-support recovery.

Every draw must lie inside the prior's constrained support; the
recovered posterior mean must match the analytical / quadrature
reference. These cases test the guide's bijector machinery:

* :class:`AutoNormalGuide` uses
  :func:`torch.distributions.transform_to`-derived bijectors to
  push unconstrained-Normal draws into the prior's support
  (sigmoid for ``Uniform(0,1)``, ``exp`` / ``softplus`` for
  positive-only, ``StickBreakingTransform`` for simplex, etc.).
* The bijector's log-det-Jacobian must be included in the guide's
  ``log_prob`` so the ELBO is correct.
"""

from __future__ import annotations

import torch

from quivers.inference import (
    AutoNormalGuide,
    ELBO,
    SVI,
)
from tests.benchmarks.datasets import (
    half_normal_scale,
    truncated_normal_recovery,
)
from tests.benchmarks.metrics import posterior_mean_error
from tests.benchmarks.references import (
    half_normal_scale_reference,
    truncated_normal_recovery_reference,
)


def _train(model, guide, obs, *, steps: int = 800, lr: float = 5e-2) -> None:
    optim = torch.optim.Adam(list(model.parameters()) + list(guide.parameters()), lr=lr)
    svi = SVI(model, guide, optim, ELBO())
    for _ in range(steps):
        svi.step(torch.zeros(1, 1), obs)


def _guide_samples(guide, n: int = 1500, *, site: str) -> torch.Tensor:
    out = torch.stack(
        [guide.rsample(torch.zeros(1, 1))[site].detach() for _ in range(n)],
        dim=0,
    )
    return out.reshape(-1)


def test_halfnormal_scale_samples_positive() -> None:
    """Every guide draw of sigma must satisfy sigma > 0."""
    torch.manual_seed(0)
    data = half_normal_scale()
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train(data.model, guide, data.observations, steps=800)
    samples = _guide_samples(guide, n=2000, site="sigma")
    assert (samples > 0).all(), "HalfNormal recovery produced non-positive samples"


def test_halfnormal_scale_recovers_mean() -> None:
    """AutoNormalGuide's posterior mean of sigma matches the
    quadrature reference."""
    torch.manual_seed(0)
    data = half_normal_scale()
    ref = half_normal_scale_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train(data.model, guide, data.observations, steps=1000)
    samples = _guide_samples(guide, n=2000, site="sigma")
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.15, (
        f"AutoNormal / HalfNormal scale: posterior mean error "
        f"{err:.4f} > 0.15 (true mean = {ref['mean']:.4f})"
    )


def test_truncated_normal_samples_in_unit_interval() -> None:
    """Every guide draw of mu must lie in (0, 1)."""
    torch.manual_seed(0)
    data = truncated_normal_recovery()
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train(data.model, guide, data.observations, steps=800)
    samples = _guide_samples(guide, n=2000, site="mu")
    assert (samples > 0).all() and (samples < 1).all(), (
        "TruncatedNormal recovery produced mu outside (0, 1)"
    )


def test_truncated_normal_recovers_mean() -> None:
    """Quadrature-derived posterior mean is matched within tolerance."""
    torch.manual_seed(0)
    data = truncated_normal_recovery()
    ref = truncated_normal_recovery_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train(data.model, guide, data.observations, steps=1000)
    samples = _guide_samples(guide, n=2000, site="mu")
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.05, (
        f"AutoNormal / TruncatedNormal: posterior mean error "
        f"{err:.4f} > 0.05 (true mean = {ref['mean']:.4f})"
    )
