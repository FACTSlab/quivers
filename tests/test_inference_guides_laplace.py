"""Tests for the Laplace-approximation guide.

The guide has two phases:

* Pre-:meth:`fit_hessian` it is a Dirac at the learnable MAP point
  with zero log-density.
* Post-:meth:`fit_hessian` it is a multivariate Gaussian with the
  cached scale_tril.

Both phases must satisfy the standard guide contract (sample shapes
match :class:`AutoNormalGuide`, gradients flow, log-density is
finite). On a conjugate Normal-Normal model the post-Hessian
covariance must match the analytical posterior precision.
"""

from __future__ import annotations
import textwrap

import math

import torch

from quivers.dsl import loads
from quivers.inference import (
    AutoNormalGuide,
    ELBO,
    SVI,
)
from quivers.inference.guides.laplace import AutoLaplaceApproximation


def _hierarchical_model():
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


def _normal_normal_model():
    """y_i ~ N(mu, 1) with mu ~ N(0, 1). Conjugate; posterior is
    N(N*y_bar / (N + 1), 1 / (N + 1))."""
    return loads(
        "object Obs : FinSet 10\n"
        "program p : Obs -> Obs\n"
        "    sample mu <- Normal(0.0, 1.0)\n"
        "    observe y : Obs <- Normal(mu, 1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def _make_obs():
    return {
        "subj_idx": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
        "r": torch.zeros(12),
    }


# ---------------------------------------------------------------------------
# Pre-Hessian (MAP) phase
# ---------------------------------------------------------------------------


def test_map_phase_rsample_shapes_match_autonormal() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    g_normal = AutoNormalGuide(model, observed_names={"r"})
    g_laplace = AutoLaplaceApproximation(model, observed_names={"r"})
    s_normal = g_normal.rsample(torch.zeros(1, 1))
    s_laplace = g_laplace.rsample(torch.zeros(1, 1))
    assert set(s_normal.keys()) == set(s_laplace.keys())
    for name in s_normal:
        assert s_normal[name].shape == s_laplace[name].shape


def test_map_phase_log_prob_is_zero() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoLaplaceApproximation(model, observed_names={"r"})
    samples = guide.rsample(torch.zeros(1, 1))
    log_q = guide.log_prob(torch.zeros(1, 1), samples)
    assert torch.all(log_q == 0.0)


def test_map_phase_gradient_flows_to_map_z() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoLaplaceApproximation(model, observed_names={"r"})
    elbo = ELBO()
    loss = elbo(model, guide, torch.zeros(1, 1), _make_obs())
    loss.backward()
    assert guide.map_z.grad is not None
    assert torch.isfinite(guide.map_z.grad).all()
    assert torch.any(guide.map_z.grad.abs() > 0)


# ---------------------------------------------------------------------------
# Hessian fitting + post-fit Gaussian phase
# ---------------------------------------------------------------------------


def test_fit_hessian_flips_state_and_produces_psd_cholesky() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoLaplaceApproximation(model, observed_names={"r"})
    assert not guide.hessian_fitted
    guide.fit_hessian(model, torch.zeros(1, 1), _make_obs())
    assert guide.hessian_fitted
    L = guide._scale_tril
    diag = torch.diagonal(L)
    assert torch.all(diag > 0), "scale_tril diagonal must be strictly positive"


def test_post_hessian_log_prob_is_finite() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoLaplaceApproximation(model, observed_names={"r"})
    guide.fit_hessian(model, torch.zeros(1, 1), _make_obs())
    samples = guide.rsample(torch.zeros(1, 1))
    log_q = guide.log_prob(torch.zeros(1, 1), samples)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()


def test_post_hessian_samples_have_correct_shapes() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    g_normal = AutoNormalGuide(model, observed_names={"r"})
    g_laplace = AutoLaplaceApproximation(model, observed_names={"r"})
    g_laplace.fit_hessian(model, torch.zeros(1, 1), _make_obs())
    s_normal = g_normal.rsample(torch.zeros(1, 1))
    s_laplace = g_laplace.rsample(torch.zeros(1, 1))
    for name in s_normal:
        assert s_normal[name].shape == s_laplace[name].shape


# ---------------------------------------------------------------------------
# Conjugate recovery
# ---------------------------------------------------------------------------


def test_normal_normal_laplace_recovers_analytical_posterior() -> None:
    """Normal(0, 1) prior + 10 Normal(mu, 1) observations gives a
    closed-form posterior N(N*y_bar/(N+1), 1/(N+1)). The Laplace
    approximation is *exact* for a Gaussian posterior, so the
    learned MAP must match N*y_bar/(N+1) and the Hessian must give
    a variance of 1/(N+1) (up to optimisation tolerance)."""
    torch.manual_seed(0)
    model = _normal_normal_model()
    N = 10
    y = torch.randn(N) + 2.0
    y_bar = float(y.mean())
    true_mean = N * y_bar / (N + 1)
    true_var = 1.0 / (N + 1)

    guide = AutoLaplaceApproximation(model, observed_names={"y"})
    optim = torch.optim.Adam(guide.parameters(), lr=5e-2)
    svi = SVI(model, guide, optim, ELBO())
    for _ in range(800):
        svi.step(torch.zeros(1, 1), {"y": y})
    guide.fit_hessian(model, torch.zeros(1, 1), {"y": y})

    map_estimate = float(guide.map_z.detach())
    variance = float(guide._scale_tril[0, 0].detach() ** 2)
    assert math.isclose(map_estimate, true_mean, abs_tol=0.05), (
        f"MAP recovered {map_estimate:.4f}; expected {true_mean:.4f}"
    )
    assert math.isclose(variance, true_var, rel_tol=0.15), (
        f"Laplace variance recovered {variance:.4f}; expected {true_var:.4f}"
    )


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_laplace_rejects_zero_dim_model() -> None:
    """A model with no continuous latents has nothing for Laplace
    to fit. Constructing the guide should raise."""
    import pytest

    # A "model" with only observed sites — i.e. no latents.
    src = (
        "object Obs : FinSet 4\n"
        "program p : Obs -> Obs\n"
        "    sample mu <- Normal(0.0, 1.0)\n"
        "    observe y : Obs <- Normal(mu, 1.0)\n"
        "    return mu\n"
        "export p\n"
    )
    model = loads(textwrap.dedent(src)).morphism
    # Mark every latent as observed to make the registry empty.
    with pytest.raises(ValueError, match="zero total"):
        AutoLaplaceApproximation(model, observed_names={"y", "mu"})
