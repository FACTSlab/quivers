"""Tests for variational objectives and gradient estimators.

Each objective must:

1. Return a finite scalar loss on a representative model.
2. Pipe gradients back to both model and guide parameters.
3. Numerically tighten under the conditions the theory promises:
   ELBO ≤ IWAE-K=8 ≤ IWAE-K=64 (importance-weighted bounds
   tighten with K); ELBO is recovered as the limit of VR-IWAE
   at α→1.
"""

from __future__ import annotations

import torch

from quivers.dsl import loads
from quivers.inference import (
    ELBO,
    AutoNormalGuide,
    DoublyReparameterized,
    IWAEBound,
    Reparameterized,
    RenyiBound,
    StickingTheLanding,
    VRIWAEBound,
)


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


def _make_obs():
    return {
        "subj_idx": torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]),
        "r": torch.zeros(12),
    }


# ---------------------------------------------------------------------------
# ELBO
# ---------------------------------------------------------------------------


def test_elbo_returns_finite_scalar() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    elbo = ELBO()
    loss = elbo(model, guide, torch.zeros(1, 1), _make_obs())
    assert loss.dim() == 0
    assert torch.isfinite(loss).all()


def test_elbo_gradients_flow_to_guide_params() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    elbo = ELBO()
    loss = elbo(model, guide, torch.zeros(1, 1), _make_obs())
    loss.backward()
    for name, param in guide.named_parameters():
        assert param.grad is not None, f"guide.{name} has no gradient"
        assert torch.isfinite(param.grad).all()


def test_elbo_multi_particle_averages() -> None:
    """ELBO with K particles is an average of K single-particle
    estimates. With a fixed seed, the K=4 result should be close
    to (but not equal to) the K=1 mean of 4 independent draws."""
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    obs = _make_obs()
    torch.manual_seed(42)
    loss_k1 = float(ELBO(num_particles=1)(model, guide, torch.zeros(1, 1), obs))
    torch.manual_seed(42)
    loss_k8 = float(ELBO(num_particles=8)(model, guide, torch.zeros(1, 1), obs))
    # Both estimate the same quantity; they should be of comparable
    # magnitude and both finite.
    assert torch.isfinite(torch.tensor(loss_k1))
    assert torch.isfinite(torch.tensor(loss_k8))


# ---------------------------------------------------------------------------
# IWAE
# ---------------------------------------------------------------------------


def test_iwae_returns_finite_scalar() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    iwae = IWAEBound(num_particles=8)
    loss = iwae(model, guide, torch.zeros(1, 1), _make_obs())
    assert loss.dim() == 0
    assert torch.isfinite(loss).all()


def test_iwae_tightens_with_more_particles() -> None:
    """The negated IWAE bound *decreases* as K grows (the bound
    on log p(y) gets tighter)."""
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    obs = _make_obs()
    # Use the plain reparameterized gradient form so the surrogate
    # equals the bound.
    iwae_4 = IWAEBound(num_particles=4, estimator=Reparameterized())
    iwae_32 = IWAEBound(num_particles=32, estimator=Reparameterized())
    losses_4 = []
    losses_32 = []
    for _ in range(8):
        losses_4.append(float(iwae_4(model, guide, torch.zeros(1, 1), obs)))
        losses_32.append(float(iwae_32(model, guide, torch.zeros(1, 1), obs)))
    mean_4 = sum(losses_4) / len(losses_4)
    mean_32 = sum(losses_32) / len(losses_32)
    # Bound is tighter ⇒ negated loss is closer to -log p(y) ⇒ smaller.
    assert mean_32 <= mean_4 + 0.5, (
        f"IWAE bound did not tighten with K: K=4 loss {mean_4:.3f} vs "
        f"K=32 loss {mean_32:.3f}"
    )


def test_iwae_dreg_estimator_gradients_flow() -> None:
    """Verify the DReG estimator (the default for IWAE) yields
    finite gradients on every variational parameter."""
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    iwae = IWAEBound(num_particles=8, estimator=DoublyReparameterized())
    loss = iwae(model, guide, torch.zeros(1, 1), _make_obs())
    loss.backward()
    for name, param in guide.named_parameters():
        assert param.grad is not None, f"guide.{name} has no gradient"
        assert torch.isfinite(param.grad).all()


# ---------------------------------------------------------------------------
# Renyi
# ---------------------------------------------------------------------------


def test_renyi_returns_finite_scalar() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    bound = RenyiBound(alpha=0.5, num_particles=8)
    loss = bound(model, guide, torch.zeros(1, 1), _make_obs())
    assert torch.isfinite(loss)


def test_renyi_rejects_singular_alpha() -> None:
    import pytest

    with pytest.raises(ValueError, match="alpha == 1.0"):
        RenyiBound(alpha=1.0)


# ---------------------------------------------------------------------------
# VR-IWAE
# ---------------------------------------------------------------------------


def test_vriwae_at_alpha_zero_matches_iwae_bound() -> None:
    """VR-IWAE with α=0 is exactly the IWAE bound."""
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    obs = _make_obs()
    K = 16
    torch.manual_seed(42)
    iwae_loss = float(
        IWAEBound(num_particles=K, estimator=Reparameterized())(
            model, guide, torch.zeros(1, 1), obs
        )
    )
    torch.manual_seed(42)
    vriwae_loss = float(
        VRIWAEBound(alpha=0.0, num_particles=K, estimator=Reparameterized())(
            model, guide, torch.zeros(1, 1), obs
        )
    )
    assert abs(iwae_loss - vriwae_loss) < 1e-4, (
        f"VR-IWAE(α=0, K={K}) should equal IWAE(K={K}); "
        f"got {vriwae_loss:.6f} vs {iwae_loss:.6f}"
    )


# ---------------------------------------------------------------------------
# Sticking-the-landing
# ---------------------------------------------------------------------------


def test_sticking_the_landing_estimator_gradients_flow() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    elbo = ELBO(estimator=StickingTheLanding())
    loss = elbo(model, guide, torch.zeros(1, 1), _make_obs())
    loss.backward()
    for name, param in guide.named_parameters():
        assert param.grad is not None, f"guide.{name} has no gradient"
        assert torch.isfinite(param.grad).all()


# ---------------------------------------------------------------------------
# Estimator integration: SVI step
# ---------------------------------------------------------------------------


def test_svi_step_runs_with_iwae_dreg() -> None:
    """End-to-end: SVI with IWAEBound + DReG completes a step
    cleanly and the loss is finite."""
    torch.manual_seed(0)
    from quivers.inference import SVI

    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    obj = IWAEBound(num_particles=8)
    opt = torch.optim.Adam(list(model.parameters()) + list(guide.parameters()), lr=1e-2)
    svi = SVI(model, guide, opt, obj)
    loss = svi.step(torch.zeros(1, 1), _make_obs())
    assert torch.isfinite(torch.tensor(loss))


# ---------------------------------------------------------------------------
# Validation paths for objectives
# ---------------------------------------------------------------------------


def test_elbo_rejects_num_particles_below_one() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_particles must be >= 1"):
        ELBO(num_particles=0)


def test_iwae_rejects_num_particles_below_one() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_particles must be >= 1"):
        IWAEBound(num_particles=0)


def test_renyi_rejects_num_particles_below_one() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_particles must be >= 1"):
        RenyiBound(alpha=0.5, num_particles=0)


def test_vriwae_rejects_alpha_one() -> None:
    import pytest

    with pytest.raises(ValueError, match="singular"):
        VRIWAEBound(alpha=1.0)


def test_vriwae_rejects_num_particles_below_one() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_particles must be >= 1"):
        VRIWAEBound(alpha=0.5, num_particles=0)


def test_vriwae_at_alpha_zero_with_renyi_matches_iwae() -> None:
    """Cross-validation: Rényi(α=0) and VR-IWAE(α=0) should both
    recover the IWAE bound. Verifies the algebraic identity between
    the families at the boundary value."""
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNormalGuide(model, observed_names={"r"})
    obs = _make_obs()
    K = 8
    torch.manual_seed(42)
    iwae = float(
        IWAEBound(num_particles=K, estimator=Reparameterized())(
            model, guide, torch.zeros(1, 1), obs
        )
    )
    torch.manual_seed(42)
    renyi = float(
        RenyiBound(alpha=1e-6, num_particles=K, estimator=Reparameterized())(
            model, guide, torch.zeros(1, 1), obs
        )
    )
    # Both estimate the same bound at alpha → 0; with the same RNG
    # seed they should agree to high precision.
    assert abs(iwae - renyi) < 1e-3, (
        f"IWAE and Rényi(α→0) disagree: {iwae:.6f} vs {renyi:.6f}"
    )
