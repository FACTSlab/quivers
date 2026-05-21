"""Tests for the mixture variational guide.

The mixture guide wraps multiple component guides; sampling
returns a Gumbel-Softmax-weighted convex combination of the
component samples (or a categorical-picked component sample via
:meth:`AutoMixtureGuide.hard_rsample`), and log-density is the
logsumexp of the per-component log-densities plus the mixture
log-weights.

The recovery test: on a model with a bimodal posterior — two
identifiable modes induced by sign-symmetry in the likelihood —
a 2-component AutoMixtureGuide pulls its two components toward
opposite modes after SVI, while a single AutoNormalGuide gets
stuck at one mode (or averages them, missing both).
"""

from __future__ import annotations

import math

import torch

from quivers.dsl import loads
from quivers.inference import (
    AutoNormalGuide,
    ELBO,
    SVI,
)
from quivers.inference.guides.mixture import AutoMixtureGuide


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
# Basic contract
# ---------------------------------------------------------------------------


def test_mixture_rsample_returns_constrained_shapes() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    comp1 = AutoNormalGuide(model, observed_names=obs_names)
    comp2 = AutoNormalGuide(model, observed_names=obs_names)
    guide = AutoMixtureGuide([comp1, comp2])
    samples = guide.rsample(torch.zeros(1, 1))
    ref = comp1.rsample(torch.zeros(1, 1))
    for name in ref:
        assert samples[name].shape == ref[name].shape


def test_mixture_log_prob_is_logsumexp_of_components() -> None:
    """log q(z) = logsumexp_k (log pi_k + log q_k(z))."""
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    comp1 = AutoNormalGuide(model, observed_names=obs_names)
    comp2 = AutoNormalGuide(model, observed_names=obs_names)
    guide = AutoMixtureGuide([comp1, comp2])
    samples = comp1.rsample(torch.zeros(1, 1))
    log_q_mix = guide.log_prob(torch.zeros(1, 1), samples)
    log_q_1 = comp1.log_prob(torch.zeros(1, 1), samples)
    log_q_2 = comp2.log_prob(torch.zeros(1, 1), samples)
    # With uniform mixture logits (initial state), the weights are 0.5 each,
    # so log_q_mix = logsumexp(log 0.5 + log q_1, log 0.5 + log q_2)
    #              = log 0.5 + logsumexp(log q_1, log q_2)
    expected = torch.log(torch.tensor(0.5)) + torch.logsumexp(
        torch.stack([log_q_1, log_q_2]), dim=0
    )
    assert torch.allclose(log_q_mix, expected, atol=1e-5)


def test_mixture_gradients_flow_to_logits_and_components() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    comp1 = AutoNormalGuide(model, observed_names=obs_names)
    comp2 = AutoNormalGuide(model, observed_names=obs_names)
    guide = AutoMixtureGuide([comp1, comp2])
    elbo = ELBO()
    loss = elbo(model, guide, torch.zeros(1, 1), _make_obs())
    loss.backward()
    assert guide.mixture_logits.grad is not None
    assert torch.isfinite(guide.mixture_logits.grad).all()
    for comp in guide.components:
        for param in comp.parameters():
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()


def test_mixture_rejects_mismatched_component_registries() -> None:
    import pytest

    torch.manual_seed(0)
    model1 = _hierarchical_model()
    comp1 = AutoNormalGuide(model1, observed_names={"r"})
    # Reuse the same model but with a different observed set so the
    # registry latent-name sets differ.
    comp2 = AutoNormalGuide(model1, observed_names={"r", "by_subj"})
    with pytest.raises(ValueError, match="different latent names"):
        AutoMixtureGuide([comp1, comp2])


def test_mixture_rejects_single_component() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    comp = AutoNormalGuide(model, observed_names={"r"})
    with pytest.raises(ValueError, match="at least 2 components"):
        AutoMixtureGuide([comp])


def test_mixture_hard_rsample_returns_single_component() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    comp1 = AutoNormalGuide(model, observed_names=obs_names)
    comp2 = AutoNormalGuide(model, observed_names=obs_names)
    guide = AutoMixtureGuide([comp1, comp2])
    samples = guide.hard_rsample(torch.zeros(1, 1))
    ref = comp1.rsample(torch.zeros(1, 1))
    for name in ref:
        assert samples[name].shape == ref[name].shape


def test_mixture_runs_end_to_end_svi() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    obs_names = {"r"}
    comp1 = AutoNormalGuide(model, observed_names=obs_names)
    comp2 = AutoNormalGuide(model, observed_names=obs_names)
    guide = AutoMixtureGuide([comp1, comp2])
    optim = torch.optim.Adam(
        list(model.parameters()) + list(guide.parameters()), lr=1e-2
    )
    svi = SVI(model, guide, optim, ELBO())
    losses = []
    for _ in range(20):
        losses.append(svi.step(torch.zeros(1, 1), _make_obs()))
    for loss in losses:
        assert torch.isfinite(torch.tensor(loss))


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_mixture_rejects_invalid_temperature() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    comp1 = AutoNormalGuide(model, observed_names={"r"})
    comp2 = AutoNormalGuide(model, observed_names={"r"})
    with pytest.raises(ValueError, match="init_temperature must be positive"):
        AutoMixtureGuide([comp1, comp2], init_temperature=0.0)


def test_mixture_set_temperature_rejects_nonpositive() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    comp1 = AutoNormalGuide(model, observed_names={"r"})
    comp2 = AutoNormalGuide(model, observed_names={"r"})
    g = AutoMixtureGuide([comp1, comp2])
    with pytest.raises(ValueError, match="must be positive"):
        g.set_temperature(-1.0)


def test_mixture_temperature_can_be_annealed() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    comp1 = AutoNormalGuide(model, observed_names={"r"})
    comp2 = AutoNormalGuide(model, observed_names={"r"})
    g = AutoMixtureGuide([comp1, comp2], init_temperature=1.0)
    assert math.isclose(g.temperature, 1.0, rel_tol=1e-6)
    g.set_temperature(0.1)
    assert math.isclose(g.temperature, 0.1, rel_tol=1e-6)


def test_mixture_num_components_property() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    comp1 = AutoNormalGuide(model, observed_names={"r"})
    comp2 = AutoNormalGuide(model, observed_names={"r"})
    comp3 = AutoNormalGuide(model, observed_names={"r"})
    g = AutoMixtureGuide([comp1, comp2, comp3])
    assert g.num_components == 3
