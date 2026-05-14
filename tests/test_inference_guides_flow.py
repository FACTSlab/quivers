"""Tests for normalizing-flow variational guides.

Verify the flow-based guides satisfy the standard guide
contract: sample shapes match the mean-field reference, gradients
flow through every flow-stack parameter under ELBO and IWAE, the
log-density round-trips (``log_prob(rsample())`` is finite), and
the IAF / NSF guides can be optimized end-to-end via SVI without
numerical blow-up.
"""

from __future__ import annotations

import torch

from quivers.dsl import loads
from quivers.inference import (
    AutoNormalGuide,
    ELBO,
    IWAEBound,
    SVI,
)
from quivers.inference.guides.flow import (
    AutoIAFGuide,
    AutoNeuralSplineGuide,
    AutoNormalizingFlow,
)
from quivers.inference.transforms import (
    MADE,
    InverseAutoregressiveTransform,
)


def _hierarchical_model():
    return loads(
        "object Subj : 4\n"
        "object Resp : 12\n"
        "program p : Resp -> Resp\n"
        "    sigma <- HalfNormal(1.0)\n"
        "    by_subj : Subj <- Normal(0.0, sigma)\n"
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
# Shape contract
# ---------------------------------------------------------------------------


def test_iaf_rsample_shapes_match_autonormal() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    g_normal = AutoNormalGuide(model, observed_names={"r"})
    g_iaf = AutoIAFGuide(model, observed_names={"r"}, num_flows=2)
    s_normal = g_normal.rsample(torch.zeros(1, 1))
    s_iaf = g_iaf.rsample(torch.zeros(1, 1))
    assert set(s_normal.keys()) == set(s_iaf.keys())
    for name in s_normal:
        assert s_normal[name].shape == s_iaf[name].shape


def test_nsf_rsample_shapes_match_autonormal() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    g_normal = AutoNormalGuide(model, observed_names={"r"})
    g_nsf = AutoNeuralSplineGuide(model, observed_names={"r"}, num_flows=2)
    s_normal = g_normal.rsample(torch.zeros(1, 1))
    s_nsf = g_nsf.rsample(torch.zeros(1, 1))
    assert set(s_normal.keys()) == set(s_nsf.keys())
    for name in s_normal:
        assert s_normal[name].shape == s_nsf[name].shape


# ---------------------------------------------------------------------------
# log_prob round-trip
# ---------------------------------------------------------------------------


def test_iaf_log_prob_finite_on_own_samples() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoIAFGuide(model, observed_names={"r"}, num_flows=2)
    samples = guide.rsample(torch.zeros(1, 1))
    log_q = guide.log_prob(torch.zeros(1, 1), samples)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()


def test_nsf_log_prob_finite_on_own_samples() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNeuralSplineGuide(model, observed_names={"r"}, num_flows=2, num_bins=4)
    samples = guide.rsample(torch.zeros(1, 1))
    log_q = guide.log_prob(torch.zeros(1, 1), samples)
    assert log_q.shape == (1,)
    assert torch.isfinite(log_q).all()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_iaf_gradients_flow_through_made_weights() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoIAFGuide(model, observed_names={"r"}, num_flows=2)
    elbo = ELBO()
    loss = elbo(model, guide, torch.zeros(1, 1), _make_obs())
    loss.backward()
    grads_seen = 0
    for name, param in guide.named_parameters():
        if param.grad is None:
            continue
        if torch.any(param.grad.abs() > 0):
            grads_seen += 1
        assert torch.isfinite(param.grad).all()
    assert grads_seen > 0, "no IAF parameter received a non-zero gradient"


def test_nsf_gradients_flow_through_coupling_mlps() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoNeuralSplineGuide(model, observed_names={"r"}, num_flows=2, num_bins=4)
    elbo = ELBO()
    loss = elbo(model, guide, torch.zeros(1, 1), _make_obs())
    loss.backward()
    grads_seen = 0
    for name, param in guide.named_parameters():
        if param.grad is None:
            continue
        if torch.any(param.grad.abs() > 0):
            grads_seen += 1
        assert torch.isfinite(param.grad).all()
    assert grads_seen > 0, "no NSF parameter received a non-zero gradient"


# ---------------------------------------------------------------------------
# End-to-end SVI
# ---------------------------------------------------------------------------


def test_iaf_svi_runs_without_blowup() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoIAFGuide(model, observed_names={"r"}, num_flows=2)
    optim = torch.optim.Adam(
        list(model.parameters()) + list(guide.parameters()), lr=1e-3
    )
    svi = SVI(model, guide, optim, ELBO())
    losses = []
    for _ in range(20):
        losses.append(svi.step(torch.zeros(1, 1), _make_obs()))
    for loss in losses:
        assert torch.isfinite(torch.tensor(loss))


def test_iaf_runs_with_iwae() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    guide = AutoIAFGuide(model, observed_names={"r"}, num_flows=2)
    iwae = IWAEBound(num_particles=4)
    loss = iwae(model, guide, torch.zeros(1, 1), _make_obs())
    loss.backward()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Custom flow stack
# ---------------------------------------------------------------------------


def test_autonormalizingflow_accepts_user_transforms() -> None:
    torch.manual_seed(0)
    model = _hierarchical_model()
    registry = AutoNormalGuide(model, observed_names={"r"}).registry
    D_total = registry.total_unconstrained_dim
    made = MADE(dim=D_total, n_per_dim=2, hidden=16, n_hidden_layers=1)
    transforms = [InverseAutoregressiveTransform(made)]
    guide = AutoNormalizingFlow(model, observed_names={"r"}, transforms=transforms)
    s = guide.rsample(torch.zeros(1, 1))
    log_q = guide.log_prob(torch.zeros(1, 1), s)
    assert torch.isfinite(log_q).all()


# ---------------------------------------------------------------------------
# Validation paths
# ---------------------------------------------------------------------------


def test_iaf_rejects_num_flows_zero() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    with pytest.raises(ValueError, match="num_flows must be >= 1"):
        AutoIAFGuide(model, observed_names={"r"}, num_flows=0)


def test_nsf_rejects_num_flows_zero() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    with pytest.raises(ValueError, match="num_flows must be >= 1"):
        AutoNeuralSplineGuide(model, observed_names={"r"}, num_flows=0)


def test_autonormalizing_flow_rejects_empty_transforms() -> None:
    import pytest

    torch.manual_seed(0)
    model = _hierarchical_model()
    with pytest.raises(ValueError, match="transforms list must be non-empty"):
        AutoNormalizingFlow(model, observed_names={"r"}, transforms=[])


def test_iaf_rejects_single_dim_model() -> None:
    """IAF needs >= 2 dimensions for the autoregressive ordering
    to make sense. A 1-dim model (single scalar latent) is rejected."""
    import pytest

    src = (
        "object Obs : 4\n"
        "program p : Obs -> Obs\n"
        "    mu <- Normal(0.0, 1.0)\n"
        "    observe y : Obs <- Normal(mu, 1.0)\n"
        "    return mu\n"
        "export p\n"
    )
    from quivers.dsl import loads

    model = loads(src).morphism
    with pytest.raises(ValueError, match=">= 2 unconstrained"):
        AutoIAFGuide(model, observed_names={"y"})


def test_nsf_rejects_single_dim_model() -> None:
    import pytest

    src = (
        "object Obs : 4\n"
        "program p : Obs -> Obs\n"
        "    mu <- Normal(0.0, 1.0)\n"
        "    observe y : Obs <- Normal(mu, 1.0)\n"
        "    return mu\n"
        "export p\n"
    )
    from quivers.dsl import loads

    model = loads(src).morphism
    with pytest.raises(ValueError, match=">= 2 unconstrained"):
        AutoNeuralSplineGuide(model, observed_names={"y"})
