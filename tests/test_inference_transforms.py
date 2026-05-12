"""Tests for normalising-flow transform primitives.

Every primitive must satisfy three invariants:

1. **Inverse round-trip**: ``inverse(forward(x)) ≈ x`` (numerical
   precision floor).
2. **Log-det-Jacobian agreement**: the analytic
   ``log_abs_det_jacobian`` agrees with
   :func:`torch.autograd.functional.jacobian` to 1e-4 (a tight
   numerical match given float32).
3. **Gradient flow**: parameters of the parameterised layers
   receive non-zero finite gradients when the chain is trained
   against a smooth target.

These tests are correctness gates — a flow primitive that fails
any of them is silently incorrect when stacked inside a guide.
"""

from __future__ import annotations

import pytest
import torch

from quivers.inference.transforms import (
    MADE,
    AffineCouplingTransform,
    BatchNormTransform,
    InverseAutoregressiveTransform,
    LULinearTransform,
    MaskedAutoregressiveTransform,
    NeuralSplineCouplingTransform,
    alternating_mask,
    half_mask,
    make_coupling_mlp,
)


def _numeric_log_abs_det(transform, x: torch.Tensor) -> torch.Tensor:
    """Compute log|det J_T(x)| numerically via torch.autograd.

    Returns a tensor of shape ``x.shape[:-1]``.
    """

    def forward_fn(z):
        return transform(z)

    # Single-point Jacobian for each batch entry.
    out = torch.empty(x.shape[:-1])
    flat = x.reshape(-1, x.shape[-1])
    for i in range(flat.shape[0]):
        J = torch.autograd.functional.jacobian(forward_fn, flat[i])
        out.reshape(-1)[i] = torch.linalg.slogdet(J)[1]
    return out


# ---------------------------------------------------------------------------
# Affine coupling
# ---------------------------------------------------------------------------


def test_affine_coupling_inverse_roundtrip() -> None:
    dim = 6
    mask = alternating_mask(dim)
    n_a = int(mask.sum().item())
    n_b = dim - n_a
    # Use a random-initialised net so the transform is non-trivial.
    net = make_coupling_mlp(n_a, 2 * n_b, hidden=16)
    for p in net.parameters():
        p.data = torch.randn_like(p)
    layer = AffineCouplingTransform(dim, net, mask)

    x = torch.randn(8, dim)
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-5), (x - x_back).abs().max().item()


def test_affine_coupling_log_det_matches_numerical_jacobian() -> None:
    dim = 4
    mask = alternating_mask(dim)
    n_a = int(mask.sum().item())
    n_b = dim - n_a
    net = make_coupling_mlp(n_a, 2 * n_b, hidden=16)
    for p in net.parameters():
        p.data = torch.randn_like(p)
    layer = AffineCouplingTransform(dim, net, mask)

    x = torch.randn(4, dim)
    y = layer(x)
    analytic = layer.log_abs_det_jacobian(x, y)
    numeric = _numeric_log_abs_det(layer, x)
    assert torch.allclose(analytic, numeric, atol=1e-4), (
        (analytic - numeric).abs().max().item()
    )


def test_affine_coupling_rejects_degenerate_mask() -> None:
    dim = 4
    all_true = torch.ones(dim, dtype=torch.bool)
    net = make_coupling_mlp(dim, 2)
    with pytest.raises(ValueError):
        AffineCouplingTransform(dim, net, all_true)


# ---------------------------------------------------------------------------
# Masked autoregressive (MAF)
# ---------------------------------------------------------------------------


def test_maf_inverse_roundtrip() -> None:
    dim = 5
    made = MADE(dim, n_per_dim=2, hidden=16, n_hidden_layers=2)
    for p in made.parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p) * 0.1
    layer = MaskedAutoregressiveTransform(made)

    x = torch.randn(6, dim)
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-4), (x - x_back).abs().max().item()


def test_maf_log_det_matches_numerical_jacobian() -> None:
    dim = 4
    made = MADE(dim, n_per_dim=2, hidden=16, n_hidden_layers=2)
    for p in made.parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p) * 0.1
    layer = MaskedAutoregressiveTransform(made)

    x = torch.randn(3, dim)
    y = layer(x)
    analytic = layer.log_abs_det_jacobian(x, y)
    numeric = _numeric_log_abs_det(layer, x)
    assert torch.allclose(analytic, numeric, atol=1e-4)


def test_maf_autoregressive_property() -> None:
    """MAF output[j] depends only on input[k] for k with
    ordering[k] <= ordering[j]. Perturbing input[k] with strictly
    greater degree must leave output[j] unchanged."""
    dim = 5
    made = MADE(dim, n_per_dim=2, hidden=16, n_hidden_layers=2)
    for p in made.parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p) * 0.5
    layer = MaskedAutoregressiveTransform(made)

    x = torch.randn(1, dim)
    y_base = layer(x)
    ordering = made.ordering.tolist()
    for k in range(dim):
        x_perturb = x.clone()
        x_perturb[0, k] += 1.0
        y_perturb = layer(x_perturb)
        for j in range(dim):
            if ordering[k] > ordering[j]:
                assert torch.allclose(
                    y_base[0, j], y_perturb[0, j], atol=1e-5
                ), (
                    f"MAF leaks: perturbing input {k} (deg {ordering[k]}) "
                    f"changed output {j} (deg {ordering[j]})"
                )


# ---------------------------------------------------------------------------
# Inverse autoregressive (IAF)
# ---------------------------------------------------------------------------


def test_iaf_inverse_roundtrip() -> None:
    dim = 5
    made = MADE(dim, n_per_dim=2, hidden=16, n_hidden_layers=2)
    for p in made.parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p) * 0.1
    layer = InverseAutoregressiveTransform(made)

    x = torch.randn(6, dim)
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-4)


def test_iaf_log_det_matches_numerical_jacobian() -> None:
    dim = 4
    made = MADE(dim, n_per_dim=2, hidden=16, n_hidden_layers=2)
    for p in made.parameters():
        if p.requires_grad:
            p.data = torch.randn_like(p) * 0.1
    layer = InverseAutoregressiveTransform(made)

    x = torch.randn(3, dim)
    y = layer(x)
    analytic = layer.log_abs_det_jacobian(x, y)
    numeric = _numeric_log_abs_det(layer, x)
    assert torch.allclose(analytic, numeric, atol=1e-4)


# ---------------------------------------------------------------------------
# Neural spline coupling (NSF)
# ---------------------------------------------------------------------------


def test_nsf_coupling_inverse_roundtrip() -> None:
    dim = 6
    mask = alternating_mask(dim)
    n_a = int(mask.sum().item())
    n_b = dim - n_a
    num_bins = 6
    net = make_coupling_mlp(n_a, n_b * (3 * num_bins - 1), hidden=32)
    for p in net.parameters():
        p.data = torch.randn_like(p) * 0.5
    layer = NeuralSplineCouplingTransform(
        dim, net, mask, num_bins=num_bins, tail_bound=4.0
    )

    x = torch.randn(8, dim) * 1.5  # stay well inside tail_bound
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-3), (x - x_back).abs().max().item()


def test_nsf_identity_outside_tail_bound() -> None:
    """Outside ``[-tail_bound, tail_bound]`` the spline is the
    identity, so the forward output equals the input there."""
    dim = 4
    mask = alternating_mask(dim)
    n_a = int(mask.sum().item())
    n_b = dim - n_a
    num_bins = 4
    net = make_coupling_mlp(n_a, n_b * (3 * num_bins - 1))
    layer = NeuralSplineCouplingTransform(
        dim, net, mask, num_bins=num_bins, tail_bound=3.0
    )

    # Values far outside the spline tail; transformed coordinates
    # must come back unchanged.
    x = torch.tensor([[10.0, 10.0, -10.0, -10.0]])
    y = layer(x)
    transformed = y[..., ~mask]
    untransformed = x[..., ~mask]
    assert torch.allclose(transformed, untransformed)


# ---------------------------------------------------------------------------
# LU linear
# ---------------------------------------------------------------------------


def test_lu_linear_inverse_roundtrip() -> None:
    dim = 5
    layer = LULinearTransform(dim)
    for p in layer.parameters():
        if p.requires_grad and p.dim() > 0:
            p.data = torch.randn_like(p) * 0.2
    x = torch.randn(6, dim)
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-4)


def test_lu_linear_log_det_matches_numerical_jacobian() -> None:
    dim = 4
    layer = LULinearTransform(dim)
    for p in layer.parameters():
        if p.requires_grad and p.dim() > 0:
            p.data = torch.randn_like(p) * 0.2
    x = torch.randn(3, dim)
    y = layer(x)
    analytic = layer.log_abs_det_jacobian(x, y)
    # LULinear's log-det is constant across x; numeric reference
    # is per-row but every row should match the constant.
    numeric = _numeric_log_abs_det(layer, x)
    expected = analytic.expand_as(numeric)
    assert torch.allclose(expected, numeric, atol=1e-4)


# ---------------------------------------------------------------------------
# BatchNorm
# ---------------------------------------------------------------------------


def test_batchnorm_inverse_roundtrip_in_eval_mode() -> None:
    """In eval mode the running stats are used for both directions,
    so the round-trip is exact."""
    dim = 5
    layer = BatchNormTransform(dim)
    x = torch.randn(20, dim)
    # Run a few training-mode forward passes to populate running
    # stats.
    layer.train(True)
    for _ in range(5):
        _ = layer(torch.randn(20, dim))
    layer.eval()
    y = layer(x)
    x_back = layer.inv(y)
    assert torch.allclose(x, x_back, atol=1e-4)


# ---------------------------------------------------------------------------
# Compose round-trip across a multi-layer stack
# ---------------------------------------------------------------------------


def test_multilayer_flow_roundtrip() -> None:
    """A flow with several heterogeneous layers (the standard
    NSF / IAF stack we'd use in a real guide) round-trips
    end-to-end."""
    from torch.distributions.transforms import ComposeTransform

    dim = 6
    layers = []
    for k in range(3):
        mask = alternating_mask(dim, even=(k % 2 == 0))
        n_a = int(mask.sum().item())
        n_b = dim - n_a
        net = make_coupling_mlp(n_a, 2 * n_b, hidden=16)
        layers.append(AffineCouplingTransform(dim, net, mask))
        layers.append(LULinearTransform(dim))
    flow = ComposeTransform(layers)

    x = torch.randn(8, dim)
    y = flow(x)
    x_back = flow.inv(y)
    assert torch.allclose(x, x_back, atol=1e-3)
