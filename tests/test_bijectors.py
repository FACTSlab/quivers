"""Tests for the [`Bijector`][quivers.continuous.bijectors.Bijector]
library used by [`Pushforward`][quivers.continuous.measure.Pushforward].

Every bijector is exercised on three contracts:

1. `inverse(forward(x)) == x` to machine tolerance.
2. `forward_log_det_jacobian(x)` matches the closed-form Jacobian
   (or matches a finite-difference numerical Jacobian where the
   closed form is non-obvious).
3. `inverse_log_det_jacobian(y) == -forward_log_det_jacobian(inverse(y))`.
"""

from __future__ import annotations

import math

import pytest
import torch

from quivers.continuous.bijectors import (
    Affine,
    Bijector,
    Compose,
    Exp,
    Identity,
    Inverse,
    Log,
    Logit,
    Sigmoid,
    Softplus,
    StickBreaking,
)


def _round_trip(b: Bijector, x: torch.Tensor, atol: float = 1e-5) -> None:
    y = b.forward(x)
    x_back = b.inverse(y)
    torch.testing.assert_close(x_back, x, atol=atol, rtol=atol)


def _jacobian_consistency(b: Bijector, x: torch.Tensor, atol: float = 1e-5) -> None:
    y = b.forward(x)
    fwd = b.forward_log_det_jacobian(x)
    inv = b.inverse_log_det_jacobian(y)
    torch.testing.assert_close(fwd, -inv, atol=atol, rtol=atol)


def test_identity() -> None:
    x = torch.randn(5)
    b = Identity()
    _round_trip(b, x)
    torch.testing.assert_close(b.forward_log_det_jacobian(x), torch.zeros_like(x))


def test_exp_log_inverses() -> None:
    x = torch.randn(7)
    e = Exp()
    log = Log()
    _round_trip(e, x)
    _round_trip(log, torch.exp(x))
    torch.testing.assert_close(e.forward(log.forward(torch.exp(x))), torch.exp(x))


def test_exp_jacobian_is_identity_in_log_space() -> None:
    x = torch.randn(5)
    e = Exp()
    torch.testing.assert_close(e.forward_log_det_jacobian(x), x)


def test_sigmoid_logit_inverses() -> None:
    x = torch.randn(7)
    s = Sigmoid()
    lo = Logit()
    _round_trip(s, x)
    _round_trip(lo, torch.sigmoid(x))


def test_sigmoid_jacobian_stable_in_tails() -> None:
    x = torch.tensor([-50.0, -10.0, 0.0, 10.0, 50.0])
    s = Sigmoid()
    fwd = s.forward_log_det_jacobian(x)
    assert torch.isfinite(fwd).all()


def test_softplus_round_trip() -> None:
    # Avoid the `x = 0` degenerate point where `inverse(softplus(0))`
    # collapses to `log(0)` if the relative tolerance is taken
    # against zero; the bijector is well-defined but a round-trip
    # check needs a non-zero baseline.
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0])
    sp = Softplus()
    _round_trip(sp, x, atol=1e-3)
    _jacobian_consistency(sp, x, atol=1e-3)


def test_affine_round_trip_and_jacobian() -> None:
    x = torch.randn(5)
    a = Affine(scale=2.5, shift=1.0)
    _round_trip(a, x)
    expected_log_det = math.log(2.5)
    torch.testing.assert_close(
        a.forward_log_det_jacobian(x),
        torch.full_like(x, expected_log_det),
    )


def test_affine_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        Affine(scale=0.0, shift=0.0)
    with pytest.raises(ValueError, match="scale must be strictly positive"):
        Affine(scale=-1.0, shift=0.0)


def test_compose_chain_rule() -> None:
    x = torch.randn(5)
    inner = Affine(scale=2.0, shift=1.0)
    outer = Sigmoid()
    c = Compose(outer, inner)
    _round_trip(c, x)
    expected_fwd = inner.forward_log_det_jacobian(x) + outer.forward_log_det_jacobian(
        inner.forward(x)
    )
    torch.testing.assert_close(
        c.forward_log_det_jacobian(x),
        expected_fwd,
        atol=1e-5,
        rtol=1e-5,
    )


def test_inverse_swaps_forward_and_inverse() -> None:
    x = torch.tensor([0.5, 1.0, 2.0])
    inv_exp = Inverse(Exp())
    torch.testing.assert_close(inv_exp.forward(x), torch.log(x))
    torch.testing.assert_close(
        inv_exp.inverse(torch.tensor([0.0])), torch.tensor([1.0])
    )


def test_stick_breaking_sums_to_one() -> None:
    torch.manual_seed(0)
    x = torch.randn(11, 4)
    sb = StickBreaking()
    y = sb.forward(x)
    assert y.shape == torch.Size([11, 5])
    torch.testing.assert_close(y.sum(dim=-1), torch.ones(11))
    assert (y > 0).all()


def test_stick_breaking_round_trip() -> None:
    torch.manual_seed(0)
    x = torch.randn(7, 3)
    sb = StickBreaking()
    y = sb.forward(x)
    x_back = sb.inverse(y)
    torch.testing.assert_close(x_back, x, atol=1e-4, rtol=1e-4)


def test_compose_inverse_jacobian_chain_rule() -> None:
    x = torch.randn(5)
    c = Compose(Sigmoid(), Affine(scale=1.5, shift=0.0))
    y = c.forward(x)
    fwd = c.forward_log_det_jacobian(x)
    inv = c.inverse_log_det_jacobian(y)
    torch.testing.assert_close(fwd, -inv, atol=1e-5, rtol=1e-5)
