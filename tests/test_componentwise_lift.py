"""Tests for componentwise_lift — the free monoid functor on morphisms.

The lift f^k of a morphism f: A -> B produces f^k: A^k -> B^k where
f^k[a1,...,ak, b1,...,bk] = prod_i f[ai, bi].

This is the functorial action of the free monoid on morphisms in the
Kleisli category of the fuzzy powerset monad.
"""

from __future__ import annotations

import torch
import pytest

from quivers.core.tensor_ops import componentwise_lift, noisy_or_contract


class TestComponentwiseLiftBasic:
    def test_k0_returns_ones(self):
        """k=0 is the empty-string stratum: trivial 1x1 tensor."""
        f = torch.rand(3, 4)
        result = componentwise_lift(f, 0)
        assert result.shape == (1, 1)
        assert result.item() == pytest.approx(1.0)

    def test_k1_is_identity(self):
        """k=1 returns the morphism unchanged."""
        f = torch.rand(3, 4)
        result = componentwise_lift(f, 1)
        torch.testing.assert_close(result, f)

    def test_k2_shape(self):
        """k=2 on (3, 4) gives shape (3, 3, 4, 4)."""
        f = torch.rand(3, 4)
        result = componentwise_lift(f, 2)
        assert result.shape == (3, 3, 4, 4)

    def test_k3_shape(self):
        """k=3 on (2, 5) gives shape (2, 2, 2, 5, 5, 5)."""
        f = torch.rand(2, 5)
        result = componentwise_lift(f, 3)
        assert result.shape == (2, 2, 2, 5, 5, 5)

    def test_negative_k_raises(self):
        f = torch.rand(3, 4)
        with pytest.raises(ValueError, match="k must be >= 0"):
            componentwise_lift(f, -1)


class TestComponentwiseLiftProduct:
    """Verify the defining property: f^k[a1,...,ak,b1,...,bk] = prod f[ai,bi]."""

    def test_k2_exhaustive(self):
        """Exhaustively check the product property for k=2."""
        f = torch.rand(3, 4)
        lifted = componentwise_lift(f, 2)

        for a1 in range(3):
            for a2 in range(3):
                for b1 in range(4):
                    for b2 in range(4):
                        expected = f[a1, b1] * f[a2, b2]
                        torch.testing.assert_close(
                            lifted[a1, a2, b1, b2],
                            expected,
                            atol=1e-6,
                            rtol=1e-6,
                        )

    def test_k3_spot_check(self):
        """Spot check a few entries for k=3."""
        f = torch.rand(2, 3)
        lifted = componentwise_lift(f, 3)

        # check several random index tuples
        for a1, a2, a3, b1, b2, b3 in [
            (0, 0, 0, 0, 0, 0),
            (1, 0, 1, 2, 1, 0),
            (0, 1, 1, 1, 2, 2),
        ]:
            expected = f[a1, b1] * f[a2, b2] * f[a3, b3]
            torch.testing.assert_close(
                lifted[a1, a2, a3, b1, b2, b3],
                expected,
                atol=1e-6,
                rtol=1e-6,
            )

    def test_k2_on_identity_is_identity(self):
        """Lifting the identity morphism gives the identity on pairs."""
        n = 4
        eye = torch.eye(n)
        lifted = componentwise_lift(eye, 2)

        # lifted[a1,a2,b1,b2] = delta(a1,b1)*delta(a2,b2)
        for a1 in range(n):
            for a2 in range(n):
                for b1 in range(n):
                    for b2 in range(n):
                        expected = 1.0 if (a1 == b1 and a2 == b2) else 0.0
                        assert lifted[a1, a2, b1, b2].item() == pytest.approx(
                            expected,
                            abs=1e-6,
                        )


class TestComponentwiseLiftGradients:
    def test_gradient_flow(self):
        """Gradients propagate through componentwise_lift."""
        f = torch.rand(3, 4, requires_grad=True)
        lifted = componentwise_lift(f, 2)
        loss = lifted.sum()
        loss.backward()

        assert f.grad is not None
        assert torch.isfinite(f.grad).all()
        assert (f.grad != 0).any()


class TestComponentwiseLiftComposition:
    """Verify that lifted morphisms compose correctly via noisy-OR."""

    def test_lifted_composition_shape(self):
        """Composing two lifted morphisms gives the right shape."""
        f = torch.rand(3, 4)
        g = torch.rand(4, 5)

        f2 = componentwise_lift(f, 2)  # (3, 3, 4, 4)
        g2 = componentwise_lift(g, 2)  # (4, 4, 5, 5)

        # compose over the shared (4, 4) dimensions
        composed = noisy_or_contract(f2, g2, n_contract=2)
        assert composed.shape == (3, 3, 5, 5)

    def test_lift_then_compose_vs_compose_then_lift(self):
        """The free monoid functor should (approximately) preserve composition.

        For boolean (crisp) morphisms, lifting commutes exactly with
        noisy-OR composition.  For fuzzy morphisms, the two paths may
        differ due to noisy-OR non-associativity, but should agree on
        boolean inputs.
        """
        # use boolean (crisp) morphisms where noisy-OR = classical OR
        f = torch.zeros(3, 4)
        f[0, 1] = 1.0
        f[1, 2] = 1.0
        f[2, 0] = 1.0

        g = torch.zeros(4, 2)
        g[0, 0] = 1.0
        g[1, 1] = 1.0
        g[2, 0] = 1.0

        # path 1: compose then lift
        fg = noisy_or_contract(f, g, n_contract=1)  # (3, 2)
        fg_lifted = componentwise_lift(fg, 2)  # (3, 3, 2, 2)

        # path 2: lift then compose
        f2 = componentwise_lift(f, 2)  # (3, 3, 4, 4)
        g2 = componentwise_lift(g, 2)  # (4, 4, 2, 2)
        lifted_fg = noisy_or_contract(f2, g2, n_contract=2)  # (3, 3, 2, 2)

        # for boolean inputs, these should agree
        torch.testing.assert_close(fg_lifted, lifted_fg, atol=1e-5, rtol=1e-5)
