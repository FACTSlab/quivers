"""Tests for morphisms and DSL operators."""

import torch
import pytest

from quivers.core.objects import FinSet, ProductSet, FreeMonoid
from quivers.core.morphisms import (
    ComposedMorphism,
    ProductMorphism,
    MarginalizedMorphism,
    morphism,
    observed,
)


class TestObservedMorphism:
    def test_basic(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        data = torch.rand(3, 4)
        m = observed(x, y, data)

        assert m.domain == x
        assert m.codomain == y
        torch.testing.assert_close(m.tensor, data)

    def test_shape_mismatch(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)

        with pytest.raises(ValueError, match="shape"):
            observed(x, y, torch.rand(3, 5))

    def test_not_learnable(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        m = observed(x, y, torch.rand(3, 4))
        mod = m.module()

        # no parameters, only buffers
        assert len(list(mod.parameters())) == 0


class TestLatentMorphism:
    def test_basic(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        m = morphism(x, y)

        assert m.domain == x
        assert m.codomain == y
        assert m.tensor.shape == (3, 4)
        # sigmoid output is in (0, 1)
        assert (m.tensor > 0).all()
        assert (m.tensor < 1).all()

    def test_has_parameters(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        m = morphism(x, y)
        mod = m.module()

        params = list(mod.parameters())
        assert len(params) == 1
        assert params[0].shape == (3, 4)

    def test_product_set_domain(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)
        p = ProductSet(x, y)
        m = morphism(p, z)

        assert m.tensor.shape == (3, 4, 2)


class TestComposition:
    def test_rshift_operator(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)

        f = morphism(x, y)
        g = morphism(y, z)
        h = f >> g

        assert isinstance(h, ComposedMorphism)
        assert h.domain == x
        assert h.codomain == z
        assert h.tensor.shape == (3, 2)

    def test_domain_mismatch_raises(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)
        w = FinSet("W", 5)

        f = morphism(x, y)
        g = morphism(z, w)

        with pytest.raises(TypeError, match="codomain"):
            f >> g

    def test_chain_of_three(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 5)
        w = FinSet("W", 2)

        f = morphism(x, y)
        g = morphism(y, z)
        h = morphism(z, w)

        result = f >> g >> h
        assert result.domain == x
        assert result.codomain == w
        assert result.tensor.shape == (3, 2)

    def test_values_in_unit_interval(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)

        f = morphism(x, y)
        g = morphism(y, z)
        h = f >> g

        assert (h.tensor >= 0).all()
        assert (h.tensor <= 1).all()

    def test_gradient_through_composition(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)

        f = morphism(x, y)
        g = morphism(y, z)
        h = f >> g

        loss = h.tensor.sum()
        loss.backward()

        assert f.raw.grad is not None
        assert g.raw.grad is not None
        assert torch.isfinite(f.raw.grad).all()
        assert torch.isfinite(g.raw.grad).all()

    def test_boolean_composition(self):
        """On {0,1} data, noisy-OR composition matches boolean matmul."""
        x = FinSet("X", 3)
        y = FinSet("Y", 2)
        z = FinSet("Z", 3)

        m_data = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        n_data = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        f = observed(x, y, m_data)
        g = observed(y, z, n_data)
        h = f >> g

        expected = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        torch.testing.assert_close(h.tensor, expected, atol=1e-5, rtol=1e-5)

    def test_higher_order_composition(self):
        """Compose morphisms between product sets."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        c = FinSet("C", 4)
        ab = ProductSet(a, b)

        f = morphism(ab, c)
        g = morphism(c, a)
        h = f >> g

        assert h.domain == ab
        assert h.codomain == a
        assert h.tensor.shape == (2, 3, 2)


class TestProductMorphism:
    def test_matmul_operator(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)
        w = FinSet("W", 5)

        f = morphism(x, y)
        g = morphism(z, w)
        fg = f @ g

        assert isinstance(fg, ProductMorphism)
        assert fg.tensor.shape == (3, 2, 4, 5)

    def test_product_values(self):
        """Product morphism tensor should be outer product of components."""
        x = FinSet("X", 2)
        y = FinSet("Y", 2)
        z = FinSet("Z", 2)
        w = FinSet("W", 2)

        f_data = torch.tensor([[0.5, 0.3], [0.8, 0.1]])
        g_data = torch.tensor([[0.4, 0.6], [0.7, 0.2]])

        f = observed(x, y, f_data)
        g = observed(z, w, g_data)
        fg = f @ g

        # fg[i, k, j, l] = f[i, j] * g[k, l]
        t = fg.tensor
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    for l in range(2):
                        expected = f_data[i, j] * g_data[k, l]
                        torch.testing.assert_close(
                            t[i, k, j, l],
                            expected,
                            atol=1e-6,
                            rtol=1e-6,
                        )


class TestMarginalization:
    def test_marginalize_codomain_component(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)
        yz = ProductSet(y, z)

        f = morphism(x, yz)
        g = f.marginalize(y)

        assert isinstance(g, MarginalizedMorphism)
        assert g.domain == x
        assert g.codomain == z
        assert g.tensor.shape == (3, 2)

    def test_marginalize_values_in_range(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)
        yz = ProductSet(y, z)

        f = morphism(x, yz)
        g = f.marginalize(y)

        assert (g.tensor >= 0).all()
        assert (g.tensor <= 1).all()

    def test_marginalize_requires_product_codomain(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        f = morphism(x, y)

        with pytest.raises(TypeError, match="ProductSet"):
            f.marginalize(y)


class TestModuleCollection:
    def test_composed_collects_params(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)

        f = morphism(x, y)
        g = morphism(y, z)
        h = f >> g
        mod = h.module()

        params = list(mod.parameters())
        # two learnable morphisms
        assert len(params) == 2

    def test_mixed_observed_latent(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)

        f = morphism(x, y)
        g = observed(y, z, torch.rand(4, 2))
        h = f >> g
        mod = h.module()

        params = list(mod.parameters())
        # only f is learnable
        assert len(params) == 1


class TestFreeMonoidComposition:
    """Verify morphisms compose over FreeMonoid-shaped domains/codomains.

    These tests ensure the primitives support the patterns needed for
    linguistic type systems (strings of primitive types) without
    encoding any particular type system.
    """

    def test_compose_through_free_monoid(self):
        """Morphisms with FreeMonoid codomains compose correctly."""
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        fm_b = FreeMonoid(b, max_length=2)

        f = morphism(a, fm_b)
        g = morphism(fm_b, a)
        h = f >> g

        assert h.domain == a
        assert h.codomain == a
        assert h.tensor.shape == (3, 3)

    def test_product_with_free_monoid(self):
        """Product morphisms work with FreeMonoid domains."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        fm_a = FreeMonoid(a, max_length=1)  # size = 1 + 2 = 3

        f = morphism(fm_a, b)
        g = morphism(fm_a, b)
        fg = f @ g

        assert fg.tensor.shape == (fm_a.size, fm_a.size, b.size, b.size)


class TestMarginalizationAsDecomposition:
    """Verify that marginalization supports predicate-decomposition patterns.

    A morphism A×B×C → D marginalized over B gives A×C → D, and the
    original relation entails the marginalized one (monotonicity).
    This is the abstract pattern behind predicate decomposition:
    give(a, r, t) -> has_after(r, t) via marginalizing out the agent.
    """

    def test_marginalize_preserves_monotonicity(self):
        """The original morphism's values are bounded by the marginalized one.

        A morphism X -> B×C×D marginalized over B gives X -> C×D.
        For every fixed b: f[x, b, c, d] <= marginalized_f[x, c, d].

        This is the abstract version of predicate decomposition:
        give(a, r, t) has a higher-dimensional codomain, and marginalizing
        over one component yields a weaker (dominated) relation.
        """
        torch.manual_seed(42)
        x = FinSet("X", 3)
        b = FinSet("B", 4)
        c = FinSet("C", 2)
        d = FinSet("D", 5)

        bcd = ProductSet(b, c, d)
        data = torch.rand(3, 4, 2, 5)
        f = observed(x, bcd, data)
        g = f.marginalize(b)  # marginalizes over B in codomain

        marginalized = g.tensor  # shape (3, 2, 5)

        # for every b_idx: f[:, b_idx, :, :] <= marginalized[:, :, :]
        for b_idx in range(4):
            assert (data[:, b_idx, :, :] <= marginalized + 1e-6).all()

    def test_marginalize_two_components(self):
        """Marginalizing over multiple components yields smaller codomain."""
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        c = FinSet("C", 2)
        d = FinSet("D", 5)

        bcd = ProductSet(b, c, d)
        f = morphism(a, bcd)
        g = f.marginalize(b, c)  # keep only D

        assert g.domain == a
        assert g.codomain == d
        assert g.tensor.shape == (3, 5)
