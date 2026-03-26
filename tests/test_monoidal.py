"""Tests for monoidal structures."""

import torch

from quivers.core.objects import FinSet, ProductSet, CoproductSet, Unit
from quivers.core.morphisms import identity
from quivers.categorical.monoidal import CartesianMonoidal, CoproductMonoidal, EmptySet


class TestCartesianMonoidal:
    def test_product(self):
        m = CartesianMonoidal()
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        result = m.product(a, b)
        assert isinstance(result, ProductSet)
        assert result.shape == (2, 3)

    def test_unit(self):
        m = CartesianMonoidal()
        assert m.unit == Unit

    def test_associator_is_identity(self):
        """Since ProductSet flattens, associator is identity."""
        m = CartesianMonoidal()
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        c = FinSet("C", 4)
        assoc = m.associator(a, b, c)

        flat = ProductSet(a, b, c)
        expected = identity(flat).tensor
        torch.testing.assert_close(assoc.tensor, expected)

    def test_left_unitor(self):
        """I × A → A should be an isomorphism."""
        m = CartesianMonoidal()
        a = FinSet("A", 3)
        lu = m.left_unitor(a)

        # should map (0, i) → i for i in range(3)
        assert lu.tensor.shape == (1, 3, 3)
        for i in range(3):
            assert lu.tensor[0, i, i].item() == 1.0

    def test_right_unitor(self):
        """A × I → A should be an isomorphism."""
        m = CartesianMonoidal()
        a = FinSet("A", 3)
        ru = m.right_unitor(a)

        assert ru.tensor.shape == (3, 1, 3)
        for i in range(3):
            assert ru.tensor[i, 0, i].item() == 1.0

    def test_braiding_is_involution(self):
        """σ_{B,A} ∘ σ_{A,B} = id."""
        m = CartesianMonoidal()
        a = FinSet("A", 2)
        b = FinSet("B", 3)

        sigma_ab = m.braiding(a, b)
        sigma_ba = m.braiding(b, a)

        result = (sigma_ab >> sigma_ba).tensor
        expected = identity(ProductSet(a, b)).tensor

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestCoproductMonoidal:
    def test_product(self):
        m = CoproductMonoidal()
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        result = m.product(a, b)
        assert isinstance(result, CoproductSet)
        assert result.size == 5

    def test_unit_is_empty(self):
        m = CoproductMonoidal()
        assert isinstance(m.unit, EmptySet)
        assert m.unit.size == 0

    def test_associator_is_identity(self):
        """Since CoproductSet flattens, associator is identity."""
        m = CoproductMonoidal()
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        c = FinSet("C", 4)
        assoc = m.associator(a, b, c)

        flat = CoproductSet(a, b, c)
        expected = identity(flat).tensor
        torch.testing.assert_close(assoc.tensor, expected)

    def test_braiding_swap(self):
        """Braiding swaps the two coproduct blocks."""
        m = CoproductMonoidal()
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        sigma = m.braiding(a, b)

        # element 0 (from A) should map to index 3 (after B's 3 elements)
        assert sigma.tensor[0, 3].item() == 1.0
        assert sigma.tensor[1, 4].item() == 1.0
        # element 2 (first of B) should map to index 0
        assert sigma.tensor[2, 0].item() == 1.0

    def test_braiding_involution(self):
        """σ_{B,A} ∘ σ_{A,B} = id."""
        m = CoproductMonoidal()
        a = FinSet("A", 2)
        b = FinSet("B", 3)

        sigma_ab = m.braiding(a, b)
        sigma_ba = m.braiding(b, a)

        result = (sigma_ab >> sigma_ba).tensor
        expected = identity(CoproductSet(a, b)).tensor
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
