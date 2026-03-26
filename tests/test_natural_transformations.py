"""Tests for natural transformations."""

import torch

from quivers.core.objects import FinSet
from quivers.core.morphisms import observed, identity
from quivers.categorical.functors import ComposedFunctor, FreeMonoidFunctor, IDENTITY
from quivers.categorical.natural_transformations import ComponentwiseNT


class TestIdentityFunctor:
    def test_map_object(self):
        a = FinSet("A", 3)
        assert IDENTITY.map_object(a) == a

    def test_map_morphism_tensor(self):
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        f_data = torch.rand(3, 4)
        f = observed(a, b, f_data)
        result = IDENTITY.map_morphism(f)
        torch.testing.assert_close(result.tensor, f_data)

    def test_map_tensor(self):
        t = torch.rand(3, 4)
        result = IDENTITY.map_tensor(t, None)
        torch.testing.assert_close(result, t)


class TestComposedFunctor:
    def test_composed_identity(self):
        """Id ∘ Id = Id on objects."""
        composed = ComposedFunctor(IDENTITY, IDENTITY)
        a = FinSet("A", 3)
        assert composed.map_object(a) == a

    def test_composed_with_free_monoid(self):
        """Id ∘ FreeMonoid maps A to A*."""
        fm = FreeMonoidFunctor(max_length=1)
        composed = ComposedFunctor(IDENTITY, fm)
        a = FinSet("A", 2)
        result = composed.map_object(a)
        assert result.size == 3  # 1 + 2

    def test_map_tensor(self):
        fm = FreeMonoidFunctor(max_length=1)
        composed = ComposedFunctor(IDENTITY, fm)
        t = torch.rand(2, 3)
        from quivers.core.quantales import PRODUCT_FUZZY

        result = composed.map_tensor(t, PRODUCT_FUZZY)
        expected = fm.map_tensor(t, PRODUCT_FUZZY)
        torch.testing.assert_close(result, expected)


class TestComponentwiseNT:
    def test_identity_nt(self):
        """An NT from Id to Id using identity components."""
        nt = ComponentwiseNT(
            IDENTITY,
            IDENTITY,
            component_fn=lambda obj: identity(obj),
        )
        a = FinSet("A", 3)
        comp = nt.component(a)
        torch.testing.assert_close(comp.tensor, torch.eye(3))

    def test_naturality_with_identity(self):
        """Verify naturality of the identity NT."""
        nt = ComponentwiseNT(
            IDENTITY,
            IDENTITY,
            component_fn=lambda obj: identity(obj),
        )
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        f_data = torch.rand(3, 4)
        f = observed(a, b, f_data)

        assert nt.verify_naturality(f, atol=1e-5)

    def test_repr(self):
        nt = ComponentwiseNT(
            IDENTITY,
            IDENTITY,
            component_fn=lambda obj: identity(obj),
        )
        assert "ComponentwiseNT" in repr(nt)
