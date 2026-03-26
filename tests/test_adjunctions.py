"""Tests for adjunctions."""

import torch
import pytest

from quivers.core.objects import FinSet, FreeMonoid
from quivers.core.morphisms import observed, identity
from quivers.categorical.adjunctions import FreeForgetfulAdjunction


class TestFreeForgetfulAdjunction:
    def test_unit_shape(self):
        adj = FreeForgetfulAdjunction(max_length=2)
        a = FinSet("A", 3)
        eta = adj.unit_component(a)

        fm = FreeMonoid(a, max_length=2)
        assert eta.tensor.shape == (3, fm.size)

    def test_unit_embeds_length_1(self):
        adj = FreeForgetfulAdjunction(max_length=2)
        a = FinSet("A", 2)
        eta = adj.unit_component(a)

        fm = FreeMonoid(a, max_length=2)
        for i in range(2):
            j = fm.encode((i,))
            assert eta.tensor[i, j].item() == 1.0

    def test_counit_shape(self):
        adj = FreeForgetfulAdjunction(max_length=2)
        a = FinSet("A", 3)
        eps = adj.counit_component(a)

        fm = FreeMonoid(a, max_length=2)
        assert eps.tensor.shape == (fm.size, 3)

    def test_counit_projects_length_1(self):
        adj = FreeForgetfulAdjunction(max_length=2)
        a = FinSet("A", 2)
        eps = adj.counit_component(a)

        fm = FreeMonoid(a, max_length=2)
        for i in range(2):
            j = fm.encode((i,))
            assert eps.tensor[j, i].item() == 1.0

    def test_triangle_left(self):
        """ε_{F(A)} ∘ F(η_A) ≈ id_{F(A)}."""
        adj = FreeForgetfulAdjunction(max_length=1)
        a = FinSet("A", 2)
        assert adj.verify_triangle_left(a, atol=1e-5)

    def test_triangle_right(self):
        """G(ε_A) ∘ η_{G(A)} ≈ id_{G(A)}."""
        adj = FreeForgetfulAdjunction(max_length=1)
        a = FinSet("A", 2)
        assert adj.verify_triangle_right(a, atol=1e-5)

    def test_unit_accepts_non_finset(self):
        """Unit accepts any SetObject via proxy FinSet."""
        adj = FreeForgetfulAdjunction(max_length=1)
        fm = FreeMonoid(FinSet("A", 2), max_length=1)  # size 3
        eta = adj.unit_component(fm)

        # domain is proxy FinSet(3), codomain is FreeMonoid(proxy(3), max_length=1)
        assert eta.tensor.shape[0] == 3
        assert eta.tensor.shape[1] == 4  # 1 + 3

    def test_counit_at_free_monoid(self):
        """Counit at FreeMonoid uses concatenation map."""
        adj = FreeForgetfulAdjunction(max_length=1)
        a = FinSet("A", 2)
        fm = FreeMonoid(a, max_length=1)  # size 3
        eps = adj.counit_component(fm)

        # domain is (A*)* = FreeMonoid(proxy(3), max_length=1), size 4
        # codomain is A*, size 3
        assert eps.tensor.shape == (4, 3)

        # empty word → empty word
        assert eps.tensor[0, 0].item() == 1.0

        # (i) → element i of A*
        for i in range(3):
            assert eps.tensor[1 + i, i].item() == 1.0
