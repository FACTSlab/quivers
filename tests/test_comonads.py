"""Tests for comonads and coKleisli categories."""

import torch
import pytest

from quivers.core.objects import FinSet, ProductSet
from quivers.core.morphisms import morphism, observed, identity
from quivers.core.quantales import PRODUCT_FUZZY, BOOLEAN
from quivers.monadic.comonads import (
    DiagonalComonad,
    CofreeComonad,
    CoKleisliCategory,
)


class TestDiagonalComonad:
    def test_counit_is_projection(self):
        """ε_A: A × A → A should be first projection."""
        w = DiagonalComonad()
        a = FinSet("A", 3)
        eps = w.counit(a)

        # shape: (3, 3, 3)
        assert eps.tensor.shape == (3, 3, 3)

        # for each (a1, a2), eps should select a1
        for a1 in range(3):
            for a2 in range(3):
                for a_out in range(3):
                    expected = 1.0 if a_out == a1 else 0.0
                    assert eps.tensor[a1, a2, a_out].item() == expected

    def test_comultiply_is_diagonal(self):
        """δ_A: A × A → (A × A) × (A × A) should duplicate pairs."""
        w = DiagonalComonad()
        a = FinSet("A", 2)
        delta = w.comultiply(a)

        # shape: (2, 2, 2, 2, 2, 2)
        assert delta.tensor.shape == (2, 2, 2, 2, 2, 2)

        # (a1, a2) maps to ((a1, a2), (a1, a2))
        for a1 in range(2):
            for a2 in range(2):
                val = delta.tensor[a1, a2, a1, a2, a1, a2].item()
                assert val == 1.0

    def test_counit_law_left(self):
        """ε_W(A) ∘ δ_A = id_W(A) (left counit)."""
        w = DiagonalComonad()
        a = FinSet("A", 2)
        wa = ProductSet(a, a)

        delta = w.comultiply(a)
        eps_wa = w.counit(wa)

        result = (delta >> eps_wa).tensor
        expected = identity(wa).tensor

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestCofreeComonad:
    def test_counit_projects(self):
        """ε_A: A × S → A should project to A."""
        s = FinSet("S", 2)
        w = CofreeComonad(store=s)
        a = FinSet("A", 3)
        eps = w.counit(a)

        assert eps.tensor.shape == (3, 2, 3)

        for ai in range(3):
            for si in range(2):
                for ao in range(3):
                    expected = 1.0 if ao == ai else 0.0
                    assert eps.tensor[ai, si, ao].item() == expected

    def test_comultiply_duplicates_store(self):
        """δ_A: A × S → (A × S) × S should duplicate S."""
        s = FinSet("S", 2)
        w = CofreeComonad(store=s)
        a = FinSet("A", 2)
        delta = w.comultiply(a)

        # shape: (2, 2, 2, 2, 2) = (A, S, A, S, S)
        assert delta.tensor.shape == (2, 2, 2, 2, 2)

        # (a, s) -> (a, s, s)
        for ai in range(2):
            for si in range(2):
                val = delta.tensor[ai, si, ai, si, si].item()
                assert val == 1.0

    def test_counit_law(self):
        """ε ∘ δ = id."""
        s = FinSet("S", 2)
        w = CofreeComonad(store=s)
        a = FinSet("A", 2)

        wa = ProductSet(a, s)
        delta = w.comultiply(a)
        eps_wa = w.counit(wa)

        result = (delta >> eps_wa).tensor
        expected = identity(wa).tensor

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestCoKleisliCategory:
    def test_identity_is_counit(self):
        """CoKleisli identity at A is ε_A: W(A) → A."""
        w = DiagonalComonad()
        cat = CoKleisliCategory(w)
        a = FinSet("A", 3)

        id_a = cat.identity(a)
        eps = w.counit(a)

        torch.testing.assert_close(
            id_a.tensor, eps.tensor, atol=1e-5, rtol=1e-5
        )
