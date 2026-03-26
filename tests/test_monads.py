"""Tests for monads and Kleisli categories."""

import torch
import pytest

from quivers.core.objects import FinSet, FreeMonoid
from quivers.core.morphisms import observed
from quivers.monadic.monads import (
    FuzzyPowersetMonad,
    FreeMonoidMonad,
    KleisliCategory,
)
from quivers.core.quantales import BOOLEAN


class TestFuzzyPowersetMonad:
    def test_unit_is_identity(self):
        m = FuzzyPowersetMonad()
        a = FinSet("A", 3)
        eta = m.unit(a)
        torch.testing.assert_close(eta.tensor, torch.eye(3))

    def test_multiply_is_identity(self):
        m = FuzzyPowersetMonad()
        a = FinSet("A", 3)
        mu = m.multiply(a)
        torch.testing.assert_close(mu.tensor, torch.eye(3))

    def test_kleisli_compose_matches_rshift(self):
        """f >=> g should equal f >> g for FuzzyPowerset."""
        torch.manual_seed(42)
        m = FuzzyPowersetMonad()
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        c = FinSet("C", 2)

        f_data = torch.rand(3, 4)
        g_data = torch.rand(4, 2)
        f = observed(a, b, f_data)
        g = observed(b, c, g_data)

        kleisli = m.kleisli_compose(f, g)
        rshift = (f >> g).tensor

        torch.testing.assert_close(kleisli.tensor, rshift, atol=1e-5, rtol=1e-5)

    def test_boolean_kleisli(self):
        m = FuzzyPowersetMonad(quantale=BOOLEAN)
        a = FinSet("A", 2)
        b = FinSet("B", 2)
        c = FinSet("C", 2)

        f_data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        g_data = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        f = observed(a, b, f_data, quantale=BOOLEAN)
        g = observed(b, c, g_data, quantale=BOOLEAN)

        result = m.kleisli_compose(f, g)

        # should be g_data (since f is identity permutation)
        torch.testing.assert_close(result.tensor, g_data, atol=1e-5, rtol=1e-5)


class TestFreeMonoidMonad:
    def test_unit_embeds_length_1(self):
        m = FreeMonoidMonad(max_length=2)
        a = FinSet("A", 3)
        eta = m.unit(a)

        # domain: A (size 3), codomain: A* (size 1+3+9=13)
        assert eta.tensor.shape == (3, 13)

        # each element a maps to the length-1 word (a)
        fm = FreeMonoid(a, max_length=2)
        for i in range(3):
            flat_idx = fm.encode((i,))
            assert eta.tensor[i, flat_idx].item() == 1.0

    def test_multiply_flattens(self):
        m = FreeMonoidMonad(max_length=2)
        a = FinSet("A", 2)
        mu = m.multiply(a)

        fm_a = FreeMonoid(a, max_length=2)

        # check that length-1 words in (A*)* that contain length-1
        # words in A* map correctly
        # word (idx_of_(0,)) in (A*)* should map to flat index of (0,) in A*
        inner_idx = fm_a.encode((0,))
        fm_fm_a = FreeMonoid(
            FinSet(f"{a.name}*", fm_a.size),
            max_length=2,
        )
        outer_idx = fm_fm_a.encode((inner_idx,))
        target_idx = fm_a.encode((0,))

        assert mu.tensor[outer_idx, target_idx].item() == 1.0

    def test_unit_requires_finset(self):
        m = FreeMonoidMonad(max_length=1)
        with pytest.raises(TypeError, match="FinSet"):
            m.unit(FreeMonoid(FinSet("A", 2), max_length=1))


class TestKleisliCategory:
    def test_identity(self):
        m = FuzzyPowersetMonad()
        cat = KleisliCategory(m)
        a = FinSet("A", 3)
        eta = cat.identity(a)
        torch.testing.assert_close(eta.tensor, torch.eye(3))

    def test_compose(self):
        torch.manual_seed(42)
        m = FuzzyPowersetMonad()
        cat = KleisliCategory(m)
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        c = FinSet("C", 2)

        f = observed(a, b, torch.rand(3, 4))
        g = observed(b, c, torch.rand(4, 2))

        result = cat.compose(f, g)
        expected = (f >> g).tensor

        torch.testing.assert_close(result.tensor, expected, atol=1e-5, rtol=1e-5)


class TestMonadLaws:
    def test_fuzzy_powerset_left_unit(self):
        """μ ∘ η_T = id (left unit)."""
        m = FuzzyPowersetMonad()
        a = FinSet("A", 3)
        # η_A >> id_A should equal id_A
        eta = m.unit(a)
        mu = m.multiply(a)
        result = (eta >> mu).tensor
        torch.testing.assert_close(result, torch.eye(3), atol=1e-5, rtol=1e-5)

    def test_fuzzy_powerset_right_unit(self):
        """μ ∘ T(η) = id (right unit)."""
        m = FuzzyPowersetMonad()
        a = FinSet("A", 3)
        # T(η) = η (since T = Id), so μ ∘ η = id
        eta = m.unit(a)
        mu = m.multiply(a)
        result = (eta >> mu).tensor
        torch.testing.assert_close(result, torch.eye(3), atol=1e-5, rtol=1e-5)
