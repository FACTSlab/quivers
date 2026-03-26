"""Tests for algebras and coalgebras."""

import torch

from quivers.core.objects import FinSet, FreeMonoid
from quivers.core.morphisms import identity
from quivers.monadic.monads import FuzzyPowersetMonad, FreeMonoidMonad
from quivers.monadic.comonads import DiagonalComonad
from quivers.monadic.algebras import (
    FreeAlgebra,
    ObservedAlgebra,
    CofreeCoalgebra,
    EilenbergMooreCategory,
)


class TestFreeAlgebra:
    def test_fuzzy_powerset_free_algebra(self):
        """Free algebra for FuzzyPowerset has carrier A and μ = id."""
        m = FuzzyPowersetMonad()
        a = FinSet("A", 3)
        alg = FreeAlgebra(m, a)

        # carrier should be T(A) = A (since T = Id)
        assert alg.carrier == a

        # structure map μ_A = identity
        alpha = alg.structure_map()
        torch.testing.assert_close(alpha.tensor, torch.eye(3))

    def test_free_algebra_unit_law(self):
        """α ∘ η_A = id for free algebra."""
        m = FuzzyPowersetMonad()
        a = FinSet("A", 3)
        alg = FreeAlgebra(m, a)

        assert alg.verify_unit_law()

    def test_free_algebra_associativity(self):
        """α ∘ μ = α ∘ T(α) for free algebra."""
        m = FuzzyPowersetMonad()
        a = FinSet("A", 3)
        alg = FreeAlgebra(m, a)

        assert alg.verify_associativity()

    def test_free_monoid_free_algebra(self):
        """Free algebra for FreeMonoid has carrier A*."""
        m = FreeMonoidMonad(max_length=1)
        a = FinSet("A", 2)
        alg = FreeAlgebra(m, a)

        # carrier should be T(A) = FreeMonoid(A, 1)
        fm = FreeMonoid(a, max_length=1)
        assert alg.carrier.size == fm.size


class TestObservedAlgebra:
    def test_identity_algebra(self):
        """The identity morphism is an algebra for FuzzyPowerset."""
        m = FuzzyPowersetMonad()
        a = FinSet("A", 3)
        alg = ObservedAlgebra(m, a, torch.eye(3))

        assert alg.verify_unit_law()
        assert alg.verify_associativity()


class TestCofreeCoalgebra:
    def test_cofree_coalgebra_structure(self):
        """Cofree coalgebra has carrier W(A) and γ = δ_A."""
        from quivers.core.objects import ProductSet

        w = DiagonalComonad()
        a = FinSet("A", 2)
        coalg = CofreeCoalgebra(w, a)

        # carrier should be W(A) = A × A
        expected_carrier = ProductSet(a, a)
        assert coalg.carrier.shape == expected_carrier.shape

    def test_cofree_coalgebra_counit_law(self):
        """ε ∘ γ = id for cofree coalgebra."""
        w = DiagonalComonad()
        a = FinSet("A", 2)
        coalg = CofreeCoalgebra(w, a)

        assert coalg.verify_counit_law()


class TestEilenbergMooreCategory:
    def test_free_algebra_creation(self):
        """EM category can create free algebras."""
        m = FuzzyPowersetMonad()
        em = EilenbergMooreCategory(m)
        a = FinSet("A", 3)
        alg = em.free_algebra(a)

        assert alg.carrier == a
        assert alg.verify_unit_law()

    def test_identity_is_homomorphism(self):
        """The identity morphism should be an algebra homomorphism."""
        m = FuzzyPowersetMonad()
        em = EilenbergMooreCategory(m)
        a = FinSet("A", 3)
        alg = em.free_algebra(a)
        id_a = identity(a)

        assert em.is_homomorphism(id_a, alg, alg)
