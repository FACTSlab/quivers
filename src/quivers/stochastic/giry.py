"""The Giry monad on finite sets (the probability monad).

The Giry monad G on FinSet maps each finite set A to the set of
probability distributions on A. In our representation:

    G(A) = A   (at the set level, since distributions are represented
                as row-stochastic tensors, not as elements of a simplex)

This is structurally identical to the fuzzy powerset monad but with
a different composition rule: sum-product (matrix multiplication)
instead of noisy-OR.

    η_A: A → A       — the Kronecker delta (deterministic distribution)
    μ_A: A → A       — identity (flattening nested distributions)
    f >=> g = f >> g  — Kleisli composition = matrix multiplication

The Kleisli category of the Giry monad on FinSet is exactly FinStoch,
the category of finite stochastic matrices.

This module provides:

    GiryMonad      — the probability monad (T, η, μ) with MarkovAlgebra
    FinStoch       — the Kleisli category of GiryMonad
"""

from __future__ import annotations

from quivers.core.objects import SetObject
from quivers.core.morphisms import Morphism, identity
from quivers.categorical.functors import Functor, IDENTITY
from quivers.monadic.monads import KleisliCategory
from quivers.monadic.typeclasses import Monad
from quivers.core.algebras import MARKOV


class GiryMonad(Monad):
    """The Giry (probability) monad on finite sets.

    At the finite-set level, G(A) = A because probability distributions
    over A are represented as tensors indexed by A (i.e., functions
    A → [0,1] that sum to 1), not as elements of a separate simplex
    object.

    The Kleisli composition uses the MarkovAlgebra (sum-product),
    yielding standard matrix multiplication of stochastic matrices.

    This is the categorical foundation for all stochastic morphisms
    in quivers: every Markov kernel A → B is a Kleisli morphism
    A → G(B) = A → B in this monad.

    Examples
    --------
    >>> from quivers import FinSet
    >>> from quivers.stochastic.giry import GiryMonad, FinStoch
    >>> G = GiryMonad()
    >>> A = FinSet(name="A", cardinality=3)
    >>> eta = G.unit(A)  # Kronecker delta: shape (3, 3)
    >>> finstoch = FinStoch()
    >>> # compose stochastic morphisms via finstoch.compose(f, g)
    """

    @property
    def endofunctor(self) -> Functor:
        """G = Id at the set level."""
        return IDENTITY

    # Typeclass interface
    def fmap_obj(self, A: SetObject) -> SetObject:
        return A

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f

    def pure(self, A: SetObject) -> Morphism:
        """``η_A = δ`` — Kronecker delta (deterministic distribution)."""
        return identity(A, algebra=MARKOV)

    def join(self, A: SetObject) -> Morphism:
        """``μ_A`` — flatten nested distributions.

        Since ``G(A) = A`` at the finite-set level, the flattening is
        the identity.
        """
        return identity(A, algebra=MARKOV)

    # Convenience aliases for the historical Eilenberg–Moore vocabulary.
    def unit(self, A: SetObject) -> Morphism:
        return self.pure(A)

    def multiply(self, A: SetObject) -> Morphism:
        return self.join(A)

    def kleisli_compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition via sum-product (matrix multiplication)."""
        return f >> g

    def __repr__(self) -> str:
        return "GiryMonad()"


class FinStoch(KleisliCategory):
    """The category FinStoch of finite sets and stochastic maps.

    This is the Kleisli category of the Giry monad: objects are
    finite sets, morphisms are stochastic matrices (Markov kernels),
    and composition is matrix multiplication.

    Examples
    --------
    >>> from quivers import FinSet
    >>> from quivers.stochastic import StochasticMorphism
    >>> from quivers.stochastic.giry import FinStoch
    >>> cat = FinStoch()
    >>> A = FinSet(name="A", cardinality=3)
    >>> B = FinSet(name="B", cardinality=4)
    >>> C = FinSet(name="C", cardinality=2)
    >>> f = StochasticMorphism(A, B)
    >>> g = StochasticMorphism(B, C)
    >>> h = cat.compose(f, g)  # A → C via matrix multiplication
    """

    def __init__(self) -> None:
        super().__init__(GiryMonad())

    def __repr__(self) -> str:
        return "FinStoch()"
