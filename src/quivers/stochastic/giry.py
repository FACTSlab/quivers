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

    GiryMonad      — the probability monad (T, η, μ) with MarkovQuantale
    FinStoch       — the Kleisli category of GiryMonad
"""

from __future__ import annotations

from quivers.core.objects import SetObject
from quivers.core.morphisms import Morphism, identity
from quivers.categorical.functors import Functor, IDENTITY
from quivers.monadic.monads import Monad, KleisliCategory
from quivers.stochastic.quantale import MARKOV


class GiryMonad(Monad):
    """The Giry (probability) monad on finite sets.

    At the finite-set level, G(A) = A because probability distributions
    over A are represented as tensors indexed by A (i.e., functions
    A → [0,1] that sum to 1), not as elements of a separate simplex
    object.

    The Kleisli composition uses the MarkovQuantale (sum-product),
    yielding standard matrix multiplication of stochastic matrices.

    This is the categorical foundation for all stochastic morphisms
    in quivers: every Markov kernel A → B is a Kleisli morphism
    A → G(B) = A → B in this monad.

    Examples
    --------
    >>> from quivers import FinSet
    >>> from quivers.stochastic.giry import GiryMonad, FinStoch
    >>> G = GiryMonad()
    >>> A = FinSet("A", 3)
    >>> eta = G.unit(A)  # Kronecker delta: shape (3, 3)
    >>> finstoch = FinStoch()
    >>> # compose stochastic morphisms via finstoch.compose(f, g)
    """

    @property
    def endofunctor(self) -> Functor:
        """G = Id at the set level."""
        return IDENTITY

    def unit(self, obj: SetObject) -> Morphism:
        """η_A = δ: the Kronecker delta (deterministic distribution).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The identity/delta morphism η_A: A → A.
        """
        return identity(obj, quantale=MARKOV)

    def multiply(self, obj: SetObject) -> Morphism:
        """μ_A = id: flatten nested distributions.

        For finite sets, flattening G(G(A)) → G(A) is just the
        identity because G(A) = A at the set level.

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The multiplication morphism μ_A: A → A.
        """
        return identity(obj, quantale=MARKOV)

    def kleisli_compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition via sum-product (matrix multiplication).

        Since G = Id, the Kleisli composition f >=> g is just
        standard composition f >> g under the MarkovQuantale.

        Parameters
        ----------
        f : Morphism
            Left Kleisli morphism A → B.
        g : Morphism
            Right Kleisli morphism B → C.

        Returns
        -------
        Morphism
            Composed morphism A → C.
        """
        return f >> g

    def _multiply_at_codomain(self, g: Morphism) -> Morphism:
        return identity(g.codomain, quantale=MARKOV)

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
    >>> A = FinSet("A", 3)
    >>> B = FinSet("B", 4)
    >>> C = FinSet("C", 2)
    >>> f = StochasticMorphism(A, B)
    >>> g = StochasticMorphism(B, C)
    >>> h = cat.compose(f, g)  # A → C via matrix multiplication
    """

    def __init__(self) -> None:
        super().__init__(GiryMonad())

    def __repr__(self) -> str:
        return "FinStoch()"
