"""Distributive laws for composing monads.

A distributive law λ: S ∘ T ⇒ T ∘ S between monads S and T allows
their composition T ∘ S to form a new monad. The component
λ_A: S(T(A)) → T(S(A)) "pushes S past T."

This module provides:

    DistributiveLaw (abstract)
    └── FreeMonoidPowersetLaw — λ: Free(P(A)) → P(Free(A))
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.objects import SetObject, FinSet, FreeMonoid
from quivers.core.morphisms import Morphism, observed
from quivers.core.quantales import PRODUCT_FUZZY, Quantale
from quivers.core.tensor_ops import componentwise_lift
from quivers.monadic.monads import Monad, FuzzyPowersetMonad, FreeMonoidMonad


class DistributiveLaw(ABC):
    """Abstract distributive law λ: S ∘ T ⇒ T ∘ S.

    Subclasses must implement outer_monad, inner_monad, and distribute.
    """

    @property
    @abstractmethod
    def outer_monad(self) -> Monad:
        """The outer monad S (applied second in S ∘ T)."""
        ...

    @property
    @abstractmethod
    def inner_monad(self) -> Monad:
        """The inner monad T (applied first in S ∘ T)."""
        ...

    @abstractmethod
    def distribute(self, obj: SetObject) -> Morphism:
        """The component λ_A: S(T(A)) → T(S(A)).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The distributive morphism λ_A.
        """
        ...


class FreeMonoidPowersetLaw(DistributiveLaw):
    """Distributive law λ: Free(P(A)) → P(Free(A)).

    A word of fuzzy sets (S₁, ..., Sₖ) in Free(P(A)) maps to
    a fuzzy set of words in P(Free(A)) with membership:

        λ(S₁,...,Sₖ)(a₁,...,aₖ) = ⊗ᵢ Sᵢ(aᵢ)

    This is exactly the componentwise lift (the functorial action
    of FreeMonoid on morphisms). The distributive law witnesses the
    fact that "a word of fuzzy sets" can be reinterpreted as "a
    fuzzy set of words" via the product t-norm.

    Parameters
    ----------
    max_length : int
        Maximum string length for the free monoid.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(
        self,
        max_length: int,
        quantale: Quantale | None = None,
    ) -> None:
        self._max_length = max_length
        self._quantale = quantale if quantale is not None else PRODUCT_FUZZY
        self._powerset = FuzzyPowersetMonad(quantale=self._quantale)
        self._free_monoid = FreeMonoidMonad(max_length)

    @property
    def outer_monad(self) -> FreeMonoidMonad:
        """The free monoid monad (applied to P(A))."""
        return self._free_monoid

    @property
    def inner_monad(self) -> FuzzyPowersetMonad:
        """The fuzzy powerset monad (applied to A)."""
        return self._powerset

    def distribute(self, obj: SetObject) -> Morphism:
        """λ_A: Free(P(A)) → P(Free(A)) = Free(A) (at set level).

        Since P is identity at set level, Free(P(A)) = Free(A) = A*.
        And P(Free(A)) = Free(A) = A* (also identity at set level).
        So λ_A: A* → A* is an endomorphism.

        The tensor is the block-diagonal matrix built from
        componentwise_lift — which is exactly the FreeMonoidFunctor's
        action on the identity morphism A → A.

        Parameters
        ----------
        obj : SetObject
            Must be a FinSet.

        Returns
        -------
        ObservedMorphism
            The distributive law component λ_A.
        """
        if not isinstance(obj, FinSet):
            raise TypeError(
                f"FreeMonoidPowersetLaw requires FinSet, got {type(obj).__name__}"
            )

        fm = FreeMonoid(obj, max_length=self._max_length)

        # the distributive law at the tensor level is the block-diagonal
        # identity: for each stratum k, the k-fold componentwise lift
        # of id_A is id_{A^k}
        n = obj.cardinality
        id_tensor = torch.eye(n)

        blocks: list[torch.Tensor] = []

        for k in range(self._max_length + 1):
            lifted = componentwise_lift(id_tensor, k, quantale=self._quantale)
            rows = n**k if k > 0 else 1
            cols = n**k if k > 0 else 1
            blocks.append(lifted.reshape(rows, cols))

        data = torch.block_diag(*blocks)

        return observed(fm, fm, data, quantale=self._quantale)
