"""Monoidal structures on V-enriched FinSet.

A monoidal structure (C, ⊗, I) consists of a bifunctor ⊗, a unit
object I, and coherence isomorphisms (associator, unitors, braiding).

This module provides:

    MonoidalStructure (abstract)
    ├── CartesianMonoidal  — (FinSet, ×, 1)
    └── CoproductMonoidal  — (FinSet, +, ∅)
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import torch

from quivers.core.objects import SetObject, FinSet, ProductSet, CoproductSet, Unit
from quivers.core.morphisms import ObservedMorphism, observed
from quivers.core.quantales import PRODUCT_FUZZY, Quantale


class MonoidalStructure(ABC):
    """Abstract monoidal structure on a category.

    Provides the monoidal product, unit object, and coherence
    isomorphisms that witness the monoidal laws.
    """

    @abstractmethod
    def product(self, a: SetObject, b: SetObject) -> SetObject:
        """The monoidal product A ⊗ B.

        Parameters
        ----------
        a : SetObject
            Left object.
        b : SetObject
            Right object.

        Returns
        -------
        SetObject
            The product object.
        """
        ...

    @property
    @abstractmethod
    def unit(self) -> SetObject:
        """The monoidal unit I."""
        ...

    @abstractmethod
    def associator(self, a: SetObject, b: SetObject, c: SetObject) -> ObservedMorphism:
        """The associator (A ⊗ B) ⊗ C → A ⊗ (B ⊗ C).

        Parameters
        ----------
        a, b, c : SetObject
            Three objects.

        Returns
        -------
        ObservedMorphism
            The associator isomorphism.
        """
        ...

    @abstractmethod
    def left_unitor(self, a: SetObject) -> ObservedMorphism:
        """The left unitor I ⊗ A → A.

        Parameters
        ----------
        a : SetObject
            The object.

        Returns
        -------
        ObservedMorphism
            The left unitor isomorphism.
        """
        ...

    @abstractmethod
    def right_unitor(self, a: SetObject) -> ObservedMorphism:
        """The right unitor A ⊗ I → A.

        Parameters
        ----------
        a : SetObject
            The object.

        Returns
        -------
        ObservedMorphism
            The right unitor isomorphism.
        """
        ...

    @abstractmethod
    def braiding(self, a: SetObject, b: SetObject) -> ObservedMorphism:
        """The braiding A ⊗ B → B ⊗ A.

        Parameters
        ----------
        a, b : SetObject
            Two objects.

        Returns
        -------
        ObservedMorphism
            The braiding (swap) isomorphism.
        """
        ...


class CartesianMonoidal(MonoidalStructure):
    """Cartesian monoidal structure: (FinSet, ×, 1).

    The product is the cartesian product (ProductSet), the unit is
    the terminal object (Unit = FinSet("1", 1)).

    Since ProductSet auto-flattens, the associator (A×B)×C → A×(B×C)
    is the identity (both sides flatten to ProductSet(A,B,C)). The
    braiding A×B → B×A is a permutation tensor.

    Parameters
    ----------
    quantale : Quantale or None
        The enrichment algebra for coherence morphisms.
    """

    def __init__(self, quantale: Quantale | None = None) -> None:
        self._quantale = quantale if quantale is not None else PRODUCT_FUZZY

    def product(self, a: SetObject, b: SetObject) -> ProductSet:
        """Cartesian product A × B."""
        return ProductSet(a, b)

    @property
    def unit(self) -> FinSet:
        """Terminal object {*}."""
        return Unit

    def associator(self, a: SetObject, b: SetObject, c: SetObject) -> ObservedMorphism:
        """(A × B) × C → A × (B × C).

        Since ProductSet flattens, both sides are ProductSet(A,B,C),
        so this is the identity morphism.
        """
        # both flatten to the same ProductSet
        flat = ProductSet(a, b, c)
        data = self._quantale.identity_tensor(flat.shape)
        return observed(flat, flat, data, quantale=self._quantale)

    def left_unitor(self, a: SetObject) -> ObservedMorphism:
        """I × A → A (project out the unit component).

        The tensor maps (1, a₁, ..., aₙ) → (a₁, ..., aₙ).
        Since Unit has size 1, this is a "squeeze" that drops
        the trivial dimension.
        """
        source = ProductSet(Unit, a)  # shape: (1, *a.shape)
        target = a

        # build identity-like tensor from source.shape to target.shape
        data = torch.zeros(*source.shape, *target.shape)

        for idx in itertools.product(*(range(s) for s in target.shape)):
            # source index: (0,) + idx
            # target index: idx
            src = (0,) + idx
            data[src + idx] = self._quantale.unit

        return observed(source, target, data, quantale=self._quantale)

    def right_unitor(self, a: SetObject) -> ObservedMorphism:
        """A × I → A (project out the unit component).

        The tensor maps (a₁, ..., aₙ, 1) → (a₁, ..., aₙ).
        """
        source = ProductSet(a, Unit)  # shape: (*a.shape, 1)
        target = a

        data = torch.zeros(*source.shape, *target.shape)

        for idx in itertools.product(*(range(s) for s in target.shape)):
            # source index: idx + (0,)
            src = idx + (0,)
            data[src + idx] = self._quantale.unit

        return observed(source, target, data, quantale=self._quantale)

    def braiding(self, a: SetObject, b: SetObject) -> ObservedMorphism:
        """A × B → B × A (swap permutation).

        The tensor has shape (*a.shape, *b.shape, *b.shape, *a.shape)
        with unit on entries where the swap holds.
        """
        source = ProductSet(a, b)
        target = ProductSet(b, a)

        data = torch.full((*source.shape, *target.shape), self._quantale.zero)

        for a_idx in itertools.product(*(range(s) for s in a.shape)):
            for b_idx in itertools.product(*(range(s) for s in b.shape)):
                src = a_idx + b_idx
                tgt = b_idx + a_idx
                data[src + tgt] = self._quantale.unit

        return observed(source, target, data, quantale=self._quantale)

    def __repr__(self) -> str:
        return "CartesianMonoidal()"


class EmptySet(SetObject):
    """The initial object (empty set) with cardinality 0.

    Used as the unit for coproduct monoidal structure.
    """

    @property
    def size(self) -> int:
        return 0

    @property
    def shape(self) -> tuple[int, ...]:
        return (0,)

    def __repr__(self) -> str:
        return "EmptySet()"

    def __hash__(self) -> int:
        return hash("EmptySet")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, EmptySet)


# module-level singleton
EMPTY = EmptySet()


class CoproductMonoidal(MonoidalStructure):
    """Coproduct monoidal structure: (FinSet, +, ∅).

    The product is the coproduct (CoproductSet), the unit is
    the initial object (EmptySet with cardinality 0).

    Parameters
    ----------
    quantale : Quantale or None
        The enrichment algebra for coherence morphisms.
    """

    def __init__(self, quantale: Quantale | None = None) -> None:
        self._quantale = quantale if quantale is not None else PRODUCT_FUZZY

    def product(self, a: SetObject, b: SetObject) -> CoproductSet:
        """Coproduct A + B."""
        return CoproductSet(a, b)

    @property
    def unit(self) -> EmptySet:
        """Initial object ∅."""
        return EMPTY

    def associator(self, a: SetObject, b: SetObject, c: SetObject) -> ObservedMorphism:
        """(A + B) + C → A + (B + C).

        Since CoproductSet flattens, both sides are CoproductSet(A,B,C),
        so this is the identity.
        """
        flat = CoproductSet(a, b, c)
        data = self._quantale.identity_tensor(flat.shape)
        return observed(flat, flat, data, quantale=self._quantale)

    def left_unitor(self, a: SetObject) -> ObservedMorphism:
        """∅ + A → A.

        Since ∅ has size 0, CoproductSet(∅, A) has the same size as A.
        The unitor is the identity (offset for A starts at 0).
        """
        source = CoproductSet(EMPTY, a)
        target = a

        # source.size == a.size (empty contributes 0)
        n = target.size
        data = torch.zeros(n, *target.shape)

        for i in range(n):
            idx = (i,)
            data[idx + idx] = self._quantale.unit

        return observed(source, target, data, quantale=self._quantale)

    def right_unitor(self, a: SetObject) -> ObservedMorphism:
        """A + ∅ → A.

        Since ∅ has size 0, CoproductSet(A, ∅) has the same size as A.
        """
        source = CoproductSet(a, EMPTY)
        target = a

        n = target.size
        data = torch.zeros(n, *target.shape)

        for i in range(n):
            idx = (i,)
            data[idx + idx] = self._quantale.unit

        return observed(source, target, data, quantale=self._quantale)

    def braiding(self, a: SetObject, b: SetObject) -> ObservedMorphism:
        """A + B → B + A (block swap).

        Swaps the two coproduct components.
        """
        source = CoproductSet(a, b)
        target = CoproductSet(b, a)

        n_a = a.size
        n_b = b.size
        n = n_a + n_b

        data = torch.full((n, n), self._quantale.zero)

        # elements 0..n_a-1 in source (from A) map to n_b..n-1 in target
        for i in range(n_a):
            data[i, n_b + i] = self._quantale.unit

        # elements n_a..n-1 in source (from B) map to 0..n_b-1 in target
        for i in range(n_b):
            data[n_a + i, i] = self._quantale.unit

        return observed(source, target, data, quantale=self._quantale)

    def __repr__(self) -> str:
        return "CoproductMonoidal()"
