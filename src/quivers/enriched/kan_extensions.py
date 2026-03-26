"""Kan extensions: generalized (re)indexing of morphisms.

The left Kan extension of a morphism R: A → B along a map p: A → A'
produces a morphism Lan_p(R): A' → B via:

    (Lan_p R)(a', b) = ⋁{a : p(a) = a'} R(a, b)

The right Kan extension uses meet (⋀) instead of join (⋁):

    (Ran_p R)(a', b) = ⋀{a : p(a) = a'} R(a, b)

Marginalization is a special case: left Kan extension along a
projection π: A₁×...×Aₙ → A_{i₁}×...×A_{iₖ}.

This module provides:

    ObjectMap (abstract) — maps between finite sets
    ├── Projection       — π: A₁×...×Aₙ → some subset of components
    └── Inclusion        — ι: A → A + B (coproduct injection)

    left_kan()  — left Kan extension
    right_kan() — right Kan extension
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import torch

from quivers.core.objects import SetObject, ProductSet, CoproductSet
from quivers.core.morphisms import Morphism, ObservedMorphism, observed
from quivers.core.quantales import PRODUCT_FUZZY, Quantale


class ObjectMap(ABC):
    """An abstract deterministic map between finite sets.

    Represents a function p: A → A' at the element level, used
    as the map along which to compute Kan extensions.
    """

    @property
    @abstractmethod
    def source(self) -> SetObject:
        """The source object A."""
        ...

    @property
    @abstractmethod
    def target(self) -> SetObject:
        """The target object A'."""
        ...

    @abstractmethod
    def apply(self, source_idx: tuple[int, ...]) -> tuple[int, ...]:
        """Map a source index to a target index.

        Parameters
        ----------
        source_idx : tuple[int, ...]
            An element of A as a multi-index.

        Returns
        -------
        tuple[int, ...]
            The corresponding element of A'.
        """
        ...

    def fiber_indices(
        self, target_idx: tuple[int, ...]
    ) -> list[tuple[int, ...]]:
        """Return all source indices that map to the given target index.

        Parameters
        ----------
        target_idx : tuple[int, ...]
            An element of A'.

        Returns
        -------
        list[tuple[int, ...]]
            All a ∈ A such that p(a) = target_idx.
        """
        result: list[tuple[int, ...]] = []

        for src_idx in itertools.product(*(range(s) for s in self.source.shape)):
            if self.apply(src_idx) == target_idx:
                result.append(src_idx)

        return result


class Projection(ObjectMap):
    """Projection from a product to a subset of its components.

    Given ProductSet(A₁, ..., Aₙ), projects onto the components
    at the specified indices: π(a₁, ..., aₙ) = (a_{i₁}, ..., a_{iₖ}).

    Parameters
    ----------
    product : ProductSet
        The source product set.
    keep_indices : tuple[int, ...]
        Indices of the components to keep (0-based).
    """

    def __init__(
        self, product: ProductSet, keep_indices: tuple[int, ...]
    ) -> None:
        if not isinstance(product, ProductSet):
            raise TypeError(
                f"Projection requires ProductSet, got {type(product).__name__}"
            )

        n = len(product.components)

        for idx in keep_indices:
            if not (0 <= idx < n):
                raise ValueError(
                    f"component index {idx} out of range [0, {n})"
                )

        self._product = product
        self._keep_indices = keep_indices
        self._drop_indices = tuple(
            i for i in range(n) if i not in keep_indices
        )

        # build target set
        kept = [product.components[i] for i in keep_indices]

        if len(kept) == 1:
            self._target = kept[0]

        else:
            self._target = ProductSet(*kept)

        # precompute dimension offsets for each component
        self._dim_offsets: list[int] = []
        offset = 0

        for comp in product.components:
            self._dim_offsets.append(offset)
            offset += comp.ndim

    @property
    def source(self) -> ProductSet:
        """The source product set."""
        return self._product

    @property
    def target(self) -> SetObject:
        """The target (projected) set."""
        return self._target

    @property
    def keep_indices(self) -> tuple[int, ...]:
        """Indices of the kept components."""
        return self._keep_indices

    @property
    def drop_indices(self) -> tuple[int, ...]:
        """Indices of the dropped (marginalized) components."""
        return self._drop_indices

    def apply(self, source_idx: tuple[int, ...]) -> tuple[int, ...]:
        """Project by keeping only the selected component indices."""
        result: list[int] = []

        for comp_idx in self._keep_indices:
            comp = self._product.components[comp_idx]
            offset = self._dim_offsets[comp_idx]

            for d in range(comp.ndim):
                result.append(source_idx[offset + d])

        return tuple(result)


class Inclusion(ObjectMap):
    """Coproduct inclusion ι_k: Aₖ → A₁ + ... + Aₙ.

    Embeds the k-th component of a CoproductSet into the full set.

    Parameters
    ----------
    coproduct : CoproductSet
        The target coproduct set.
    component_index : int
        Which component to include (0-based).
    """

    def __init__(
        self, coproduct: CoproductSet, component_index: int
    ) -> None:
        if not isinstance(coproduct, CoproductSet):
            raise TypeError(
                f"Inclusion requires CoproductSet, got "
                f"{type(coproduct).__name__}"
            )

        n = len(coproduct.components)

        if not (0 <= component_index < n):
            raise ValueError(
                f"component_index {component_index} out of range [0, {n})"
            )

        self._coproduct = coproduct
        self._component_index = component_index
        self._component = coproduct.components[component_index]
        self._offset = coproduct.offset(component_index)

    @property
    def source(self) -> SetObject:
        """The source component."""
        return self._component

    @property
    def target(self) -> CoproductSet:
        """The target coproduct set."""
        return self._coproduct

    def apply(self, source_idx: tuple[int, ...]) -> tuple[int, ...]:
        """Embed into the coproduct with offset."""
        # coproduct has shape (total_size,), so indices are flat
        (flat_idx,) = source_idx
        return (self._offset + flat_idx,)


def left_kan(
    morph: Morphism,
    along: ObjectMap,
    quantale: Quantale | None = None,
) -> ObservedMorphism:
    """Left Kan extension of a morphism along an object map.

    Computes: (Lan_p R)(a', b) = ⋁{a : p(a) = a'} R(a, b)

    Parameters
    ----------
    morph : Morphism
        The morphism R: A → B.
    along : ObjectMap
        The map p: A → A'.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    ObservedMorphism
        The left Kan extension Lan_p(R): A' → B.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY

    target = along.target
    codomain = morph.codomain
    result_shape = (*target.shape, *codomain.shape)
    result = torch.full(result_shape, q.zero)
    source_tensor = morph.tensor.detach()

    # for each target index, join over the fiber
    for tgt_idx in itertools.product(*(range(s) for s in target.shape)):
        fiber = along.fiber_indices(tgt_idx)

        if not fiber:
            continue

        # collect R(a, :) for all a in the fiber
        slices = torch.stack(
            [source_tensor[src_idx] for src_idx in fiber]
        )

        # join over the fiber dimension (dim=0)
        joined = q.join(slices, dim=0)
        result[tgt_idx] = joined

    return observed(target, codomain, result, quantale=q)


def right_kan(
    morph: Morphism,
    along: ObjectMap,
    quantale: Quantale | None = None,
) -> ObservedMorphism:
    """Right Kan extension of a morphism along an object map.

    Computes: (Ran_p R)(a', b) = ⋀{a : p(a) = a'} R(a, b)

    Parameters
    ----------
    morph : Morphism
        The morphism R: A → B.
    along : ObjectMap
        The map p: A → A'.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    ObservedMorphism
        The right Kan extension Ran_p(R): A' → B.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY

    target = along.target
    codomain = morph.codomain
    result_shape = (*target.shape, *codomain.shape)
    result = torch.full(result_shape, q.unit)
    source_tensor = morph.tensor.detach()

    for tgt_idx in itertools.product(*(range(s) for s in target.shape)):
        fiber = along.fiber_indices(tgt_idx)

        if not fiber:
            # empty fiber: meet over empty set = unit
            continue

        slices = torch.stack(
            [source_tensor[src_idx] for src_idx in fiber]
        )

        # meet over the fiber dimension (dim=0)
        met = q.meet(slices, dim=0)
        result[tgt_idx] = met

    return observed(target, codomain, result, quantale=q)
