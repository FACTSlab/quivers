"""Weighted (co)limits for V-enriched categories.

In a V-enriched category, the correct notion of limit is the
*weighted* limit. Given a V-functor D: J → C (diagram) and a
V-functor W: J → V (weight), the weighted limit {W, D} is
characterized by:

    C(X, {W, D}) ≅ [J, V](W, C(X, D-))

and the weighted colimit W ⊗_J D by:

    C(W ⊗_J D, X) ≅ [J, V](W, C(D-, X))

In our finite setting, these reduce to end/coend computations:

    {W, D} = ∫_j [W(j), D(j)]     (weighted limit)
    W ⊗_J D = ∫^j W(j) ⊗ D(j)    (weighted colimit)

where [-, -] denotes the internal hom in V and ⊗ the tensor.

This module provides:

    Weight                  — a V-valued presheaf (weight functor)
    Diagram                 — a finite diagram of objects/morphisms
    weighted_limit()        — compute {W, D}
    weighted_colimit()      — compute W ⊗_J D
    representable_weight()  — weight represented by an object
    terminal_weight()       — constant weight at the unit
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import itertools

import torch

from quivers.core.objects import SetObject, FinSet
from quivers.core.morphisms import Morphism, ObservedMorphism, observed
from quivers.core.quantales import PRODUCT_FUZZY, Quantale


@dataclass
class Weight:
    """A V-valued presheaf on a finite indexing category.

    Represents a weight functor W: J → V where J is a finite set of
    indices, and W assigns a V-value (scalar in the quantale's lattice)
    to each index.

    Parameters
    ----------
    values : torch.Tensor
        A 1D tensor of shape (|J|,) with values in L (the quantale's
        lattice). W(j) = values[j].
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    values: torch.Tensor
    quantale: Quantale | None = None

    def __post_init__(self) -> None:
        if self.values.ndim != 1:
            raise ValueError(
                f"weight values must be 1D, got shape {self.values.shape}"
            )

        if self.quantale is None:
            self.quantale = PRODUCT_FUZZY

    @property
    def size(self) -> int:
        """Number of indices in J."""
        return self.values.shape[0]


@dataclass
class Diagram:
    """A finite diagram of morphisms sharing a common domain or codomain.

    For weighted limits, this is a collection of morphisms
    D(j): X → A_j for each index j (cone-shaped).
    For weighted colimits, this is a collection of morphisms
    D(j): A_j → X for each index j (cocone-shaped).

    In the simplest case (discrete diagram), this is just a list of
    objects with no connecting morphisms.

    Parameters
    ----------
    objects : Sequence[SetObject]
        The objects A_j in the diagram, one per index j.
    """

    objects: Sequence[SetObject]

    @property
    def size(self) -> int:
        """Number of objects in the diagram."""
        return len(self.objects)


def weighted_limit(
    weight: Weight,
    diagram: Diagram,
    quantale: Quantale | None = None,
) -> torch.Tensor:
    """Compute the weighted limit {W, D} for a discrete diagram.

    For a discrete diagram (no connecting morphisms), the weighted
    limit reduces to:

        {W, D} = ∫_j [W(j), D(j)] = ⋀_j [W(j), D(j)]

    where [w, x] is the internal hom in the quantale (residuation):

        [w, x] = sup{v : w ⊗ v ≤ x}

    For the product-fuzzy quantale, [w, x] = min(1, x/w) when w > 0,
    and the unit when w = 0.

    For discrete diagrams with objects A_0, ..., A_{n-1}, this
    computes a tensor of shape (∏_j |A_j|,) representing the
    weighted product.

    Parameters
    ----------
    weight : Weight
        The weight W with values W(j) for each j.
    diagram : Diagram
        The diagram of objects.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        The weighted limit tensor. For discrete diagrams, shape is
        the meet of identity tensors scaled by the weights.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY

    if weight.size != diagram.size:
        raise ValueError(
            f"weight size {weight.size} != diagram size {diagram.size}"
        )

    n = weight.size

    if n == 0:
        return torch.tensor(q.unit)

    # for a discrete diagram, the weighted limit is the meet
    # (product) of the identities scaled by internal homs
    # {W, D}_j gives a "weighted identity" for each component
    components: list[torch.Tensor] = []

    for j in range(n):
        obj = diagram.objects[j]
        w_j = weight.values[j]

        # identity tensor for object j, scaled by the weight
        id_j = q.identity_tensor(obj.shape)
        scaled = _internal_hom_scalar(w_j, id_j, q)
        components.append(scaled)

    # the weighted limit is the meet over all components
    # for independent objects, this is just the collection
    # return as a list of component tensors
    # for the product, stack and meet over the first dim
    if len(components) == 1:
        return components[0]

    # for a discrete diagram, return the component-wise meet
    # the result represents the product weighted by W
    # we return as a dictionary-like structure encoded as a tuple
    # but for simplicity, we return the tensor for the product
    # which is the outer-meet of all scaled identities
    result = components[0]

    for comp in components[1:]:
        # outer product via tensor_op, then reshape
        n_r = result.ndim
        n_c = comp.ndim

        r_exp = result.reshape(*result.shape, *([1] * n_c))
        c_exp = comp.reshape(*([1] * n_r), *comp.shape)

        result = q.tensor_op(r_exp, c_exp)

    return result


def weighted_colimit(
    weight: Weight,
    diagram: Diagram,
    quantale: Quantale | None = None,
) -> torch.Tensor:
    """Compute the weighted colimit W ⊗_J D for a discrete diagram.

    For a discrete diagram, the weighted colimit reduces to:

        W ⊗_J D = ∫^j W(j) ⊗ D(j) = ⋁_j W(j) ⊗ D(j)

    This is the join (existential) over j of the tensor product of
    the weight value with the identity at each object.

    Parameters
    ----------
    weight : Weight
        The weight W with values W(j) for each j.
    diagram : Diagram
        The diagram of objects.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        The weighted colimit tensor.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY

    if weight.size != diagram.size:
        raise ValueError(
            f"weight size {weight.size} != diagram size {diagram.size}"
        )

    n = weight.size

    if n == 0:
        return torch.tensor(q.zero)

    components: list[torch.Tensor] = []

    for j in range(n):
        obj = diagram.objects[j]
        w_j = weight.values[j]

        # identity tensor scaled by weight
        id_j = q.identity_tensor(obj.shape)
        scaled = q.tensor_op(id_j, w_j)
        components.append(scaled)

    if len(components) == 1:
        return components[0]

    # join over all components (outer-join)
    result = components[0]

    for comp in components[1:]:
        n_r = result.ndim
        n_c = comp.ndim

        r_exp = result.reshape(*result.shape, *([1] * n_c))
        c_exp = comp.reshape(*([1] * n_r), *comp.shape)

        # join of tensor products
        result = q.tensor_op(r_exp, c_exp)

    return result


def weighted_limit_morphisms(
    weight: Weight,
    morphisms: Sequence[Morphism],
    quantale: Quantale | None = None,
) -> torch.Tensor:
    """Compute a weighted limit from a family of morphisms.

    Given morphisms f_j: X → A_j and weights W(j), computes the
    weighted meet:

        result[x, ...] = ⋀_j [W(j), f_j(x, ...)]

    This is the tensor-level computation of the weighted limit
    of a cone.

    Parameters
    ----------
    weight : Weight
        The weight W.
    morphisms : Sequence[Morphism]
        The morphisms f_j comprising the cone.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        The weighted limit tensor.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY

    if weight.size != len(morphisms):
        raise ValueError(
            f"weight size {weight.size} != "
            f"number of morphisms {len(morphisms)}"
        )

    n = len(morphisms)

    if n == 0:
        raise ValueError("need at least one morphism")

    # compute [W(j), f_j] for each j and meet over j
    components: list[torch.Tensor] = []

    for j in range(n):
        w_j = weight.values[j]
        f_j = morphisms[j].tensor
        hom = _internal_hom_scalar(w_j, f_j, q)
        components.append(hom)

    # meet over all components (they should all have the same shape)
    stacked = torch.stack(components, dim=0)
    return q.meet(stacked, dim=0)


def weighted_colimit_morphisms(
    weight: Weight,
    morphisms: Sequence[Morphism],
    quantale: Quantale | None = None,
) -> torch.Tensor:
    """Compute a weighted colimit from a family of morphisms.

    Given morphisms f_j: A_j → X and weights W(j), computes the
    weighted join:

        result[..., x] = ⋁_j W(j) ⊗ f_j(..., x)

    Parameters
    ----------
    weight : Weight
        The weight W.
    morphisms : Sequence[Morphism]
        The morphisms f_j comprising the cocone.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    torch.Tensor
        The weighted colimit tensor.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY

    if weight.size != len(morphisms):
        raise ValueError(
            f"weight size {weight.size} != "
            f"number of morphisms {len(morphisms)}"
        )

    n = len(morphisms)

    if n == 0:
        raise ValueError("need at least one morphism")

    components: list[torch.Tensor] = []

    for j in range(n):
        w_j = weight.values[j]
        f_j = morphisms[j].tensor
        scaled = q.tensor_op(f_j, w_j)
        components.append(scaled)

    stacked = torch.stack(components, dim=0)
    return q.join(stacked, dim=0)


def representable_weight(
    index_set: FinSet,
    represented_at: int,
    quantale: Quantale | None = None,
) -> Weight:
    """Create a representable weight (Yoneda-style).

    The representable weight at index k is W(j) = I if j == k,
    and W(j) = ⊥ otherwise. Weighted limits with representable
    weights recover evaluation: {y(k), D} ≅ D(k).

    Parameters
    ----------
    index_set : FinSet
        The indexing set J.
    represented_at : int
        The representing index k.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    Weight
        The representable weight at k.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY

    values = torch.full((index_set.cardinality,), q.zero)
    values[represented_at] = q.unit

    return Weight(values=values, quantale=q)


def terminal_weight(
    index_set: FinSet,
    quantale: Quantale | None = None,
) -> Weight:
    """Create the terminal (constant unit) weight.

    W(j) = I for all j. Weighted limits with the terminal weight
    recover ordinary (conical) limits.

    Parameters
    ----------
    index_set : FinSet
        The indexing set J.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.

    Returns
    -------
    Weight
        The terminal weight.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY
    values = torch.full((index_set.cardinality,), q.unit)

    return Weight(values=values, quantale=q)


def _internal_hom_scalar(
    w: torch.Tensor | float,
    x: torch.Tensor,
    quantale: Quantale,
) -> torch.Tensor:
    """Compute the internal hom [w, x] for a scalar weight w.

    For the product-fuzzy quantale: [w, x] = min(1, x/w) when w > 0,
        and 1 when w = 0.
    For the boolean quantale: [w, x] = ¬w ∨ x = max(1-w, x).
    For general quantales: use residuation [w, x] = ¬(w ⊗ ¬x) as
        a default approximation.

    Parameters
    ----------
    w : torch.Tensor or float
        The weight value (scalar in L).
    x : torch.Tensor
        The tensor to compute the hom into.
    quantale : Quantale
        The enrichment algebra.

    Returns
    -------
    torch.Tensor
        The internal hom tensor [w, x].
    """
    from quivers.core.quantales import ProductFuzzy, BooleanQuantale

    w_t = torch.as_tensor(w, dtype=x.dtype)

    if isinstance(quantale, ProductFuzzy):
        # [w, x] = min(1, x / w) for w > 0, else 1
        safe_w = w_t.clamp(min=1e-7)
        result = (x / safe_w).clamp(max=1.0)

        # where w ≈ 0, return unit
        return torch.where(w_t < 1e-7, torch.ones_like(x), result)

    elif isinstance(quantale, BooleanQuantale):
        # [w, x] = ¬w ∨ x = max(1 - w, x)
        return torch.max(1.0 - w_t, x)

    else:
        # general fallback: ¬(w ⊗ ¬x)
        return quantale.negate(
            quantale.tensor_op(w_t, quantale.negate(x))
        )
