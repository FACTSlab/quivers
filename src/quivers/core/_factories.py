"""Concrete morphism constructors for the V-enriched category of finite sets.

Provides explicit V-relation tensors for the standard categorical
structure that every typeclass instance, free-monad construction, and
arrow bridge depends on: coproduct injections / case eliminators,
product projections / pairings, distributivity, terminal / initial
maps, and parallel pair morphisms over arbitrary `SetObject`
shapes. Tensors are constructed with the unit / zero of a given
algebra so the constructions work uniformly across
`ProductFuzzyAlgebra`, `BooleanAlgebra`, and the other
algebras registered in [`quivers.core.algebras`][quivers.core.algebras].

These constructors are the alphabet from which the stdlib monad,
arrow, and comonad instances are built. Each returns an
`ObservedMorphism` whose tensor is deterministic and whose
entries are the categorical structure maps.
"""

from __future__ import annotations

import itertools

import torch

from quivers.core.morphisms import Morphism, ObservedMorphism, observed, identity
from quivers.core.objects import (
    CoproductSet,
    ProductSet,
    SetObject,
    Unit,
)
from quivers.core.algebras import PRODUCT_FUZZY, Algebra


def _q(algebra: Algebra | None) -> Algebra:
    return algebra if algebra is not None else PRODUCT_FUZZY


def _flat_offsets(components: tuple[SetObject, ...]) -> tuple[int, ...]:
    """Cumulative offsets of components within a coproduct."""
    out: list[int] = []
    running = 0
    for c in components:
        out.append(running)
        running += c.size
    return tuple(out)


def _flat_indices(obj: SetObject) -> list[tuple[int, ...]]:
    """All index tuples in row-major order over an object's tensor shape."""
    return list(itertools.product(*(range(s) for s in obj.shape)))


def inj(
    components: tuple[SetObject, ...],
    index: int,
    algebra: Algebra | None = None,
) -> ObservedMorphism:
    """Coproduct injection ``ι_i : A_i → A_0 + ... + A_n``.

    The flat coproduct has shape ``(sum_j |A_j|,)``. The injection places
    each ``a ∈ A_i`` at flat position ``offset(i) + flat(a)``.

    Parameters
    ----------
    components
        The coproduct's components in declared order.
    index
        Which component this injection embeds (``0 <= index < len(components)``).
    algebra
        Enrichment algebra.
    """
    if not 0 <= index < len(components):
        raise ValueError(f"injection index {index} out of range [0, {len(components)})")
    q = _q(algebra)
    source = components[index]
    target = CoproductSet(components=components)
    offsets = _flat_offsets(components)
    base = offsets[index]
    data = torch.full((*source.shape, target.size), q.zero)
    src_size = source.size
    src_shape = source.shape
    for local_flat in range(src_size):
        # Decode flat index back to multi-index over source shape.
        idx: list[int] = []
        rem = local_flat
        for dim in reversed(src_shape):
            idx.append(rem % dim)
            rem //= dim
        idx.reverse()
        tup = tuple(idx)
        data[tup + (base + local_flat,)] = q.unit
    return observed(source, target, data, algebra=q)


def case(
    components: tuple[SetObject, ...],
    branches: tuple[Morphism, ...],
    target: SetObject,
    algebra: Algebra | None = None,
) -> ObservedMorphism:
    """Coproduct case eliminator ``[f_0, ..., f_n] : ⨿_i A_i → B``.

    Each branch ``f_i : A_i → B`` contributes the rows of the result
    corresponding to its component's range of flat coproduct indices.

    The result is realised as an ``ObservedMorphism`` whose tensor
    concatenates the branches' tensors along the flat-coproduct axis.

    Parameters
    ----------
    components
        Component objects.
    branches
        Tuple of length ``len(components)`` of morphisms each
        ``f_i : A_i → target``.
    target
        Common codomain.
    algebra
        Enrichment algebra (must match each branch's algebra).
    """
    if len(branches) != len(components):
        raise ValueError(
            f"case eliminator needs {len(components)} branches; got {len(branches)}"
        )
    q = _q(algebra)
    source = CoproductSet(components=components)
    data = torch.full((source.size, *target.shape), q.zero)
    offsets = _flat_offsets(components)
    for i, (comp, branch) in enumerate(zip(components, branches)):
        if branch.domain != comp:
            raise TypeError(
                f"case branch {i} has domain {branch.domain!r}; expected {comp!r}"
            )
        if branch.codomain != target:
            raise TypeError(
                f"case branch {i} has codomain {branch.codomain!r}; expected {target!r}"
            )
        # Flatten the branch's source-shape dimensions into a single
        # axis of length comp.size, then place into the corresponding
        # slice of the result.
        branch_t = branch.tensor.reshape(comp.size, *target.shape)
        base = offsets[i]
        data[base : base + comp.size] = branch_t
    return observed(source, target, data, algebra=q)


def pi(
    components: tuple[SetObject, ...],
    index: int,
    algebra: Algebra | None = None,
) -> ObservedMorphism:
    """Product projection ``π_i : A_0 × ... × A_n → A_i``.

    The tensor at ``(a_0, ..., a_n, b)`` is ``unit`` iff the multi-index
    of ``b`` equals the multi-index of ``a_i`` and ``zero`` otherwise.
    """
    if not 0 <= index < len(components):
        raise ValueError(
            f"projection index {index} out of range [0, {len(components)})"
        )
    q = _q(algebra)
    source = ProductSet(components=components)
    target = components[index]
    # Cumulative offsets within the flat-product shape so we can pick
    # out the dimensions belonging to component `index`.
    dim_offsets: list[int] = []
    running = 0
    for c in components:
        dim_offsets.append(running)
        running += c.ndim
    start = dim_offsets[index]
    width = target.ndim
    data = torch.full((*source.shape, *target.shape), q.zero)
    # Iterate over every input index tuple; check the projection slice.
    src_index_iter = itertools.product(*(range(s) for s in source.shape))
    for src in src_index_iter:
        tgt = src[start : start + width]
        data[src + tgt] = q.unit
    return observed(source, target, data, algebra=q)


def pair(
    legs: tuple[Morphism, ...],
    algebra: Algebra | None = None,
) -> ObservedMorphism:
    """Universal product pairing ``⟨f_0, ..., f_n⟩ : A → B_0 × ... × B_n``.

    Given morphisms ``f_i : A → B_i`` that share a domain, returns the
    morphism into the product whose ``(a, b_0, ..., b_n)`` entry is
    ``⊗_i f_i(a, b_i)`` (the V-algebra tensor product of the per-leg
    entries). Concretely realised by an outer-product across legs of
    the leg tensors.
    """
    if not legs:
        raise ValueError("pair requires at least one leg")
    q = _q(algebra)
    domain = legs[0].domain
    for i, leg in enumerate(legs):
        if leg.domain != domain:
            raise TypeError(
                f"pair leg {i} has domain {leg.domain!r}; expected {domain!r}"
            )
    target = ProductSet(components=tuple(leg.codomain for leg in legs))
    # Outer product across legs, then aggregate via the algebra's tensor_op.
    # We compute term-by-term in a vectorized manner: each leg tensor has
    # shape (*domain.shape, *codomain_i.shape); we reshape to align the
    # codomain dimensions and use the algebra's tensor_op to combine.
    dom_ndim = domain.ndim
    result_t = legs[0].tensor
    for leg in legs[1:]:
        # Expand existing result with extra trailing dims for leg's codomain
        # and leg with extra middle dims for already-accumulated codomains.
        cur_extra = leg.codomain.ndim
        prev_extra = result_t.ndim - dom_ndim
        result_view = result_t.reshape(*result_t.shape, *([1] * cur_extra))
        leg_view = leg.tensor.reshape(
            *leg.tensor.shape[:dom_ndim],
            *([1] * prev_extra),
            *leg.codomain.shape,
        )
        result_t = q.tensor_op(result_view, leg_view)
    return observed(domain, target, result_t, algebra=q)


def terminal(domain: SetObject, algebra: Algebra | None = None) -> ObservedMorphism:
    """The unique morphism ``! : A → 1`` into the terminal object.

    Sends every ``a ∈ A`` to the single element of `Unit`.
    """
    q = _q(algebra)
    data = torch.full((*domain.shape, 1), q.unit)
    return observed(domain, Unit, data, algebra=q)


def constant(
    domain: SetObject,
    codomain: SetObject,
    target_index: int,
    algebra: Algebra | None = None,
) -> ObservedMorphism:
    """Constant morphism ``const_b : A → B`` selecting element ``b ∈ B``.

    The tensor has ``unit`` on every ``(a, b)`` and ``zero`` everywhere
    else, where ``b`` corresponds to the flat index ``target_index``.
    """
    if not 0 <= target_index < codomain.size:
        raise ValueError(
            f"target_index {target_index} out of range [0, {codomain.size})"
        )
    q = _q(algebra)
    # Decode target_index back to multi-index over codomain.
    rem = target_index
    tgt_idx: list[int] = []
    for dim in reversed(codomain.shape):
        tgt_idx.append(rem % dim)
        rem //= dim
    tgt_idx.reverse()
    tgt = tuple(tgt_idx)
    data = torch.full((*domain.shape, *codomain.shape), q.zero)
    for src in itertools.product(*(range(s) for s in domain.shape)):
        data[src + tgt] = q.unit
    return observed(domain, codomain, data, algebra=q)


def distrib_right(
    a: SetObject,
    bs: tuple[SetObject, ...],
    algebra: Algebra | None = None,
) -> ObservedMorphism:
    """Distributivity ``A × (B_0 + ... + B_n) → (A × B_0) + ... + (A × B_n)``.

    The canonical isomorphism that distributes a Cartesian product
    on the right over a coproduct, in any distributive monoidal
    category. The tensor identifies ``(a, ι_i(b_i))`` with
    ``ι_i((a, b_i))``.
    """
    q = _q(algebra)
    coprod = CoproductSet(components=bs)
    source = ProductSet(components=(a, coprod))
    summands = tuple(ProductSet(components=(a, b)) for b in bs)
    target = CoproductSet(components=summands)
    src_shape = source.shape
    data = torch.full((*src_shape, target.size), q.zero)
    coprod_offsets = _flat_offsets(bs)
    target_offsets = _flat_offsets(summands)
    a_idxs = _flat_indices(a)
    for i, b in enumerate(bs):
        b_idxs = _flat_indices(b)
        coprod_base = coprod_offsets[i]
        target_base = target_offsets[i]
        # In a flat product index of (a, coprod), the coprod axis is a
        # single dim — but a may have multi-dim shape. So source idx is
        # (*a_idx, coprod_flat); target flat idx is target_base + (a_flat * |b| + b_flat).
        a_size = a.size
        b_size = b.size
        for local_a, a_idx in enumerate(a_idxs):
            for local_b, b_idx in enumerate(b_idxs):
                src_idx = a_idx + (coprod_base + local_b,)
                # Encode (a, b) within ProductSet(a, b) as flat
                # row-major over a then b dimensions.
                # Local flat within summand:
                local_tgt = local_a * b_size + local_b
                _ = local_tgt  # tuple-index form below; suppress lint
                # Build full target-shape multi-index, then since
                # target is a CoproductSet, the codomain has shape
                # (target.size,) — a single flat dimension.
                _ = a_size, b_size
                data[src_idx + (target_base + local_a * b_size + local_b,)] = q.unit
    return observed(source, target, data, algebra=q)


def parallel(
    f: Morphism,
    g: Morphism,
    algebra: Algebra | None = None,
) -> ObservedMorphism:
    """Parallel pair ``f ⊗ g : A × C → B × D`` of two morphisms.

    The tensor is the V-algebra outer product of ``f`` and ``g``:
    ``(f ⊗ g)((a, c), (b, d)) = f(a, b) ⊗ g(c, d)``.
    """
    q = _q(algebra)
    domain = ProductSet(components=(f.domain, g.domain))
    codomain = ProductSet(components=(f.codomain, g.codomain))
    # Build the outer product via reshaping for broadcast under tensor_op.
    f_dom_n = f.domain.ndim
    f_cod_n = f.codomain.ndim
    g_dom_n = g.domain.ndim
    g_cod_n = g.codomain.ndim
    # Target layout: (*f.domain, *g.domain, *f.codomain, *g.codomain).
    # f tensor has shape (*f.domain, *f.codomain); reshape to insert g
    # axes as size-1 broadcast dims at the correct positions.
    f_t = f.tensor.reshape(
        *f.domain.shape,
        *([1] * g_dom_n),
        *f.codomain.shape,
        *([1] * g_cod_n),
    )
    g_t = g.tensor.reshape(
        *([1] * f_dom_n),
        *g.domain.shape,
        *([1] * f_cod_n),
        *g.codomain.shape,
    )
    data = q.tensor_op(f_t, g_t)
    return observed(domain, codomain, data, algebra=q)


def coproduct_map(
    branches: tuple[Morphism, ...],
    algebra: Algebra | None = None,
) -> ObservedMorphism:
    """Coproduct functorial action ``f_0 + ... + f_n : ⨿_i A_i → ⨿_i B_i``.

    Given ``f_i : A_i → B_i``, builds the coproduct morphism that acts
    as ``f_i`` on the ``i``-th component.
    """
    if not branches:
        raise ValueError("coproduct_map requires at least one branch")
    q = _q(algebra)
    src_components = tuple(b.domain for b in branches)
    tgt_components = tuple(b.codomain for b in branches)
    source = CoproductSet(components=src_components)
    target = CoproductSet(components=tgt_components)
    data = torch.full((source.size, target.size), q.zero)
    src_offsets = _flat_offsets(src_components)
    tgt_offsets = _flat_offsets(tgt_components)
    for i, branch in enumerate(branches):
        # branch.tensor has shape (*A_i.shape, *B_i.shape); flatten both.
        a = branch.domain
        b = branch.codomain
        flat = branch.tensor.reshape(a.size, b.size)
        sa = src_offsets[i]
        sb = tgt_offsets[i]
        data[sa : sa + a.size, sb : sb + b.size] = flat
    return observed(source, target, data, algebra=q)


def identity_morphism(
    obj: SetObject, algebra: Algebra | None = None
) -> ObservedMorphism:
    """Identity morphism on ``obj``. Re-export of `core.morphisms.identity`."""
    return identity(obj, algebra=algebra)


def reshape_to(
    source: SetObject,
    target: SetObject,
    inner: Morphism,
    algebra: Algebra | None = None,
) -> ObservedMorphism:
    """Re-stamp ``inner``'s tensor onto ``source → target`` shapes.

    Requires ``source.size == inner.domain.size`` and
    ``target.size == inner.codomain.size``; reshapes the tensor data
    onto the new shape. Used to bridge between equivalent presentations
    (e.g. ``ProductSet(A, B)`` vs. a single ``FinSet(|A|·|B|)``) without
    rebuilding the morphism.
    """
    q = _q(algebra)
    if source.size != inner.domain.size:
        raise ValueError(
            f"reshape_to: source size {source.size} != inner domain size {inner.domain.size}"
        )
    if target.size != inner.codomain.size:
        raise ValueError(
            f"reshape_to: target size {target.size} != inner codomain size {inner.codomain.size}"
        )
    data = inner.tensor.reshape(*source.shape, *target.shape)
    return observed(source, target, data, algebra=q)


__all__ = [
    "inj",
    "case",
    "pi",
    "pair",
    "terminal",
    "constant",
    "distrib_right",
    "parallel",
    "coproduct_map",
    "identity_morphism",
    "reshape_to",
]
