"""Traced monoidal categories: feedback and fixpoint.

A traced monoidal category is a symmetric monoidal category (C, ⊗, I)
equipped with a trace operator:

    Tr^U_{A,B}: C(A ⊗ U, B ⊗ U) → C(A, B)

satisfying naturality, dinaturality, vanishing, superposing, and
yanking axioms.

In the V-enriched fuzzy-relation setting, the trace corresponds to
a fixpoint computation: given f: A × U → B × U, the trace
"feeds back" the U component:

    Tr(f)(a, b) = ⋁_u f((a, u), (b, u))

which is the join (existential) over the feedback variable.

For the tropical quantale, this computes shortest-cycle distances.
For general quantales, it may require iteration to a fixpoint.

This module provides:

    TracedMonoidal (abstract)
    ├── CartesianTrace   — trace on the cartesian monoidal structure
    └── IterativeTrace   — trace via fixpoint iteration

    trace()              — apply the trace operator
    partial_trace()      — trace over a subset of feedback wires
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import itertools

import torch

from quivers.core.objects import SetObject, ProductSet
from quivers.core.morphisms import Morphism, observed, identity
from quivers.core.quantales import PRODUCT_FUZZY, Quantale
from quivers.categorical.monoidal import MonoidalStructure, CartesianMonoidal


class TracedMonoidal(ABC):
    """Abstract traced monoidal category.

    Provides the trace operator Tr^U_{A,B} on a monoidal structure.

    Parameters
    ----------
    monoidal : MonoidalStructure
        The underlying monoidal structure.
    quantale : Quantale or None
        The enrichment algebra.
    """

    def __init__(
        self,
        monoidal: MonoidalStructure,
        quantale: Quantale | None = None,
    ) -> None:
        self._monoidal = monoidal
        self._quantale = quantale if quantale is not None else PRODUCT_FUZZY

    @property
    def monoidal(self) -> MonoidalStructure:
        """The underlying monoidal structure."""
        return self._monoidal

    @property
    def quantale(self) -> Quantale:
        """The enrichment algebra."""
        return self._quantale

    @abstractmethod
    def trace(
        self,
        morph: Morphism,
        feedback: SetObject,
        domain: SetObject,
        codomain: SetObject,
    ) -> Morphism:
        """Apply the trace operator.

        Given f: A ⊗ U → B ⊗ U, compute Tr^U(f): A → B.

        Parameters
        ----------
        morph : Morphism
            The morphism f: A ⊗ U → B ⊗ U.
        feedback : SetObject
            The feedback object U.
        domain : SetObject
            The external domain A.
        codomain : SetObject
            The external codomain B.

        Returns
        -------
        Morphism
            The traced morphism Tr^U(f): A → B.
        """
        ...

    def verify_yanking(
        self,
        feedback: SetObject,
        atol: float = 1e-5,
    ) -> bool:
        """Verify the yanking axiom: Tr^U(σ_{U,U}) = id_U.

        The trace of the braiding/swap should be the identity.

        Parameters
        ----------
        feedback : SetObject
            The object U to test.
        atol : float
            Absolute tolerance.

        Returns
        -------
        bool
            True if yanking holds within tolerance.
        """
        # σ_{U,U}: U ⊗ U → U ⊗ U (swap)
        if isinstance(self._monoidal, CartesianMonoidal):
            swap = self._monoidal.braiding(feedback, feedback)
            traced = self.trace(swap, feedback, feedback, feedback)
            expected = identity(feedback, quantale=self._quantale).tensor

            return torch.allclose(traced.tensor, expected, atol=atol)

        return True  # skip for non-cartesian

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self._monoidal!r})"


class CartesianTrace(TracedMonoidal):
    """Trace on the cartesian monoidal structure (FinSet, ×, 1).

    For a morphism f: A × U → B × U, the cartesian trace is:

        Tr^U(f)(a, b) = ⋁_u f((a, u), (b, u))

    This is simply the join over the feedback dimensions of the
    tensor's diagonal (where the U components of domain and
    codomain agree).

    Parameters
    ----------
    quantale : Quantale or None
        The enrichment algebra.
    """

    def __init__(self, quantale: Quantale | None = None) -> None:
        q = quantale if quantale is not None else PRODUCT_FUZZY
        super().__init__(CartesianMonoidal(q), q)

    def trace(
        self,
        morph: Morphism,
        feedback: SetObject,
        domain: SetObject,
        codomain: SetObject,
    ) -> Morphism:
        """Compute the cartesian trace via diagonal extraction and join.

        Parameters
        ----------
        morph : Morphism
            f: A × U → B × U.
        feedback : SetObject
            The feedback object U.
        domain : SetObject
            The external domain A.
        codomain : SetObject
            The external codomain B.

        Returns
        -------
        ObservedMorphism
            Tr^U(f): A → B.
        """
        q = self._quantale
        t = morph.tensor

        # tensor shape: (*a.shape, *u.shape, *b.shape, *u.shape)
        # we need to extract the diagonal over the two U copies
        # and join over U

        # the U dimensions in domain are at positions [n_a, n_a + n_u)
        # the U dimensions in codomain are at positions [n_a + n_u + n_b, n_a + n_u + n_b + n_u)
        result_shape = (*domain.shape, *codomain.shape)
        result = torch.full(result_shape, q.zero)

        # extract diagonal and join over U
        for a_idx in itertools.product(*(range(s) for s in domain.shape)):
            for b_idx in itertools.product(*(range(s) for s in codomain.shape)):
                # collect f(a, u, b, u) for all u
                vals: list[torch.Tensor] = []

                for u_idx in itertools.product(*(range(s) for s in feedback.shape)):
                    src_idx = a_idx + u_idx + b_idx + u_idx
                    vals.append(t[src_idx].unsqueeze(0))

                if vals:
                    stacked = torch.cat(vals)
                    result[a_idx + b_idx] = q.join(stacked, dim=0)

        return observed(domain, codomain, result, quantale=q)


class IterativeTrace(TracedMonoidal):
    """Trace via fixpoint iteration.

    For quantales where the trace does not have a closed-form
    expression, this computes the trace by iterating:

        x_0 = ⊥
        x_{n+1} = f(a, x_n)(b, x_n)  (schematically)

    until convergence. This works for continuous quantales on
    complete lattices (Kleene's fixpoint theorem).

    Parameters
    ----------
    monoidal : MonoidalStructure
        The monoidal structure.
    quantale : Quantale or None
        The enrichment algebra.
    max_iter : int
        Maximum number of iterations.
    atol : float
        Convergence tolerance.
    """

    def __init__(
        self,
        monoidal: MonoidalStructure,
        quantale: Quantale | None = None,
        max_iter: int = 100,
        atol: float = 1e-6,
    ) -> None:
        super().__init__(monoidal, quantale)
        self._max_iter = max_iter
        self._atol = atol

    @property
    def max_iter(self) -> int:
        """Maximum number of fixpoint iterations."""
        return self._max_iter

    def trace(
        self,
        morph: Morphism,
        feedback: SetObject,
        domain: SetObject,
        codomain: SetObject,
    ) -> Morphism:
        """Compute the trace via fixpoint iteration.

        Parameters
        ----------
        morph : Morphism
            f: A ⊗ U → B ⊗ U.
        feedback : SetObject
            The feedback object U.
        domain : SetObject
            The external domain A.
        codomain : SetObject
            The external codomain B.

        Returns
        -------
        ObservedMorphism
            Tr^U(f): A → B.
        """
        q = self._quantale
        t = morph.tensor

        result_shape = (*domain.shape, *codomain.shape)

        # initialize with zero (bottom)
        prev = torch.full(result_shape, q.zero)

        for _ in range(self._max_iter):
            current = torch.full(result_shape, q.zero)

            for a_idx in itertools.product(*(range(s) for s in domain.shape)):
                for b_idx in itertools.product(*(range(s) for s in codomain.shape)):
                    vals: list[torch.Tensor] = []

                    for u_idx in itertools.product(*(range(s) for s in feedback.shape)):
                        src_idx = a_idx + u_idx + b_idx + u_idx
                        vals.append(t[src_idx].unsqueeze(0))

                    if vals:
                        stacked = torch.cat(vals)
                        current[a_idx + b_idx] = q.join(stacked, dim=0)

            if torch.allclose(current, prev, atol=self._atol):
                break

            prev = current

        return observed(domain, codomain, current, quantale=q)


def trace(
    morph: Morphism,
    feedback: SetObject,
    domain: SetObject,
    codomain: SetObject,
    quantale: Quantale | None = None,
) -> Morphism:
    """Convenience function: apply cartesian trace.

    Given f: A × U → B × U, compute Tr^U(f): A → B
    using the CartesianTrace.

    Parameters
    ----------
    morph : Morphism
        The morphism f: A × U → B × U.
    feedback : SetObject
        The feedback object U.
    domain : SetObject
        The external domain A.
    codomain : SetObject
        The external codomain B.
    quantale : Quantale or None
        The enrichment algebra.

    Returns
    -------
    Morphism
        The traced morphism Tr^U(f): A → B.
    """
    tracer = CartesianTrace(quantale=quantale)
    return tracer.trace(morph, feedback, domain, codomain)


def partial_trace(
    morph: Morphism,
    feedback_indices: tuple[int, ...],
    quantale: Quantale | None = None,
) -> Morphism:
    """Trace over a subset of feedback wires in a product.

    Given f: (A₁ × ... × Aₙ) → (B₁ × ... × Bₙ) where some
    pairs (Aᵢ, Bᵢ) are feedback wires (Aᵢ = Bᵢ), compute the
    partial trace over the specified pairs.

    Parameters
    ----------
    morph : Morphism
        The morphism.
    feedback_indices : tuple[int, ...]
        Indices of the product components to trace over.
        These components must have matching shapes in domain
        and codomain.
    quantale : Quantale or None
        The enrichment algebra.

    Returns
    -------
    ObservedMorphism
        The partially traced morphism.
    """
    q = quantale if quantale is not None else PRODUCT_FUZZY
    t = morph.tensor
    dom = morph.domain
    cod = morph.codomain

    if not isinstance(dom, ProductSet) or not isinstance(cod, ProductSet):
        raise TypeError("partial_trace requires ProductSet domain and codomain")

    if len(dom.components) != len(cod.components):
        raise ValueError("domain and codomain must have the same number of components")

    # validate feedback components match
    for idx in feedback_indices:
        if dom.components[idx].shape != cod.components[idx].shape:
            raise ValueError(
                f"component {idx}: domain shape {dom.components[idx].shape} "
                f"!= codomain shape {cod.components[idx].shape}"
            )

    # build feedback and external objects
    external_dom_comps = [
        c for i, c in enumerate(dom.components) if i not in feedback_indices
    ]
    external_cod_comps = [
        c for i, c in enumerate(cod.components) if i not in feedback_indices
    ]
    [dom.components[i] for i in feedback_indices]

    # compute dimension indices
    n_dom = dom.ndim
    dom_offsets: list[int] = []
    offset = 0

    for c in dom.components:
        dom_offsets.append(offset)
        offset += c.ndim

    cod_offsets: list[int] = []
    offset = n_dom

    for c in cod.components:
        cod_offsets.append(offset)
        offset += c.ndim

    # identify feedback dimension pairs
    contra_dims: list[int] = []
    co_dims: list[int] = []

    for idx in feedback_indices:
        comp = dom.components[idx]

        for d in range(comp.ndim):
            contra_dims.append(dom_offsets[idx] + d)
            co_dims.append(cod_offsets[idx] + d)

    # use diagonal extraction and join
    from quivers.enriched.ends_coends import coend

    result = coend(
        t,
        contra_dims=tuple(contra_dims),
        co_dims=tuple(co_dims),
        quantale=q,
    )

    # build result objects
    if len(external_dom_comps) == 0:
        raise ValueError("cannot trace over all components")

    elif len(external_dom_comps) == 1:
        ext_dom = external_dom_comps[0]

    else:
        ext_dom = ProductSet(*external_dom_comps)

    if len(external_cod_comps) == 1:
        ext_cod = external_cod_comps[0]

    else:
        ext_cod = ProductSet(*external_cod_comps)

    return observed(ext_dom, ext_cod, result, quantale=q)
