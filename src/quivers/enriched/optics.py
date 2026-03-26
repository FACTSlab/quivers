"""Optics: composable bidirectional transformations.

Optics generalize lenses and prisms to provide composable
"accessors" that can get, set, and transform parts of a
structure. In a V-enriched setting, optics are formulated via
profunctors or as pairs of morphisms with specific laws.

The general optic from (S, T) to (A, B) in a monoidal category
is a coend:

    Optic(S, T, A, B) = ∫^M S → M ⊗ A  ×  M ⊗ B → T

which factors a transformation through a residual M.

This module provides concrete optic types:

    Optic (abstract)
    ├── Lens     — get/put on product structures
    ├── Prism    — match/build on coproduct structures
    ├── Adapter  — invertible optics (isomorphisms)
    └── Grate    — closed-structure optics

    compose_optics() — compose two optics sequentially
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import itertools

import torch

from quivers.core.objects import SetObject, ProductSet, CoproductSet
from quivers.core.morphisms import Morphism, observed, identity
from quivers.core.quantales import PRODUCT_FUZZY, Quantale
from quivers.enriched.profunctors import Profunctor


class Optic(ABC):
    """Abstract optic from (S, T) to (A, B).

    An optic provides a bidirectional transformation between a
    "whole" (S/T) and a "part" (A/B). S is the source whole type,
    T is the target whole type, A is the source part type, and B
    is the target part type.

    For simple (non-polymorphic) optics, S = T and A = B.

    Parameters
    ----------
    source : SetObject
        The source whole S.
    target : SetObject
        The target whole T.
    focus_source : SetObject
        The source part A.
    focus_target : SetObject
        The target part B.
    quantale : Quantale or None
        The enrichment algebra.
    """

    def __init__(
        self,
        source: SetObject,
        target: SetObject,
        focus_source: SetObject,
        focus_target: SetObject,
        quantale: Quantale | None = None,
    ) -> None:
        self._source = source
        self._target = target
        self._focus_source = focus_source
        self._focus_target = focus_target
        self._quantale = quantale if quantale is not None else PRODUCT_FUZZY

    @property
    def source(self) -> SetObject:
        """The source whole S."""
        return self._source

    @property
    def target(self) -> SetObject:
        """The target whole T."""
        return self._target

    @property
    def focus_source(self) -> SetObject:
        """The source part A."""
        return self._focus_source

    @property
    def focus_target(self) -> SetObject:
        """The target part B."""
        return self._focus_target

    @property
    def quantale(self) -> Quantale:
        """The enrichment algebra."""
        return self._quantale

    @abstractmethod
    def forward(self) -> Morphism:
        """The forward (get/match) morphism.

        Returns
        -------
        Morphism
            A morphism extracting the focus from the source.
        """
        ...

    @abstractmethod
    def backward(self) -> Morphism:
        """The backward (put/build) morphism.

        Returns
        -------
        Morphism
            A morphism putting the focus back into the whole.
        """
        ...

    def as_profunctor(self) -> Profunctor:
        """View this optic as a profunctor S ↛ T.

        Computes the profunctor representation by composing
        forward and backward through the focus.

        Returns
        -------
        Profunctor
            The profunctor representation of this optic.
        """
        fwd = self.forward()
        bwd = self.backward()

        # the profunctor view depends on the optic type,
        # but generally it's fwd >> bwd or a similar composition
        tensor = self._quantale.compose(fwd.tensor, bwd.tensor, self._focus_source.ndim)

        return Profunctor(
            contra=self._source,
            co=self._target,
            tensor=tensor,
            quantale=self._quantale,
        )

    def __repr__(self) -> str:
        cls = type(self).__name__
        return (
            f"{cls}(({self._source!r}, {self._target!r}) "
            f"→ ({self._focus_source!r}, {self._focus_target!r}))"
        )


class Lens(Optic):
    """A lens focusing on a component of a product.

    A lens from (S, S) to (A, A) on a ProductSet S = A × C consists of:

        get: S → A           (extract the focus)
        put: A × C → S       (replace the focus, keeping complement)

    In the V-enriched setting, get is a projection morphism and put
    reconstructs the product.

    Parameters
    ----------
    whole : ProductSet
        The whole product S = A × C.
    focus_index : int
        Index of the focus component A in the product.
    quantale : Quantale or None
        The enrichment algebra.
    """

    def __init__(
        self,
        whole: ProductSet,
        focus_index: int,
        quantale: Quantale | None = None,
    ) -> None:
        if not isinstance(whole, ProductSet):
            raise TypeError(f"Lens requires ProductSet, got {type(whole).__name__}")

        if not (0 <= focus_index < len(whole.components)):
            raise ValueError(
                f"focus_index {focus_index} out of range [0, {len(whole.components)})"
            )

        focus = whole.components[focus_index]
        super().__init__(
            source=whole,
            target=whole,
            focus_source=focus,
            focus_target=focus,
            quantale=quantale,
        )
        self._focus_index = focus_index

    @property
    def focus_index(self) -> int:
        """Index of the focus component."""
        return self._focus_index

    def forward(self) -> Morphism:
        """get: S → A (projection to focus component).

        Returns
        -------
        ObservedMorphism
            The projection morphism.
        """
        q = self._quantale
        whole = self._source
        focus = self._focus_source

        data = torch.full((*whole.shape, *focus.shape), q.zero)

        # build projection tensor
        assert isinstance(whole, ProductSet)
        components = whole.components

        # compute dimension offset for focus
        offset = 0

        for i in range(self._focus_index):
            offset += components[i].ndim

        # for each element of the whole, project to the focus
        for idx in itertools.product(*(range(s) for s in whole.shape)):
            focus_idx = idx[offset : offset + focus.ndim]
            data[idx + focus_idx] = q.unit

        return observed(whole, focus, data, quantale=q)

    def backward(self) -> Morphism:
        """put: A → S (embed focus back, averaging over complement).

        In the V-enriched setting, the "put" creates a morphism
        A → S where fixing the focus component and joining over
        the complement gives a fuzzy relation.

        Returns
        -------
        ObservedMorphism
            The embedding morphism.
        """
        q = self._quantale
        whole = self._source
        focus = self._focus_source

        # put: A → S where S = A × C
        # for each a, put(a, (a', c)) = δ(a, a') for the focus,
        # and the unit for the complement
        assert isinstance(whole, ProductSet)

        data = torch.full((*focus.shape, *whole.shape), q.zero)

        # compute dimension offset for focus
        components = whole.components
        offset = 0

        for i in range(self._focus_index):
            offset += components[i].ndim

        for focus_idx in itertools.product(*(range(s) for s in focus.shape)):
            # for each complement index, set the delta
            complement_shapes: list[tuple[int, ...]] = []

            for i, comp in enumerate(components):
                if i != self._focus_index:
                    complement_shapes.append(tuple(range(s) for s in comp.shape))

            # iterate over all whole indices where focus matches
            for whole_idx in itertools.product(*(range(s) for s in whole.shape)):
                w_focus = whole_idx[offset : offset + focus.ndim]

                if w_focus == focus_idx:
                    data[focus_idx + whole_idx] = q.unit

        return observed(focus, whole, data, quantale=q)


class Prism(Optic):
    """A prism focusing on a component of a coproduct.

    A prism from (S, S) to (A, A) on a CoproductSet S = A + C consists of:

        match: S → A + C    (attempt to extract the focus)
        build: A → S         (embed the focus into the whole)

    Parameters
    ----------
    whole : CoproductSet
        The whole coproduct S = A + C.
    focus_index : int
        Index of the focus component A in the coproduct.
    quantale : Quantale or None
        The enrichment algebra.
    """

    def __init__(
        self,
        whole: CoproductSet,
        focus_index: int,
        quantale: Quantale | None = None,
    ) -> None:
        if not isinstance(whole, CoproductSet):
            raise TypeError(f"Prism requires CoproductSet, got {type(whole).__name__}")

        if not (0 <= focus_index < len(whole.components)):
            raise ValueError(
                f"focus_index {focus_index} out of range [0, {len(whole.components)})"
            )

        focus = whole.components[focus_index]
        super().__init__(
            source=whole,
            target=whole,
            focus_source=focus,
            focus_target=focus,
            quantale=quantale,
        )
        self._focus_index = focus_index

    @property
    def focus_index(self) -> int:
        """Index of the focus component."""
        return self._focus_index

    def forward(self) -> Morphism:
        """match: S → A (partial extraction, zero on non-matching).

        Returns
        -------
        ObservedMorphism
            The matching morphism.
        """
        q = self._quantale
        whole = self._source
        focus = self._focus_source

        assert isinstance(whole, CoproductSet)
        start, end = whole.component_range(self._focus_index)

        data = torch.full((whole.size, focus.size), q.zero)

        for i in range(focus.size):
            data[start + i, i] = q.unit

        return observed(whole, focus, data, quantale=q)

    def backward(self) -> Morphism:
        """build: A → S (inject focus into the coproduct).

        Returns
        -------
        ObservedMorphism
            The injection morphism.
        """
        q = self._quantale
        whole = self._source
        focus = self._focus_source

        assert isinstance(whole, CoproductSet)
        start, end = whole.component_range(self._focus_index)

        data = torch.full((focus.size, whole.size), q.zero)

        for i in range(focus.size):
            data[i, start + i] = q.unit

        return observed(focus, whole, data, quantale=q)


class Adapter(Optic):
    """An adapter (isomorphism optic).

    An adapter between (S, T) and (A, B) consists of an isomorphism
    pair from: S → A and to: B → T. Every adapter is both a lens
    and a prism.

    Parameters
    ----------
    from_morph : Morphism
        The forward isomorphism S → A.
    to_morph : Morphism
        The backward isomorphism B → T.
    quantale : Quantale or None
        The enrichment algebra.
    """

    def __init__(
        self,
        from_morph: Morphism,
        to_morph: Morphism,
        quantale: Quantale | None = None,
    ) -> None:
        super().__init__(
            source=from_morph.domain,
            target=to_morph.codomain,
            focus_source=from_morph.codomain,
            focus_target=to_morph.domain,
            quantale=quantale,
        )
        self._from_morph = from_morph
        self._to_morph = to_morph

    def forward(self) -> Morphism:
        """from: S → A."""
        return self._from_morph

    def backward(self) -> Morphism:
        """to: B → T."""
        return self._to_morph

    def verify_isomorphism(self, atol: float = 1e-5) -> bool:
        """Verify that forward and backward form an isomorphism.

        Checks from >> to ≈ id_S and to >> from ≈ id_A (when S=T, A=B).

        Parameters
        ----------
        atol : float
            Absolute tolerance.

        Returns
        -------
        bool
            True if the adapter is an isomorphism.
        """
        if self._source != self._target or self._focus_source != self._focus_target:
            return False

        fwd = self._from_morph
        bwd = self._to_morph

        # fwd >> bwd ≈ id_S
        roundtrip_s = (fwd >> bwd).tensor
        id_s = identity(self._source, quantale=self._quantale).tensor

        if not torch.allclose(roundtrip_s, id_s, atol=atol):
            return False

        # bwd >> fwd ≈ id_A
        roundtrip_a = (bwd >> fwd).tensor
        id_a = identity(self._focus_source, quantale=self._quantale).tensor

        return torch.allclose(roundtrip_a, id_a, atol=atol)


class Grate(Optic):
    """A grate optic for closed/exponential structures.

    A grate from S to A through a "coindexing" object I
    encapsulates the pattern:

        cotraverse: (I → A) → S

    In the V-enriched setting, the grate is represented by
    a morphism from the internal hom [I, A] to S.

    Parameters
    ----------
    source : SetObject
        The whole S.
    focus : SetObject
        The focus A.
    index : SetObject
        The coindex I.
    cotraverse_tensor : torch.Tensor
        The tensor for (I → A) → S.
    quantale : Quantale or None
        The enrichment algebra.
    """

    def __init__(
        self,
        source: SetObject,
        focus: SetObject,
        index: SetObject,
        cotraverse_tensor: torch.Tensor,
        quantale: Quantale | None = None,
    ) -> None:
        super().__init__(
            source=source,
            target=source,
            focus_source=focus,
            focus_target=focus,
            quantale=quantale,
        )
        self._index = index
        self._cotraverse_tensor = cotraverse_tensor

    @property
    def index(self) -> SetObject:
        """The coindex object I."""
        return self._index

    def forward(self) -> Morphism:
        """Extract focus by evaluating at each index.

        Produces S → A by marginalizing over the index.
        """
        q = self._quantale
        source = self._source
        focus = self._focus_source

        # use the cotraverse tensor transposed as the getter
        # simplified: just project
        data = q.identity_tensor(source.shape)

        # if source and focus match, return identity
        if source.shape == focus.shape:
            return observed(source, focus, data, quantale=q)

        # otherwise, use marginalization
        return observed(source, focus, self._cotraverse_tensor, quantale=q)

    def backward(self) -> Morphism:
        """Rebuild whole from focus via cotraverse."""
        q = self._quantale
        focus = self._focus_target
        source = self._target

        if source.shape == focus.shape:
            data = q.identity_tensor(source.shape)
            return observed(focus, source, data, quantale=q)

        # transpose the cotraverse
        n_src = len(source.shape)
        n_foc = len(focus.shape)
        perm = list(range(n_src, n_src + n_foc)) + list(range(n_src))

        return observed(
            focus,
            source,
            self._cotraverse_tensor.permute(*perm),
            quantale=q,
        )


def compose_optics(outer: Optic, inner: Optic) -> Optic:
    """Compose two optics sequentially.

    Given outer: (S, T) → (M, N) and inner: (M, N) → (A, B),
    produces the composed optic (S, T) → (A, B).

    The composition is returned as an Adapter wrapping the composed
    forward and backward morphisms.

    Parameters
    ----------
    outer : Optic
        The outer optic (S, T) → (M, N).
    inner : Optic
        The inner optic (M, N) → (A, B).

    Returns
    -------
    Adapter
        The composed optic.
    """
    fwd = outer.forward() >> inner.forward()
    bwd = inner.backward() >> outer.backward()

    return Adapter(
        from_morph=fwd,
        to_morph=bwd,
        quantale=outer.quantale,
    )
