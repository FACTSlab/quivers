"""Adjunctions between endofunctors.

An adjunction F ⊣ G consists of functors F (left adjoint) and G
(right adjoint), together with natural transformations:

    η: Id ⇒ G ∘ F    (unit)
    ε: F ∘ G ⇒ Id    (counit)

satisfying the triangle identities:

    ε_{F(A)} ∘ F(η_A) = id_{F(A)}
    G(ε_A) ∘ η_{G(A)} = id_{G(A)}

Every adjunction induces a monad T = G ∘ F with η as unit and
μ = G(ε_F) as multiplication.

This module provides:

    Adjunction (abstract)
    └── FreeForgetfulAdjunction — Free monoid ⊣ Forgetful
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.objects import SetObject, FinSet, FreeMonoid
from quivers.core.morphisms import Morphism, identity, observed
from quivers.categorical.functors import (
    Functor,
    IdentityFunctor,
    FreeMonoidFunctor,
    IDENTITY,
)


class ForgetfulFunctor(Functor):
    """The forgetful functor from FreeMonoid-algebras to FinSet.

    On objects: A* ↦ A* (the underlying set).
    Since FreeMonoid is already a CoproductSet (hence a SetObject),
    the forgetful functor is the identity on the set level.
    On morphisms: identity on tensors.

    In the adjunction Free ⊣ Forget, "Forget" maps A* back to A*
    as a bare set (no monoid structure). Since we represent everything
    as bare sets already, this is the identity functor.
    """

    def map_object(self, obj: SetObject) -> SetObject:
        """Identity on objects."""
        return obj

    def map_morphism(self, morph: Morphism) -> Morphism:
        """Identity on morphisms."""
        from quivers.core.morphisms import FunctorMorphism

        return FunctorMorphism(self, morph, morph.domain, morph.codomain)

    def map_tensor(self, tensor: torch.Tensor, quantale: object) -> torch.Tensor:
        """Identity on tensors."""
        return tensor

    def __repr__(self) -> str:
        return "ForgetfulFunctor()"


class Adjunction(ABC):
    """Abstract adjunction F ⊣ G.

    Subclasses must implement left, right, unit_component, and
    counit_component.
    """

    @property
    @abstractmethod
    def left(self) -> Functor:
        """The left adjoint F."""
        ...

    @property
    @abstractmethod
    def right(self) -> Functor:
        """The right adjoint G."""
        ...

    @abstractmethod
    def unit_component(self, obj: SetObject) -> Morphism:
        """The unit component η_A: A → G(F(A)).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The unit morphism η_A.
        """
        ...

    @abstractmethod
    def counit_component(self, obj: SetObject) -> Morphism:
        """The counit component ε_A: F(G(A)) → A.

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The counit morphism ε_A.
        """
        ...

    def verify_triangle_left(
        self, obj: SetObject, atol: float = 1e-5
    ) -> bool:
        """Verify left triangle identity: ε_{F(A)} ∘ F(η_A) ≈ id_{F(A)}.

        Parameters
        ----------
        obj : SetObject
            The object A.
        atol : float
            Absolute tolerance.

        Returns
        -------
        bool
            True if the identity holds within tolerance.
        """
        fa = self.left.map_object(obj)
        eta_a = self.unit_component(obj)
        f_eta = self.left.map_morphism(eta_a)
        eps_fa = self.counit_component(fa)
        result = (f_eta >> eps_fa).tensor
        expected = identity(fa).tensor

        return torch.allclose(result, expected, atol=atol)

    def verify_triangle_right(
        self, obj: SetObject, atol: float = 1e-5
    ) -> bool:
        """Verify right triangle identity: G(ε_A) ∘ η_{G(A)} ≈ id_{G(A)}.

        Parameters
        ----------
        obj : SetObject
            The object A.
        atol : float
            Absolute tolerance.

        Returns
        -------
        bool
            True if the identity holds within tolerance.
        """
        ga = self.right.map_object(obj)
        eps_a = self.counit_component(obj)
        g_eps = self.right.map_morphism(eps_a)
        eta_ga = self.unit_component(ga)
        result = (eta_ga >> g_eps).tensor
        expected = identity(ga).tensor

        return torch.allclose(result, expected, atol=atol)

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"{cls}({self.left!r} ⊣ {self.right!r})"


class FreeForgetfulAdjunction(Adjunction):
    """The free-forgetful adjunction: Free ⊣ Forget.

    Free: FinSet → FinSet via FreeMonoidFunctor (A ↦ A*)
    Forget: identity at set level (A* ↦ A* as bare set)

    η_A: A → A* embeds each element as a length-1 word.
    ε_A: A* → A projects onto length-1 words (for A a FinSet).

    Parameters
    ----------
    max_length : int
        Maximum string length for the free monoid.
    """

    def __init__(self, max_length: int) -> None:
        self._max_length = max_length
        self._free = FreeMonoidFunctor(max_length)
        self._forget = ForgetfulFunctor()

    @property
    def left(self) -> Functor:
        """The free functor."""
        return self._free

    @property
    def right(self) -> Functor:
        """The forgetful functor."""
        return self._forget

    def unit_component(self, obj: SetObject) -> Morphism:
        """η_A: A → A* embeds as length-1 words.

        Parameters
        ----------
        obj : SetObject
            Must be a FinSet.

        Returns
        -------
        Morphism
            The embedding morphism.
        """
        if not isinstance(obj, FinSet):
            obj = FinSet(getattr(obj, "name", repr(obj)), obj.size)

        fm = FreeMonoid(obj, max_length=self._max_length)
        n = obj.cardinality
        data = torch.zeros(n, fm.size)
        offset = fm.offset(1)

        for a in range(n):
            data[a, offset + a] = 1.0

        return observed(obj, fm, data)

    def counit_component(self, obj: SetObject) -> Morphism:
        """ε_A: Free(Forget(A)) → A.

        For a FinSet A, this projects A* onto A by extracting the
        length-1 stratum. Length-1 words map to their element; all
        others (empty string, longer words) map to zero.

        For a FreeMonoid A* (needed by the left triangle identity),
        this is the monoid evaluation map (A*)* → A* that
        concatenates formal words of words into single words.

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The counit morphism.
        """
        if isinstance(obj, FreeMonoid):
            return self._counit_at_free_monoid(obj)

        if not isinstance(obj, FinSet):
            obj = FinSet(getattr(obj, "name", repr(obj)), obj.size)

        fm = FreeMonoid(obj, max_length=self._max_length)
        data = torch.zeros(fm.size, obj.cardinality)

        # length-1 words: project to the corresponding element
        offset = fm.offset(1)

        for a in range(obj.cardinality):
            data[offset + a, a] = 1.0

        return observed(fm, obj, data)

    def _counit_at_free_monoid(self, fm: FreeMonoid) -> Morphism:
        """ε_{A*}: (A*)* → A* via concatenation.

        Each element of (A*)* is a formal word of elements of A*.
        The counit evaluates by concatenating the component words.
        If the concatenated word exceeds max_length, the row is
        zero (the word is outside the truncated monoid).

        Parameters
        ----------
        fm : FreeMonoid
            The free monoid A* (the object at which ε is evaluated).

        Returns
        -------
        Morphism
            The concatenation/evaluation map.
        """
        m = fm.size
        proxy = FinSet(repr(fm), m)
        fmfm = FreeMonoid(proxy, max_length=self._max_length)

        data = torch.zeros(fmfm.size, fm.size)

        for idx in range(fmfm.size):
            # decode idx in (A*)* to a word of proxy indices
            word_of_indices = fmfm.decode(idx)

            # each proxy index is an element of A*; decode to
            # get the word it represents, then concatenate
            concatenated: list[int] = []

            for proxy_idx in word_of_indices:
                inner_word = fm.decode(proxy_idx)
                concatenated.extend(inner_word)

            # encode back if within max_length
            if len(concatenated) <= fm.max_length:
                target_idx = fm.encode(tuple(concatenated))
                data[idx, target_idx] = 1.0

            # else: row stays zero (exceeds truncation)

        return observed(fmfm, fm, data)
