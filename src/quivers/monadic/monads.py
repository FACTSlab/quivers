"""Monads and Kleisli categories on V-enriched FinSet.

A monad (T, η, μ) consists of an endofunctor T, a unit η: Id ⇒ T,
and a multiplication μ: T² ⇒ T satisfying:

    μ ∘ T(μ) = μ ∘ μ_T    (associativity)
    μ ∘ η_T = id           (left unit)
    μ ∘ T(η) = id          (right unit)

The Kleisli category of a monad has the same objects but morphisms
A → B are morphisms A → T(B) in the base category, composed via:

    f >=> g = μ_C ∘ T(g) ∘ f

This module provides:

    Monad (abstract)
    ├── FuzzyPowersetMonad — Kleisli category = FuzzyRel
    └── FreeMonoidMonad    — Kleisli category = string-valued relations

    KleisliCategory — wraps a monad for composition
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.objects import SetObject, FinSet, FreeMonoid
from quivers.core.quantales import PRODUCT_FUZZY, Quantale
from quivers.core.morphisms import (
    Morphism,
    identity,
    observed,
)
from quivers.categorical.functors import (
    Functor,
    FreeMonoidFunctor,
    IDENTITY,
)


class Monad(ABC):
    """Abstract monad (T, η, μ) on V-enriched FinSet.

    Subclasses must implement endofunctor, unit, and multiply.
    Kleisli composition is derived.
    """

    @property
    @abstractmethod
    def endofunctor(self) -> Functor:
        """The underlying endofunctor T."""
        ...

    @abstractmethod
    def unit(self, obj: SetObject) -> Morphism:
        """The unit component η_A: A → T(A).

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
    def multiply(self, obj: SetObject) -> Morphism:
        """The multiplication component μ_A: T(T(A)) → T(A).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The multiplication morphism μ_A.
        """
        ...

    def kleisli_compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition: f >=> g = μ_C ∘ T(g) ∘ f.

        For f: A → T(B) and g: B → T(C), produces A → T(C).

        Parameters
        ----------
        f : Morphism
            Left Kleisli morphism A → T(B).
        g : Morphism
            Right Kleisli morphism B → T(C).

        Returns
        -------
        Morphism
            Composed Kleisli morphism A → T(C).
        """
        # T(g): T(B) → T(T(C))
        tg = self.endofunctor.map_morphism(g)

        # f >> T(g): A → T(T(C))
        f_then_tg = f >> tg

        # μ_C: T(T(C)) → T(C)
        # need to extract C from g's codomain T(C)
        # for FuzzyPowerset: T = Id so C = g.codomain
        # for FreeMonoid: g.codomain = FreeMonoid(C), C = g.codomain.generators
        mu = self._multiply_at_codomain(g)

        return f_then_tg >> mu

    def _multiply_at_codomain(self, g: Morphism) -> Morphism:
        """Get μ for the appropriate object given g's codomain structure.

        Subclasses can override for efficiency.
        """
        # default: extract the base object from g's codomain
        # T(C) = g.codomain, so we need the C such that T(C) = g.codomain
        # this is functor-specific
        raise NotImplementedError(
            "Subclasses must implement _multiply_at_codomain or override "
            "kleisli_compose"
        )


class KleisliCategory:
    """The Kleisli category of a monad.

    Objects are the same as the base category. Morphisms A → B
    in the Kleisli category are morphisms A → T(B) in the base
    category.

    Parameters
    ----------
    monad : Monad
        The underlying monad.
    """

    def __init__(self, monad: Monad) -> None:
        self._monad = monad

    @property
    def monad(self) -> Monad:
        """The underlying monad."""
        return self._monad

    def identity(self, obj: SetObject) -> Morphism:
        """Kleisli identity at object A: η_A: A → T(A).

        Parameters
        ----------
        obj : SetObject
            The object A.

        Returns
        -------
        Morphism
            The Kleisli identity morphism.
        """
        return self._monad.unit(obj)

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition: f >=> g.

        Parameters
        ----------
        f : Morphism
            Left morphism A → T(B).
        g : Morphism
            Right morphism B → T(C).

        Returns
        -------
        Morphism
            Composed morphism A → T(C).
        """
        return self._monad.kleisli_compose(f, g)


class FuzzyPowersetMonad(Monad):
    """The fuzzy powerset monad with a given quantale.

    At the set level, T(A) = A because fuzzy subsets are represented
    as membership function tensors, not as elements of a powerset.
    The unit η_A = identity(A) and the multiplication μ_A = identity(A).

    The Kleisli composition f >=> g reduces to quantale.compose(f, g),
    which is exactly the >> operator on morphisms.

    Parameters
    ----------
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(self, quantale: Quantale | None = None) -> None:
        self._quantale = quantale if quantale is not None else PRODUCT_FUZZY

    @property
    def quantale(self) -> Quantale:
        """The enrichment algebra."""
        return self._quantale

    @property
    def endofunctor(self) -> Functor:
        """T = Id (identity functor at set level)."""
        return IDENTITY

    def unit(self, obj: SetObject) -> Morphism:
        """η_A = identity(A): Kronecker delta."""
        return identity(obj, quantale=self._quantale)

    def multiply(self, obj: SetObject) -> Morphism:
        """μ_A = identity(A): union of fuzzy sets."""
        return identity(obj, quantale=self._quantale)

    def kleisli_compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition = V-enriched composition via >>."""
        # since T = Id, f >=> g = μ ∘ T(g) ∘ f = id ∘ g ∘ f = f >> g
        return f >> g

    def _multiply_at_codomain(self, g: Morphism) -> Morphism:
        return identity(g.codomain, quantale=self._quantale)

    def __repr__(self) -> str:
        return f"FuzzyPowersetMonad({self._quantale!r})"


class FreeMonoidMonad(Monad):
    """The free monoid monad, truncated to max_length.

    T(A) = FreeMonoid(A, max_length) = 1 + A + A² + ... + A^max_length.
    η_A: A → A* embeds each element as a length-1 word.
    μ_A: (A*)* → A* flattens nested words by concatenation
    (truncated to max_length).

    Parameters
    ----------
    max_length : int
        Maximum string length for the truncated free monoid.
    """

    def __init__(self, max_length: int) -> None:
        self._max_length = max_length
        self._functor = FreeMonoidFunctor(max_length)

    @property
    def max_length(self) -> int:
        """Maximum string length."""
        return self._max_length

    @property
    def endofunctor(self) -> Functor:
        """T = FreeMonoidFunctor(max_length)."""
        return self._functor

    def unit(self, obj: SetObject) -> Morphism:
        """η_A: A → A* embeds each element as a length-1 word.

        The tensor has shape (|A|, |A*|) with 1 at positions
        (a, offset_1 + a) and 0 elsewhere.
        """
        if not isinstance(obj, FinSet):
            raise TypeError(
                f"FreeMonoidMonad.unit requires FinSet, got {type(obj).__name__}"
            )

        fm = FreeMonoid(obj, max_length=self._max_length)
        n = obj.cardinality
        data = torch.zeros(n, fm.size)

        # length-1 stratum starts at offset(1)
        offset = fm.offset(1)

        for a in range(n):
            data[a, offset + a] = 1.0

        return observed(obj, fm, data)

    def multiply(self, obj: SetObject) -> Morphism:
        """μ_A: (A*)* → A* flattens nested words by concatenation.

        A word in (A*)* at stratum k is a k-tuple of words in A*.
        Flattening concatenates them. If the total length exceeds
        max_length, the entry is 0 (truncation).
        """
        if not isinstance(obj, FinSet):
            raise TypeError(
                f"FreeMonoidMonad.multiply requires FinSet, got {type(obj).__name__}"
            )

        fm_a = FreeMonoid(obj, max_length=self._max_length)
        fm_fm_a = FreeMonoid(
            FinSet(f"{obj.name}*", fm_a.size),
            max_length=self._max_length,
        )

        data = torch.zeros(fm_fm_a.size, fm_a.size)

        # for each element in (A*)*: decode as a word of A*-indices,
        # decode each A*-index as a word of A-indices, concatenate,
        # encode the result in A*
        for i in range(fm_fm_a.size):
            # decode i as a word of fm_a-indices
            outer_word = fm_fm_a.decode(i)

            # decode each inner index
            inner_words: list[tuple[int, ...]] = []

            for idx in outer_word:
                inner_words.append(fm_a.decode(idx))

            # concatenate
            flat: tuple[int, ...] = ()

            for w in inner_words:
                flat = flat + w

            # encode if within length bound
            if len(flat) <= self._max_length:
                j = fm_a.encode(flat)
                data[i, j] = 1.0

        return observed(fm_fm_a, fm_a, data)

    def kleisli_compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition: f >=> g = μ_C ∘ T(g) ∘ f."""
        tg = self._functor.map_morphism(g)
        f_then_tg = f >> tg
        mu = self._multiply_at_codomain(g)
        return f_then_tg >> mu

    def _multiply_at_codomain(self, g: Morphism) -> Morphism:
        """Extract the base object C from g: B → T(C) = FreeMonoid(C)."""
        cod = g.codomain

        if isinstance(cod, FreeMonoid):
            return self.multiply(cod.generators)

        # if codomain is already a FinSet, assume T(C) = C (length 0)
        if isinstance(cod, FinSet):
            return self.multiply(cod)

        raise TypeError(f"Cannot extract base object from codomain {cod!r}")

    def __repr__(self) -> str:
        return f"FreeMonoidMonad(max_length={self._max_length})"
