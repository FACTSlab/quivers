"""Concrete monad instances on V-enriched FinSet.

The :class:`Monad` typeclass itself lives in
:mod:`quivers.monadic.typeclasses`; this module provides two concrete
monad instances together with the :class:`KleisliCategory` adapter.

- :class:`FuzzyPowersetMonad` — the powerset monad over a quantale,
  whose Kleisli category is the V-enriched relation category.
- :class:`FreeMonoidMonad` — the free monoid monad on a finite alphabet,
  truncated to a maximum length.
- :class:`KleisliCategory` — wraps any :class:`Monad` instance for
  composition.

Both concrete monads subclass :class:`Monad` and implement the
required ``fmap_obj`` / ``fmap`` / ``pure`` / ``join`` operations.
``apply`` is inherited from :class:`Applicative` and raises when the
internal-hom construction is not supplied per-instance.
"""

from __future__ import annotations

from quivers.categorical.functors import (
    IDENTITY,
    Functor,
    FreeMonoidFunctor,
)
from quivers.core.morphisms import Morphism, identity, observed
from quivers.core.objects import FinSet, FreeMonoid, SetObject
from quivers.core.quantales import PRODUCT_FUZZY, Quantale
from quivers.monadic.typeclasses import Monad


class FuzzyPowersetMonad(Monad):
    """The fuzzy powerset monad with a given quantale.

    At the set level, ``T(A) = A`` because fuzzy subsets are
    represented as membership-function tensors, not as elements of a
    powerset. The unit ``η_A = identity(A)`` and the multiplication
    ``μ_A = identity(A)``.

    Kleisli composition is V-enriched composition (``>>`` on morphisms).

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

    # Typeclass interface
    def fmap_obj(self, A: SetObject) -> SetObject:
        return A

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f

    def pure(self, A: SetObject) -> Morphism:
        return identity(A, quantale=self._quantale)

    def join(self, A: SetObject) -> Morphism:
        return identity(A, quantale=self._quantale)

    # Eilenberg–Moore vocabulary aliases.
    def unit(self, A: SetObject) -> Morphism:
        """``η_A : A → T(A)``; alias for :meth:`pure`."""
        return self.pure(A)

    def multiply(self, A: SetObject) -> Morphism:
        """``μ_A : T(T(A)) → T(A)``; alias for :meth:`join`."""
        return self.join(A)

    def kleisli_compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition; V-enriched composition via ``>>``."""
        return f >> g

    def __repr__(self) -> str:
        return f"FuzzyPowersetMonad({self._quantale!r})"


class FreeMonoidMonad(Monad):
    """The free monoid monad, truncated to ``max_length``.

    ``T(A) = FreeMonoid(generators=A, max_length=max_length) =
    1 + A + A² + ... + A^max_length``.

    - ``η_A : A → A*`` embeds each element as a length-1 word.
    - ``μ_A : (A*)* → A*`` flattens nested words by concatenation
      (truncated to ``max_length``).

    Parameters
    ----------
    max_length : int
        Maximum word length (inclusive). Defaults to 4.
    quantale : Quantale or None
        The enrichment algebra. Defaults to PRODUCT_FUZZY.
    """

    def __init__(self, max_length: int = 4, quantale: Quantale | None = None) -> None:
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")
        self._max_length = max_length
        self._quantale = quantale if quantale is not None else PRODUCT_FUZZY

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def quantale(self) -> Quantale:
        return self._quantale

    @property
    def endofunctor(self) -> Functor:
        return FreeMonoidFunctor(max_length=self._max_length)

    # Typeclass interface
    def fmap_obj(self, A: SetObject) -> SetObject:
        if not isinstance(A, FinSet):
            raise TypeError(
                f"FreeMonoidMonad.fmap_obj requires a FinSet, got {type(A).__name__}"
            )
        return FreeMonoid(generators=A, max_length=self._max_length)

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return self.endofunctor.map_morphism(f)

    def pure(self, A: SetObject) -> Morphism:
        """``η_A : A → A*`` — embed elements as length-1 words.

        Returns a morphism whose tensor is ``[0, I, 0, ..., 0]`` along
        the codomain's component-axis: zero on the empty word, identity
        on the length-1 component, zero elsewhere.
        """
        import torch

        if not isinstance(A, FinSet):
            raise TypeError(
                f"FreeMonoidMonad.pure requires a FinSet, got {type(A).__name__}"
            )
        ta = self.fmap_obj(A)
        n = A.cardinality
        total = ta.size  # type: ignore[attr-defined]
        # offset of the length-1 component
        start, _ = ta.component_range(1)  # type: ignore[attr-defined]
        data = torch.zeros((n, total))
        for i in range(n):
            data[i, start + i] = 1.0
        return observed(A, ta, data, quantale=self._quantale)

    def join(self, A: SetObject) -> Morphism:
        """``μ_A : (A*)* → A*`` — flatten nested words by concatenation.

        The flattened-result indexing follows the canonical word-encoding
        of :class:`FreeMonoid`; flattenings whose total length exceeds
        ``max_length`` are dropped.
        """
        import torch

        if not isinstance(A, FinSet):
            raise TypeError(
                f"FreeMonoidMonad.join requires a FinSet, got {type(A).__name__}"
            )
        ta = self.fmap_obj(A)  # FreeMonoid(A, max_length)
        # The outer free monoid is over an alphabet whose elements are
        # the indices of the inner free monoid; that alphabet is a
        # FinSet of cardinality ta.size.
        outer_alphabet = FinSet(
            name=f"{A.name}*",
            cardinality=ta.size,  # type: ignore[attr-defined]
        )
        tta = FreeMonoid(generators=outer_alphabet, max_length=self._max_length)
        outer_size = tta.size  # type: ignore[attr-defined]
        inner_size = ta.size  # type: ignore[attr-defined]
        data = torch.zeros((outer_size, inner_size))
        for k in range(outer_size):
            outer_word = tta.decode(k)  # type: ignore[attr-defined]
            # Each entry of outer_word is an index into ta; decode it
            # to obtain the inner A-word, then concatenate.
            concatenated: list[int] = []
            for inner_idx in outer_word:
                inner_word = ta.decode(inner_idx)  # type: ignore[attr-defined]
                concatenated.extend(inner_word)
            if len(concatenated) > self._max_length:
                continue
            flat = ta.encode(tuple(concatenated))  # type: ignore[attr-defined]
            data[k, flat] = 1.0
        return observed(tta, ta, data, quantale=self._quantale)

    def unit(self, A: SetObject) -> Morphism:
        """Alias for :meth:`pure`."""
        return self.pure(A)

    def multiply(self, A: SetObject) -> Morphism:
        """Alias for :meth:`join`."""
        return self.join(A)

    def __repr__(self) -> str:
        return (
            f"FreeMonoidMonad(max_length={self._max_length}, "
            f"quantale={self._quantale!r})"
        )


class KleisliCategory:
    """The Kleisli category of a monad.

    Objects are the same as the base category. Morphisms ``A → B`` in
    the Kleisli category are morphisms ``A → T(B)`` in the base
    category. Composition uses the underlying monad's
    :meth:`Monad.join`, falling through to a closed-form
    ``kleisli_compose`` method on the monad when available.

    Parameters
    ----------
    monad : Monad
        The underlying monad instance.
    """

    def __init__(self, monad: Monad) -> None:
        self._monad = monad

    @property
    def monad(self) -> Monad:
        return self._monad

    def identity(self, obj: SetObject) -> Morphism:
        """Kleisli identity: ``η_A : A → T(A)``."""
        return self._monad.pure(obj)

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        """Kleisli composition.

        For ``f : A → T(B)`` and ``g : B → T(C)``, returns
        ``A → T(C)`` via the monad's bind / join construction.
        """
        # Defer to a closed-form kleisli_compose when the monad ships
        # one; otherwise the generic join-based construction requires
        # an internal-hom representation supplied per instance.
        kc = getattr(self._monad, "kleisli_compose", None)
        if callable(kc):
            return kc(f, g)
        raise NotImplementedError(
            f"KleisliCategory.compose: {type(self._monad).__name__} "
            "exposes no closed-form kleisli composition; supply one "
            "via the bridges in quivers.monadic.bridges"
        )


__all__ = [
    "FuzzyPowersetMonad",
    "FreeMonoidMonad",
    "KleisliCategory",
]
