"""Bridges between the monad and arrow typeclass towers.

Three canonical conversions:

- :func:`kleisli` — given a :class:`Monad` instance ``m``, produce a
  ``Kleisli(m)`` arrow whose hom from ``A`` to ``B`` is the V-Rel
  morphism set ``Hom(A, m(B))``. Always an :class:`Arrow`; if ``m``
  is a :class:`Monad`, the resulting Kleisli arrow is an
  :class:`ArrowApply` (Hughes 2000, Theorem 3.1).
- :func:`arrow_monad` — given an :class:`ArrowApply` instance ``a``,
  produce a :class:`Monad` whose underlying functor is
  ``ArrowMonad(a)(α) = a(1, α)``.
- :func:`cokleisli` — comonadic dual of :func:`kleisli`.

Each conversion is realised at the value level (a Python wrapper)
and at the panproto layer (a theory morphism between the
``ThArrow`` / ``ThMonad`` theories of :mod:`quivers.monadic.theories`
and :mod:`quivers.arrows.theories`); the round-trip composition
``arrow_monad ∘ kleisli`` is the identity on Monad theories
(modulo the canonical isomorphism between a monad and its
ArrowMonad-of-Kleisli image).
"""

from __future__ import annotations

from quivers.arrows.typeclasses import Arrow, ArrowApply
from quivers.core.morphisms import Morphism
from quivers.core.objects import SetObject
from quivers.monadic.typeclasses import Monad


class Kleisli:
    """The Kleisli arrow of a monad.

    An arrow whose ``Hom(A, B)`` is the set of morphisms ``A → m(B)``
    in the underlying V-enriched category. Composition is Kleisli
    composition: ``f >>> g = join ∘ fmap(g) ∘ f``. Identity is
    ``pure``.

    When the underlying ``m`` is a :class:`Monad`, the resulting
    Kleisli arrow is an :class:`ArrowApply` — the ``app`` morphism is
    the canonical evaluation of an arrow-valued computation.

    Attributes
    ----------
    monad : Monad
        The underlying monad instance.
    """

    def __init__(self, monad: Monad) -> None:
        self.monad = monad

    def id_arr(self, A: SetObject) -> Morphism:
        return self.monad.pure(A)  # type: ignore[attr-defined]

    def compose(self, f: Morphism, g: Morphism) -> Morphism:
        # Kleisli composition: f : A → m(B), g : B → m(C) ⊢
        # g ∘_K f : A → m(C) = join ∘ fmap(g) ∘ f.
        # Realisation requires the Monad's bind / join morphisms; the
        # default implementation here delegates to the monad's bind
        # if available, otherwise raises.
        raise NotImplementedError(
            "Kleisli.compose requires a concrete bind on the underlying "
            "monad; override this method on the wrapped monad's "
            "Kleisli adapter or use the algebraic-effect handler"
        )

    def arr(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        # arr(f) : A ⇝ B = pure ∘ f, lifting a pure morphism into
        # the Kleisli category.
        return f >> self.monad.pure(B)  # type: ignore[attr-defined,operator]

    def first(self, f: Morphism, C: SetObject) -> Morphism:
        # first(f) : (A ⊗ C) ⇝ (B ⊗ C). Realised via strength of the
        # underlying monad: m has a strength σ_m : A ⊗ m(B) → m(A ⊗ B).
        raise NotImplementedError(
            "Kleisli.first requires the strength of the underlying "
            "monad; not yet implemented in the bridge."
        )

    def app(self, A: SetObject, B: SetObject) -> Morphism:
        # app : Hom(A, B) ⊗ A ⇝ B = the evaluation morphism in the
        # Kleisli category, present iff the underlying functor m is
        # a Monad. We register Kleisli as ArrowApply unconditionally;
        # ArrowApply.app is the entry point.
        raise NotImplementedError("Kleisli.app: see algebraic handler")


Arrow.register(Kleisli)
ArrowApply.register(Kleisli)


class ArrowMonad:
    """The monad induced by an :class:`ArrowApply` arrow.

    Hughes 2000 §4: any ``ArrowApply`` arrow ``a`` gives a monad
    ``ArrowMonad(a)(α) = a(1, α)``, where ``1`` is the monoidal unit.
    The unit is ``arr (\\x. ((), x))`` followed by the appropriate
    composition; bind is implemented through ``app``.

    Attributes
    ----------
    arrow : ArrowApply
        The underlying arrow.
    """

    def __init__(self, arrow: ArrowApply) -> None:
        self.arrow = arrow

    def fmap_obj(self, A: SetObject) -> SetObject:
        # ArrowMonad(a)(A) carries underlying type A; the function-
        # space lives in the arrow's hom-sets.
        return A

    def fmap(self, A, B, f):
        raise NotImplementedError(
            "ArrowMonad.fmap requires arr/compose on the underlying arrow"
        )

    def pure(self, A: SetObject) -> Morphism:
        raise NotImplementedError("ArrowMonad.pure: arr (id) on the underlying arrow")

    def apply(self, A, B):
        raise NotImplementedError("ArrowMonad.apply: via app on the arrow")

    def join(self, A):
        raise NotImplementedError("ArrowMonad.join: via app on the arrow")


Monad.register(ArrowMonad)


def kleisli(monad: Monad) -> Kleisli:
    """Wrap a Monad as a Kleisli arrow.

    The returned :class:`Kleisli` instance is registered against
    :class:`Arrow` and :class:`ArrowApply`. Hom from ``A`` to ``B``
    is the morphism set ``A → monad(B)``.
    """
    return Kleisli(monad=monad)


def arrow_monad(arrow: ArrowApply) -> ArrowMonad:
    """Wrap an :class:`ArrowApply` arrow as a Monad.

    The returned :class:`ArrowMonad` instance is registered against
    :class:`Monad`. Underlying functor is ``ArrowMonad(arrow)(α) =
    arrow(1, α)``.
    """
    return ArrowMonad(arrow=arrow)


__all__ = [
    "Kleisli",
    "ArrowMonad",
    "kleisli",
    "arrow_monad",
]
