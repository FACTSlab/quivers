"""Haskell-style typeclass hierarchy for compositional effects.

This module defines the abstract interfaces that effect instances
implement. Each typeclass declares a set of operations that an
instance must provide; default derivations are supplied where
Haskell admits them. The class-extension lattice is:

    Functor
    ├── Applicative
    │   ├── Monad
    │   └── Alternative
    │       └── MonadPlus  (also extends Monad)
    └── Traversable      (also extends Foldable)

    Foldable

    MonadTrans   — orthogonal: lifts an inner monad through a stacked
                   transformer; not a sub-class of any of the above.

The :func:`quivers.stochastic.effect_lifts.class_directed_lifts`
schema-lifting machinery dispatches on which classes an effect
inhabits, never on the effect's identity, so user-defined effects
extend the framework automatically.

Each typeclass is also mirrored by a :mod:`panproto.Theory` in
:mod:`quivers.monadic.theories`; class-extension is realised there as
theory inclusion via :func:`panproto.colimit`, making every instance
a verifiable theory morphism.

References
----------
- Bumford, D. and Charlow, S. (2026). *Effect-Driven Interpretation:
  Functors for Natural Language Composition*. Cambridge Elements in
  Semantics. Cambridge University Press. Online ISBN 9781009285377;
  preprint arXiv:2504.00316.
- Hughes, J. (2000). *Generalising monads to arrows*. Science of
  Computer Programming, 37(1–3), 67–111.
  doi:10.1016/S0167-6423(99)00023-4.
- Bauer, A. and Pretnar, M. (2015). *Programming with algebraic
  effects and handlers*. Journal of Logical and Algebraic Methods in
  Programming, 84(1), 108–123. doi:10.1016/j.jlamp.2014.02.001.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from quivers.core.morphisms import Morphism
from quivers.core.objects import SetObject


class Functor(ABC):
    """Functor: an endofunctor on (V-enriched) finite sets.

    A :class:`Functor` instance carries:

    - :meth:`fmap_obj` — the type-level action ``F : SetObject → SetObject``.
    - :meth:`fmap` — the morphism-level action sending
      ``f : A → B`` to ``F(f) : F(A) → F(B)``.

    Functor laws:

    - identity:    ``F(id_A) = id_{F(A)}``
    - composition: ``F(g ∘ f) = F(g) ∘ F(f)``

    Both laws are documented and runtime-checkable in
    :mod:`quivers.monadic.laws`.
    """

    @abstractmethod
    def fmap_obj(self, A: SetObject) -> SetObject:
        """The type-level action ``F(A)``."""

    @abstractmethod
    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        """The morphism-level action ``F(f) : F(A) → F(B)``."""


class Applicative(Functor, ABC):
    """Applicative functor: pure + apply.

    Adds:

    - :meth:`pure` — the unit ``η_A : A → F(A)``.
    - :meth:`apply` — the application combinator
      ``F(A → B) ⊗ F(A) → F(B)``.

    Applicative laws (Hughes 2000 §3.1, after McBride and Paterson):

    - identity:       ``apply(pure(id), v) = v``
    - homomorphism:   ``apply(pure(f), pure(x)) = pure(f x)``
    - interchange:    ``apply(u, pure(y)) = apply(pure(λf. f y), u)``
    - composition:    ``apply(apply(apply(pure(∘), u), v), w) =
                       apply(u, apply(v, w))``

    The default :meth:`fmap` derivation
    ``fmap(f) = apply(pure(f), -)`` is provided.
    """

    @abstractmethod
    def pure(self, A: SetObject) -> Morphism:
        """``η_A : A → F(A)``."""

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        """``F(A → B) ⊗ F(A) → F(B)``.

        Realised in V-Rel as a parameterised binary operation; the
        concrete construction depends on the instance and on the
        internal-hom representation of the underlying enriched
        category. The default implementation raises
        :exc:`NotImplementedError`; concrete instances override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.apply requires an internal-hom "
            "construction; override on the concrete instance"
        )

    def lift_a2(
        self, A: SetObject, B: SetObject, C: SetObject, f: Morphism
    ) -> Morphism:
        """Apply a binary function under the Applicative.

        Default derivation: ``liftA2 f x y = apply(apply(pure f, x), y)``.
        Concrete instances may override for efficiency.
        """
        raise NotImplementedError(
            "lift_a2 default derivation requires a panproto-side "
            "currying construction; override on the concrete instance"
        )


class Monad(Applicative, ABC):
    """Monad: pure + join (or bind).

    Adds:

    - :meth:`join` — the multiplication ``μ_A : F(F(A)) → F(A)``.

    The :meth:`bind` operation is derived via ``bind(m, k) = join(fmap(k)(m))``.

    Monad laws:

    - left unit:     ``bind(pure(a), k) = k(a)``
    - right unit:    ``bind(m, pure) = m``
    - associativity: ``bind(bind(m, k), h) = bind(m, λx. bind(k(x), h))``

    These are equivalent to:

    - ``join ∘ pure_F = id``                       (left unit)
    - ``join ∘ F(pure) = id``                      (right unit)
    - ``join ∘ F(join) = join ∘ join``             (associativity)
    """

    @abstractmethod
    def join(self, A: SetObject) -> Morphism:
        """``μ_A : F(F(A)) → F(A)``."""

    def bind(self, A: SetObject, B: SetObject, k: Morphism) -> Morphism:
        """Monadic bind: ``F(A) ⊗ (A → F(B)) → F(B)``.

        Default derivation: ``bind(m, k) = join_B(fmap(k)(m))``. Override
        on the concrete instance for a more direct realisation.
        """
        raise NotImplementedError(
            "bind default derivation requires panproto-side composition; "
            "override on the concrete instance"
        )


class Alternative(Applicative, ABC):
    """Alternative: an Applicative with empty + alt.

    Adds:

    - :meth:`empty` — the bottom element ``⊥_A : 1 → F(A)``.
    - :meth:`alt` — pointwise alternative ``F(A) ⊗ F(A) → F(A)``.

    Alternative laws:

    - identity:    ``alt(empty, x) = x = alt(x, empty)``
    - associativity: ``alt(alt(x, y), z) = alt(x, alt(y, z))``
    - left zero (when also a Monad): ``bind(empty, k) = empty``
    """

    @abstractmethod
    def empty(self, A: SetObject) -> Morphism:
        """``⊥_A : 1 → F(A)``."""

    @abstractmethod
    def alt(self, A: SetObject) -> Morphism:
        """``F(A) ⊗ F(A) → F(A)``."""


class MonadPlus(Monad, Alternative, ABC):
    """Monad + Alternative: a Monad whose ``F`` is also Alternative.

    No new operations; the diamond inheritance picks up both
    :meth:`join` (from Monad) and :meth:`empty` / :meth:`alt`
    (from Alternative). Adds the *left-zero* law:
    ``bind(empty, k) = empty`` (already documented under Alternative).
    """


class Foldable(ABC):
    """Foldable: support a catamorphism-like fold.

    Used by Hamblin-style alternative semantics to collapse a
    structure of alternatives down to a single value. The minimal
    interface is :meth:`foldr`; richer combinators are derivable.
    """

    @abstractmethod
    def foldr(self, A: SetObject, B: SetObject) -> Morphism:
        """``(A ⊗ B → B) ⊗ B ⊗ F(A) → B``."""


class Traversable(Functor, Foldable, ABC):
    """Traversable: distribute an Applicative action through a structure.

    Adds :meth:`traverse`: given an Applicative ``G`` and a morphism
    ``f : A → G(B)``, produces ``F(A) → G(F(B))``. This is Charlow's
    central tool for distributing scope through alternatives.

    Traversable laws (after McBride and Paterson 2008):

    - naturality:   ``t ∘ traverse(f) = traverse(t ∘ f)``
                    for any applicative natural transformation t.
    - identity:     ``traverse(pure_Identity) = pure_Identity``
    - composition:  ``traverse(Compose ∘ f) =
                     Compose ∘ fmap(traverse(g)) ∘ traverse(f)``
    """

    @abstractmethod
    def traverse(
        self, A: SetObject, B: SetObject, applicative: Applicative, f: Morphism
    ) -> Morphism:
        """``F(A) → G(F(B))`` for an Applicative ``G`` and ``f : A → G(B)``."""


class MonadTrans(ABC):
    """Monad transformer: lift an inner monad to a stacked monad.

    A :class:`MonadTrans` instance ``T`` provides :meth:`lift`,
    embedding an inner-monad Kleisli arrow into the transformer.
    The class is orthogonal to the Functor/Applicative/Monad tower:
    transformers are themselves monads (when applied to a base monad),
    but the *transformer* interface is the lift, not the bind.

    Lift laws:

    - ``lift ∘ pure_m = pure_{T(m)}``
    - ``lift(bind_m(x, k)) = bind_{T(m)}(lift(x), lift ∘ k)``
    """

    @abstractmethod
    def lift(self, m: Monad, A: SetObject) -> Morphism:
        """``m(A) → T(m)(A)``."""


__all__ = [
    "Functor",
    "Applicative",
    "Monad",
    "Alternative",
    "MonadPlus",
    "Foldable",
    "Traversable",
    "MonadTrans",
]
