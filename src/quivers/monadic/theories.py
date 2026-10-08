"""Theories mirroring the typeclass hierarchy.

For each typeclass in [`quivers.monadic.typeclasses`][quivers.monadic.typeclasses], this module
declares a corresponding `TypeclassTheory`, a record of the sorts,
operations, and laws the typeclass requires:

- `ThFunctor`: sorts ``Carrier`` and ``Hom``, the operation ``fmap``,
  and the two functor laws as equations.
- `ThApplicative`: extends `ThFunctor` with ``pure`` and ``apply``
  and the four applicative laws.
- `ThMonad`: extends `ThApplicative` with ``join`` (equivalently
  ``bind``) and the three monad laws.
- `ThAlternative`, `ThMonadPlus`, `ThMonadTrans`, `ThFoldable`, and
  `ThTraversable`: likewise.

Class extension is recorded by name in each theory's ``extends``
field, mirroring theory inclusion via a panproto colimit.

The arrow tower in [`quivers.arrows.theories`][quivers.arrows.theories] mirrors this
construction for the Hughes-style arrow typeclasses.
"""

from __future__ import annotations

import didactic.api as dx


class TypeclassTheory(dx.Model):
    """A record of the sorts, operations, and laws of one typeclass.

    Attributes
    ----------
    name : str
        The theory's identifier (e.g. ``"ThFunctor"``).
    sorts : tuple of str
        The sorts the theory introduces.
    operations : tuple of str
        The operations the theory introduces, each written as a
        signature such as ``"fmap : Hom → Hom"``.
    equations : tuple of str
        The equations registered with the theory; each is a free-form
        statement of a typeclass law.
    extends : tuple of str
        Names of the theories this one extends.
    """

    name: str
    sorts: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    equations: tuple[str, ...] = ()
    extends: tuple[str, ...] = ()


ThFunctor = TypeclassTheory(
    name="ThFunctor",
    sorts=("Carrier", "Hom"),
    operations=("fmap_obj : Carrier → Carrier", "fmap : Hom → Hom"),
    equations=(
        "fmap(id) = id",
        "fmap(g ∘ f) = fmap(g) ∘ fmap(f)",
    ),
)
"""The theory of `Functor`."""

ThApplicative = TypeclassTheory(
    name="ThApplicative",
    operations=(
        "pure : Carrier → F(Carrier)",
        "apply : F(Hom) ⊗ F(Carrier) → F(Carrier)",
    ),
    equations=(
        "apply(pure(id), v) = v",
        "apply(pure(f), pure(x)) = pure(f x)",
        "apply(u, pure(y)) = apply(pure(λf. f y), u)",
        "apply(apply(apply(pure(∘), u), v), w) = apply(u, apply(v, w))",
    ),
    extends=("ThFunctor",),
)
"""The theory of `Applicative`."""

ThMonad = TypeclassTheory(
    name="ThMonad",
    operations=("join : F(F(Carrier)) → F(Carrier)",),
    equations=(
        "join ∘ pure = id",
        "join ∘ fmap(pure) = id",
        "join ∘ fmap(join) = join ∘ join",
    ),
    extends=("ThApplicative",),
)
"""The theory of `Monad`."""

ThAlternative = TypeclassTheory(
    name="ThAlternative",
    operations=("empty : 1 → F(Carrier)", "alt : F(Carrier) ⊗ F(Carrier) → F(Carrier)"),
    equations=(
        "alt(empty, x) = x",
        "alt(x, empty) = x",
        "alt(alt(x, y), z) = alt(x, alt(y, z))",
    ),
    extends=("ThApplicative",),
)
"""The theory of `Alternative`."""

ThMonadPlus = TypeclassTheory(
    name="ThMonadPlus",
    equations=("bind(empty, k) = empty",),
    extends=("ThMonad", "ThAlternative"),
)
"""The theory of `MonadPlus`."""

ThMonadTrans = TypeclassTheory(
    name="ThMonadTrans",
    operations=("lift : m(Carrier) → t(m)(Carrier)",),
    equations=(
        "lift ∘ pure_m = pure_{T(m)}",
        "lift(bind_m(x, k)) = bind_{T(m)}(lift(x), lift ∘ k)",
    ),
)
"""The theory of `MonadTrans`."""

ThFoldable = TypeclassTheory(
    name="ThFoldable",
    operations=("foldr : (Carrier ⊗ B → B) ⊗ B ⊗ F(Carrier) → B",),
)
"""The theory of `Foldable`."""

ThTraversable = TypeclassTheory(
    name="ThTraversable",
    operations=("traverse : (A → G(B)) ⊗ F(A) → G(F(B))",),
    equations=(
        "naturality: t ∘ traverse(f) = traverse(t ∘ f)",
        "identity: traverse(pure_Identity) = pure_Identity",
        "composition: traverse(Compose ∘ f) = Compose ∘ fmap(traverse(g)) ∘ traverse(f)",
    ),
    extends=("ThFunctor", "ThFoldable"),
)
"""The theory of `Traversable`."""


__all__ = [
    "TypeclassTheory",
    "ThFunctor",
    "ThApplicative",
    "ThMonad",
    "ThAlternative",
    "ThMonadPlus",
    "ThMonadTrans",
    "ThFoldable",
    "ThTraversable",
]
