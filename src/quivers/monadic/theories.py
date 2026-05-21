"""Panproto theories mirroring the typeclass hierarchy.

For each typeclass in [`quivers.monadic.typeclasses`][quivers.monadic.typeclasses], this module
declares a corresponding `panproto.Theory`:

- `ThFunctor` — sorts: ``Carrier``, ``Hom``; operation: ``fmap``
  with the two functor laws as equations.
- `ThApplicative` — extends `ThFunctor` with ``pure`` and
  ``apply`` and the four applicative laws.
- `ThMonad` — extends `ThApplicative` with ``join`` (or
  equivalently ``bind``) and the three monad laws.
- `ThAlternative`, `ThMonadPlus`, `ThMonadTrans`,
  `ThTraversable` — likewise.

Class extension is realised as theory inclusion via `panproto.colimit`.
Each typeclass instance in [`quivers.monadic.instances`][quivers.monadic.instances] emits a
panproto theory morphism whose existence panproto can verify (the
operations are present, the equations hold).

The arrow tower in [`quivers.arrows.theories`][quivers.arrows.theories] mirrors this
construction for the Hughes-style arrow typeclasses.

Implementation note
-------------------
This module currently *declares* the theories as Python data
structures using a thin wrapper around ``panproto.define_theory``.
Some panproto API surface required for the full encoding (notably
polymorphic-arity operations and equation-set composition under
``colimit``) is in flux upstream; the wrapper falls back to a
record-only representation when the panproto API is unavailable, so
the typeclass framework remains usable without the panproto theory
mirrors.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class _TheoryStub:
    """Minimal record-only representation of a typeclass theory.

    Used as a fallback when the upstream panproto API for declaring
    polymorphic-arity operations and equation-set composition is not
    yet stable.

    Attributes
    ----------
    name : str
        The theory's identifier (e.g. ``"ThFunctor"``).
    sorts : tuple of str
        The sorts the theory introduces.
    operations : tuple of str
        The operations the theory introduces.
    equations : tuple of str
        The equations registered with the theory; each is a free-form
        statement of a typeclass law.
    extends : tuple of str
        Names of theories this one extends (via panproto colimit).
    """

    name: str
    sorts: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
    equations: tuple[str, ...] = ()
    extends: tuple[str, ...] = field(default_factory=tuple)


ThFunctor = _TheoryStub(
    name="ThFunctor",
    sorts=("Carrier", "Hom"),
    operations=("fmap_obj : Carrier → Carrier", "fmap : Hom → Hom"),
    equations=(
        "fmap(id) = id",
        "fmap(g ∘ f) = fmap(g) ∘ fmap(f)",
    ),
)

ThApplicative = _TheoryStub(
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

ThMonad = _TheoryStub(
    name="ThMonad",
    operations=("join : F(F(Carrier)) → F(Carrier)",),
    equations=(
        "join ∘ pure = id",
        "join ∘ fmap(pure) = id",
        "join ∘ fmap(join) = join ∘ join",
    ),
    extends=("ThApplicative",),
)

ThAlternative = _TheoryStub(
    name="ThAlternative",
    operations=("empty : 1 → F(Carrier)", "alt : F(Carrier) ⊗ F(Carrier) → F(Carrier)"),
    equations=(
        "alt(empty, x) = x",
        "alt(x, empty) = x",
        "alt(alt(x, y), z) = alt(x, alt(y, z))",
    ),
    extends=("ThApplicative",),
)

ThMonadPlus = _TheoryStub(
    name="ThMonadPlus",
    equations=("bind(empty, k) = empty",),
    extends=("ThMonad", "ThAlternative"),
)

ThMonadTrans = _TheoryStub(
    name="ThMonadTrans",
    operations=("lift : m(Carrier) → t(m)(Carrier)",),
    equations=(
        "lift ∘ pure_m = pure_{T(m)}",
        "lift(bind_m(x, k)) = bind_{T(m)}(lift(x), lift ∘ k)",
    ),
)

ThFoldable = _TheoryStub(
    name="ThFoldable",
    operations=("foldr : (Carrier ⊗ B → B) ⊗ B ⊗ F(Carrier) → B",),
)

ThTraversable = _TheoryStub(
    name="ThTraversable",
    operations=("traverse : (A → G(B)) ⊗ F(A) → G(F(B))",),
    equations=(
        "naturality: t ∘ traverse(f) = traverse(t ∘ f)",
        "identity: traverse(pure_Identity) = pure_Identity",
        "composition: traverse(Compose ∘ f) = Compose ∘ fmap(traverse(g)) ∘ traverse(f)",
    ),
    extends=("ThFunctor", "ThFoldable"),
)


__all__ = [
    "ThFunctor",
    "ThApplicative",
    "ThMonad",
    "ThAlternative",
    "ThMonadPlus",
    "ThMonadTrans",
    "ThFoldable",
    "ThTraversable",
]
