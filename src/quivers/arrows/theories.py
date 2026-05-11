"""Panproto theories mirroring the arrow typeclass tower.

Each Hughes-style arrow typeclass corresponds to a panproto theory
that declares the operations and laws the typeclass requires:

- :data:`ThCategory_` — composition + identity, with the three
  category laws.
- :data:`ThArrow` — adds ``arr`` and ``first``; the seven Hughes
  arrow laws.
- :data:`ThArrowChoice`, :data:`ThArrowApply`, :data:`ThArrowLoop`,
  :data:`ThArrowZero`, :data:`ThArrowPlus` — the additional
  operations and laws of each extension.

The bridges in :mod:`quivers.monadic.bridges` correspond to panproto
theory morphisms ``ThMonad → ThArrowApply`` (Kleisli construction)
and back (ArrowMonad construction); the round-trip composition is
the identity on the appropriate theory image.
"""

from __future__ import annotations

from quivers.monadic.theories import _TheoryStub


ThCategory_ = _TheoryStub(
    name="ThCategory_",
    sorts=("Object", "Arr"),
    operations=(
        "id_arr : Object → Arr",
        "compose : Arr ⊗ Arr → Arr",
    ),
    equations=(
        "compose(id, f) = f",
        "compose(f, id) = f",
        "compose(compose(f, g), h) = compose(f, compose(g, h))",
    ),
)

ThArrow = _TheoryStub(
    name="ThArrow",
    operations=(
        "arr : (Object → Object) → Arr",
        "first : Arr → Arr",
    ),
    equations=(
        "arr(id) = id_arr",
        "arr(g ∘ f) = compose(arr(f), arr(g))",
        "first(arr(f)) = arr(f × id)",
        "compose(first(f), arr(id × g)) = compose(arr(id × g), first(f))",
        "compose(first(f), arr(fst)) = compose(arr(fst), f)",
        "first(first(f)) ≡ first(f) (assoc)",
        "first(compose(f, g)) = compose(first(f), first(g))",
    ),
    extends=("ThCategory_",),
)

ThArrowChoice = _TheoryStub(
    name="ThArrowChoice",
    operations=("left_arr : Arr → Arr",),
    equations=(
        "left(arr(f)) = arr(left(f))",
        "left(compose(f, g)) = compose(left(f), left(g))",
    ),
    extends=("ThArrow",),
)

ThArrowApply = _TheoryStub(
    name="ThArrowApply",
    operations=("app : (Arr ⊗ Object) → Object",),
    equations=("compose(arr(λx. (arr(λy. (x, y)), z)), app) ≡ id",),
    extends=("ThArrow",),
)

ThArrowLoop = _TheoryStub(
    name="ThArrowLoop",
    operations=("loop_arr : Arr → Arr",),
    equations=(
        "left tightening: loop(compose(arr(first(h)), f)) = compose(h, loop(f))",
        "right tightening: loop(compose(f, arr(first(h)))) = compose(loop(f), h)",
        "sliding: loop(compose(f, arr(id × k))) = loop(compose(arr(id × k), f))",
        "vanishing: loop(arr(id)) = arr(id)",
        "superposing: first(loop(f)) = loop(arr(assoc) ∘ first(f) ∘ arr(assoc⁻¹))",
    ),
    extends=("ThArrow",),
)

ThArrowZero = _TheoryStub(
    name="ThArrowZero",
    operations=("zero_arr : Arr",),
    extends=("ThArrow",),
)

ThArrowPlus = _TheoryStub(
    name="ThArrowPlus",
    operations=("alt_arr : Arr ⊗ Arr → Arr",),
    equations=(
        "alt(zero, f) = f",
        "alt(f, zero) = f",
        "alt(alt(f, g), h) = alt(f, alt(g, h))",
    ),
    extends=("ThArrowZero",),
)


__all__ = [
    "ThCategory_",
    "ThArrow",
    "ThArrowChoice",
    "ThArrowApply",
    "ThArrowLoop",
    "ThArrowZero",
    "ThArrowPlus",
]
