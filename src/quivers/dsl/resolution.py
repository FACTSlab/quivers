"""Bidirectional resolution lenses for QVR type and space expressions.

The compiler's ``_resolve_type`` and ``_resolve_space`` map syntactic AST
trees (:class:`~quivers.dsl.ast_nodes.TypeExpr` and
:class:`~quivers.dsl.ast_nodes.SpaceExpr` variants) to *resolved* values
(:class:`~quivers.core.objects.SetObject` and
:class:`~quivers.continuous.spaces.ContinuousSpace` variants). This module
expresses those mappings as :class:`didactic.api.Lens` instances so the
resolution layer becomes a first-class lens family rather than ad-hoc
visitor code.

Each lens carries the resolution *environment* it needs (the object
inventory and, for spaces, the previously-declared space inventory) on
its own instance — that's the dependent-optics shape: the lens is
parameterized by context that determines how each leaf vertex resolves.
The forward direction performs the value-dependent lookup; the
complement holds the original AST node so :meth:`backward` recovers it
verbatim. Round-trip laws hold by construction:

- ``backward(*forward(t)) == t`` (GetPut)
- ``forward(backward(s, c)) == (s, c)`` (PutGet)

:class:`didactic.api.Lens` is a pure-Python authoring surface; the
panproto-side compilation (each ``Lens`` becoming a ``panproto.Lens``
with formal complement-bearing get/put and runtime law-checking) is a
later didactic feature. Today these lenses behave as ordinary Python
objects with ``forward`` / ``backward`` methods and ``>>`` composition.
"""

import didactic.api as dx

from quivers.continuous.spaces import (
    ContinuousSpace,
    Euclidean,
    PositiveReals,
    ProductSpace,
    Simplex,
    UnitInterval,
)
from quivers.core.objects import (
    CoproductSet,
    FinSet,
    ProductSet,
    SetObject,
)
from quivers.dsl.ast_nodes import (
    SpaceConstructor,
    SpaceExpr,
    SpaceName,
    SpaceProduct,
    TypeCoproduct,
    TypeEffectApply,
    TypeExpr,
    TypeName,
    TypeProduct,
    TypeSlash,
)


__all__ = [
    "TypeExprToSetObject",
    "SpaceExprToContinuousSpace",
]


# ---------------------------------------------------------------------------
# TypeExpr -> SetObject
# ---------------------------------------------------------------------------


class TypeExprToSetObject(dx.Lens[TypeExpr, SetObject, TypeExpr]):
    """Resolve a :class:`TypeExpr` AST tree to a :class:`SetObject`.

    The forward direction dispatches on the variant ``kind``:

    - ``TypeName(name="X")`` → ``self.env[X]`` if X is bound, else
      ``FinSet(name=X, cardinality=int(X))`` if X is an integer literal.
    - ``TypeProduct(components=…)`` → ``ProductSet(components=…)`` with
      each component recursively resolved.
    - ``TypeCoproduct(components=…)`` → ``CoproductSet(components=…)``
      with each component recursively resolved.

    Parameters
    ----------
    env : dict[str, SetObject]
        The object-name → :class:`SetObject` environment built from
        ``object`` declarations earlier in the program.
    """

    def __init__(self, env: dict[str, SetObject]) -> None:
        self._env = env

    def forward(self, t: TypeExpr, /) -> tuple[SetObject, TypeExpr]:
        return self._resolve(t), t

    def backward(self, s: SetObject, complement: TypeExpr, /) -> TypeExpr:
        return complement

    def _resolve(self, t: TypeExpr) -> SetObject:
        if isinstance(t, TypeName):
            if t.name in self._env:
                return self._env[t.name]
            try:
                cardinality = int(t.name)
            except ValueError as e:
                raise KeyError(
                    f"undefined object {t.name!r} (line {t.line}, col {t.col})"
                ) from e
            # Synthesise a uniqueable FinSet name for bare integer literals
            # (`_8`-style) so two distinct ``object X : 8`` and ``object Y : 8``
            # declarations share the same FinSet by structure but the literal
            # case still gets a valid str name.
            return FinSet(name=f"_{t.name}", cardinality=cardinality)

        if isinstance(t, TypeProduct):
            components = tuple(self._resolve(c) for c in t.components)
            return ProductSet(components=components)

        if isinstance(t, TypeCoproduct):
            components = tuple(self._resolve(c) for c in t.components)
            return CoproductSet(components=components)

        if isinstance(t, TypeSlash):
            raise TypeError(
                f"residuated TypeSlash {t.direction!r} is not a "
                "resolvable SetObject; slash patterns appear only in "
                "rule declarations, where they are matched against "
                "the runtime Category system rather than resolved to "
                "set objects (line {}, col {})".format(t.line, t.col)
            )

        if isinstance(t, TypeEffectApply):
            raise TypeError(
                f"effect-typed TypeEffectApply {t.effect!r} is not "
                "resolvable as a SetObject; effect-typed categories "
                "live in the chart parser's effect-lifting layer "
                f"(line {t.line}, col {t.col})"
            )

        raise TypeError(f"unexpected TypeExpr variant: {type(t).__name__}")


# ---------------------------------------------------------------------------
# SpaceExpr -> ContinuousSpace
# ---------------------------------------------------------------------------


def _build_space_constructor(
    constructor: str, args: tuple[str, ...], kwargs: dict[str, str], name: str
) -> ContinuousSpace:
    """Translate a parsed SpaceConstructor to its runtime ContinuousSpace."""
    if constructor == "Euclidean":
        if not args:
            raise ValueError("Euclidean requires a dimension argument")
        dim = int(args[0])
        low = float(kwargs["low"]) if "low" in kwargs else None
        high = float(kwargs["high"]) if "high" in kwargs else None
        return Euclidean(name=name, dim=dim, low=low, high=high)

    if constructor == "Simplex":
        if not args:
            raise ValueError("Simplex requires a dimension argument")
        return Simplex(name=name, dim=int(args[0]))

    if constructor == "PositiveReals":
        if not args:
            raise ValueError("PositiveReals requires a dimension argument")
        return PositiveReals(name=name, dim=int(args[0]))

    if constructor == "UnitInterval":
        dim = int(args[0]) if args else 1
        return UnitInterval(name, dim)

    raise ValueError(f"unsupported space constructor {constructor!r}")


class SpaceExprToContinuousSpace(
    dx.Lens[SpaceExpr, ContinuousSpace | SetObject, SpaceExpr]
):
    """Resolve a :class:`SpaceExpr` AST tree to a :class:`ContinuousSpace`.

    Bare identifiers (:class:`SpaceName`) may resolve to either a
    previously declared continuous space or, for mixed-domain programs,
    a discrete :class:`SetObject` declared earlier — the forward
    direction dispatches accordingly:

    - :class:`SpaceConstructor` → invoke the named constructor with the
      parsed numeric args.
    - :class:`SpaceName` → look up in ``env_spaces`` (continuous) or
      ``env_objects`` (discrete fallback).
    - :class:`SpaceProduct` → recurse into components.

    Parameters
    ----------
    env_spaces : dict[str, ContinuousSpace]
        The space-name → :class:`ContinuousSpace` environment from
        previously declared ``space`` blocks.
    env_objects : dict[str, SetObject]
        The object-name → :class:`SetObject` environment, consulted when
        a bare identifier does not name a continuous space.
    name : str
        The binding name for the new space (used by
        :class:`SpaceConstructor` outputs); callers passing a
        :class:`SpaceConstructor` should provide the declaration name.
    """

    def __init__(
        self,
        env_spaces: dict[str, ContinuousSpace],
        env_objects: dict[str, SetObject],
        name: str,
    ) -> None:
        self._env_spaces = env_spaces
        self._env_objects = env_objects
        self._name = name

    def forward(self, s: SpaceExpr, /) -> tuple[ContinuousSpace | SetObject, SpaceExpr]:
        return self._resolve(s, self._name), s

    def backward(
        self, r: ContinuousSpace | SetObject, complement: SpaceExpr, /
    ) -> SpaceExpr:
        return complement

    def _resolve(self, s: SpaceExpr, scope_name: str) -> ContinuousSpace | SetObject:
        if isinstance(s, SpaceConstructor):
            return _build_space_constructor(s.constructor, s.args, s.kwargs, scope_name)

        if isinstance(s, SpaceName):
            if s.name in self._env_spaces:
                return self._env_spaces[s.name]
            if s.name in self._env_objects:
                return self._env_objects[s.name]
            raise KeyError(f"undefined space {s.name!r} (line {s.line}, col {s.col})")

        if isinstance(s, SpaceProduct):
            components = tuple(
                self._resolve(c, f"{scope_name}_{i}")
                for i, c in enumerate(s.components)
            )
            return ProductSpace(components=components)

        raise TypeError(f"unexpected space-expression variant: {type(s).__name__}")
