"""Compiler mixin: unified type resolution.

A single `_resolve_any_space` walks any `ObjectExpr` to either a
`SetObject` (discrete) or a `ContinuousSpace` (continuous). The
``_resolve_type`` and ``_resolve_space`` forwarders are
type-narrowing wrappers so callers that already know they want a
discrete object can demand one without re-encoding the constraint
in every call site.
"""

from __future__ import annotations

from quivers.continuous import spaces as continuous_spaces
from quivers.continuous.spaces import ContinuousSpace, ProductSpace
from quivers.core.objects import CoproductSet, FinSet, ProductSet, SetObject
from quivers.dsl.ast_nodes import (
    ContinuousConstructor,
    DiscreteConstructor,
    ObjectCoproduct,
    ObjectEffectApply,
    ObjectExpr,
    TypeName,
    ObjectProduct,
    ObjectSlash,
)
from quivers.dsl.compiler._prelude import CompileError


def _prod(xs: list[int]) -> int:
    out = 1
    for x in xs:
        out *= x
    return out


_CONTINUOUS_FACTORIES: dict[str, str] = {
    "Real": "Euclidean",
    "Simplex": "Simplex",
    "Sphere": "Sphere",
    "Ball": "Ball",
    "CholeskyFactor": "CholeskyFactor",
    "Covariance": "Covariance",
    "Correlation": "Correlation",
    "Orthogonal": "Orthogonal",
    "Stiefel": "Stiefel",
    "LowerTriangular": "LowerTriangular",
    "Diagonal": "Diagonal",
}


class _ResolutionMixin:
    """Mixin: unified resolution of type expressions.

    The compiler base provides ``_objects`` and ``_spaces``; the
    annotations below pin them so the type checker can verify every
    access from a mixin method.
    """

    _objects: dict[str, SetObject]
    _spaces: dict[str, ContinuousSpace]

    def _resolve_index_size(self, texpr: ObjectExpr) -> int:
        """Resolve a type expression in index position to a cardinality.

        Used by the let-expression factor evaluator to determine the
        axis size of each binder at compile time. Any resolved
        `SetObject` exposes its cardinality directly; a
        `ContinuousSpace` is illegal in this position.
        """
        obj = self._resolve_type(texpr)
        card = getattr(obj, "cardinality", None)
        if card is None:
            line = getattr(texpr, "line", 0)
            col = getattr(texpr, "col", 0)
            raise CompileError(
                f"index must be a finite-set object, got {type(obj).__name__}",
                line,
                col,
            )
        return int(card)

    def _resolve_type(
        self,
        texpr: ObjectExpr,
        bind_name: str | None = None,
    ) -> SetObject:
        """Resolve a type expression that must denote a discrete object.

        Calls `_resolve_any_space` and rejects continuous-space
        results. ``bind_name`` is consulted when the type is an
        anonymous integer literal so the synthesised `FinSet`
        carries the declaration's name.
        """
        if (
            isinstance(texpr, TypeName)
            and texpr.name.isdigit()
            and texpr.name not in self._objects
            and bind_name is not None
        ):
            return FinSet(name=bind_name, cardinality=int(texpr.name))
        obj = self._resolve_any_space(texpr)
        if isinstance(obj, ContinuousSpace):
            raise CompileError(
                f"expected a discrete object here, got continuous "
                f"space {type(obj).__name__}",
                getattr(texpr, "line", 0),
                getattr(texpr, "col", 0),
            )
        return obj

    def _resolve_any_space(self, texpr: ObjectExpr):
        """Resolve a type expression to a SetObject or ContinuousSpace.

        Dispatches on the `ObjectExpr` variant. Product types
        mix discrete and continuous components: a product whose
        every factor is discrete becomes a `ProductSet`; any
        continuous component lifts the whole product into a
        `ProductSpace`.
        """
        if isinstance(texpr, TypeName):
            return self._resolve_type_name(texpr)
        if isinstance(texpr, DiscreteConstructor):
            return self._resolve_discrete_constructor(texpr)
        if isinstance(texpr, ContinuousConstructor):
            return self._resolve_continuous_constructor(texpr)
        if isinstance(texpr, ObjectProduct):
            components = [self._resolve_any_space(c) for c in texpr.components]
            if any(isinstance(c, ContinuousSpace) for c in components):
                return ProductSpace(components=tuple(components))
            return ProductSet(components=tuple(components))
        if isinstance(texpr, ObjectCoproduct):
            components = [self._resolve_any_space(c) for c in texpr.components]
            if any(isinstance(c, ContinuousSpace) for c in components):
                raise CompileError(
                    "coproduct of continuous spaces is not supported",
                    getattr(texpr, "line", 0),
                    getattr(texpr, "col", 0),
                )
            return CoproductSet(components=tuple(components))
        if isinstance(texpr, (ObjectSlash, ObjectEffectApply)):
            raise CompileError(
                f"{type(texpr).__name__}: residuated / effect-typed "
                f"expressions do not resolve to a concrete object or "
                f"space outside a schema pattern context",
                getattr(texpr, "line", 0),
                getattr(texpr, "col", 0),
            )
        raise CompileError(
            f"unsupported type expression: {type(texpr).__name__}",
            getattr(texpr, "line", 0),
            getattr(texpr, "col", 0),
        )

    # -----------------------------------------------------------------
    # specialised resolvers
    # -----------------------------------------------------------------

    def _resolve_type_name(
        self,
        texpr: TypeName,
    ) -> SetObject | ContinuousSpace:
        name = texpr.name
        if name.isdigit():
            return FinSet(name=f"_{name}", cardinality=int(name))
        if name in self._objects:
            return self._objects[name]
        if name in self._spaces:
            return self._spaces[name]
        raise CompileError(
            f"undefined object or space {name!r}",
            texpr.line,
            texpr.col,
        )

    def _resolve_discrete_constructor(
        self,
        texpr: DiscreteConstructor,
    ) -> SetObject:
        if texpr.constructor != "FinSet":
            raise CompileError(
                f"unknown discrete constructor {texpr.constructor!r}",
                texpr.line,
                texpr.col,
            )
        if len(texpr.args) != 1:
            raise CompileError(
                f"FinSet(N) takes exactly one argument; got {len(texpr.args)}",
                texpr.line,
                texpr.col,
            )
        arg = texpr.args[0]
        if arg.isdigit():
            return FinSet(name=f"_FinSet_{arg}", cardinality=int(arg))
        if arg in self._objects:
            return self._objects[arg]
        raise CompileError(
            f"FinSet({arg!r}): argument must be an integer literal or "
            f"a previously-declared object name",
            texpr.line,
            texpr.col,
        )

    def _resolve_continuous_constructor(self, texpr: ContinuousConstructor):
        ctor_name = texpr.constructor
        cls = getattr(
            continuous_spaces,
            _CONTINUOUS_FACTORIES[ctor_name],
            None,
        )
        if cls is None:
            raise CompileError(
                f"continuous constructor {ctor_name!r} is not "
                f"available in quivers.continuous.spaces",
                texpr.line,
                texpr.col,
            )
        positional: list[int] = []
        for arg in texpr.args:
            positional.append(self._eval_size_arg(arg, texpr))
        kwargs: dict[str, float | int] = {}
        for key, val in texpr.kwargs.items():
            kwargs[key] = self._eval_scalar_kwarg(val, texpr)
        synth_name = f"_{ctor_name}_" + "_".join(str(p) for p in positional)
        if ctor_name == "Real":
            if len(positional) == 1:
                return cls(name=synth_name, dim=positional[0], **kwargs)
            if len(positional) >= 2:
                return cls(
                    name=synth_name,
                    dim=int(_prod(positional)),
                    **kwargs,
                )
            raise CompileError(
                "Real takes at least one dimension argument; got 0",
                texpr.line,
                texpr.col,
            )
        try:
            return cls(name=synth_name, *positional, **kwargs)
        except TypeError:
            try:
                return cls(*positional, **kwargs)
            except TypeError as exc:
                raise CompileError(
                    f"{ctor_name}: invalid arguments "
                    f"({positional!r}, {kwargs!r}): {exc}",
                    texpr.line,
                    texpr.col,
                ) from exc

    def _eval_size_arg(self, arg: str, texpr) -> int:
        if arg.isdigit():
            return int(arg)
        if arg in self._objects:
            card = getattr(self._objects[arg], "cardinality", None)
            if card is None:
                raise CompileError(
                    f"object {arg!r} has no cardinality",
                    texpr.line,
                    texpr.col,
                )
            return int(card)
        raise CompileError(
            f"size argument {arg!r}: not an integer literal or a "
            f"previously-declared finite object",
            texpr.line,
            texpr.col,
        )

    def _eval_scalar_kwarg(self, val: str, texpr) -> float | int:
        try:
            if any(ch in val for ch in ".eE"):
                return float(val)
            return int(val)
        except ValueError as exc:
            raise CompileError(
                f"keyword argument {val!r}: not a numeric literal",
                texpr.line,
                texpr.col,
            ) from exc


__all__ = ["_ResolutionMixin"]
