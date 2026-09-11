"""Compiler mixin: unified type resolution.

A single `_resolve_any_space` walks any `ObjectExpr` to either a
`SetObject` (discrete) or a `ContinuousSpace` (continuous). The
``_resolve_type`` and ``_resolve_space`` forwarders are
type-narrowing wrappers so callers that already know they want a
discrete object can demand one without re-encoding the constraint
in every call site.
"""

from __future__ import annotations

import didactic.api as dx

from quivers.continuous.spaces import (
    Ball,
    CholeskyFactor,
    ContinuousSpace,
    Correlation,
    Covariance,
    Diagonal,
    Euclidean,
    LowerTriangular,
    Orthogonal,
    ProductSpace,
    Simplex,
    Sphere,
    Stiefel,
)
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


def _finset_literal(bind_name: str, digits: str, line: int, col: int) -> FinSet:
    """Build the `FinSet` an integer literal in type position denotes.

    A `FinSet` carries at least one element, so a literal ``0`` names
    no object at all; reporting that here keeps the diagnostic on the
    literal's own line and column instead of surfacing as a bare
    validation error from the object constructor.
    """
    cardinality = int(digits)
    if cardinality < 1:
        raise CompileError(
            f"FinSet cardinality must be at least 1, got {cardinality}; "
            f"the empty finite set is not an inhabitable type",
            line,
            col,
        )
    return FinSet(name=bind_name, cardinality=cardinality)


class ContinuousCtorSpec(dx.Model):
    """The calling convention of one surface continuous constructor.

    Attributes
    ----------
    size_fields : tuple of str
        Names of the size arguments, in surface order. Their count
        is the constructor's positional arity and their names appear
        verbatim in the arity diagnostic.
    option_fields : tuple of str
        Names the brace-block keyword arguments may take.
    fuses_sizes : bool
        Whether extra size arguments fuse into the single size slot
        by multiplication (``Real 4 8`` denotes
        :math:`\\mathbb{R}^{32}`) rather than being an arity error.
    """

    size_fields: tuple[str, ...]
    option_fields: tuple[str, ...] = ()
    fuses_sizes: bool = False


_CONTINUOUS_CTORS: dict[str, ContinuousCtorSpec] = {
    "Real": ContinuousCtorSpec(
        size_fields=("dim",),
        option_fields=("low", "high"),
        fuses_sizes=True,
    ),
    "Simplex": ContinuousCtorSpec(size_fields=("dim",)),
    "Sphere": ContinuousCtorSpec(size_fields=("dim",)),
    "Ball": ContinuousCtorSpec(size_fields=("dim",), option_fields=("radius",)),
    "CholeskyFactor": ContinuousCtorSpec(size_fields=("dim",)),
    "Covariance": ContinuousCtorSpec(size_fields=("dim",)),
    "Correlation": ContinuousCtorSpec(size_fields=("dim",)),
    "Orthogonal": ContinuousCtorSpec(size_fields=("dim",)),
    "Stiefel": ContinuousCtorSpec(size_fields=("rows", "cols")),
    "LowerTriangular": ContinuousCtorSpec(size_fields=("dim",)),
    "Diagonal": ContinuousCtorSpec(size_fields=("dim",)),
}


def _build_continuous_space(
    ctor_name: str,
    synth_name: str,
    sizes: list[int],
    options: dict[str, float | int],
) -> ContinuousSpace:
    """Construct the space a validated constructor invocation denotes.

    Every `ContinuousSpace` subclass takes keyword-only fields, so
    the sizes are bound by name here rather than forwarded
    positionally. Arity, option names, and size positivity are
    already checked by the caller.
    """
    if ctor_name == "Real":
        return Euclidean(
            name=synth_name,
            dim=sizes[0],
            low=options.get("low"),
            high=options.get("high"),
        )
    if ctor_name == "Simplex":
        return Simplex(name=synth_name, dim=sizes[0])
    if ctor_name == "Sphere":
        return Sphere(name=synth_name, dim=sizes[0])
    if ctor_name == "Ball":
        return Ball(name=synth_name, dim=sizes[0], radius=options.get("radius", 1.0))
    if ctor_name == "CholeskyFactor":
        return CholeskyFactor(name=synth_name, dim=sizes[0])
    if ctor_name == "Covariance":
        return Covariance(name=synth_name, dim=sizes[0])
    if ctor_name == "Correlation":
        return Correlation(name=synth_name, dim=sizes[0])
    if ctor_name == "Orthogonal":
        return Orthogonal(name=synth_name, dim=sizes[0])
    if ctor_name == "Stiefel":
        return Stiefel(name=synth_name, rows=sizes[0], cols=sizes[1])
    if ctor_name == "LowerTriangular":
        return LowerTriangular(name=synth_name, dim=sizes[0])
    if ctor_name == "Diagonal":
        return Diagonal(name=synth_name, dim=sizes[0])
    raise CompileError(
        f"unknown continuous constructor {ctor_name!r}; "
        f"available: {sorted(_CONTINUOUS_CTORS)}"
    )


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
            return _finset_literal(
                bind_name,
                texpr.name,
                texpr.line,
                texpr.col,
            )
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
            return _finset_literal(f"_{name}", name, texpr.line, texpr.col)
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
            return _finset_literal(
                f"_FinSet_{arg}",
                arg,
                texpr.line,
                texpr.col,
            )
        if arg in self._objects:
            return self._objects[arg]
        raise CompileError(
            f"FinSet({arg!r}): argument must be an integer literal or "
            f"a previously-declared object name",
            texpr.line,
            texpr.col,
        )

    def _resolve_continuous_constructor(
        self,
        texpr: ContinuousConstructor,
    ) -> ContinuousSpace:
        """Resolve ``Real 3``, ``Simplex 4``, ``Stiefel 5 2``, ... to a space.

        The surface writes size arguments by juxtaposition and
        options in a brace block; both are bound onto the target
        class's keyword fields through the constructor's
        `ContinuousCtorSpec`. Every arity, option-name, and
        size-positivity violation is reported against the
        constructor's own line and column.
        """
        ctor_name = texpr.constructor
        spec = _CONTINUOUS_CTORS.get(ctor_name)
        if spec is None:
            raise CompileError(
                f"unknown continuous constructor {ctor_name!r}; "
                f"available: {sorted(_CONTINUOUS_CTORS)}",
                texpr.line,
                texpr.col,
            )
        sizes = [self._eval_size_arg(arg, texpr) for arg in texpr.args]
        self._check_ctor_arity(ctor_name, spec, sizes, texpr)
        for size in sizes:
            if size < 1:
                raise CompileError(
                    f"{ctor_name}: dimension must be at least 1, got {size}; "
                    f"a zero-dimensional space has no values to sample",
                    texpr.line,
                    texpr.col,
                )
        bound = [_prod(sizes)] if spec.fuses_sizes else list(sizes)
        options: dict[str, float | int] = {}
        for key, val in texpr.kwargs.items():
            if key not in spec.option_fields:
                allowed = (
                    ", ".join(repr(k) for k in spec.option_fields)
                    if spec.option_fields
                    else "none"
                )
                raise CompileError(
                    f"{ctor_name}: unknown option {key!r}; accepted options: {allowed}",
                    texpr.line,
                    texpr.col,
                )
            options[key] = self._eval_scalar_kwarg(val, texpr)
        if ctor_name == "Stiefel" and bound[1] > bound[0]:
            raise CompileError(
                f"Stiefel: cols={bound[1]} must not exceed rows={bound[0]}; "
                f"a Stiefel manifold needs at least as many ambient "
                f"dimensions as frame vectors",
                texpr.line,
                texpr.col,
            )
        low = options.get("low")
        high = options.get("high")
        if low is not None and high is not None and high <= low:
            raise CompileError(
                f"{ctor_name}: high={high} must be strictly greater than low={low}",
                texpr.line,
                texpr.col,
            )
        synth_name = f"_{ctor_name}_" + "_".join(str(size) for size in sizes)
        return _build_continuous_space(ctor_name, synth_name, bound, options)

    def _check_ctor_arity(
        self,
        ctor_name: str,
        spec: ContinuousCtorSpec,
        sizes: list[int],
        texpr: ContinuousConstructor,
    ) -> None:
        """Reject a size-argument count the constructor cannot bind."""
        wanted = len(spec.size_fields)
        if spec.fuses_sizes:
            if len(sizes) >= 1:
                return
            raise CompileError(
                f"{ctor_name} takes at least one dimension argument; got 0",
                texpr.line,
                texpr.col,
            )
        if len(sizes) == wanted:
            return
        names = " ".join(spec.size_fields)
        raise CompileError(
            f"{ctor_name} takes exactly {wanted} size argument(s) "
            f"({names}); got {len(sizes)}",
            texpr.line,
            texpr.col,
        )

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

    def _eval_scalar_kwarg(self, val: float | int | str, texpr) -> float | int:
        if isinstance(val, (int, float)):
            return val
        raise CompileError(
            f"keyword argument {val!r}: not a numeric literal",
            texpr.line,
            texpr.col,
        )


__all__ = ["ContinuousCtorSpec", "_ResolutionMixin"]
