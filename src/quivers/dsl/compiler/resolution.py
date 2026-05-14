"""Compiler mixin: type and space resolution."""

from __future__ import annotations
from quivers.core.objects import FinSet, SetObject
from quivers.dsl.ast_nodes import (
    SpaceConstructor,
    SpaceExpr,
    TypeExpr,
    TypeName,
    TypeProduct,
)
from quivers.dsl.compiler._prelude import (
    CompileError,
    _get_space_constructors,
)


class _ResolutionMixin:
    """Mixin: type and space resolution methods."""

    def _resolve_index_size(self, texpr: TypeExpr) -> int:
        """Resolve a TypeExpr in finite-set-object position to its
        cardinality. 
        
        Used by the let-expression factor evaluator
        to determine the axis size of each binder at compile time.
        """
        obj = self._resolve_type(texpr)
        card = getattr(obj, "cardinality", None)
        if card is None:
            line = getattr(texpr, "line", 0)
            col = getattr(texpr, "col", 0)
            raise CompileError(
                f"factor binder's index must be a finite-set object, "
                f"got {type(obj).__name__}",
                line,
                col,
            )
        return int(card)

    def _resolve_type(self, texpr: TypeExpr, bind_name: str | None = None) -> SetObject:
        """Resolve a type expression into a SetObject.

        Delegates to :class:`~quivers.dsl.resolution.TypeExprToSetObject`,
        a :class:`didactic.api.Lens` parameterized by the current object
        environment. Integer-literal :class:`TypeName` nodes that aren't
        in the environment use ``bind_name`` (falling back to
        ``"_<value>"``) as the synthesized :class:`FinSet` name; this
        thin wrapper is kept so the literal-naming policy stays in
        compiler control.
        """
        from quivers.dsl.resolution import TypeExprToSetObject

        if (
            isinstance(texpr, TypeName)
            and texpr.name.isdigit()
            and texpr.name not in self._objects
            and bind_name is not None
        ):
            return FinSet(name=bind_name, cardinality=int(texpr.name))

        try:
            resolved, _ = TypeExprToSetObject(self._objects).forward(texpr)
        except KeyError as e:
            line = getattr(texpr, "line", 0)
            col = getattr(texpr, "col", 0)
            raise CompileError(str(e).strip("'\""), line, col) from e
        return resolved

    def _resolve_any_space(self, texpr: TypeExpr):
        """Resolve a type expression to either a SetObject or ContinuousSpace.

        Continuous morphism domains/codomains can be either discrete
        objects, continuous spaces, or product types.

        Parameters
        ----------
        texpr : TypeExpr
            The type expression to resolve (TypeName, TypeProduct, etc.).

        Returns
        -------
        SetObject or ContinuousSpace
            The resolved domain/codomain.
        """
        if isinstance(texpr, TypeProduct):
            from quivers.core.objects import ProductSet
            from quivers.continuous.spaces import ContinuousSpace, ProductSpace

            components = [self._resolve_any_space(c) for c in texpr.components]
            if any((isinstance(c, ContinuousSpace) for c in components)):
                return ProductSpace(components=tuple(components))
            return ProductSet(components=tuple(components))
        if not isinstance(texpr, TypeName):
            raise CompileError(
                f"unsupported type expression in domain/codomain: {type(texpr).__name__}",
                getattr(texpr, "line", 0),
                getattr(texpr, "col", 0),
            )
        name = texpr.name
        if name in self._objects:
            return self._objects[name]
        if name in self._spaces:
            return self._spaces[name]
        raise CompileError(f"undefined object or space {name!r}", texpr.line, texpr.col)

    def _resolve_space(self, sexpr: SpaceExpr, bind_name: str | None = None):
        """Resolve a space expression into a ContinuousSpace.

        Delegates to :class:`~quivers.dsl.resolution.SpaceExprToContinuousSpace`,
        a :class:`didactic.api.Lens` parameterized by both the space and
        object environments (a bare identifier may resolve to either).
        """
        from quivers.dsl.resolution import SpaceExprToContinuousSpace

        if isinstance(sexpr, SpaceConstructor):
            constructors = _get_space_constructors()
            cname = sexpr.constructor
            if cname not in constructors:
                raise CompileError(
                    f"unknown space constructor {cname!r}; available: "
                    f"{', '.join(sorted(constructors))}",
                    sexpr.line,
                    sexpr.col,
                )

        try:
            resolved, _ = SpaceExprToContinuousSpace(
                env_spaces=self._spaces,
                env_objects=self._objects,
                name=bind_name or "_anon",
            ).forward(sexpr)
        except (KeyError, ValueError) as e:
            line = getattr(sexpr, "line", 0)
            col = getattr(sexpr, "col", 0)
            raise CompileError(str(e).strip("'\""), line, col) from e
        return resolved
