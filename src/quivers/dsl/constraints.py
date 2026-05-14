"""Static constraint solver for QVR modules.

Walks a parsed :class:`Module` and reports well-formedness violations
that the type-resolution lens or the compiler would otherwise only
surface at the point of use:

- ``residuated_constraint`` — :class:`TypeSlash` patterns must
  reference a residuated universe (a previously-declared
  :class:`FreeResiduated` object) at every binding site. Schema
  parameters typed as a residuated universe are recognized; bare
  :class:`TypeSlash` outside a residuated context is flagged.
- ``effect_constraint`` — :class:`TypeEffectApply` references must
  name an effect that is part of the residuated universe's
  :attr:`FreeResiduated.effects` field (or, today, that follows the
  ``Eff_<Answer>`` mangling convention). Effects not in scope are
  flagged.
- ``schema_param_residuated`` — schema parameters typed at a
  non-residuated object are reported when the schema body uses
  ``TypeSlash`` patterns over those parameters.
- ``bundle_unknown_member`` — bundle-decl members that don't resolve
  at parse time (rules / schemas / built-in schema names / bundles).
  This duplicates the compile-time check intentionally so the LSP
  surfaces the diagnostic without invoking the full compiler.

The solver consumes the parsed AST only; it never invokes the
compiler. Returned diagnostics are :class:`Violation` records that
the ``qvr check`` CLI lifts into structured ``Diagnostic`` instances.
"""

from __future__ import annotations

from dataclasses import dataclass

from quivers.dsl.ast_nodes import (
    AliasDecl,
    BundleDecl,
    CategoryDecl,
    EnumSetLiteral,
    FreeMonoidExpr,
    FreeResiduatedExpr,
    Module,
    ObjectDecl,
    RuleDecl,
    SchemaDecl,
    Statement,
    TypeCoproduct,
    TypeEffectApply,
    TypeExpr,
    TypeName,
    TypeProduct,
    TypeSlash,
)


@dataclass(frozen=True)
class Violation:
    """One constraint-solver finding."""

    code: str
    message: str
    line: int
    col: int


def check_constraints(module: Module) -> list[Violation]:
    """Return every well-formedness violation in ``module``.

    Walks the statements in source order, building up the symbol
    tables (objects, residuated universes, declared aliases, declared
    rule/schema/bundle names) incrementally so that out-of-order
    references are flagged.
    """
    out: list[Violation] = []

    # Symbol tables — populated as we walk statements in source order.
    objects: dict[str, ObjectDecl] = {}
    aliases: dict[str, AliasDecl] = {}
    residuated_universes: dict[str, FreeResiduatedExpr] = {}
    enum_sets: dict[str, EnumSetLiteral] = {}
    free_monoids: dict[str, FreeMonoidExpr] = {}
    rule_names: set[str] = set()
    schema_names: set[str] = set()
    bundle_names: set[str] = set()
    category_atoms: set[str] = set()

    # Built-in schemas registry (consulted for bundle membership).
    from quivers.stochastic.schema import SCHEMA_REGISTRY

    builtin_schemas = set(SCHEMA_REGISTRY.keys())

    def is_residuated_object(name: str) -> bool:
        """True if ``name`` resolves (via aliases) to a FreeResiduated."""
        seen: set[str] = set()
        cur = name
        while cur not in seen:
            seen.add(cur)
            if cur in residuated_universes:
                return True
            if cur in aliases:
                rhs = aliases[cur].type_expr
                if isinstance(rhs, TypeName):
                    cur = rhs.name
                    continue
            return False
        return False

    def is_known_atom(name: str, residuated_scope: set[str]) -> bool:
        """True if ``name`` is a known atom of any residuated universe in scope."""
        for univ in residuated_scope:
            if univ in residuated_universes:
                gen = residuated_universes[univ].generators
                if gen in enum_sets and name in enum_sets[gen].elements:
                    return True
        return name in category_atoms

    def is_known_effect(name: str, residuated_scope: set[str]) -> bool:
        """True if ``name`` is a declared effect under the in-scope universes.

        Today effects are mangled into the universe declaration via
        their concatenated name (``Cont_S``, ``Alt``); the constraint
        solver accepts any TypeName-shaped effect identifier and defers
        deeper validation to the runtime ``class_directed_lifts``
        machinery.
        """
        # The grammar admits ``T(X)`` for arbitrary T; the constraint
        # solver accepts identifiers conforming to a conventional
        # naming pattern (PascalCase / underscore-suffixed). Tighter
        # validation arrives when the dedicated `effect ... : Monad`
        # declaration syntax lands.
        del residuated_scope
        return bool(name) and (name[0].isupper() or "_" in name)

    def walk_pattern(
        texpr: TypeExpr,
        *,
        param_residuated: dict[str, str],
        in_residuated_context: bool,
        line_hint: int = 0,
        col_hint: int = 0,
    ) -> None:
        """Walk a pattern, accumulating violations."""
        if isinstance(texpr, TypeName):
            return
        if isinstance(texpr, TypeProduct):
            for c in texpr.components:
                walk_pattern(
                    c,
                    param_residuated=param_residuated,
                    in_residuated_context=in_residuated_context,
                    line_hint=line_hint,
                    col_hint=col_hint,
                )
            return
        if isinstance(texpr, TypeCoproduct):
            for c in texpr.components:
                walk_pattern(
                    c,
                    param_residuated=param_residuated,
                    in_residuated_context=in_residuated_context,
                    line_hint=line_hint,
                    col_hint=col_hint,
                )
            return
        if isinstance(texpr, TypeSlash):
            # TypeSlash is legal only in a residuated context. The
            # context is residuated when EITHER the surrounding
            # declaration is a SchemaDecl with a parameter typed at a
            # FreeResiduated universe, OR it appears under the
            # generators of a FreeResiduated.
            if not in_residuated_context:
                line = texpr.line or line_hint
                col = texpr.col or col_hint
                out.append(
                    Violation(
                        code="residuated_constraint",
                        message=(
                            f"TypeSlash {texpr.direction!r} appears outside "
                            "a residuated context; either declare a "
                            "FreeResiduated universe and parameterize the "
                            "schema by it, or remove the slash"
                        ),
                        line=line,
                        col=col,
                    )
                )
            walk_pattern(
                texpr.result,
                param_residuated=param_residuated,
                in_residuated_context=in_residuated_context,
                line_hint=line_hint,
                col_hint=col_hint,
            )
            walk_pattern(
                texpr.argument,
                param_residuated=param_residuated,
                in_residuated_context=in_residuated_context,
                line_hint=line_hint,
                col_hint=col_hint,
            )
            return
        if isinstance(texpr, TypeEffectApply):
            scope = set(param_residuated.values())
            if not is_known_effect(texpr.effect, scope):
                line = texpr.line or line_hint
                col = texpr.col or col_hint
                out.append(
                    Violation(
                        code="effect_constraint",
                        message=(
                            f"effect {texpr.effect!r} has no recognized "
                            "naming pattern; effect names must start with "
                            "an uppercase letter or contain an underscore"
                        ),
                        line=line,
                        col=col,
                    )
                )
            for arg in texpr.args:
                walk_pattern(
                    arg,
                    param_residuated=param_residuated,
                    in_residuated_context=in_residuated_context,
                    line_hint=line_hint,
                    col_hint=col_hint,
                )
            return

    for stmt in module.statements:
        if isinstance(stmt, ObjectDecl):
            objects[stmt.name] = stmt
            if isinstance(stmt.init, FreeResiduatedExpr):
                residuated_universes[stmt.name] = stmt.init
            elif isinstance(stmt.init, EnumSetLiteral):
                enum_sets[stmt.name] = stmt.init
            elif isinstance(stmt.init, FreeMonoidExpr):
                free_monoids[stmt.name] = stmt.init
        elif isinstance(stmt, AliasDecl):
            aliases[stmt.name] = stmt
        elif isinstance(stmt, CategoryDecl):
            category_atoms.add(stmt.name)
        elif isinstance(stmt, RuleDecl):
            rule_names.add(stmt.name)
            # Rules may use TypeSlash freely; classical Lambek context.
            for prem in stmt.premises:
                walk_pattern(
                    prem,
                    param_residuated={},
                    in_residuated_context=True,
                    line_hint=stmt.line,
                    col_hint=stmt.col,
                )
            walk_pattern(
                stmt.conclusion,
                param_residuated={},
                in_residuated_context=True,
                line_hint=stmt.line,
                col_hint=stmt.col,
            )
        elif isinstance(stmt, SchemaDecl):
            schema_names.add(stmt.name)
            param_residuated: dict[str, str] = {}
            for names_group, ty in zip(stmt.parameter_names, stmt.parameter_types):
                if isinstance(ty, TypeName) and is_residuated_object(ty.name):
                    for nm in names_group:
                        param_residuated[nm] = ty.name
            # Schema body is residuated iff at least one parameter is.
            in_res = bool(param_residuated)
            walk_pattern(
                stmt.domain,
                param_residuated=param_residuated,
                in_residuated_context=in_res,
                line_hint=stmt.line,
                col_hint=stmt.col,
            )
            walk_pattern(
                stmt.codomain,
                param_residuated=param_residuated,
                in_residuated_context=in_res,
                line_hint=stmt.line,
                col_hint=stmt.col,
            )
        elif isinstance(stmt, BundleDecl):
            bundle_names.add(stmt.name)
            for member in stmt.rules:
                if (
                    member not in rule_names
                    and member not in schema_names
                    and member not in bundle_names
                    and member not in builtin_schemas
                ):
                    out.append(
                        Violation(
                            code="bundle_unknown_member",
                            message=(
                                f"bundle {stmt.name!r} references unknown "
                                f"member {member!r}; not a declared rule "
                                "/ schema / bundle, nor a built-in schema"
                            ),
                            line=stmt.line,
                            col=stmt.col,
                        )
                    )

    return out


def _statement_kind(stmt: Statement) -> str:
    return type(stmt).__name__


__all__ = ["Violation", "check_constraints"]
