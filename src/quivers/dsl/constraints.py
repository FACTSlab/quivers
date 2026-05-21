"""Static constraint solver for QVR modules.

Walks a parsed `Module` and reports well-formedness violations
that the unified resolver or the compiler would otherwise only
surface at the point of use:

* ``residuated_constraint``: `ObjectSlash` patterns must
  reference a residuated universe (a previously-declared
  `TypeFreeResiduated`) at every binding site. Schema
  parameters typed as a residuated universe are recognized; bare
  `ObjectSlash` outside a residuated context is flagged.
* ``effect_constraint``: `ObjectEffectApply` references must
  name an effect with a conventional name (PascalCase or
  underscore-containing); other identifiers are flagged.
* ``bundle_unknown_member``: bundle-decl members that do not
  resolve at parse time (rules / schemas / built-in schema names /
  other bundles). The LSP relies on this so it can surface bundle
  problems without invoking the compiler.

The solver consumes the parsed AST only; it never invokes the
compiler. Returned diagnostics are `Violation` records that
the ``qvr check`` CLI lifts into structured ``Diagnostic`` instances.
"""

from __future__ import annotations

import didactic.api as dx

from quivers.dsl.ast_nodes import (
    BundleDecl,
    CategoryDecl,
    Module,
    RuleDecl,
    SchemaDecl,
    ObjectCoproduct,
    ObjectDecl,
    ObjectEffectApply,
    TypeEnumSet,
    ObjectExpr,
    TypeFreeMonoid,
    TypeFreeResiduated,
    TypeFromExpr,
    TypeName,
    ObjectProduct,
    ObjectSlash,
)
from quivers.stochastic.schema import SCHEMA_REGISTRY


class Violation(dx.Model):
    """One constraint-solver finding."""

    code: str
    message: str
    line: int
    col: int


def check_constraints(module: Module) -> list[Violation]:
    """Return every well-formedness violation in ``module``.

    Walks the statements in source order, building up the symbol
    tables (residuated universes, enum sets, free monoids, declared
    rule / schema / bundle names) incrementally so that out-of-order
    references are flagged.
    """
    out: list[Violation] = []

    residuated_universes: dict[str, TypeFreeResiduated] = {}
    enum_sets: dict[str, TypeEnumSet] = {}
    free_monoids: dict[str, TypeFreeMonoid] = {}
    aliased_names: dict[str, ObjectExpr] = {}
    rule_names: set[str] = set()
    schema_names: set[str] = set()
    bundle_names: set[str] = set()
    category_atoms: set[str] = set()

    builtin_schemas = set(SCHEMA_REGISTRY.keys())

    def is_residuated_object(name: str) -> bool:
        """True iff ``name`` resolves (via aliases) to a residuated universe."""
        seen: set[str] = set()
        cur = name
        while cur not in seen:
            seen.add(cur)
            if cur in residuated_universes:
                return True
            if cur in aliased_names:
                rhs = aliased_names[cur]
                if isinstance(rhs, TypeName):
                    cur = rhs.name
                    continue
            return False
        return False

    def is_known_effect(name: str) -> bool:
        """True iff ``name`` matches the conventional effect-naming pattern."""
        return bool(name) and (name[0].isupper() or "_" in name)

    def walk_pattern(
        texpr: ObjectExpr,
        *,
        in_residuated_context: bool,
        line_hint: int = 0,
        col_hint: int = 0,
    ) -> None:
        if isinstance(texpr, TypeName):
            return
        if isinstance(texpr, ObjectProduct):
            for c in texpr.components:
                walk_pattern(
                    c,
                    in_residuated_context=in_residuated_context,
                    line_hint=line_hint,
                    col_hint=col_hint,
                )
            return
        if isinstance(texpr, ObjectCoproduct):
            for c in texpr.components:
                walk_pattern(
                    c,
                    in_residuated_context=in_residuated_context,
                    line_hint=line_hint,
                    col_hint=col_hint,
                )
            return
        if isinstance(texpr, ObjectSlash):
            if not in_residuated_context:
                line = texpr.line or line_hint
                col = texpr.col or col_hint
                out.append(
                    Violation(
                        code="residuated_constraint",
                        message=(
                            f"ObjectSlash {texpr.direction!r} appears "
                            "outside a residuated context; either "
                            "declare a FreeResiduated universe and "
                            "parameterize the schema by it, or remove "
                            "the slash"
                        ),
                        line=line,
                        col=col,
                    )
                )
            walk_pattern(
                texpr.result,
                in_residuated_context=in_residuated_context,
                line_hint=line_hint,
                col_hint=col_hint,
            )
            walk_pattern(
                texpr.argument,
                in_residuated_context=in_residuated_context,
                line_hint=line_hint,
                col_hint=col_hint,
            )
            return
        if isinstance(texpr, ObjectEffectApply):
            if not is_known_effect(texpr.effect):
                line = texpr.line or line_hint
                col = texpr.col or col_hint
                out.append(
                    Violation(
                        code="effect_constraint",
                        message=(
                            f"effect {texpr.effect!r} has no "
                            "recognized naming pattern; effect names "
                            "must start with an uppercase letter or "
                            "contain an underscore"
                        ),
                        line=line,
                        col=col,
                    )
                )
            for arg in texpr.args:
                walk_pattern(
                    arg,
                    in_residuated_context=in_residuated_context,
                    line_hint=line_hint,
                    col_hint=col_hint,
                )
            return

    for stmt in module.statements:
        if isinstance(stmt, ObjectDecl):
            init = stmt.init
            if isinstance(init, TypeFreeResiduated):
                residuated_universes[stmt.name] = init
            elif isinstance(init, TypeEnumSet):
                enum_sets[stmt.name] = init
            elif isinstance(init, TypeFreeMonoid):
                free_monoids[stmt.name] = init
            elif isinstance(init, TypeFromExpr):
                aliased_names[stmt.name] = init.expr
        elif isinstance(stmt, CategoryDecl):
            for name in stmt.names:
                category_atoms.add(name)
        elif isinstance(stmt, RuleDecl):
            rule_names.add(stmt.name)
            for prem in stmt.premises:
                walk_pattern(
                    prem,
                    in_residuated_context=True,
                    line_hint=stmt.line,
                    col_hint=stmt.col,
                )
            walk_pattern(
                stmt.conclusion,
                in_residuated_context=True,
                line_hint=stmt.line,
                col_hint=stmt.col,
            )
        elif isinstance(stmt, SchemaDecl):
            schema_names.add(stmt.name)
            in_res = any(
                isinstance(p.type_expr, TypeName)
                and is_residuated_object(p.type_expr.name)
                for p in stmt.parameters
            )
            walk_pattern(
                stmt.domain,
                in_residuated_context=in_res,
                line_hint=stmt.line,
                col_hint=stmt.col,
            )
            walk_pattern(
                stmt.codomain,
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
                                f"bundle {stmt.name!r} references "
                                f"unknown member {member!r}; not a "
                                "declared rule / schema / bundle, "
                                "nor a built-in schema"
                            ),
                            line=stmt.line,
                            col=stmt.col,
                        )
                    )

    return out


__all__ = ["Violation", "check_constraints"]
