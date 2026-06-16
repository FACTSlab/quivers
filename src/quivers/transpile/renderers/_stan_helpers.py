"""Render `LetExprNode` to Stan tree-sitter schema vertices.

The Stan grammar exposes the following expression-level vertex
kinds the helper builds:

* `real_literal` / `integer_literal` for numeric leaves
* `variable_expression` wrapping an `identifier` for variable refs
* `infix_op_expression` (per-operator alts via `chose-alt-fingerprint`)
* `prefix_op_expression` for unary minus
* `function_application` with an `identifier` callee + `expression_list`
* `index_expression` for `arr[i1][i2]...`
* `array_expression` for `{e0, e1, ...}` list literals

Some `LetExprNode` kinds do not map to Stan user-program syntax
(strings, lambdas, method calls). The helper raises
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
with a precise kind rather than emitting a fake placeholder token.
`LetExprFactor` is unrolled at render time: the cases form becomes
an `array_expression` whose children are the case bodies in label
order; the uniform-body multi-binder form becomes nested
`array_expression` vertices populated by substituting each binder
through its axis's static cardinality (looked up via the
`_StanLetCtx`'s `cards` map sourced from
[`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]).
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLambda,
    LetExprList,
    LetExprLiteral,
    LetExprMethodCall,
    LetExprNode,
    LetExprString,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.dsl.ast_nodes.let_expressions import LetFactorBinder, LetFactorCase
from quivers.dsl.ast_nodes.objects import TypeName
from quivers.transpile._api import UnsupportedConstruct


def render_let_expr_stan(ctx, expr: LetExprNode) -> str:
    """Build a Stan expression schema for `expr` in `ctx`. Returns
    the root vertex id."""
    if isinstance(expr, LetExprLiteral):
        if isinstance(expr.value, float) or "." in repr(expr.value):
            v = ctx.vertex(ctx.fresh("rl"), "real_literal")
        else:
            v = ctx.vertex(ctx.fresh("il"), "integer_literal")
        ctx.literal(v, str(expr.value))
        return v
    if isinstance(expr, LetExprVar):
        v = ctx.vertex(ctx.fresh("vex"), "variable_expression")
        ident = ctx.vertex(ctx.fresh("id"), "identifier")
        ctx.literal(ident, expr.name)
        ctx.edge(v, ident, "child_of")
        return v
    if isinstance(expr, LetExprBinOp):
        b = ctx.vertex(ctx.fresh("bin"), "infix_op_expression")
        ctx.constraint(b, "chose-alt-fingerprint", expr.op)
        ctx.edge(b, render_let_expr_stan(ctx, expr.left), "child_of")
        ctx.edge(b, render_let_expr_stan(ctx, expr.right), "child_of")
        return b
    if isinstance(expr, LetExprUnaryOp):
        u = ctx.vertex(ctx.fresh("uop"), "prefix_op_expression")
        ctx.constraint(u, "chose-alt-fingerprint", "-")
        ctx.edge(u, render_let_expr_stan(ctx, expr.operand), "child_of")
        return u
    if isinstance(expr, LetExprCall):
        c = ctx.vertex(ctx.fresh("call"), "function_application")
        fn = ctx.vertex(ctx.fresh("fn"), "identifier")
        ctx.literal(fn, expr.func)
        ctx.edge(c, fn, "child_of")
        args = ctx.vertex(ctx.fresh("args"), "expression_list")
        for a in expr.args:
            ctx.edge(args, render_let_expr_stan(ctx, a), "child_of")
        ctx.edge(c, args, "child_of")
        return c
    if isinstance(expr, LetExprIndex):
        s = ctx.vertex(ctx.fresh("idx"), "index_expression")
        ctx.edge(s, render_let_expr_stan(ctx, expr.array), "child_of")
        for idx in expr.indices:
            ctx.edge(s, render_let_expr_stan(ctx, idx), "child_of")
        return s
    if isinstance(expr, LetExprList):
        arr = ctx.vertex(ctx.fresh("arr"), "array_expression")
        for item in expr.items:
            ctx.edge(arr, render_let_expr_stan(ctx, item), "child_of")
        return arr
    if isinstance(expr, LetExprFactor):
        return _render_factor_stan(ctx, expr)
    if isinstance(expr, LetExprString):
        raise UnsupportedConstruct(
            "qvr-stan-helper",
            [
                "let-expr:LetExprString: Stan has no string literal "
                "in expression position"
            ],
        )
    if isinstance(expr, LetExprLambda):
        raise UnsupportedConstruct(
            "qvr-stan-helper",
            [
                "let-expr:LetExprLambda: Stan has no anonymous "
                "function syntax in user-program expression position"
            ],
        )
    if isinstance(expr, LetExprMethodCall):
        raise UnsupportedConstruct(
            "qvr-stan-helper",
            [
                "let-expr:LetExprMethodCall: Stan has no method "
                "dispatch syntax"
            ],
        )
    raise UnsupportedConstruct(
        "qvr-stan-helper",
        [f"let-expr:{type(expr).__name__}: unhandled node kind"],
    )


def _render_factor_stan(ctx, expr: LetExprFactor) -> str:
    """Unroll a `LetExprFactor` into nested `array_expression`
    vertices.

    The cases form (binders contain a single axis, body is None,
    cases enumerate labels in [0, |axis|)) emits an
    `array_expression` whose children are each case's body in
    label order.

    The uniform-body form (one or more binders, body is the
    repeated expression, cases is empty) emits a tower of
    `array_expression` vertices of shape
    `(|b0|, |b1|, ..., |bn-1|)`, where the innermost element is
    the body with each binder substituted for its 1-indexed
    integer value (Stan arrays are 1-indexed and QVR's surface
    indexing is mapped through directly).
    """
    if expr.cases and expr.body is None:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                "qvr-stan-helper",
                [
                    "let-expr:LetExprFactor: cases form requires "
                    f"exactly one binder; got {len(expr.binders)}"
                ],
            )
        ordered = sorted(expr.cases, key=lambda c: c.label)
        arr = ctx.vertex(ctx.fresh("farr"), "array_expression")
        for case in ordered:
            ctx.edge(arr, render_let_expr_stan(ctx, case.value), "child_of")
        return arr
    if expr.body is not None and not expr.cases:
        sizes = tuple(_card_for(ctx, b) for b in expr.binders)
        return _build_nested_array_stan(
            ctx, expr.binders, sizes, expr.body, ()
        )
    raise UnsupportedConstruct(
        "qvr-stan-helper",
        [
            "let-expr:LetExprFactor: mixed cases-plus-body form "
            "is not a valid surface construct"
        ],
    )


def _card_for(ctx, binder: LetFactorBinder) -> int:
    """Resolve the static cardinality of `binder.index`.

    `LetExprFactor` only unrolls when every binder's axis has a
    statically-known size; the helper consults `ctx.cards`
    (populated from
    [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]).
    """
    idx = binder.index
    if isinstance(idx, TypeName):
        cards = getattr(ctx, "cards", None)
        if cards is None or idx.name not in cards:
            raise UnsupportedConstruct(
                "qvr-stan-helper",
                [
                    f"let-expr:LetExprFactor: binder {binder.var!r} "
                    f"references object {idx.name!r} whose cardinality "
                    "is unknown at render time"
                ],
            )
        return cards[idx.name]
    raise UnsupportedConstruct(
        "qvr-stan-helper",
        [
            f"let-expr:LetExprFactor: binder {binder.var!r} index is "
            f"{type(idx).__name__}; only TypeName binders unroll"
        ],
    )


def _build_nested_array_stan(
    ctx,
    binders: tuple[LetFactorBinder, ...],
    sizes: tuple[int, ...],
    body: LetExprNode,
    fixed: tuple[int, ...],
) -> str:
    """Recursive helper that materialises the nested
    `array_expression` tower for the uniform-body factor form."""
    if len(fixed) == len(binders):
        subst = body
        for binder, value in zip(binders, fixed, strict=True):
            subst = _substitute_let_expr(
                subst, binder.var, LetExprLiteral(value=value + 1)
            )
        return render_let_expr_stan(ctx, subst)
    level = len(fixed)
    arr = ctx.vertex(ctx.fresh("farr"), "array_expression")
    for i in range(sizes[level]):
        child = _build_nested_array_stan(
            ctx, binders, sizes, body, fixed + (i,)
        )
        ctx.edge(arr, child, "child_of")
    return arr


def _substitute_let_expr(
    expr: LetExprNode, name: str, value: LetExprNode
) -> LetExprNode:
    """Capture-avoiding substitution of every free occurrence of
    `LetExprVar(name=name)` in `expr` with `value`.

    Shared substitution helper for every per-target renderer that
    needs to unroll `LetExprFactor` by binding indices to integer
    literals. Lives in `_stan_helpers` because Stan was the first
    target to need it; other helper modules import from here when
    they grow the same need (one source of truth for the walk).
    """
    if isinstance(expr, LetExprVar):
        return value if expr.name == name else expr
    if isinstance(expr, LetExprLiteral):
        return expr
    if isinstance(expr, LetExprString):
        return expr
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_substitute_let_expr(expr.left, name, value),
            right=_substitute_let_expr(expr.right, name, value),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            op=expr.op,
            operand=_substitute_let_expr(expr.operand, name, value),
        )
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(
                _substitute_let_expr(a, name, value) for a in expr.args
            ),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_substitute_let_expr(expr.array, name, value),
            indices=tuple(
                _substitute_let_expr(i, name, value) for i in expr.indices
            ),
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(
                _substitute_let_expr(i, name, value) for i in expr.items
            ),
        )
    if isinstance(expr, LetExprLambda):
        if expr.param == name:
            return expr
        return LetExprLambda(
            param=expr.param,
            body=_substitute_let_expr(expr.body, name, value),
        )
    if isinstance(expr, LetExprFactor):
        if any(b.var == name for b in expr.binders):
            return expr
        return LetExprFactor(
            binders=expr.binders,
            body=(
                _substitute_let_expr(expr.body, name, value)
                if expr.body is not None
                else None
            ),
            cases=tuple(
                LetFactorCase(
                    label=c.label,
                    value=_substitute_let_expr(c.value, name, value),
                    line=c.line,
                    col=c.col,
                )
                for c in expr.cases
            ),
        )
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_substitute_let_expr(expr.receiver, name, value),
            method=expr.method,
            args=tuple(
                _substitute_let_expr(a, name, value) for a in expr.args
            ),
        )
    raise UnsupportedConstruct(
        "qvr-let-substitution",
        [f"let-expr:{type(expr).__name__}: substitution unhandled"],
    )


__all__ = ["render_let_expr_stan"]
