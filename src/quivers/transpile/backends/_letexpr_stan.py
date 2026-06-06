"""Render `LetExprNode` to Stan tree-sitter schema vertices."""

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


def render_let_expr_stan(ctx, expr: LetExprNode) -> str:
    """Build a Stan expression schema for `expr` in `ctx`. Returns
    the root vertex id."""
    if isinstance(expr, LetExprLiteral):
        # Stan: real / int literal. Use `real_literal` for floats,
        # `integer_literal` for ints.
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
        b = ctx.vertex(ctx.fresh("bin"), "binary_expression")
        ctx.edge(b, render_let_expr_stan(ctx, expr.left), "child_of")
        op = ctx.vertex(ctx.fresh("op"), "operator")
        ctx.literal(op, expr.op)
        ctx.edge(b, op, "child_of")
        ctx.edge(b, render_let_expr_stan(ctx, expr.right), "child_of")
        return b
    if isinstance(expr, LetExprUnaryOp):
        u = ctx.vertex(ctx.fresh("uop"), "unary_expression")
        op = ctx.vertex(ctx.fresh("op"), "operator")
        ctx.literal(op, "-")
        ctx.edge(u, op, "child_of")
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
    if isinstance(expr, (LetExprList, LetExprLambda, LetExprFactor,
                         LetExprMethodCall, LetExprString)):
        # Stan has no native list-literal / lambda / factor /
        # method-call syntax in the user-program surface; fall back
        # to a named placeholder so the structure is visible.
        v = ctx.vertex(ctx.fresh("vex"), "variable_expression")
        ident = ctx.vertex(ctx.fresh("id"), "identifier")
        ctx.literal(ident, "__placeholder__")
        ctx.edge(v, ident, "child_of")
        return v
    raise NotImplementedError(
        f"render_let_expr_stan: unhandled LetExprNode kind "
        f"{type(expr).__name__!r}"
    )


__all__ = ["render_let_expr_stan"]
