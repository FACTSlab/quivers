"""Render `LetExprNode` to Julia tree-sitter schema vertices."""

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


def render_let_expr_julia(ctx, expr: LetExprNode) -> str:
    """Build a Julia expression schema for `expr` in `ctx` (a `JlCtx`)."""
    if isinstance(expr, LetExprLiteral):
        if isinstance(expr.value, float) or "." in repr(expr.value):
            v = ctx.v(ctx.fresh("fl"), "float_literal")
        else:
            v = ctx.v(ctx.fresh("il"), "integer_literal")
        ctx.lit(v, str(expr.value))
        return v
    if isinstance(expr, LetExprVar):
        v = ctx.v(ctx.fresh("id"), "identifier")
        ctx.lit(v, expr.name)
        return v
    if isinstance(expr, LetExprBinOp):
        b = ctx.v(ctx.fresh("be"), "binary_expression")
        ctx.e(b, render_let_expr_julia(ctx, expr.left))
        op = ctx.v(ctx.fresh("op"), "operator")
        ctx.lit(op, expr.op)
        ctx.e(b, op)
        ctx.e(b, render_let_expr_julia(ctx, expr.right))
        return b
    if isinstance(expr, LetExprUnaryOp):
        u = ctx.v(ctx.fresh("ue"), "unary_expression")
        op = ctx.v(ctx.fresh("op"), "operator")
        ctx.lit(op, "-")
        ctx.e(u, op)
        ctx.e(u, render_let_expr_julia(ctx, expr.operand))
        return u
    if isinstance(expr, LetExprCall):
        c = ctx.v(ctx.fresh("ce"), "call_expression")
        fn = ctx.v(ctx.fresh("id"), "identifier")
        ctx.lit(fn, expr.func)
        ctx.e(c, fn)
        args = ctx.v(ctx.fresh("al"), "argument_list")
        for a in expr.args:
            ctx.e(args, render_let_expr_julia(ctx, a))
        ctx.e(c, args)
        return c
    if isinstance(expr, LetExprIndex):
        ix = ctx.v(ctx.fresh("ix"), "index_expression")
        ctx.e(ix, render_let_expr_julia(ctx, expr.array))
        for idx in expr.indices:
            ctx.e(ix, render_let_expr_julia(ctx, idx))
        return ix
    if isinstance(expr, (LetExprList, LetExprLambda, LetExprFactor,
                         LetExprMethodCall, LetExprString)):
        # Fall back to a named placeholder; richer constructs would
        # need dedicated emitters per Julia surface form.
        v = ctx.v(ctx.fresh("id"), "identifier")
        ctx.lit(v, "__placeholder__")
        return v
    raise NotImplementedError(
        f"render_let_expr_julia: unhandled LetExprNode kind "
        f"{type(expr).__name__!r}"
    )


__all__ = ["render_let_expr_julia"]
