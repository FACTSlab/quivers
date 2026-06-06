"""Render `LetExprNode` to BUGS/JAGS tree-sitter schema vertices.

BUGS / JAGS use infix arithmetic similar to Stan but with the
deterministic-assignment operator `<-` rather than `=`.
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


def render_let_expr_bugs(ctx, expr: LetExprNode) -> str:
    """Build a BUGS/JAGS expression schema for `expr` in `ctx`."""
    if isinstance(expr, LetExprLiteral):
        v = ctx.v(ctx.fresh("num"), "number")
        ctx.lit(v, str(expr.value))
        return v
    if isinstance(expr, LetExprVar):
        v = ctx.v(ctx.fresh("id"), "identifier")
        ctx.lit(v, expr.name)
        return v
    if isinstance(expr, LetExprBinOp):
        b = ctx.v(ctx.fresh("be"), "binary_expression")
        ctx.e(b, render_let_expr_bugs(ctx, expr.left))
        op = ctx.v(ctx.fresh("op"), "operator")
        ctx.lit(op, expr.op)
        ctx.e(b, op)
        ctx.e(b, render_let_expr_bugs(ctx, expr.right))
        return b
    if isinstance(expr, LetExprUnaryOp):
        u = ctx.v(ctx.fresh("ue"), "unary_expression")
        op = ctx.v(ctx.fresh("op"), "operator")
        ctx.lit(op, "-")
        ctx.e(u, op)
        ctx.e(u, render_let_expr_bugs(ctx, expr.operand))
        return u
    if isinstance(expr, LetExprCall):
        c = ctx.v(ctx.fresh("call"), "function_call")
        fn = ctx.v(ctx.fresh("id"), "identifier")
        ctx.lit(fn, expr.func)
        ctx.e(c, fn)
        for a in expr.args:
            ctx.e(c, render_let_expr_bugs(ctx, a))
        return c
    if isinstance(expr, LetExprIndex):
        s = ctx.v(ctx.fresh("ix"), "index_expression")
        ctx.e(s, render_let_expr_bugs(ctx, expr.array))
        for idx in expr.indices:
            ctx.e(s, render_let_expr_bugs(ctx, idx))
        return s
    if isinstance(expr, (LetExprList, LetExprLambda, LetExprFactor,
                         LetExprMethodCall, LetExprString)):
        v = ctx.v(ctx.fresh("id"), "identifier")
        ctx.lit(v, "__placeholder__")
        return v
    raise NotImplementedError(
        f"render_let_expr_bugs: unhandled LetExprNode kind "
        f"{type(expr).__name__!r}"
    )


__all__ = ["render_let_expr_bugs"]
