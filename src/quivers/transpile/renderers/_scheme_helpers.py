"""Render `LetExprNode` to Scheme/Church tree-sitter schema vertices."""

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


def render_let_expr_scheme(ctx, expr: LetExprNode) -> str:
    """Build a Scheme expression schema for `expr`.

    Scheme is prefix: `a + b` becomes `(+ a b)`. Most expression forms
    map to a `list` vertex with the operator/function name as the
    first child and the arguments as subsequent children.
    """
    if isinstance(expr, LetExprLiteral):
        v = ctx.v(ctx.fresh("num"), "number")
        ctx.lit(v, str(expr.value))
        return v
    if isinstance(expr, LetExprVar):
        v = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(v, expr.name)
        return v
    if isinstance(expr, LetExprString):
        v = ctx.v(ctx.fresh("str"), "string")
        ctx.lit(v, f'"{expr.value}"')
        return v
    if isinstance(expr, LetExprBinOp):
        lst = ctx.v(ctx.fresh("list"), "list")
        op = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(op, expr.op)
        ctx.e(lst, op)
        ctx.e(lst, render_let_expr_scheme(ctx, expr.left))
        ctx.e(lst, render_let_expr_scheme(ctx, expr.right))
        return lst
    if isinstance(expr, LetExprUnaryOp):
        lst = ctx.v(ctx.fresh("list"), "list")
        op = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(op, "-")
        ctx.e(lst, op)
        ctx.e(lst, render_let_expr_scheme(ctx, expr.operand))
        return lst
    if isinstance(expr, LetExprCall):
        lst = ctx.v(ctx.fresh("list"), "list")
        fn = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(fn, expr.func)
        ctx.e(lst, fn)
        for a in expr.args:
            ctx.e(lst, render_let_expr_scheme(ctx, a))
        return lst
    if isinstance(expr, LetExprIndex):
        # (list-ref array index ...)
        lst = ctx.v(ctx.fresh("list"), "list")
        fn = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(fn, "list-ref")
        ctx.e(lst, fn)
        ctx.e(lst, render_let_expr_scheme(ctx, expr.array))
        for idx in expr.indices:
            ctx.e(lst, render_let_expr_scheme(ctx, idx))
        return lst
    if isinstance(expr, LetExprList):
        lst = ctx.v(ctx.fresh("list"), "list")
        fn = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(fn, "list")
        ctx.e(lst, fn)
        for item in expr.items:
            ctx.e(lst, render_let_expr_scheme(ctx, item))
        return lst
    if isinstance(expr, (LetExprLambda, LetExprFactor, LetExprMethodCall)):
        v = ctx.v(ctx.fresh("sym"), "symbol")
        ctx.lit(v, "__placeholder__")
        return v
    raise NotImplementedError(
        f"render_let_expr_scheme: unhandled LetExprNode kind "
        f"{type(expr).__name__!r}"
    )


__all__ = ["render_let_expr_scheme"]
