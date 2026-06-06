"""Render `LetExprNode` to JavaScript/WebPPL tree-sitter schema vertices."""

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


def render_let_expr_javascript(ctx, expr: LetExprNode) -> str:
    """Build a JavaScript expression schema for `expr` (in a webppl
    `_Ctx`)."""
    if isinstance(expr, LetExprLiteral):
        v = ctx.v(ctx.fresh("num"), "number")
        ctx.lit(v, str(expr.value))
        return v
    if isinstance(expr, LetExprVar):
        v = ctx.v(ctx.fresh("id"), "identifier")
        ctx.lit(v, expr.name)
        return v
    if isinstance(expr, LetExprString):
        v = ctx.v(ctx.fresh("str"), "string")
        ctx.lit(v, f'"{expr.value}"')
        return v
    if isinstance(expr, LetExprBinOp):
        b = ctx.v(ctx.fresh("bin"), "binary_expression")
        ctx.e(b, render_let_expr_javascript(ctx, expr.left), "left")
        op = ctx.v(ctx.fresh("op"), "operator")
        ctx.lit(op, expr.op)
        ctx.e(b, op, "operator")
        ctx.e(b, render_let_expr_javascript(ctx, expr.right), "right")
        return b
    if isinstance(expr, LetExprUnaryOp):
        u = ctx.v(ctx.fresh("uop"), "unary_expression")
        op = ctx.v(ctx.fresh("op"), "operator")
        ctx.lit(op, "-")
        ctx.e(u, op, "operator")
        ctx.e(u, render_let_expr_javascript(ctx, expr.operand), "argument")
        return u
    if isinstance(expr, LetExprCall):
        c = ctx.v(ctx.fresh("call"), "call_expression")
        fn = ctx.v(ctx.fresh("id"), "identifier")
        ctx.lit(fn, expr.func)
        ctx.e(c, fn, "function")
        args = ctx.v(ctx.fresh("args"), "arguments")
        for a in expr.args:
            ctx.e(args, render_let_expr_javascript(ctx, a), "child_of")
        ctx.e(c, args, "arguments")
        return c
    if isinstance(expr, LetExprIndex):
        s = ctx.v(ctx.fresh("subs"), "subscript_expression")
        ctx.e(s, render_let_expr_javascript(ctx, expr.array), "object")
        for idx in expr.indices:
            ctx.e(s, render_let_expr_javascript(ctx, idx), "index")
        return s
    if isinstance(expr, LetExprList):
        a = ctx.v(ctx.fresh("arr"), "array")
        for item in expr.items:
            ctx.e(a, render_let_expr_javascript(ctx, item), "child_of")
        return a
    if isinstance(expr, (LetExprLambda, LetExprFactor, LetExprMethodCall)):
        v = ctx.v(ctx.fresh("id"), "identifier")
        ctx.lit(v, "__placeholder__")
        return v
    raise NotImplementedError(
        f"render_let_expr_javascript: unhandled LetExprNode kind "
        f"{type(expr).__name__!r}"
    )


__all__ = ["render_let_expr_javascript"]
