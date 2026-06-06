"""Render `LetExprNode` to Python tree-sitter schema vertices.

Used by every Python-grammar backend (NumPyro / Pyro / PyMC /
Edward2) to emit `let <name> = <expr>` as a deterministic
`assignment` inside the model body.

The mapping is recursive on the `LetExprNode` discriminator. Each
case produces a single vertex (plus any descendants) representing the
expression in Python's `binary_operator` / `unary_operator` /
`call` / `subscript` / etc. node-types.
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


def render_let_expr_python(ctx, expr: LetExprNode) -> str:
    """Recursively build a Python expression schema for `expr` in
    `ctx` (a `PyCtx`). Returns the root vertex id."""
    if isinstance(expr, LetExprLiteral):
        v = ctx.v(ctx.fresh("lit"), "float" if "." in repr(expr.value) else "integer")
        ctx.literal(v, str(expr.value))
        return v
    if isinstance(expr, LetExprVar):
        v = ctx.v(ctx.fresh("id"), "identifier")
        ctx.literal(v, expr.name)
        return v
    if isinstance(expr, LetExprString):
        v = ctx.v(ctx.fresh("str"), "string")
        ctx.literal(v, f'"{expr.value}"')
        return v
    if isinstance(expr, LetExprBinOp):
        b = ctx.v(ctx.fresh("bop"), "binary_operator")
        ctx.e(b, render_let_expr_python(ctx, expr.left), "left")
        op = ctx.v(ctx.fresh("op"), "operator")
        ctx.literal(op, expr.op)
        ctx.e(b, op, "operator")
        ctx.e(b, render_let_expr_python(ctx, expr.right), "right")
        return b
    if isinstance(expr, LetExprUnaryOp):
        u = ctx.v(ctx.fresh("uop"), "unary_operator")
        op = ctx.v(ctx.fresh("op"), "operator")
        ctx.literal(op, "-")
        ctx.e(u, op, "operator")
        ctx.e(u, render_let_expr_python(ctx, expr.operand), "argument")
        return u
    if isinstance(expr, LetExprCall):
        c = ctx.v(ctx.fresh("call"), "call")
        fn = ctx.v(ctx.fresh("fn"), "identifier")
        ctx.literal(fn, expr.func)
        ctx.e(c, fn, "function")
        args = ctx.v(ctx.fresh("args"), "argument_list")
        for a in expr.args:
            ctx.e(args, render_let_expr_python(ctx, a), "child_of")
        ctx.e(c, args, "arguments")
        return c
    if isinstance(expr, LetExprIndex):
        s = ctx.v(ctx.fresh("subs"), "subscript")
        ctx.e(s, render_let_expr_python(ctx, expr.array), "value")
        for idx in expr.indices:
            ctx.e(s, render_let_expr_python(ctx, idx), "subscript")
        return s
    if isinstance(expr, LetExprList):
        lst = ctx.v(ctx.fresh("list"), "list")
        for item in expr.items:
            ctx.e(lst, render_let_expr_python(ctx, item), "child_of")
        return lst
    if isinstance(expr, LetExprLambda):
        lam = ctx.v(ctx.fresh("lam"), "lambda")
        params = ctx.v(ctx.fresh("ps"), "lambda_parameters")
        pid = ctx.v(ctx.fresh("p"), "identifier")
        ctx.literal(pid, expr.param)
        ctx.e(params, pid, "child_of")
        ctx.e(lam, params, "parameters")
        ctx.e(lam, render_let_expr_python(ctx, expr.body), "body")
        return lam
    if isinstance(expr, LetExprMethodCall):
        a = ctx.v(ctx.fresh("attr"), "attribute")
        ctx.e(a, render_let_expr_python(ctx, expr.receiver), "object")
        m = ctx.v(ctx.fresh("m"), "identifier")
        ctx.literal(m, expr.method)
        ctx.e(a, m, "attribute")
        c = ctx.v(ctx.fresh("call"), "call")
        ctx.e(c, a, "function")
        args = ctx.v(ctx.fresh("args"), "argument_list")
        for a_node in expr.args:
            ctx.e(args, render_let_expr_python(ctx, a_node), "child_of")
        ctx.e(c, args, "arguments")
        return c
    if isinstance(expr, LetExprFactor):
        # Render as a no-op identifier reference; factor expressions
        # need backend-specific tensor assembly that is out of scope
        # here. Surface the construct name for visibility.
        v = ctx.v(ctx.fresh("id"), "identifier")
        ctx.literal(v, "__factor__")
        return v
    raise NotImplementedError(
        f"render_let_expr_python: unhandled LetExprNode kind "
        f"{type(expr).__name__!r}"
    )


__all__ = ["render_let_expr_python"]
