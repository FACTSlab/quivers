"""Shared SchemaBuilder helpers for the Python tree-sitter grammar.

Used by every renderer whose target is Python source (NumPyro, Pyro,
PyMC, Edward2). Each helper takes a builder context and produces a
vertex id for the constructed sub-AST. Vertex kinds match Python's
tree-sitter `node-types.json` exactly.

This module also exposes
[`render_let_expr_python`][quivers.transpile.renderers._python_helpers.render_let_expr_python]
which lowers a [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode]
sub-tree into the same Python schema, used by every Python-grammar
renderer to emit `let <name> = <expr>` as a deterministic
`assignment` inside the model body.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    import panproto


class PyCtx:
    """Owns a [`panproto.SchemaBuilder`][panproto.SchemaBuilder] plus
    a fresh-id counter."""

    def __init__(self, sb: panproto.SchemaBuilder) -> None:
        self._sb = sb
        self._n = 0

    def fresh(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def v(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str) -> None:
        self._sb.edge(src, tgt, kind)

    def literal(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)


def identifier(ctx: PyCtx, text: str) -> str:
    """Emit an ``identifier`` vertex carrying ``text``."""
    vid = ctx.v(ctx.fresh("id"), "identifier")
    ctx.literal(vid, text)
    return vid


def string_literal(ctx: PyCtx, text: str) -> str:
    """Emit a double-quoted ``string`` vertex with ``text`` as its content."""
    s = ctx.v(ctx.fresh("s"), "string")
    start = ctx.v(ctx.fresh("ss"), "string_start")
    ctx.literal(start, '"')
    content = ctx.v(ctx.fresh("sc"), "string_content")
    ctx.literal(content, text)
    end = ctx.v(ctx.fresh("se"), "string_end")
    ctx.literal(end, '"')
    ctx.e(s, start, "child_of")
    ctx.e(s, content, "child_of")
    ctx.e(s, end, "child_of")
    return s


def number_literal(ctx: PyCtx, value: float) -> str:
    """Emit ``integer`` if ``value`` is a whole number, else ``float``."""
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        vid = ctx.v(ctx.fresh("int"), "integer")
        ctx.literal(vid, str(int(value)))
    else:
        vid = ctx.v(ctx.fresh("flt"), "float")
        ctx.literal(vid, repr(float(value)))
    return vid


def arg_expr(ctx: PyCtx, raw: str | float) -> str:
    """Build an expression vertex for a sample argument."""
    if isinstance(raw, str):
        return identifier(ctx, raw)
    return number_literal(ctx, raw)


def attribute(ctx: PyCtx, chain: tuple[str, ...]) -> str:
    """Build a left-recursive ``a.b.c.d`` attribute access.

    Tree-sitter Python represents `a.b.c` as ``attribute(object:
    attribute(object: id 'a', attribute: 'b'), attribute: 'c')``.
    """
    if len(chain) < 2:
        msg = f"attribute needs at least 2 names; got {chain!r}"
        raise ValueError(msg)
    current = identifier(ctx, chain[0])
    for attr_name in chain[1:]:
        attr = ctx.v(ctx.fresh("attr"), "attribute")
        attr_id = identifier(ctx, attr_name)
        ctx.e(attr, current, "object")
        ctx.e(attr, attr_id, "attribute")
        current = attr
    return current


def call(
    ctx: PyCtx,
    function: str,
    *,
    positional: tuple[str, ...] = (),
    keyword: tuple[tuple[str, str], ...] = (),
) -> str:
    """Build a ``call`` vertex with positional and keyword args."""
    c = ctx.v(ctx.fresh("call"), "call")
    args = ctx.v(ctx.fresh("args"), "argument_list")
    ctx.e(c, function, "function")
    ctx.e(c, args, "arguments")
    for pid in positional:
        ctx.e(args, pid, "child_of")
    for name, vid in keyword:
        kw = ctx.v(ctx.fresh("kw"), "keyword_argument")
        kw_name = identifier(ctx, name)
        ctx.e(kw, kw_name, "name")
        ctx.e(kw, vid, "value")
        ctx.e(args, kw, "child_of")
    return c


def assignment(ctx: PyCtx, *, lhs_name: str, rhs: str) -> str:
    """Build ``<lhs_name> = <rhs>``."""
    asn = ctx.v(ctx.fresh("asn"), "assignment")
    lhs = identifier(ctx, lhs_name)
    ctx.e(asn, lhs, "left")
    ctx.e(asn, rhs, "right")
    return asn


def function_def(
    ctx: PyCtx,
    *,
    name: str,
    default_params: tuple[str, ...],
    body_vid: str,
) -> str:
    """Build ``def <name>(<p1>=None, <p2>=None, ...): <body>``."""
    func = ctx.v(ctx.fresh("fn"), "function_definition")
    fname = identifier(ctx, name)
    params = ctx.v(ctx.fresh("ps"), "parameters")
    ctx.e(func, fname, "name")
    ctx.e(func, params, "parameters")
    ctx.e(func, body_vid, "body")
    for pname in default_params:
        dp = ctx.v(ctx.fresh("dp"), "default_parameter")
        dp_name = identifier(ctx, pname)
        dp_val = ctx.v(ctx.fresh("none"), "none")
        ctx.literal(dp_val, "None")
        ctx.e(dp, dp_name, "name")
        ctx.e(dp, dp_val, "value")
        ctx.e(params, dp, "child_of")
    return func


def with_statement(
    ctx: PyCtx,
    *,
    expression: str,
    alias: str | None,
    body_vid: str,
) -> str:
    """Build ``with <expression> [as <alias>]: <body>``.

    Tree-sitter Python's shape for ``with X as Y: body`` is:

    ```text
    with_statement
      with_clause
        with_item   (value field -> either expression, or as_pattern wrapping it)
      body field    -> block
    ```

    When ``alias`` is set, the ``with_item``'s value field points at
    the as_pattern (which itself owns the expression via its child_of
    edge and the target via its alias field). Crucially, the
    expression is referenced exactly ONCE in the schema graph; routing
    it under both the with_item's value field and the as_pattern's
    child_of edge would cause `emit_pretty` to traverse it twice and
    emit the call twice (producing
    ``with pymc.Model() pymc.Model() as: ...``).
    """
    ws = ctx.v(ctx.fresh("with"), "with_statement")
    clause = ctx.v(ctx.fresh("wc"), "with_clause")
    item = ctx.v(ctx.fresh("wi"), "with_item")
    if alias is not None:
        as_pat = ctx.v(ctx.fresh("asp"), "as_pattern")
        target = ctx.v(ctx.fresh("astgt"), "as_pattern_target")
        ctx.e(target, identifier(ctx, alias), "child_of")
        ctx.e(as_pat, expression, "child_of")
        ctx.e(as_pat, target, "alias")
        ctx.e(item, as_pat, "value")
    else:
        ctx.e(item, expression, "value")
    ctx.e(clause, item, "child_of")
    ctx.e(ws, clause, "child_of")
    ctx.e(ws, body_vid, "body")
    return ws


def render_let_expr_python(ctx: PyCtx, expr: LetExprNode) -> str:
    """Recursively build a Python expression schema for `expr` in
    `ctx` (a [`PyCtx`][quivers.transpile.renderers._python_helpers.PyCtx]).
    Returns the root vertex id."""
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
        # Python tree-sitter's `binary_operator` has CHOICE alts per
        # operator (`+`, `-`, `*`, `**`, etc.). The walker discriminates
        # via the `field:operator` constraint on the binary_operator
        # vertex itself; no separate operator vertex is needed.
        b = ctx.v(ctx.fresh("bop"), "binary_operator")
        ctx.constraint(b, "field:operator", expr.op)
        ctx.constraint(b, "chose-alt-fingerprint", expr.op)
        ctx.e(b, render_let_expr_python(ctx, expr.left), "left")
        ctx.e(b, render_let_expr_python(ctx, expr.right), "right")
        return b
    if isinstance(expr, LetExprUnaryOp):
        u = ctx.v(ctx.fresh("uop"), "unary_operator")
        ctx.constraint(u, "field:operator", "-")
        ctx.constraint(u, "chose-alt-fingerprint", "-")
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


__all__ = [
    "PyCtx",
    "arg_expr",
    "assignment",
    "attribute",
    "call",
    "function_def",
    "identifier",
    "number_literal",
    "render_let_expr_python",
    "string_literal",
    "with_statement",
]
