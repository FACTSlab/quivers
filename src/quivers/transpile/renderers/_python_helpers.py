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
from quivers.dsl.ast_nodes.let_expressions import LetFactorBinder
from quivers.dsl.ast_nodes.objects import TypeName
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.renderers._stan_helpers import (
    _substitute_let_expr,
)

if TYPE_CHECKING:
    import panproto


class PyCtx:
    """Owns a [`panproto.SchemaBuilder`][panproto.SchemaBuilder] plus
    a fresh-id counter and the per-render
    [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards] map.

    `cards` is consulted when unrolling
    [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor] binders;
    every backend that wires the IR-walk into this ctx is expected
    to pass `cards` at construction.
    """

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        cards: dict[str, int] | None = None,
    ) -> None:
        self._sb = sb
        self._n = 0
        self.cards: dict[str, int] = dict(cards or {})

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
        return _render_factor_python(ctx, expr)
    raise UnsupportedConstruct(
        "qvr-python-helper",
        [
            f"let-expr:{type(expr).__name__}: unhandled node kind"
        ],
    )


def _render_factor_python(ctx: PyCtx, expr: LetExprFactor) -> str:
    """Unroll a `LetExprFactor` into a Python list literal.

    Cases form (single binder, body=None): emit `[case_0, case_1,
    ...]` in label order. Uniform-body form (one or more binders):
    substitute each binder for its 0-indexed integer value (Python
    arrays are 0-based, matching QVR's surface convention) and
    unroll into a nested list literal of shape
    (|b0|, |b1|, ..., |bn-1|). The shared
    [`_substitute_let_expr`][quivers.transpile.renderers._stan_helpers._substitute_let_expr]
    walk takes the same value for both `index_value` and
    `scalar_value` because no index-base shift is needed here.
    """
    if expr.cases and expr.body is None:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                "qvr-python-helper",
                [
                    "let-expr:LetExprFactor: cases form requires "
                    f"exactly one binder; got {len(expr.binders)}"
                ],
            )
        ordered = sorted(expr.cases, key=lambda c: c.label)
        items = tuple(
            render_let_expr_python(ctx, c.value) for c in ordered
        )
        return _emit_python_list(ctx, items)
    if expr.body is not None and not expr.cases:
        sizes = tuple(_card_for(ctx, b) for b in expr.binders)
        return _build_nested_python(
            ctx, expr.binders, sizes, expr.body, ()
        )
    raise UnsupportedConstruct(
        "qvr-python-helper",
        [
            "let-expr:LetExprFactor: mixed cases-plus-body form is "
            "not a valid surface construct"
        ],
    )


def _emit_python_list(ctx: PyCtx, items: tuple[str, ...]) -> str:
    """Emit a Python list literal `[e0, e1, ...]`."""
    vid = ctx.v(ctx.fresh("list"), "list")
    for item in items:
        ctx.e(vid, item, "child_of")
    return vid


def _card_for(ctx: PyCtx, binder: LetFactorBinder) -> int:
    """Resolve the static cardinality of `binder.index` via the
    `PyCtx.cards` snapshot of `IRProgram.cards`."""
    idx = binder.index
    if isinstance(idx, TypeName):
        size = ctx.cards.get(idx.name)
        if size is None:
            raise UnsupportedConstruct(
                "qvr-python-helper",
                [
                    f"let-expr:LetExprFactor: binder {binder.var!r} "
                    f"references object {idx.name!r} whose cardinality "
                    "is unknown at render time"
                ],
            )
        return size
    raise UnsupportedConstruct(
        "qvr-python-helper",
        [
            f"let-expr:LetExprFactor: binder {binder.var!r} index is "
            f"{type(idx).__name__}; only TypeName binders unroll"
        ],
    )


def _build_nested_python(
    ctx: PyCtx,
    binders: tuple[LetFactorBinder, ...],
    sizes: tuple[int, ...],
    body: LetExprNode,
    fixed: tuple[int, ...],
) -> str:
    """Recursively materialise the nested Python list tower for the
    uniform-body factor form."""
    if len(fixed) == len(binders):
        subst = body
        for binder, value in zip(binders, fixed, strict=True):
            literal = LetExprLiteral(value=value)
            subst = _substitute_let_expr(
                subst,
                binder.var,
                index_value=literal,
                scalar_value=literal,
            )
        return render_let_expr_python(ctx, subst)
    level = len(fixed)
    items = tuple(
        _build_nested_python(
            ctx, binders, sizes, body, fixed + (i,)
        )
        for i in range(sizes[level])
    )
    return _emit_python_list(ctx, items)


def shape_tuple(ctx: PyCtx, shape: tuple[int, ...]) -> str:
    """Build a Python ``tuple`` node from an integer shape.

    Emits ``()`` for an empty shape, ``(<n>,)`` for a singleton (with
    the required trailing comma), and ``(<r>, <c>, ...)`` for higher
    arity. Tree-sitter Python's `tuple` production needs an explicit
    `ptrace-*` punctuation trace to render the comma; without it the
    emitter drops the comma and produces a `parenthesized_expression`.
    """
    tup = ctx.v(ctx.fresh("tup"), "tuple")
    n = len(shape)
    if n == 0:
        ctx.constraint(tup, "chose-alt-fingerprint", "()")
        ctx.constraint(tup, "ptrace-0", "T(")
        ctx.constraint(tup, "ptrace-1", "T)")
        return tup
    kind_list = " ".join("integer" for _ in range(n))
    if n == 1:
        ctx.constraint(tup, "chose-alt-fingerprint", "( ,)")
        ctx.constraint(tup, "ptrace-0", "T(")
        ctx.constraint(tup, "ptrace-1", "Cinteger")
        ctx.constraint(tup, "ptrace-2", "T,")
        ctx.constraint(tup, "ptrace-3", "T)")
    else:
        fingerprint = "( " + " ".join("," for _ in range(n - 1)) + " )"
        ctx.constraint(tup, "chose-alt-fingerprint", fingerprint)
        ctx.constraint(tup, "ptrace-0", "T(")
        slot = 1
        for i in range(n):
            ctx.constraint(tup, f"ptrace-{slot}", "Cinteger")
            slot += 1
            if i < n - 1:
                ctx.constraint(tup, f"ptrace-{slot}", "T,")
                slot += 1
        ctx.constraint(tup, f"ptrace-{slot}", "T)")
    ctx.constraint(tup, "chose-alt-child-kinds", kind_list)
    for size in shape:
        ctx.e(tup, number_literal(ctx, size), "child_of")
    return tup


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
    "shape_tuple",
    "string_literal",
    "with_statement",
]
