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

import json
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
        target: str = "python",
    ) -> None:
        self._sb = sb
        self._n = 0
        self.cards: dict[str, int] = dict(cards or {})
        # The concrete Python backend ("pyro" / "numpyro" / "pymc" /
        # "edward2"); selects the per-target builtin symbol table when a
        # ``LetExprCall`` is lowered.
        self.target = target
        # Imports a lowered ``LetExprCall`` symbol requires, as
        # ``(dotted-chain, alias)`` pairs. Only NumPyro emits an import
        # block, so only it drains this; the other backends assume their
        # framework runtime is already in scope.
        self.required_imports: set[tuple[tuple[str, ...], str]] = set()

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


def python_binary_op(
    ctx: PyCtx, op: str, left: str, right: str,
) -> str:
    """Build a ``binary_operator`` vertex carrying ``op`` as the
    field:operator constraint.

    The Python tree-sitter `binary_operator` has CHOICE alternatives
    per operator; both the field constraint and the alt fingerprint
    must be set so the pretty printer picks the right surface form.
    """
    b = ctx.v(ctx.fresh("bop"), "binary_operator")
    ctx.constraint(b, "field:operator", op)
    ctx.constraint(b, "chose-alt-fingerprint", op)
    ctx.e(b, left, "left")
    ctx.e(b, right, "right")
    return b


def python_unary_minus(ctx: PyCtx, operand: str) -> str:
    """Build a ``unary_operator`` vertex for ``-<operand>``."""
    u = ctx.v(ctx.fresh("uop"), "unary_operator")
    ctx.constraint(u, "field:operator", "-")
    ctx.constraint(u, "chose-alt-fingerprint", "-")
    ctx.e(u, operand, "argument")
    return u


def python_paren(ctx: PyCtx, inner: str) -> str:
    """Build a ``parenthesized_expression`` vertex around ``inner``.

    The Python pretty printer drops parens around nested
    binary_operator children, which scrambles operator precedence;
    callers that depend on explicit grouping wrap subexpressions in
    parenthesized_expression to force the printer to emit ``( ... )``.
    """
    p = ctx.v(ctx.fresh("paren"), "parenthesized_expression")
    ctx.constraint(p, "chose-alt-fingerprint", "( )")
    ctx.e(p, inner, "child_of")
    return p


def python_method_call(
    ctx: PyCtx,
    receiver: str,
    method: str,
    args: tuple[str, ...],
) -> str:
    """Build ``<receiver>.<method>(<args>)``."""
    a = ctx.v(ctx.fresh("attr"), "attribute")
    ctx.e(a, receiver, "object")
    method_id = identifier(ctx, method)
    ctx.e(a, method_id, "attribute")
    c = ctx.v(ctx.fresh("call"), "call")
    arglist = ctx.v(ctx.fresh("args"), "argument_list")
    ctx.e(c, a, "function")
    ctx.e(c, arglist, "arguments")
    for arg in args:
        ctx.e(arglist, arg, "child_of")
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


#: ``(dotted-import chain, alias)`` -> ``import <chain> as <alias>``.
#: Used only by the NumPyro renderer, whose ``jnp`` namespace lacks the
#: special functions and activations; ``torch`` / ``pymc`` / ``tensorflow``
#: expose theirs on the framework namespace the fragment already assumes.
_JAX_SPECIAL: tuple[tuple[str, ...], str] = (("jax", "scipy", "special"), "jsp")
_JAX_NN: tuple[tuple[str, ...], str] = (("jax", "nn"), "jnn")

#: A builtin's per-target lowering: the symbol path to emit as a dotted
#: attribute, plus the import it needs (``None`` when the symbol lives on
#: a namespace the target already has in scope).
_CallEntry = tuple[tuple[str, ...], "tuple[tuple[str, ...], str] | None"]


def _torch(name: str) -> _CallEntry:
    return (("torch", name), None)


def _torch_fn(name: str) -> _CallEntry:
    return (("torch", "nn", "functional", name), None)


def _jnp(name: str) -> _CallEntry:
    return (("jnp", name), None)


def _jsp(name: str) -> _CallEntry:
    return (("jsp", name), _JAX_SPECIAL)


def _jnn(name: str) -> _CallEntry:
    return (("jnn", name), _JAX_NN)


def _pmath(name: str) -> _CallEntry:
    return (("pymc", "math", name), None)


def _tfmath(name: str) -> _CallEntry:
    return (("tf", "math", name), None)


def _tfnn(name: str) -> _CallEntry:
    return (("tf", "nn", name), None)


#: Element-wise ``torch`` primitives that live at the top level with the
#: same name a user writes; verified against the installed ``torch``.
_TORCH_TOPLEVEL: tuple[str, ...] = (
    "exp", "expm1", "log", "log1p", "log2", "log10", "sqrt", "rsqrt",
    "square", "abs", "sign", "reciprocal", "sin", "cos", "tan", "asin",
    "acos", "atan", "sinh", "cosh", "asinh", "acosh", "atanh", "floor",
    "ceil", "round", "trunc", "erf", "erfc", "erfinv", "lgamma", "digamma",
    "tanh", "sigmoid", "neg",
)
#: Activations that live under ``torch.nn.functional``.
_TORCH_FUNCTIONAL: tuple[str, ...] = (
    "relu", "relu6", "elu", "selu", "gelu", "silu", "mish", "softplus",
    "logsigmoid", "softsign",
)

#: ``target -> {builtin name -> lowering}``. A builtin absent from a
#: target's table has no faithful single-call rendering there (e.g. it
#: needs a ``dim`` argument, or the library exposes no matching symbol),
#: so lowering it raises rather than emit an undefined name.
_LET_CALL_SYMBOLS: dict[str, dict[str, _CallEntry]] = {
    "pyro": {
        **{n: _torch(n) for n in _TORCH_TOPLEVEL},
        **{n: _torch_fn(n) for n in _TORCH_FUNCTIONAL},
    },
    "numpyro": {
        "exp": _jnp("exp"), "expm1": _jnp("expm1"), "log": _jnp("log"),
        "log1p": _jnp("log1p"), "log2": _jnp("log2"), "log10": _jnp("log10"),
        "sqrt": _jnp("sqrt"), "square": _jnp("square"), "abs": _jnp("abs"),
        "sign": _jnp("sign"), "reciprocal": _jnp("reciprocal"),
        "sin": _jnp("sin"), "cos": _jnp("cos"), "tan": _jnp("tan"),
        "asin": _jnp("arcsin"), "acos": _jnp("arccos"), "atan": _jnp("arctan"),
        "sinh": _jnp("sinh"), "cosh": _jnp("cosh"), "asinh": _jnp("arcsinh"),
        "acosh": _jnp("arccosh"), "atanh": _jnp("arctanh"),
        "floor": _jnp("floor"), "ceil": _jnp("ceil"), "round": _jnp("round"),
        "trunc": _jnp("trunc"), "tanh": _jnp("tanh"), "neg": _jnp("negative"),
        "erf": _jsp("erf"), "erfc": _jsp("erfc"), "erfinv": _jsp("erfinv"),
        "lgamma": _jsp("gammaln"), "digamma": _jsp("digamma"),
        "sigmoid": _jnn("sigmoid"), "relu": _jnn("relu"), "elu": _jnn("elu"),
        "selu": _jnn("selu"), "gelu": _jnn("gelu"), "silu": _jnn("silu"),
        "softplus": _jnn("softplus"), "logsigmoid": _jnn("log_sigmoid"),
        "softsign": _jnn("soft_sign"),
    },
    "pymc": {
        "exp": _pmath("exp"), "expm1": _pmath("expm1"), "log": _pmath("log"),
        "log1p": _pmath("log1p"), "log2": _pmath("log2"), "sqrt": _pmath("sqrt"),
        "abs": _pmath("abs"), "sin": _pmath("sin"), "cos": _pmath("cos"),
        "tan": _pmath("tan"), "sinh": _pmath("sinh"), "cosh": _pmath("cosh"),
        "tanh": _pmath("tanh"), "floor": _pmath("floor"), "ceil": _pmath("ceil"),
        "erf": _pmath("erf"), "erfc": _pmath("erfc"), "erfinv": _pmath("erfinv"),
        "sigmoid": _pmath("sigmoid"),
    },
    "edward2": {
        "exp": _tfmath("exp"), "expm1": _tfmath("expm1"), "log": _tfmath("log"),
        "log1p": _tfmath("log1p"), "sqrt": _tfmath("sqrt"), "rsqrt": _tfmath("rsqrt"),
        "square": _tfmath("square"), "abs": _tfmath("abs"), "sign": _tfmath("sign"),
        "reciprocal": _tfmath("reciprocal"), "sin": _tfmath("sin"),
        "cos": _tfmath("cos"), "tan": _tfmath("tan"), "asin": _tfmath("asin"),
        "acos": _tfmath("acos"), "atan": _tfmath("atan"), "sinh": _tfmath("sinh"),
        "cosh": _tfmath("cosh"), "asinh": _tfmath("asinh"), "acosh": _tfmath("acosh"),
        "atanh": _tfmath("atanh"), "floor": _tfmath("floor"), "ceil": _tfmath("ceil"),
        "round": _tfmath("round"), "tanh": _tfmath("tanh"), "sigmoid": _tfmath("sigmoid"),
        "erf": _tfmath("erf"), "erfc": _tfmath("erfc"), "erfinv": _tfmath("erfinv"),
        "lgamma": _tfmath("lgamma"), "digamma": _tfmath("digamma"),
        "neg": _tfmath("negative"), "logsigmoid": _tfmath("log_sigmoid"),
        "relu": _tfnn("relu"), "elu": _tfnn("elu"), "selu": _tfnn("selu"),
        "gelu": _tfnn("gelu"), "silu": _tfnn("silu"), "softplus": _tfnn("softplus"),
        "softsign": _tfnn("softsign"),
    },
}


#: Names of the tensor primitives a let-expression body may call, mirrored
#: from ``quivers.dsl.compiler.programs._LET_EXPR_BUILTINS`` (the native
#: torch dispatch table). Kept as a literal so this schema-building helper
#: stays decoupled from the torch execution path; a drift guard in the test
#: suite asserts it matches the compiler's table.
_MATH_BUILTIN_NAMES: frozenset[str] = frozenset({
    "relu", "relu6", "leaky_relu", "prelu", "rrelu", "elu", "selu", "celu",
    "gelu", "silu", "swish", "mish", "hardtanh", "hardshrink", "hardsigmoid",
    "hardswish", "softplus", "softshrink", "softsign", "softmax",
    "log_softmax", "softmin", "tanh", "tanhshrink", "sigmoid", "logsigmoid",
    "threshold", "glu", "normalize", "exp", "expm1", "log", "log1p", "log2",
    "log10", "sqrt", "rsqrt", "square", "abs", "neg", "sign", "reciprocal",
    "clamp", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh",
    "asinh", "acosh", "atanh", "floor", "ceil", "round", "trunc", "erf",
    "erfc", "erfinv", "lgamma", "digamma", "sum", "mean", "var", "std",
    "min", "max", "argmin", "argmax", "prod", "amax", "amin", "logsumexp",
    "norm", "cumsum", "cumprod", "cummax", "cummin", "flip", "sort",
    "dropout", "alpha_dropout", "layer_norm", "rms_norm",
})


def _resolve_python_call(ctx: PyCtx, func: str) -> str:
    """Emit the callee vertex for a ``LetExprCall`` to ``func``.

    When ``func`` is one of the target's mapped math builtins, emit its
    symbol as a dotted attribute (e.g. ``torch.erf``) and record any
    import the symbol needs on ``ctx.required_imports``. When ``func`` is
    a math builtin with no symbol for this target (e.g. a reduction that
    needs a ``dim`` argument), raise
    [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
    rather than emit an undefined name. Any other callee (a domain
    function such as a chart-parser ``parse``, or a user helper) is
    emitted verbatim, as before.
    """
    table = _LET_CALL_SYMBOLS.get(ctx.target, {})
    entry = table.get(func)
    if entry is not None:
        segments, import_spec = entry
        if import_spec is not None:
            ctx.required_imports.add(import_spec)
        if len(segments) == 1:
            return identifier(ctx, segments[0])
        return attribute(ctx, segments)
    if func in _MATH_BUILTIN_NAMES:
        raise UnsupportedConstruct(
            "qvr-python-helper",
            [
                f"let-expr:LetExprCall:{ctx.target}: builtin {func!r} has "
                f"no {ctx.target} symbol mapping; it cannot be rendered as "
                f"a single call in this target"
            ],
        )
    return identifier(ctx, func)


def _render_python_operand(ctx: PyCtx, expr: LetExprNode) -> str:
    """Render `expr` as an operand of a binary or unary operator.

    A nested [`LetExprBinOp`][quivers.dsl.ast_nodes.LetExprBinOp] or
    [`LetExprUnaryOp`][quivers.dsl.ast_nodes.LetExprUnaryOp] operand is
    wrapped in a `parenthesized_expression` via
    [`python_paren`][quivers.transpile.renderers._python_helpers.python_paren].
    The Python pretty printer drops parens around nested
    `binary_operator` children, so ``(a + b) * c`` would otherwise print
    as ``a + b * c`` and reassociate under Python's precedence. Wrapping
    every nested operator preserves the source grouping, matching the
    Stan and Julia renderers.
    """
    vid = render_let_expr_python(ctx, expr)
    if isinstance(expr, (LetExprBinOp, LetExprUnaryOp)):
        return python_paren(ctx, vid)
    return vid


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
        # Use json.dumps for a fully escaped Python string literal
        # (handles embedded quotes, backslashes, and control chars
        # without letting a payload break out of the quoted region).
        v = ctx.v(ctx.fresh("str"), "string")
        ctx.literal(v, json.dumps(expr.value))
        return v
    if isinstance(expr, LetExprBinOp):
        # Python tree-sitter's `binary_operator` has CHOICE alts per
        # operator (`+`, `-`, `*`, `**`, etc.). The walker discriminates
        # via the `field:operator` constraint on the binary_operator
        # vertex itself; no separate operator vertex is needed.
        b = ctx.v(ctx.fresh("bop"), "binary_operator")
        ctx.constraint(b, "field:operator", expr.op)
        ctx.constraint(b, "chose-alt-fingerprint", expr.op)
        ctx.e(b, _render_python_operand(ctx, expr.left), "left")
        ctx.e(b, _render_python_operand(ctx, expr.right), "right")
        return b
    if isinstance(expr, LetExprUnaryOp):
        u = ctx.v(ctx.fresh("uop"), "unary_operator")
        ctx.constraint(u, "field:operator", "-")
        ctx.constraint(u, "chose-alt-fingerprint", "-")
        ctx.e(u, _render_python_operand(ctx, expr.operand), "argument")
        return u
    if isinstance(expr, LetExprCall):
        fn = _resolve_python_call(ctx, expr.func)
        c = ctx.v(ctx.fresh("call"), "call")
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
    "python_binary_op",
    "python_method_call",
    "python_paren",
    "python_unary_minus",
    "render_let_expr_python",
    "shape_tuple",
    "string_literal",
    "with_statement",
]
