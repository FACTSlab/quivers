"""Render [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode] subtrees
to Julia tree-sitter schema vertices.

The Julia grammar exposes the following expression-level vertex kinds
the helper builds:

* `integer_literal` / `float_literal` for numeric leaves
* `string_literal` wrapping a `content` child for double-quoted strings
* `identifier` for variable references
* `binary_expression` (per-operator alts via the `operator` child's
  `chose-alt-fingerprint`)
* `unary_expression` for the unary minus
* `call_expression` whose children are `identifier argument_list`
* `field_expression` for the `receiver.method` access path used to
  encode method-dispatch chains (Julia is multiple-dispatch; the
  conventional surface for `chart.goal_weight()` is
  `goal_weight(chart)`, but a parse-roundtrip helper must preserve the
  source syntax; the helper renders the dot form to keep the
  user-written shape)
* `index_expression` whose children are `identifier vector_expression`
  where the inner `vector_expression` carries the comma-separated
  index expressions
* `vector_expression` for `[a, b, c]` list literals
* `arrow_function_expression` for `param -> body` (single-param uses an
  `identifier` for the param; multi-param uses an `argument_list`)

Every vertex sets `chose-alt-child-kinds` to the space-separated
sequence of its children's surface kinds, and (where the grammar
distinguishes alternatives by punctuation) sets
`chose-alt-fingerprint` to the canonical punctuation skeleton. The
panproto pretty-printer consults both slots to pick the right grammar
production; missing constraints cause silent emission gaps. The helper
returns `(vertex_id, vertex_kind)` from each recursive call so parents
can build the `chose-alt-child-kinds` string from real child kinds.

`LetExprFactor` is unrolled at render time:

* the **cases form** (binders contain a single axis, body is `None`,
  cases enumerate labels in `[0, |axis|)`) becomes a
  `vector_expression` whose children are the case bodies in label
  order;
* the **uniform-body form** (one or more binders, body is the repeated
  expression, cases is empty) becomes a tower of `vector_expression`
  vertices of shape `(|b0|, |b1|, ..., |bn-1|)`, with each binder
  substituted for its 1-indexed integer value (Julia arrays are
  1-indexed, matching QVR's surface indexing).

Static cardinalities for binder axes come from the ctx's `cards` map
(populated from [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]).
When an axis size is missing, the helper raises
[`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
tagged with the ctx's `target` ("turing" / "gen") rather than emitting
a placeholder.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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
from quivers.dsl.ast_nodes.objects import (
    DiscreteConstructor,
    ObjectExpr,
    TypeName,
)
from quivers.transpile._api import UnsupportedConstruct


@runtime_checkable
class _JlLetCtx(Protocol):
    """Structural protocol for the helper's ctx parameter.

    `target` tags the error messages with the backend name
    (``"turing"`` / ``"gen"``); `cards` is the static axis-size table
    sourced from [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]
    and consulted when unrolling a
    [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor].
    """

    target: str
    cards: dict[str, int]

    def fresh(self, prefix: str) -> str: ...
    def v(self, vid: str, kind: str) -> str: ...
    def e(self, src: str, tgt: str, kind: str = ...) -> None: ...
    def lit(self, vid: str, text: str) -> None: ...
    def constraint(self, vid: str, sort: str, value: str) -> None: ...


def render_let_expr_julia(ctx: _JlLetCtx, expr: LetExprNode) -> str:
    """Build a Julia expression schema for ``expr`` in ``ctx``.

    Returns the root vertex id. Wraps `_render` to discard the kind
    return value at the public boundary so callers see the same
    signature as the other per-target helpers.
    """
    vid, _kind = _render(ctx, expr)
    return vid


def _render(ctx: _JlLetCtx, expr: LetExprNode) -> tuple[str, str]:
    """Recursive renderer returning ``(vertex_id, vertex_kind)`` so
    parents can populate ``chose-alt-child-kinds`` accurately."""
    if isinstance(expr, LetExprLiteral):
        return _emit_literal(ctx, expr.value)
    if isinstance(expr, LetExprString):
        return _emit_string(ctx, expr.value)
    if isinstance(expr, LetExprVar):
        return _emit_identifier(ctx, expr.name)
    if isinstance(expr, LetExprBinOp):
        return _emit_binop(ctx, expr)
    if isinstance(expr, LetExprUnaryOp):
        return _emit_unary(ctx, expr)
    if isinstance(expr, LetExprCall):
        return _emit_call(ctx, expr.func, expr.args)
    if isinstance(expr, LetExprIndex):
        return _emit_index(ctx, expr)
    if isinstance(expr, LetExprList):
        return _emit_vector(
            ctx, tuple(_render(ctx, item) for item in expr.items)
        )
    if isinstance(expr, LetExprLambda):
        return _emit_lambda(ctx, expr)
    if isinstance(expr, LetExprMethodCall):
        return _emit_method_call(ctx, expr)
    if isinstance(expr, LetExprFactor):
        return _render_factor(ctx, expr)
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [f"let-expr:{type(expr).__name__}: unhandled node kind"],
    )


# ---------------------------------------------------------------------------
# Per-kind emitters.
# ---------------------------------------------------------------------------


def _emit_literal(ctx: _JlLetCtx, value: object) -> tuple[str, str]:
    """Emit `integer_literal` for whole numbers, `float_literal` otherwise.

    Whole-number floats (`1.0`, `2.0`) emit as `integer_literal` so
    that array indices substituted from factor binders satisfy Julia's
    `arr[Int]` typing rule.
    """
    if isinstance(value, bool):
        # `bool` is a subclass of `int` in Python; treat as integer.
        vid = ctx.v(ctx.fresh("il"), "integer_literal")
        ctx.lit(vid, str(int(value)))
        ctx.constraint(vid, "chose-alt-fingerprint", str(int(value)))
        return vid, "integer_literal"
    if isinstance(value, int):
        vid = ctx.v(ctx.fresh("il"), "integer_literal")
        ctx.lit(vid, str(value))
        ctx.constraint(vid, "chose-alt-fingerprint", str(value))
        return vid, "integer_literal"
    if isinstance(value, float) and value.is_integer():
        vid = ctx.v(ctx.fresh("il"), "integer_literal")
        ctx.lit(vid, str(int(value)))
        ctx.constraint(vid, "chose-alt-fingerprint", str(int(value)))
        return vid, "integer_literal"
    if isinstance(value, float):
        text = repr(value)
        vid = ctx.v(ctx.fresh("fl"), "float_literal")
        ctx.lit(vid, text)
        ctx.constraint(vid, "chose-alt-fingerprint", text)
        return vid, "float_literal"
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [
            f"let-expr:LetExprLiteral:{_target(ctx)}: literal value "
            f"{value!r} of type {type(value).__name__} has no Julia "
            f"numeric literal representation"
        ],
    )


def _emit_string(ctx: _JlLetCtx, text: str) -> tuple[str, str]:
    """Emit a `string_literal` whose single `content` child carries
    the unescaped text body.

    The Julia tree-sitter grammar models a double-quoted string as
    `string_literal -> '"' content '"'`; the fingerprint
    ``'" "'`` selects the double-quoted alternative and the
    `content` vertex's `literal-value` slot holds the text body.
    """
    vid = ctx.v(ctx.fresh("sl"), "string_literal")
    ctx.constraint(vid, "chose-alt-fingerprint", '" "')
    ctx.constraint(vid, "chose-alt-child-kinds", "content")
    content = ctx.v(ctx.fresh("sc"), "content")
    ctx.lit(content, text)
    ctx.constraint(content, "chose-alt-fingerprint", text)
    ctx.e(vid, content, "child_of")
    return vid, "string_literal"


def _emit_identifier(ctx: _JlLetCtx, name: str) -> tuple[str, str]:
    """Emit a bare `identifier` vertex carrying ``name``."""
    vid = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(vid, name)
    ctx.constraint(vid, "chose-alt-fingerprint", name)
    return vid, "identifier"


def _emit_operator(ctx: _JlLetCtx, op: str) -> tuple[str, str]:
    """Emit an `operator` vertex whose literal text is ``op``."""
    vid = ctx.v(ctx.fresh("op"), "operator")
    ctx.lit(vid, op)
    ctx.constraint(vid, "chose-alt-fingerprint", op)
    return vid, "operator"


def _emit_binop(ctx: _JlLetCtx, expr: LetExprBinOp) -> tuple[str, str]:
    """Emit a `binary_expression` whose children are
    ``<left> operator <right>``."""
    left_vid, left_kind = _render(ctx, expr.left)
    op_vid, _op_kind = _emit_operator(ctx, expr.op)
    right_vid, right_kind = _render(ctx, expr.right)
    vid = ctx.v(ctx.fresh("be"), "binary_expression")
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        f"{left_kind} operator {right_kind}",
    )
    ctx.e(vid, left_vid, "child_of")
    ctx.e(vid, op_vid, "child_of")
    ctx.e(vid, right_vid, "child_of")
    return vid, "binary_expression"


def _emit_unary(ctx: _JlLetCtx, expr: LetExprUnaryOp) -> tuple[str, str]:
    """Emit a `unary_expression` whose children are ``operator <operand>``.

    [`LetExprUnaryOp`][quivers.dsl.ast_nodes.LetExprUnaryOp] only
    encodes unary minus (the parser does not produce other unary
    operators in let expressions), so the helper emits a literal
    ``-`` operator.
    """
    op_vid, _op_kind = _emit_operator(ctx, "-")
    operand_vid, operand_kind = _render(ctx, expr.operand)
    vid = ctx.v(ctx.fresh("ue"), "unary_expression")
    ctx.constraint(
        vid, "chose-alt-child-kinds", f"operator {operand_kind}"
    )
    ctx.e(vid, op_vid, "child_of")
    ctx.e(vid, operand_vid, "child_of")
    return vid, "unary_expression"


def _emit_call(
    ctx: _JlLetCtx, func: str, args: tuple[LetExprNode, ...]
) -> tuple[str, str]:
    """Emit ``<func>(<arg_0>, <arg_1>, ...)`` as a `call_expression`
    whose children are ``identifier argument_list``."""
    rendered = tuple(_render(ctx, a) for a in args)
    callee_vid, _callee_kind = _emit_identifier(ctx, func)
    al_vid = _emit_argument_list(ctx, rendered)
    vid = ctx.v(ctx.fresh("ce"), "call_expression")
    ctx.constraint(
        vid, "chose-alt-child-kinds", "identifier argument_list"
    )
    ctx.e(vid, callee_vid, "child_of")
    ctx.e(vid, al_vid, "child_of")
    return vid, "call_expression"


def _emit_argument_list(
    ctx: _JlLetCtx, rendered: tuple[tuple[str, str], ...]
) -> str:
    """Emit an `argument_list` with the right comma fingerprint.

    Julia's `argument_list` fingerprint is ``( )`` for zero args,
    ``( , )`` for one arg, ``( , , )`` for two args, etc. (one comma
    per argument including the closing one); the panproto pretty
    printer reads the fingerprint to pick the right grammar
    alternative.
    """
    vid = ctx.v(ctx.fresh("al"), "argument_list")
    n = len(rendered)
    if n == 0:
        ctx.constraint(vid, "chose-alt-fingerprint", "()")
        ctx.lit(vid, "()")
        return vid
    fingerprint = "( " + " ".join("," for _ in range(n)) + " )"
    ctx.constraint(vid, "chose-alt-fingerprint", fingerprint)
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        " ".join(kind for _vid, kind in rendered),
    )
    for child_vid, _kind in rendered:
        ctx.e(vid, child_vid, "child_of")
    return vid


def _emit_index(ctx: _JlLetCtx, expr: LetExprIndex) -> tuple[str, str]:
    """Emit `arr[i0, i1, ...]` as an `index_expression` whose children
    are ``<arr> <vector_expression of indices>``.

    Julia's tree-sitter grammar represents subscripting as
    `arr[i,j]` -> `index_expression(arr, vector_expression(i, j))`
    where the inner `vector_expression` carries the comma-separated
    index list. Emitting the indices as direct children of the
    `index_expression` (without the `vector_expression` wrapper)
    causes the pretty-printer to silently drop them.
    """
    arr_vid, arr_kind = _render(ctx, expr.array)
    inner_rendered = tuple(_render(ctx, i) for i in expr.indices)
    inner_vid, _inner_kind = _emit_vector(ctx, inner_rendered)
    vid = ctx.v(ctx.fresh("ix"), "index_expression")
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        f"{arr_kind} vector_expression",
    )
    ctx.e(vid, arr_vid, "child_of")
    ctx.e(vid, inner_vid, "child_of")
    return vid, "index_expression"


def _emit_vector(
    ctx: _JlLetCtx, rendered: tuple[tuple[str, str], ...]
) -> tuple[str, str]:
    """Emit a `vector_expression` ``[e0, e1, ...]`` list literal.

    Fingerprint is ``[ ]`` for empty, ``[ , ]`` for one element,
    ``[ , , ]`` for two elements, etc. (one comma per element
    including the closing one).
    """
    vid = ctx.v(ctx.fresh("ve"), "vector_expression")
    n = len(rendered)
    if n == 0:
        ctx.constraint(vid, "chose-alt-fingerprint", "[ ]")
        return vid, "vector_expression"
    fingerprint = "[ " + " ".join("," for _ in range(n)) + " ]"
    ctx.constraint(vid, "chose-alt-fingerprint", fingerprint)
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        " ".join(kind for _vid, kind in rendered),
    )
    for child_vid, _kind in rendered:
        ctx.e(vid, child_vid, "child_of")
    return vid, "vector_expression"


def _emit_lambda(
    ctx: _JlLetCtx, expr: LetExprLambda
) -> tuple[str, str]:
    """Emit ``param -> body`` as an `arrow_function_expression`.

    The Julia grammar's single-parameter form uses a bare `identifier`
    for the parameter; the multi-parameter form wraps the parameter
    list in an `argument_list`. [`LetExprLambda`][quivers.dsl.ast_nodes.LetExprLambda]
    only carries a single `param` field, so the single-parameter form
    is the only shape the helper emits.
    """
    param_vid, param_kind = _emit_identifier(ctx, expr.param)
    body_vid, body_kind = _render(ctx, expr.body)
    vid = ctx.v(ctx.fresh("af"), "arrow_function_expression")
    ctx.constraint(vid, "chose-alt-fingerprint", "->")
    ctx.constraint(
        vid, "chose-alt-child-kinds", f"{param_kind} {body_kind}"
    )
    ctx.e(vid, param_vid, "child_of")
    ctx.e(vid, body_vid, "child_of")
    return vid, "arrow_function_expression"


def _emit_method_call(
    ctx: _JlLetCtx, expr: LetExprMethodCall
) -> tuple[str, str]:
    """Emit ``receiver.method(args...)`` as a `call_expression` whose
    callee is a `field_expression`.

    Julia is multiple-dispatch; the dot-syntax form ``a.b(c, d)``
    parses as a `call_expression` with a `field_expression` callee.
    The helper preserves the source surface (rather than rewriting to
    the equivalent ``b(a, c, d)`` Julia function-call form) so that
    format-preserving round-trips of QVR programs reach the same
    Julia surface they were authored in.
    """
    receiver_vid, receiver_kind = _render(ctx, expr.receiver)
    method_vid, _method_kind = _emit_identifier(ctx, expr.method)
    field_vid = ctx.v(ctx.fresh("fe"), "field_expression")
    ctx.constraint(field_vid, "chose-alt-fingerprint", ".")
    ctx.constraint(
        field_vid,
        "chose-alt-child-kinds",
        f"{receiver_kind} identifier",
    )
    ctx.e(field_vid, receiver_vid, "child_of")
    ctx.e(field_vid, method_vid, "child_of")
    rendered = tuple(_render(ctx, a) for a in expr.args)
    al_vid = _emit_argument_list(ctx, rendered)
    vid = ctx.v(ctx.fresh("ce"), "call_expression")
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        "field_expression argument_list",
    )
    ctx.e(vid, field_vid, "child_of")
    ctx.e(vid, al_vid, "child_of")
    return vid, "call_expression"


def _render_factor(
    ctx: _JlLetCtx, expr: LetExprFactor
) -> tuple[str, str]:
    """Unroll a `LetExprFactor` into nested `vector_expression` vertices.

    The cases form (binders contain a single axis, body is `None`,
    cases enumerate labels in `[0, |axis|)`) emits a
    `vector_expression` whose children are each case's body in label
    order.

    The uniform-body form (one or more binders, body is the repeated
    expression, cases is empty) emits a tower of `vector_expression`
    vertices of shape `(|b0|, |b1|, ..., |bn-1|)`, with each binder
    substituted for its 1-indexed integer value (Julia arrays are
    1-indexed).
    """
    if not expr.binders:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: empty binder "
                f"list is structurally ill-formed"
            ],
        )
    if expr.cases and expr.body is None:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                f"qvr-{_target(ctx)}-helper",
                [
                    f"let-expr:LetExprFactor:{_target(ctx)}: cases form "
                    f"requires exactly one binder; got "
                    f"{len(expr.binders)}"
                ],
            )
        size = _factor_axis_size(ctx, expr.binders[0])
        return _emit_factor_cases(ctx, expr.binders[0], size, expr.cases)
    if expr.body is not None and not expr.cases:
        sizes = tuple(_factor_axis_size(ctx, b) for b in expr.binders)
        return _build_nested_vector(
            ctx, expr.binders, sizes, expr.body, ()
        )
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [
            f"let-expr:LetExprFactor:{_target(ctx)}: mixed "
            f"cases-plus-body form is not a valid surface construct"
        ],
    )


def _emit_factor_cases(
    ctx: _JlLetCtx,
    binder: LetFactorBinder,
    size: int,
    cases: tuple[LetFactorCase, ...],
) -> tuple[str, str]:
    """Build `[<case[0].value>, ..., <case[size-1].value>]` from the
    label-keyed case list."""
    by_label = {c.label: c.value for c in cases}
    missing = sorted(set(range(size)) - by_label.keys())
    if missing:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: cases form "
                f"missing labels {missing} for binder "
                f"{binder.var!r} of size {size}"
            ],
        )
    rendered = tuple(
        _render(ctx, by_label[label]) for label in range(size)
    )
    return _emit_vector(ctx, rendered)


def _build_nested_vector(
    ctx: _JlLetCtx,
    binders: tuple[LetFactorBinder, ...],
    sizes: tuple[int, ...],
    body: LetExprNode,
    fixed: tuple[int, ...],
) -> tuple[str, str]:
    """Recursive helper that materialises the nested `vector_expression`
    tower for the uniform-body factor form.

    Returns ``(vertex_id, vertex_kind)`` so the outer call site can
    populate ``chose-alt-child-kinds`` with the right child kinds.
    """
    if len(fixed) == len(binders):
        subst = body
        for binder, value in zip(binders, fixed, strict=True):
            subst = _substitute(
                subst, binder.var, LetExprLiteral(value=value + 1)
            )
        return _render(ctx, subst)
    level = len(fixed)
    rendered: list[tuple[str, str]] = []
    for i in range(sizes[level]):
        rendered.append(
            _build_nested_vector(
                ctx, binders, sizes, body, fixed + (i,)
            )
        )
    return _emit_vector(ctx, tuple(rendered))


# ---------------------------------------------------------------------------
# Factor-binder support: axis-size lookup and body substitution.
# ---------------------------------------------------------------------------


def _factor_axis_size(
    ctx: _JlLetCtx, binder: LetFactorBinder
) -> int:
    """Resolve a factor binder's axis to a static integer size.

    Looks up the binder's index expression in ``ctx.cards``. Raises
    [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
    when the axis is a constructor with no static size or when the
    name is unknown.
    """
    name = _object_expr_axis_name(ctx, binder.index)
    size = ctx.cards.get(name)
    if size is None:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: axis "
                f"{name!r} (binder {binder.var!r}) has no static "
                f"cardinality in `ctx.cards`"
            ],
        )
    return int(size)


def _object_expr_axis_name(
    ctx: _JlLetCtx, obj: ObjectExpr
) -> str:
    """Resolve an `ObjectExpr` to the axis name a `cards` lookup wants.

    Handles `TypeName` directly and
    `DiscreteConstructor("FinSet", N)` as a literal anonymous axis
    (the size is the integer arg).
    """
    if isinstance(obj, TypeName):
        return obj.name
    if isinstance(obj, DiscreteConstructor) and obj.constructor == "FinSet":
        if len(obj.args) == 1 and obj.args[0].isdigit():
            return obj.args[0]
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: "
                f"non-literal FinSet binder {obj.args!r}"
            ],
        )
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [
            f"let-expr:LetExprFactor:{_target(ctx)}: binder "
            f"index {type(obj).__name__} not supported"
        ],
    )


def _substitute(
    expr: LetExprNode, name: str, value: LetExprNode
) -> LetExprNode:
    """Capture-avoiding substitution of every free occurrence of
    `LetExprVar(name=name)` in `expr` with `value`.

    Mirrors the shared substitution walk in `_stan_helpers`; the two
    helpers carry independent copies so the Stan helper can stay tied
    to its Stan-specific imports while the Julia helper stays tied
    to its Julia-specific protocol ctx.
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
            left=_substitute(expr.left, name, value),
            right=_substitute(expr.right, name, value),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            operand=_substitute(expr.operand, name, value),
        )
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(
                _substitute(a, name, value) for a in expr.args
            ),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_substitute(expr.array, name, value),
            indices=tuple(
                _substitute(i, name, value) for i in expr.indices
            ),
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(
                _substitute(i, name, value) for i in expr.items
            ),
        )
    if isinstance(expr, LetExprLambda):
        if expr.param == name:
            return expr
        return LetExprLambda(
            param=expr.param,
            body=_substitute(expr.body, name, value),
        )
    if isinstance(expr, LetExprFactor):
        if any(b.var == name for b in expr.binders):
            return expr
        return LetExprFactor(
            binders=expr.binders,
            body=(
                _substitute(expr.body, name, value)
                if expr.body is not None
                else None
            ),
            cases=tuple(
                LetFactorCase(
                    label=c.label,
                    value=_substitute(c.value, name, value),
                    line=c.line,
                    col=c.col,
                )
                for c in expr.cases
            ),
        )
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_substitute(expr.receiver, name, value),
            method=expr.method,
            args=tuple(
                _substitute(a, name, value) for a in expr.args
            ),
        )
    raise UnsupportedConstruct(
        "qvr-let-substitution",
        [f"let-expr:{type(expr).__name__}: substitution unhandled"],
    )


def _target(ctx: _JlLetCtx) -> str:
    """Read the ctx's `target` tag for error messages."""
    return getattr(ctx, "target", "julia")


__all__ = ["render_let_expr_julia"]
