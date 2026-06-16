"""Render `LetExprNode` to Stan tree-sitter schema vertices.

The Stan grammar exposes the following expression-level vertex
kinds the helper builds:

* `real_literal` / `integer_literal` for numeric leaves
* `variable_expression` wrapping an `identifier` for variable refs
* `infix_op_expression` (per-operator alts via `chose-alt-fingerprint`)
* `prefix_op_expression` for unary minus
* `function_expression` with `name`-edged `identifier` callee and a
  `argument_list` child whose fingerprint encodes the comma count
* `indexed_expression` with `[ ]` fingerprint, `variable_expression`
  callee, and `index`-wrapped index children
* `array_expression` for `{e0, e1, ...}` list literals

Every vertex sets `chose-alt-child-kinds` to the space-separated
sequence of its children's kinds; the pretty-printer uses this to
disambiguate grammar productions and silently drops vertices whose
constraint is missing or stale. The helper returns
`(vertex_id, kind)` from every recursive call so parents can build
the `chose-alt-child-kinds` string from real child kinds.

Some `LetExprNode` kinds do not map to Stan user-program syntax
(strings, lambdas, method calls). The helper raises
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
with a precise kind rather than emitting a fake placeholder token.
`LetExprFactor` is unrolled at render time: the cases form becomes
an `array_expression` whose children are the case bodies in label
order; the uniform-body multi-binder form becomes nested
`array_expression` vertices populated by substituting each binder
through its axis's static cardinality (looked up via the
`_StanLetCtx`'s `cards` map sourced from
[`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]).
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
from quivers.dsl.ast_nodes.let_expressions import LetFactorBinder, LetFactorCase
from quivers.dsl.ast_nodes.objects import TypeName
from quivers.transpile._api import UnsupportedConstruct


def render_let_expr_stan(ctx, expr: LetExprNode) -> str:
    """Build a Stan expression schema for `expr` in `ctx`. Returns
    the root vertex id.

    Wraps `_render` to discard the kind return value at the public
    boundary so callers see the same signature as the other
    per-target helpers.
    """
    vid, _kind = _render(ctx, expr)
    return vid


def _render(ctx, expr: LetExprNode) -> tuple[str, str]:
    """Recursive renderer returning ``(vertex_id, vertex_kind)`` so
    parents can populate ``chose-alt-child-kinds`` accurately."""
    if isinstance(expr, LetExprLiteral):
        return _emit_literal(ctx, expr.value)
    if isinstance(expr, LetExprVar):
        return _emit_variable_expression(ctx, expr.name)
    if isinstance(expr, LetExprBinOp):
        return _emit_infix(ctx, expr)
    if isinstance(expr, LetExprUnaryOp):
        return _emit_prefix(ctx, expr)
    if isinstance(expr, LetExprCall):
        return _emit_function_expression(ctx, expr.func, expr.args)
    if isinstance(expr, LetExprIndex):
        return _emit_indexed(ctx, expr)
    if isinstance(expr, LetExprList):
        return _emit_array_expression(
            ctx, tuple(_render(ctx, item) for item in expr.items)
        )
    if isinstance(expr, LetExprFactor):
        return _render_factor(ctx, expr)
    if isinstance(expr, LetExprString):
        raise UnsupportedConstruct(
            "qvr-stan-helper",
            [
                "let-expr:LetExprString: Stan has no string literal "
                "in expression position"
            ],
        )
    if isinstance(expr, LetExprLambda):
        raise UnsupportedConstruct(
            "qvr-stan-helper",
            [
                "let-expr:LetExprLambda: Stan has no anonymous "
                "function syntax in user-program expression position"
            ],
        )
    if isinstance(expr, LetExprMethodCall):
        raise UnsupportedConstruct(
            "qvr-stan-helper",
            [
                "let-expr:LetExprMethodCall: Stan has no method "
                "dispatch syntax"
            ],
        )
    raise UnsupportedConstruct(
        "qvr-stan-helper",
        [f"let-expr:{type(expr).__name__}: unhandled node kind"],
    )


def _emit_literal(ctx, value: object) -> tuple[str, str]:
    """Emit a `real_literal` or `integer_literal` vertex.

    Whole-number floats (`1.0`, `2.0`) emit as `integer_literal`
    so that array indices substituted from factor binders satisfy
    Stan's strict `arr[int]` typing rule.
    """
    if isinstance(value, float) and value == int(value):
        vid = ctx.vertex(ctx.fresh("il"), "integer_literal")
        ctx.literal(vid, str(int(value)))
        return vid, "integer_literal"
    if isinstance(value, float):
        vid = ctx.vertex(ctx.fresh("rl"), "real_literal")
        ctx.literal(vid, str(value))
        return vid, "real_literal"
    vid = ctx.vertex(ctx.fresh("il"), "integer_literal")
    ctx.literal(vid, str(value))
    return vid, "integer_literal"


def _emit_variable_expression(ctx, name: str) -> tuple[str, str]:
    """Emit a `variable_expression` wrapping an `identifier`."""
    vid = ctx.vertex(ctx.fresh("vex"), "variable_expression")
    ctx.constraint(vid, "chose-alt-child-kinds", "identifier")
    ident = ctx.vertex(ctx.fresh("id"), "identifier")
    ctx.literal(ident, name)
    ctx.edge(vid, ident, "child_of")
    return vid, "variable_expression"


def _emit_infix(ctx, expr: LetExprBinOp) -> tuple[str, str]:
    """Emit an `infix_op_expression` for a binary operator."""
    left_vid, left_kind = _render(ctx, expr.left)
    right_vid, right_kind = _render(ctx, expr.right)
    vid = ctx.vertex(ctx.fresh("bin"), "infix_op_expression")
    ctx.constraint(vid, "chose-alt-fingerprint", expr.op)
    ctx.constraint(
        vid, "chose-alt-child-kinds", f"{left_kind} {right_kind}"
    )
    ctx.edge(vid, left_vid, "child_of")
    ctx.edge(vid, right_vid, "child_of")
    return vid, "infix_op_expression"


def _emit_prefix(ctx, expr: LetExprUnaryOp) -> tuple[str, str]:
    """Emit a `prefix_op_expression` for the unary minus."""
    operand_vid, operand_kind = _render(ctx, expr.operand)
    vid = ctx.vertex(ctx.fresh("uop"), "prefix_op_expression")
    ctx.constraint(vid, "chose-alt-fingerprint", "-")
    ctx.constraint(vid, "chose-alt-child-kinds", operand_kind)
    ctx.edge(vid, operand_vid, "child_of")
    return vid, "prefix_op_expression"


# QVR function names that map to a different identifier in Stan's
# stdlib. Most pure-math names (`log`, `exp`, `sqrt`, `abs`,
# `softmax`, ...) coincide across targets and need no rewrite.
_STAN_FUNCTION_RENAMES: dict[str, str] = {
    "sigmoid": "inv_logit",
}


def _emit_function_expression(
    ctx, func: str, args: tuple[LetExprNode, ...]
) -> tuple[str, str]:
    """Emit a `function_expression` with `name` edge to the callee
    identifier and `child_of` edge to the `argument_list`.

    Applies the
    [`_STAN_FUNCTION_RENAMES`][quivers.transpile.renderers._stan_helpers._STAN_FUNCTION_RENAMES]
    table so QVR-named math primitives (`sigmoid`, ...) reach Stan
    under their stdlib identifiers (`inv_logit`, ...).
    """
    rendered = tuple(_render(ctx, a) for a in args)
    vid = ctx.vertex(ctx.fresh("call"), "function_expression")
    ctx.constraint(
        vid, "chose-alt-child-kinds", "identifier argument_list"
    )
    fn = ctx.vertex(ctx.fresh("fn"), "identifier")
    ctx.literal(fn, _STAN_FUNCTION_RENAMES.get(func, func))
    ctx.edge(vid, fn, "name")
    al_vid = _emit_argument_list(ctx, rendered)
    ctx.edge(vid, al_vid, "child_of")
    return vid, "function_expression"


def _emit_argument_list(
    ctx, rendered: tuple[tuple[str, str], ...]
) -> str:
    """Emit an `argument_list` with the right comma fingerprint and
    child-kinds string."""
    vid = ctx.vertex(ctx.fresh("args"), "argument_list")
    if rendered:
        fingerprint = "( " + ", ".join(["" for _ in rendered]) + " )"
        # Stan's grammar prints the fingerprint as `( , , )` with
        # N-1 commas for N args (one comma between each pair).
        fingerprint = "( " + ", ".join("" for _ in rendered).rstrip() + " )"
        # Build the canonical form: "( )" for one arg, "( , )" for
        # two args, "( , , )" for three args, etc.
        if len(rendered) == 1:
            fingerprint = "( )"
        else:
            commas = ", " * (len(rendered) - 1)
            fingerprint = f"( {commas.rstrip()} )"
    else:
        fingerprint = "( )"
    ctx.constraint(vid, "chose-alt-fingerprint", fingerprint)
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        " ".join(kind for _vid, kind in rendered) or "",
    )
    for child_vid, _kind in rendered:
        ctx.edge(vid, child_vid, "child_of")
    return vid


def _emit_indexed(
    ctx, expr: LetExprIndex
) -> tuple[str, str]:
    """Emit an `indexed_expression` (the `arr[i][j]...` form)."""
    arr_vid, arr_kind = _render(ctx, expr.array)
    index_vids: list[str] = []
    child_kinds: list[str] = [arr_kind]
    for idx in expr.indices:
        inner_vid, inner_kind = _render(ctx, idx)
        wrap = ctx.vertex(ctx.fresh("idx"), "index")
        ctx.constraint(wrap, "chose-alt-child-kinds", inner_kind)
        ctx.edge(wrap, inner_vid, "child_of")
        index_vids.append(wrap)
        child_kinds.append("index")
    vid = ctx.vertex(ctx.fresh("ix"), "indexed_expression")
    ctx.constraint(vid, "chose-alt-fingerprint", "[ ]")
    ctx.constraint(
        vid, "chose-alt-child-kinds", " ".join(child_kinds)
    )
    ctx.edge(vid, arr_vid, "child_of")
    for wrap in index_vids:
        ctx.edge(vid, wrap, "child_of")
    return vid, "indexed_expression"


def _emit_array_expression(
    ctx, rendered: tuple[tuple[str, str], ...]
) -> tuple[str, str]:
    """Emit an `array_expression` ``{e0, e1, ...}`` list literal."""
    vid = ctx.vertex(ctx.fresh("arr"), "array_expression")
    ctx.constraint(
        vid,
        "chose-alt-child-kinds",
        " ".join(kind for _vid, kind in rendered),
    )
    for child_vid, _kind in rendered:
        ctx.edge(vid, child_vid, "child_of")
    return vid, "array_expression"


def _render_factor(ctx, expr: LetExprFactor) -> tuple[str, str]:
    """Unroll a `LetExprFactor` into nested `array_expression`
    vertices.

    The cases form (binders contain a single axis, body is None,
    cases enumerate labels in [0, |axis|)) emits an
    `array_expression` whose children are each case's body in
    label order.

    The uniform-body form (one or more binders, body is the
    repeated expression, cases is empty) emits a tower of
    `array_expression` vertices of shape
    `(|b0|, |b1|, ..., |bn-1|)`, with each binder substituted for
    its 1-indexed integer value (Stan arrays are 1-indexed and
    QVR's surface indexing is mapped through directly).
    """
    if expr.cases and expr.body is None:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                "qvr-stan-helper",
                [
                    "let-expr:LetExprFactor: cases form requires "
                    f"exactly one binder; got {len(expr.binders)}"
                ],
            )
        ordered = sorted(expr.cases, key=lambda c: c.label)
        rendered = tuple(_render(ctx, c.value) for c in ordered)
        return _emit_array_expression(ctx, rendered)
    if expr.body is not None and not expr.cases:
        sizes = tuple(_card_for(ctx, b) for b in expr.binders)
        return _build_nested_array(
            ctx, expr.binders, sizes, expr.body, ()
        )
    raise UnsupportedConstruct(
        "qvr-stan-helper",
        [
            "let-expr:LetExprFactor: mixed cases-plus-body form "
            "is not a valid surface construct"
        ],
    )


def _card_for(ctx, binder: LetFactorBinder) -> int:
    """Resolve the static cardinality of `binder.index`.

    `LetExprFactor` only unrolls when every binder's axis has a
    statically-known size; the helper consults `ctx.cards`
    (populated from
    [`IRProgram.cards`][quivers.transpile.ir.IRProgram.cards]).
    """
    idx = binder.index
    if isinstance(idx, TypeName):
        cards = getattr(ctx, "cards", None)
        if cards is None or idx.name not in cards:
            raise UnsupportedConstruct(
                "qvr-stan-helper",
                [
                    f"let-expr:LetExprFactor: binder {binder.var!r} "
                    f"references object {idx.name!r} whose cardinality "
                    "is unknown at render time"
                ],
            )
        return cards[idx.name]
    raise UnsupportedConstruct(
        "qvr-stan-helper",
        [
            f"let-expr:LetExprFactor: binder {binder.var!r} index is "
            f"{type(idx).__name__}; only TypeName binders unroll"
        ],
    )


def _build_nested_array(
    ctx,
    binders: tuple[LetFactorBinder, ...],
    sizes: tuple[int, ...],
    body: LetExprNode,
    fixed: tuple[int, ...],
) -> tuple[str, str]:
    """Recursive helper that materialises the nested
    `array_expression` tower for the uniform-body factor form.

    Returns ``(vertex_id, vertex_kind)`` so the outer call site can
    populate ``chose-alt-child-kinds`` with the right child kinds.
    """
    if len(fixed) == len(binders):
        subst = body
        for binder, value in zip(binders, fixed, strict=True):
            subst = _substitute_let_expr(
                subst, binder.var, LetExprLiteral(value=value + 1)
            )
        return _render(ctx, subst)
    level = len(fixed)
    rendered: list[tuple[str, str]] = []
    for i in range(sizes[level]):
        rendered.append(
            _build_nested_array(
                ctx, binders, sizes, body, fixed + (i,)
            )
        )
    return _emit_array_expression(ctx, tuple(rendered))


def _substitute_let_expr(
    expr: LetExprNode, name: str, value: LetExprNode
) -> LetExprNode:
    """Capture-avoiding substitution of every free occurrence of
    `LetExprVar(name=name)` in `expr` with `value`.

    Shared substitution helper for every per-target renderer that
    needs to unroll `LetExprFactor` by binding indices to integer
    literals. Lives in `_stan_helpers` because Stan was the first
    target to need it; other helper modules import from here when
    they grow the same need (one source of truth for the walk).
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
            left=_substitute_let_expr(expr.left, name, value),
            right=_substitute_let_expr(expr.right, name, value),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            op=expr.op,
            operand=_substitute_let_expr(expr.operand, name, value),
        )
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(
                _substitute_let_expr(a, name, value) for a in expr.args
            ),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_substitute_let_expr(expr.array, name, value),
            indices=tuple(
                _substitute_let_expr(i, name, value) for i in expr.indices
            ),
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(
                _substitute_let_expr(i, name, value) for i in expr.items
            ),
        )
    if isinstance(expr, LetExprLambda):
        if expr.param == name:
            return expr
        return LetExprLambda(
            param=expr.param,
            body=_substitute_let_expr(expr.body, name, value),
        )
    if isinstance(expr, LetExprFactor):
        if any(b.var == name for b in expr.binders):
            return expr
        return LetExprFactor(
            binders=expr.binders,
            body=(
                _substitute_let_expr(expr.body, name, value)
                if expr.body is not None
                else None
            ),
            cases=tuple(
                LetFactorCase(
                    label=c.label,
                    value=_substitute_let_expr(c.value, name, value),
                    line=c.line,
                    col=c.col,
                )
                for c in expr.cases
            ),
        )
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_substitute_let_expr(expr.receiver, name, value),
            method=expr.method,
            args=tuple(
                _substitute_let_expr(a, name, value) for a in expr.args
            ),
        )
    raise UnsupportedConstruct(
        "qvr-let-substitution",
        [f"let-expr:{type(expr).__name__}: substitution unhandled"],
    )


__all__ = ["render_let_expr_stan"]
