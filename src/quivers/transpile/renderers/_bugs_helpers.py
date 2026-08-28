"""Render [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode] subtrees
to BUGS / JAGS tree-sitter schema vertices.

BUGS / JAGS share an identical expression grammar for the
deterministic-assignment idiom ``<name> <- <expr>``: ``binary_expression``,
``unary_expression``, ``function_call`` (with ``name`` and
``argument_list``), ``identifier``, ``number``, ``indexed_variable``
(with ``index_list`` carrying integer / identifier / range children),
and ``parenthesized_expression``. Neither language has a native
string literal, lambda, method-call, or list-literal at the model-body
level; those LetExpr kinds either get a structural unrolling
(``list -> c(...)`` combine, ``factor -> c(<body[0]>, <body[1]>, ...)``
unroll) or raise [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct].

The helper consumes a lightweight context exposing four bound methods:

* ``ctx.fresh(prefix: str) -> str``
* ``ctx.v(vid: str, kind: str) -> str``
* ``ctx.e(src: str, tgt: str, kind: str) -> None``
* ``ctx.lit(vid: str, text: str) -> None``
* ``ctx.constraint(vid: str, sort: str, value: str) -> None``

The renderer (BUGS or JAGS) is responsible for the surrounding
``deterministic_relation`` / ``stochastic_relation`` and for binding
the helper context. When unrolling factor binders, the helper also
reads ``ctx.cards`` (a ``dict[str, int]``) and ``ctx.target`` (one of
``"bugs"`` / ``"jags"``) to resolve axis sizes and to label the error
tag with the correct backend.
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
    LetFactorBinder,
    LetFactorCase,
)
from quivers.dsl.ast_nodes.objects import (
    DiscreteConstructor,
    ObjectExpr,
    TypeName,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.ir import (
    Dim,
    DimDynamic,
    DimStatic,
    IRArg,
    IRArgBroadcast,
    IRArgList,
    IRArgMatrix,
    IRArgRef,
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRArgNumber,
    IRSample,
    Plate,
)


#: QVR families supported on the non-negative reals whose BUGS / JAGS
#: target distribution is supported on all of R. ``HalfNormal(scale)``
#: lowers to ``dnorm(0, 1/scale^2)`` and ``HalfCauchy(scale)`` to
#: ``dt(0, 1/scale^2, 1)``; both need an explicit lower truncation at
#: zero so the emitted support matches the family's, and so the
#: normalising constant picks up the factor of two the folded density
#: carries. The value is the lower bound of the family's support.
HALF_SUPPORT_LOWER_BOUND: dict[str, float] = {
    "HalfNormal": 0.0,
    "HalfCauchy": 0.0,
}

#: The truncation-suffix keyword each backend spells. Both the ``jags``
#: and the ``bugs`` backend execute through the JAGS engine (the
#: ``bugs`` probe image installs the ``jags`` binary and pyjags, and
#: the gallery / numeric-equivalence harness maps ``"bugs"`` to the
#: JAGS probe script), so both emit JAGS's renormalized truncation
#: ``T(lower, upper)``. JAGS accepts the alternative ``I(lower, upper)``
#: only when every parameter of the truncated distribution is a
#: compile-time constant, and on a latent-parent node (the canonical
#: hierarchical scale prior ``sigma ~ dnorm(0, 1/(tau*tau)) T(0,)`` with
#: latent ``tau``) it rejects ``I(,)`` at compile time; on real
#: OpenBUGS ``I(,)`` on a latent-parent node is interval censoring, a
#: different likelihood, not the folded prior measure the family
#: denotes. ``T(,)`` is the only spelling that both compiles on the
#: JAGS engine and preserves the renormalized one-sided fold, so it is
#: the correct emission for both backends. The grammar exposes both
#: spellings as alternatives of the same ``truncation`` rule, selected
#: through the ``chose-alt-fingerprint`` constraint.
TRUNCATION_FINGERPRINT: dict[str, str] = {
    "jags": "T( , )",
    "bugs": "T( , )",
}


#: The BUGS / JAGS log-gamma and log-factorial builtins the
#: beta-binomial log-pmf is written in terms of. Both live in the
#: ``bugs`` module, which every JAGS engine loads by default and which
#: the OpenBUGS / WinBUGS function library also ships, so the closed
#: form below needs no optional module.
_LOG_GAMMA: str = "loggam"
_LOG_FACTORIAL: str = "logfact"


def _let_add(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    return LetExprBinOp(op="+", left=left, right=right)


def _let_sub(left: LetExprNode, right: LetExprNode) -> LetExprNode:
    return LetExprBinOp(op="-", left=left, right=right)


def _let_signed_sum(
    head: LetExprNode, *tail: tuple[str, LetExprNode]
) -> LetExprNode:
    """Left-fold a signed term list into one ``+`` / ``-`` chain.

    Each entry of `tail` pairs the operator that joins it to the
    running total with the term itself, so the result associates
    exactly as the source reads it.
    """
    total = head
    for op, term in tail:
        total = LetExprBinOp(op=op, left=total, right=term)
    return total


def _scalar_arg_expr(
    backend: str, family: str, slot: str, arg: IRArg
) -> LetExprNode:
    """Read one distribution argument back as a let-expression.

    The closed-form densities this module writes out consume their
    arguments as ordinary arithmetic operands rather than as
    distribution-call children, so each one has to come back as a
    `LetExprNode`. A bare reference and a numeric literal both do; an
    index expression, a list, a matrix, or a broadcast wrapper carries
    structure the scalar closed form cannot place, so it raises.
    """
    if isinstance(arg, IRArgNumber):
        return LetExprLiteral(value=arg.value)
    if isinstance(arg, IRArgRef) and not arg.indices:
        return LetExprVar(name=arg.name)
    raise UnsupportedConstruct(
        f"qvr-{backend}",
        [
            f"family:{family}:non-scalar-arg:{slot}: the closed-form "
            f"density reads each argument as a scalar operand, and "
            f"this slot carries a {type(arg).__name__}"
        ],
    )


def beta_binomial_log_pmf(
    backend: str,
    *,
    variate: str,
    args: tuple[IRArg, ...],
    arg_names: tuple[str, ...],
) -> LetExprNode:
    """Build ``log BetaBinomial(<variate>; n, a, b)`` in closed form.

    Neither the BUGS function library nor the JAGS modules a stock
    engine loads (``basemod``, ``bugs``, ``dic``) ships a
    beta-binomial distribution: JAGS carries one only in the optional
    ``mix`` module, and OpenBUGS / WinBUGS carry none at all. The
    density is nonetheless an ordinary expression in ``loggam`` and
    ``logfact``, both of which the ``bugs`` module supplies, so the
    renderer writes it out rather than naming a distribution the
    engine may not have.

    Writing ``B`` for the beta function, the pmf is

        p(y; n, a, b) = C(n, y) * B(a + y, b + n - y) / B(a, b),

    and expanding both the binomial coefficient and each beta function
    into log-gammas gives

        logfact(n) - logfact(y) - logfact(n - y)
        + loggam(a + y) + loggam(b + n - y) - loggam(a + b + n)
        - loggam(a) - loggam(b) + loggam(a + b),

    which is exactly the term this returns. It is the family's own
    log-density, not a surrogate for it: the latent conversion rate is
    integrated out analytically, so no auxiliary node enters the model
    and the joint the engine scores is the marginal QVR names.
    """
    by_name = dict(zip(arg_names, args, strict=False))
    missing = [
        slot
        for slot in ("total_count", "concentration1", "concentration0")
        if slot not in by_name
    ]
    if missing:
        raise UnsupportedConstruct(
            f"qvr-{backend}",
            [
                f"family:BetaBinomial:missing-arg:{','.join(missing)}: "
                f"the closed-form density needs the trial count and "
                f"both concentrations; the site supplies "
                f"{list(arg_names)}"
            ],
        )
    n = _scalar_arg_expr(
        backend, "BetaBinomial", "total_count", by_name["total_count"]
    )
    a = _scalar_arg_expr(
        backend,
        "BetaBinomial",
        "concentration1",
        by_name["concentration1"],
    )
    b = _scalar_arg_expr(
        backend,
        "BetaBinomial",
        "concentration0",
        by_name["concentration0"],
    )
    y = LetExprVar(name=variate)

    def loggam(inner: LetExprNode) -> LetExprNode:
        return LetExprCall(func=_LOG_GAMMA, args=(inner,))

    def logfact(inner: LetExprNode) -> LetExprNode:
        return LetExprCall(func=_LOG_FACTORIAL, args=(inner,))

    return _let_signed_sum(
        logfact(n),
        ("-", logfact(y)),
        ("-", logfact(_let_sub(n, y))),
        ("+", loggam(_let_add(a, y))),
        ("+", loggam(_let_sub(_let_add(b, n), y))),
        ("-", loggam(_let_add(_let_add(a, b), n))),
        ("-", loggam(a)),
        ("-", loggam(b)),
        ("+", loggam(_let_add(a, b))),
    )


def half_support_truncation(family: str, *, observed: bool) -> tuple[IRArg, ...] | None:
    """Return the one-sided truncation bounds a latent draw from
    ``family`` needs, or `None` when no truncation applies.

    Only latent draws are truncated. On an observed node the JAGS
    engine reads the truncation suffix as censoring rather than
    truncation, which is a different likelihood; an observed
    half-support variate already lies in the support, so the omitted
    suffix costs only the constant ``log 2`` per observation.
    """
    if observed:
        return None
    lower = HALF_SUPPORT_LOWER_BOUND.get(family)
    if lower is None:
        return None
    return (IRArgNumber(value=lower),)


@runtime_checkable
class _BugsLetCtx(Protocol):
    """Structural protocol for the helper's ctx parameter.

    `range_1_to` builds the `1:<upper>` range vertex each backend's
    grammar wants; the two grammars disagree on the alternative-
    selection constraints a `range` carries, so the renderer owns the
    construction and the helper only asks for one. It is reached only
    when the caller supplies a declared-plate table, so a caller
    rendering a standalone expression never needs it.
    """

    target: str
    cards: dict[str, int]

    def fresh(self, prefix: str) -> str: ...
    def v(self, vid: str, kind: str) -> str: ...
    def e(self, src: str, tgt: str, kind: str) -> None: ...
    def lit(self, vid: str, text: str) -> None: ...
    def constraint(self, vid: str, sort: str, value: str) -> None: ...
    def range_1_to(self, upper: str) -> str: ...


class _LetEnv:
    """The helper's per-render environment.

    Bundles the backend ctx with the declared-plate table the index
    and reduction emitters read. The table is a caller-supplied input
    rather than a property of the ctx, because a caller may render a
    standalone expression that sits in no program: every name then has
    no declared shape, and the emitters read every operand as a
    scalar, which is what such an expression means.
    """

    def __init__(
        self, ctx: _BugsLetCtx, decl_plates: dict[str, Plate]
    ) -> None:
        self._ctx = ctx
        self.decl_plates = decl_plates
        self.target = _target(ctx)
        self.cards = ctx.cards

    def fresh(self, prefix: str) -> str:
        return self._ctx.fresh(prefix)

    def v(self, vid: str, kind: str) -> str:
        return self._ctx.v(vid, kind)

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._ctx.e(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._ctx.lit(vid, text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._ctx.constraint(vid, sort, value)

    def range_1_to(self, upper: str) -> str:
        return self._ctx.range_1_to(upper)


def render_let_expr_bugs(
    ctx: _BugsLetCtx,
    expr: LetExprNode,
    *,
    decl_plates: dict[str, Plate] | None = None,
) -> str:
    """Build a BUGS / JAGS expression schema for ``expr`` in ``ctx``.

    Returns the root vertex id. Recurses into nested
    [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode] values.
    Raises [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
    when the construct has no representation in the BUGS / JAGS
    family (``LetExprString``, ``LetExprLambda``, ``LetExprMethodCall``,
    or a [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor]
    whose binders reference an axis of unknown static cardinality).

    `decl_plates` maps every bound name to its declared
    [`Plate`][quivers.transpile.ir.Plate], which is what tells the
    index emitter how many axes a subscript leaves unconsumed: a
    gather of a matrix row (`Z_mat[item_idx]` against a 32-by-2
    declaration) needs the trailing full-axis slice spelled out,
    because BUGS / JAGS read a single subscript on a rank-2 node as a
    rank error rather than a row. Omitting it renders the expression
    with no declared shapes in scope, which is the right reading for
    an expression that sits in no program.
    """
    return _render(_LetEnv(ctx, decl_plates or {}), expr)


def _render(ctx: _LetEnv, expr: LetExprNode) -> str:
    """Dispatch one [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode]
    to its per-kind emitter."""
    if isinstance(expr, LetExprLiteral):
        return _emit_number(ctx, expr.value)
    if isinstance(expr, LetExprVar):
        return _emit_identifier(ctx, expr.name)
    if isinstance(expr, LetExprBinOp):
        return _emit_binop(ctx, expr)
    if isinstance(expr, LetExprUnaryOp):
        return _emit_unary(ctx, expr)
    if isinstance(expr, LetExprCall):
        return _emit_reduction_or_call(ctx, expr)
    if isinstance(expr, LetExprIndex):
        return _emit_index(ctx, expr)
    if isinstance(expr, LetExprList):
        return _emit_list(ctx, expr.items)
    if isinstance(expr, LetExprFactor):
        return _emit_factor(ctx, expr)
    if isinstance(expr, LetExprString):
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprString:{_target(ctx)}: BUGS / JAGS "
                f"have no native string literal in the model body"
            ],
        )
    if isinstance(expr, LetExprLambda):
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprLambda:{_target(ctx)}: BUGS / JAGS "
                f"have no anonymous function syntax"
            ],
        )
    if isinstance(expr, LetExprMethodCall):
        # BUGS and JAGS have no `receiver.method(args)` dispatch
        # syntax, and unlike Stan they also have no path to a
        # user-defined-function definition the renderer could graft:
        #
        # 1. BUGS / OpenBUGS forbid user-defined functions in the
        #    model body; only built-in distributions and the
        #    standard math library are callable, and there is no
        #    `functions { ... }` block.
        # 2. JAGS allows user functions only via compiled C++
        #    modules linked at JAGS startup (the JAGS Module API);
        #    they cannot be declared inline in the model file.
        # 3. The model body cannot express the inside-algorithm
        #    chart parser anyway: BUGS / JAGS deterministic
        #    relations are non-recursive scalar/array updates over
        #    static index ranges, with no support for the variable-
        #    length span enumeration the CKY-style fixed-point
        #    requires.
        #
        # `deduction_decl` is also stripped by the IR pipeline (see
        # [`CATEGORICAL_METADATA_IGNORABLE`][quivers.transpile._api.CATEGORICAL_METADATA_IGNORABLE]),
        # so even if the model body could host a chart parser the
        # renderer has no access to the rules to compile against.
        # Rewriting `m.f(a)` as the static call `f(m, a)` without
        # supplying `f` produces an undefined-symbol model that
        # JAGS / OpenBUGS reject at parse time, so the helper
        # raises instead of emitting a bogus call.
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprMethodCall:{_target(ctx)}: BUGS / "
                f"JAGS have no method-dispatch syntax; the chart-"
                f"parser deduction graft that would supply the "
                f"called function is also impossible because BUGS "
                f"forbids user-defined model-body functions and "
                f"JAGS exposes them only through compiled C++ "
                f"modules linked at startup, not inline"
            ],
        )
    raise UnsupportedConstruct(
        f"qvr-{_target(ctx)}-helper",
        [f"let-expr:{type(expr).__name__}:{_target(ctx)}: unhandled"],
    )


# ---------------------------------------------------------------------------
# Per-kind emitters.
# ---------------------------------------------------------------------------


def _emit_number(ctx: _BugsLetCtx, value: float) -> str:
    """Emit a `number` vertex carrying the textual rendering of `value`."""
    v = ctx.v(ctx.fresh("num"), "number")
    text = str(int(value)) if float(value).is_integer() else repr(value)
    ctx.lit(v, text)
    return v


def _emit_identifier(ctx: _BugsLetCtx, name: str) -> str:
    """Emit a bare `identifier` vertex carrying `name`."""
    v = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(v, name)
    return v


def _render_bugs_operand(ctx: _LetEnv, expr: LetExprNode) -> str:
    """Render `expr` as an operand of a binary / unary operator.

    A nested [`LetExprBinOp`][quivers.dsl.ast_nodes.LetExprBinOp] or
    [`LetExprUnaryOp`][quivers.dsl.ast_nodes.LetExprUnaryOp] operand is
    wrapped in a `parenthesized_expression`. The BUGS / JAGS pretty
    printer emits children in source order without re-grouping, so
    ``(a + b) * c`` would otherwise print as ``a + b * c`` and
    reassociate under the language's precedence.
    """
    vid = _render(ctx, expr)
    if isinstance(expr, (LetExprBinOp, LetExprUnaryOp)):
        return _emit_paren(ctx, vid, _arg_edge_kind(expr))
    return vid


def _emit_paren(ctx: _BugsLetCtx, inner: str, inner_kind: str) -> str:
    """Wrap `inner` (of grammar kind `inner_kind`) in a
    `parenthesized_expression` vertex."""
    p = ctx.v(ctx.fresh("paren"), "parenthesized_expression")
    ctx.constraint(p, "chose-alt-fingerprint", "( )")
    ctx.constraint(p, "chose-alt-child-kinds", inner_kind)
    ctx.e(p, inner, "child_of")
    return p


def _emit_binop(ctx: _LetEnv, expr: LetExprBinOp) -> str:
    """Emit a `binary_expression` with `left`/`right` field edges.

    BUGS / JAGS `binary_expression` discriminates the operator via the
    grammar's CHOICE alternative; the panproto walker picks the alt
    from the `field:operator` + `chose-alt-fingerprint` pair.

    Neither language lifts an infix operator over an axis: `a * b` on
    two vector nodes is a rank error, not the elementwise product QVR
    denotes. The one axis-carrying product both languages do express
    is the contraction `inprod(a, b)`, which
    [`_emit_reduction_or_call`][quivers.transpile.renderers._bugs_helpers._emit_reduction_or_call]
    recognises before reaching here; any other axis-carrying operand
    pair raises.
    """
    left_rank = axis_rank(ctx.decl_plates, expr.left)
    right_rank = axis_rank(ctx.decl_plates, expr.right)
    if left_rank > 0 and right_rank > 0:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:elementwise-axis-operator:{_target(ctx)}: "
                f"{expr.op!r} between a rank-{left_rank} and a "
                f"rank-{right_rank} operand has no BUGS / JAGS form; "
                f"only the contracted product `sum(a * b)` lowers, to "
                f"`inprod(a, b)`"
            ],
        )
    b = ctx.v(ctx.fresh("be"), "binary_expression")
    ctx.constraint(b, "field:operator", expr.op)
    ctx.constraint(b, "chose-alt-fingerprint", expr.op)
    ctx.e(b, _render_bugs_operand(ctx, expr.left), "left")
    ctx.e(b, _render_bugs_operand(ctx, expr.right), "right")
    return b


def _emit_unary(ctx: _LetEnv, expr: LetExprUnaryOp) -> str:
    """Emit a unary-minus `unary_expression` whose single child rides
    the `operand` field."""
    u = ctx.v(ctx.fresh("ue"), "unary_expression")
    ctx.constraint(u, "field:operator", "-")
    ctx.constraint(u, "chose-alt-fingerprint", "-")
    ctx.e(u, _render_bugs_operand(ctx, expr.operand), "operand")
    return u


#: QVR function names that map to a different identifier in the
#: BUGS / JAGS math library. Most pure-math names (``exp``, ``log``,
#: ``sqrt``, ``abs``, ``pow``, ...) coincide across targets and need
#: no rewrite; the logistic sigmoid is the inverse-logit link, which
#: BUGS / JAGS expose as ``ilogit``.
_BUGS_FUNCTION_RENAMES: dict[str, str] = {
    "sigmoid": "ilogit",
}


def _emit_call(
    ctx: _BugsLetCtx,
    func: str,
    arg_ids: tuple[str, ...],
    arg_kinds: tuple[str, ...],
) -> str:
    """Emit ``<func>(<arg_0>, <arg_1>, ...)`` as a `function_call`.

    The `name` field is an `identifier` vertex; the `arguments` field
    is an `argument_list` whose children carry their grammar kind as
    the edge label (so the panproto walker can pick the right child
    alternative). QVR-named math primitives (``sigmoid``, ...) are
    rewritten to their BUGS / JAGS library identifiers (``ilogit``,
    ...) via
    [`_BUGS_FUNCTION_RENAMES`][quivers.transpile.renderers._bugs_helpers._BUGS_FUNCTION_RENAMES].
    """
    c = ctx.v(ctx.fresh("call"), "function_call")
    name_id = _emit_identifier(ctx, _BUGS_FUNCTION_RENAMES.get(func, func))
    ctx.e(c, name_id, "name")
    if not arg_ids:
        return c
    al = ctx.v(ctx.fresh("al"), "argument_list")
    ctx.e(c, al, "arguments")
    for aid, akind in zip(arg_ids, arg_kinds, strict=True):
        ctx.e(al, aid, akind)
    return c


#: QVR reduction primitives whose BUGS / JAGS counterpart contracts
#: every axis of its argument. Applied to a rank-1 operand -- the
#: shape every axis reduction in the gallery reduces -- the target
#: builtin computes exactly the QVR reduction; applied to a
#: higher-rank operand it would collapse axes the source keeps, so
#: [`_emit_reduction_or_call`][quivers.transpile.renderers._bugs_helpers._emit_reduction_or_call]
#: raises instead.
_AXIS_REDUCING_CALLS: frozenset[str] = frozenset({"sum", "mean", "prod"})


def _emit_reduction_or_call(ctx: _LetEnv, expr: LetExprCall) -> str:
    """Emit a call, lowering an axis reduction to its target spelling.

    `sum(a * b)` over two rank-1 operands is the inner product, which
    BUGS / JAGS spell `inprod(a, b)`; the elementwise product it is
    written in terms of has no target form on its own. A reduction of
    a single rank-1 operand passes straight through, because the
    target builtin already contracts the one axis the operand
    carries. Every other axis-carrying reduction raises.
    """
    if expr.func in _AXIS_REDUCING_CALLS and len(expr.args) == 1:
        arg = expr.args[0]
        rank = axis_rank(ctx.decl_plates, arg)
        if rank > 1:
            raise UnsupportedConstruct(
                f"qvr-{_target(ctx)}-helper",
                [
                    f"let-expr:axis-reduction:{_target(ctx)}: "
                    f"{expr.func}(...) over a rank-{rank} operand "
                    f"reduces only the innermost axis in QVR, and "
                    f"BUGS / JAGS `{expr.func}` contracts every axis; "
                    f"there is no target spelling for the partial "
                    f"reduction"
                ],
            )
        if rank == 1 and isinstance(arg, LetExprBinOp):
            return _emit_contracted_binop(ctx, expr.func, arg)
    return _emit_call(
        ctx,
        expr.func,
        tuple(_render(ctx, a) for a in expr.args),
        tuple(_arg_edge_kind(a) for a in expr.args),
    )


def _emit_contracted_binop(
    ctx: _LetEnv, func: str, arg: LetExprBinOp
) -> str:
    """Lower `sum(<a> * <b>)` over two rank-1 operands to
    `inprod(<a>, <b>)`, and raise on every other shape."""
    left_rank = axis_rank(ctx.decl_plates, arg.left)
    right_rank = axis_rank(ctx.decl_plates, arg.right)
    if func == "sum" and arg.op == "*" and left_rank == 1 and right_rank == 1:
        return _emit_call(
            ctx,
            "inprod",
            (
                _render(ctx, arg.left),
                _render(ctx, arg.right),
            ),
            (_arg_edge_kind(arg.left), _arg_edge_kind(arg.right)),
        )
    if left_rank > 0 and right_rank > 0:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:axis-reduction:{_target(ctx)}: "
                f"{func}(<a> {arg.op} <b>) over two axis-carrying "
                f"operands has no BUGS / JAGS form; only the "
                f"contracted product `sum(a * b)` lowers, to "
                f"`inprod(a, b)`"
            ],
        )
    return _emit_call(
        ctx,
        func,
        (_render(ctx, arg),),
        (_arg_edge_kind(arg),),
    )


def _emit_index(ctx: _LetEnv, expr: LetExprIndex) -> str:
    """Emit `arr[i0, i1, ...]` as an `indexed_variable` with an
    `index_list` child.

    BUGS / JAGS index expressions are not a separate node kind: the
    grammar reuses `indexed_variable` for both LHS index targets and
    nested expression-position subscripts. The `name` field must be
    an `identifier`; multi-level array nesting (`a[i][j]`) is rare in
    practice for these languages and so the helper expects the outer
    expression's `array` slot to resolve to an `identifier` vertex.
    When `expr.array` produces an `indexed_variable` instead, the
    helper raises rather than silently rewriting the access path.
    """
    if not isinstance(expr.array, LetExprVar):
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprIndex:{_target(ctx)}: BUGS / JAGS "
                f"`indexed_variable` requires a bare `identifier` "
                f"name, got {type(expr.array).__name__}"
            ],
        )
    iv = ctx.v(ctx.fresh("iv"), "indexed_variable")
    name_id = _emit_identifier(ctx, expr.array.name)
    ctx.e(iv, name_id, "name")
    il = ctx.v(ctx.fresh("il"), "index_list")
    ctx.e(iv, il, "indices")
    for idx in expr.indices:
        cid = _emit_index_slot(ctx, idx)
        ctx.e(il, cid, _arg_edge_kind(idx))
    for dim in residual_event_dims(
        ctx.decl_plates, expr.array.name, len(expr.indices)
    ):
        ctx.e(il, ctx.range_1_to(dim_upper_text(dim)), "range")
    return iv


def _emit_index_slot(ctx: _LetEnv, idx: LetExprNode) -> str:
    """Emit one index-list child, rebased to the target's origin.

    QVR subscripts count from zero; BUGS and JAGS count from one. A
    subscript the source spells as an integer literal (a factor
    binder already substituted to its integer coordinate, or a
    literal the model wrote itself) is therefore emitted one higher.
    A subscript that names a variable is left alone: a loop variable
    already runs `1:N`, and an index-valued covariate arrives from
    the host already lifted to one-based.

    An arithmetic subscript mixes the two conventions with no way to
    tell which operand carries the origin, so it raises rather than
    emitting an unadjusted expression.
    """
    if isinstance(idx, LetExprLiteral):
        return _emit_number(ctx, float(int(idx.value) + 1))
    if isinstance(idx, (LetExprBinOp, LetExprUnaryOp)):
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprIndex:{_target(ctx)}: arithmetic "
                f"subscript {type(idx).__name__} cannot be rebased "
                f"from the zero-based QVR origin to the one-based "
                f"{_target(ctx).upper()} origin"
            ],
        )
    return _render(ctx, idx)


def split_event_dims(
    event_dims: tuple[Dim, ...], family_event_rank: int
) -> tuple[tuple[Dim, ...], tuple[Dim, ...]]:
    """Split a site's event dims into the family's own event shape and
    the residual axes the renderer must replicate over.

    A family produces the trailing `family_event_rank` dims natively:
    a Dirichlet over a Topic axis is one draw on the simplex, so the
    Topic axis is the family's own. Every leading dim is residual: it
    replicates the family independently, and a scalar family with an
    `over=` axis is all residual. BUGS and JAGS have no vector form
    for a scalar family, so a residual axis becomes a loop rather
    than a slice.

    Returns `(native, residual)`.
    """
    rank = max(0, min(family_event_rank, len(event_dims)))
    if rank == 0:
        return (), event_dims
    return event_dims[len(event_dims) - rank :], event_dims[
        : len(event_dims) - rank
    ]


def dim_upper_text(dim: Dim) -> str:
    """Return the upper-bound text of one plate dim."""
    if isinstance(dim, DimStatic):
        return str(int(dim.size))
    if isinstance(dim, DimDynamic):
        return str(dim.size_name)
    raise UnsupportedConstruct(
        "qvr-bugs-helper",
        [f"dim:{type(dim).__name__}: unknown shape"],
    )


def residual_event_dims(
    decl_plates: dict[str, Plate], name: str, supplied: int
) -> tuple[Dim, ...]:
    """Return the event dims a subscript of `name` leaves unconsumed.

    A gather that supplies exactly one subscript per declared batch
    dim addresses a whole event block, which BUGS / JAGS spell as an
    explicit trailing `1:E` slice per event axis. A subscript list
    that already reaches into the event axes consumes them itself and
    needs no slice.
    """
    plate = decl_plates.get(name)
    if plate is None or not plate.event_dims:
        return ()
    if supplied == len(plate.batch_dims):
        return plate.event_dims
    return ()


def axis_rank(decl_plates: dict[str, Plate], expr: LetExprNode) -> int:
    """Return the number of axes the value of `expr` still carries.

    A bare name carries every axis of its declared plate; a subscript
    consumes one axis per index it supplies. Operators propagate the
    widest operand. Anything the helper cannot resolve reads as a
    scalar, which is the conservative answer: the reduction rewrites
    below only fire on a positively-ranked operand.
    """
    if isinstance(expr, LetExprVar):
        plate = decl_plates.get(expr.name)
        if plate is None:
            return 0
        return len(plate.batch_dims) + len(plate.event_dims)
    if isinstance(expr, LetExprIndex):
        if not isinstance(expr.array, LetExprVar):
            return 0
        plate = decl_plates.get(expr.array.name)
        if plate is None:
            return 0
        declared = len(plate.batch_dims) + len(plate.event_dims)
        return max(declared - len(expr.indices), 0)
    if isinstance(expr, LetExprBinOp):
        return max(
            axis_rank(decl_plates, expr.left),
            axis_rank(decl_plates, expr.right),
        )
    if isinstance(expr, LetExprUnaryOp):
        return axis_rank(decl_plates, expr.operand)
    if isinstance(expr, (LetExprList, LetExprFactor)):
        return 1
    return 0


def _emit_list(ctx: _LetEnv, items: tuple[LetExprNode, ...]) -> str:
    """Render a list literal as the BUGS / JAGS `c(...)` combine call.

    Neither language has an inline list-literal surface form; the
    canonical concatenation idiom is the built-in `c(...)` function
    (S-style combine), which both languages parse as a regular
    `function_call`.
    """
    return _emit_call(
        ctx,
        "c",
        tuple(_render(ctx, item) for item in items),
        tuple(_arg_edge_kind(item) for item in items),
    )


def _emit_factor(ctx: _LetEnv, expr: LetExprFactor) -> str:
    """Unroll a rank-1 `factor` expression into the `c(...)` combine
    call BUGS / JAGS use for a vector literal.

    Only the single-binder form has a vector spelling. A multi-binder
    factor denotes a rank-`n` tensor, and neither language has a
    reshape that would turn a flat `c(...)` back into one, so the
    tensor form is emitted as one relation per cell by
    [`factor_cells`][quivers.transpile.renderers._bugs_helpers.factor_cells]
    at the statement level rather than as an expression here.
    """
    cells = factor_cells(ctx, expr)
    if len(expr.binders) != 1:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: a "
                f"{len(expr.binders)}-binder factor denotes a rank-"
                f"{len(expr.binders)} tensor, which has no BUGS / "
                f"JAGS expression form; only a statement-level "
                f"binding unrolls it"
            ],
        )
    elements = tuple(_render(ctx, body) for _, body in cells)
    element_kinds = tuple(_arg_edge_kind(body) for _, body in cells)
    return _emit_call(ctx, "c", elements, element_kinds)


def factor_cells(
    ctx: _BugsLetCtx, expr: LetExprFactor
) -> tuple[tuple[tuple[int, ...], LetExprNode], ...]:
    """Enumerate a `factor` expression's cells.

    Returns one `(indices, body)` pair per point of the binders'
    product index space, row-major, with every binder substituted by
    its integer coordinate in the body. `indices` counts from zero,
    matching the QVR origin; the caller rebases it when it writes the
    left-hand subscripts.

    The single-axis `cases` form labels each case body by integer;
    the helper enumerates `0, ..., |I|-1` and looks up the matching
    case, raising when the label set is incomplete.
    """
    if not expr.binders:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: empty binder "
                f"list is structurally ill-formed"
            ],
        )
    sizes = tuple(_factor_axis_size(ctx, b) for b in expr.binders)
    if expr.cases:
        if len(expr.binders) != 1:
            raise UnsupportedConstruct(
                f"qvr-{_target(ctx)}-helper",
                [
                    f"let-expr:LetExprFactor:{_target(ctx)}: cases "
                    f"form requires exactly one binder, got "
                    f"{len(expr.binders)}"
                ],
            )
        return _factor_case_cells(
            ctx, expr.binders[0], sizes[0], expr.cases
        )
    if expr.body is None:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [f"let-expr:LetExprFactor:{_target(ctx)}: missing body and no cases"],
        )
    body = expr.body
    return tuple(
        (
            indices,
            _substitute(
                body,
                dict(
                    zip(
                        (b.var for b in expr.binders), indices, strict=True
                    )
                ),
            ),
        )
        for indices in _enumerate_indices(sizes)
    )


def _factor_case_cells(
    ctx: _BugsLetCtx,
    binder: LetFactorBinder,
    size: int,
    cases: tuple[LetFactorCase, ...],
) -> tuple[tuple[tuple[int, ...], LetExprNode], ...]:
    """Enumerate the label-keyed case list as `((label,), body)` cells."""
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
    return tuple(((label,), by_label[label]) for label in range(size))


def factor_axis_sizes(
    ctx: _BugsLetCtx, expr: LetExprFactor
) -> tuple[int, ...]:
    """Return the static extent of each of `expr`'s binder axes."""
    return tuple(_factor_axis_size(ctx, b) for b in expr.binders)


# ---------------------------------------------------------------------------
# Factor-binder support: axis-size lookup and body substitution.
# ---------------------------------------------------------------------------


def _factor_axis_size(ctx: _BugsLetCtx, binder: LetFactorBinder) -> int:
    """Resolve a factor binder's axis to a static integer size.

    Looks up the binder's index expression in ``ctx.cards``. Raises
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


def _object_expr_axis_name(ctx: _BugsLetCtx, obj: ObjectExpr) -> str:
    """Resolve an `ObjectExpr` to the axis name a `cards` lookup wants.

    Handles `TypeName` directly and `DiscreteConstructor("FinSet", N)`
    as a literal anonymous axis (the size is the integer arg).
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


def _enumerate_indices(
    sizes: tuple[int, ...],
) -> list[tuple[int, ...]]:
    """Row-major enumeration of `product(range(s) for s in sizes)`."""
    if not sizes:
        return [()]
    head, *rest = sizes
    rest_tup = tuple(rest)
    tails = _enumerate_indices(rest_tup)
    return [(i, *tail) for i in range(head) for tail in tails]


def _substitute(expr: LetExprNode, env: dict[str, int]) -> LetExprNode:
    """Substitute integer literals for every `LetExprVar` whose name
    appears in `env`. Recurses through every `LetExprNode` shape."""
    if isinstance(expr, LetExprVar):
        v = env.get(expr.name)
        if v is None:
            return expr
        return LetExprLiteral(value=float(v))
    if isinstance(expr, LetExprLiteral):
        return expr
    if isinstance(expr, LetExprString):
        return expr
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_substitute(expr.left, env),
            right=_substitute(expr.right, env),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(operand=_substitute(expr.operand, env))
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(_substitute(a, env) for a in expr.args),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_substitute(expr.array, env),
            indices=tuple(_substitute(i, env) for i in expr.indices),
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(_substitute(i, env) for i in expr.items),
        )
    if isinstance(expr, LetExprLambda):
        # Shadowing: drop the bound name from the substitution.
        inner_env = {k: v for k, v in env.items() if k != expr.param}
        return LetExprLambda(
            param=expr.param,
            body=_substitute(expr.body, inner_env),
        )
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_substitute(expr.receiver, env),
            method=expr.method,
            args=tuple(_substitute(a, env) for a in expr.args),
        )
    if isinstance(expr, LetExprFactor):
        # Shadowing: drop any rebound names from the substitution.
        bound = {b.var for b in expr.binders}
        inner_env = {k: v for k, v in env.items() if k not in bound}
        return LetExprFactor(
            binders=expr.binders,
            body=(_substitute(expr.body, inner_env) if expr.body is not None else None),
            cases=tuple(
                LetFactorCase(
                    label=c.label,
                    value=_substitute(c.value, inner_env),
                    line=c.line,
                    col=c.col,
                )
                for c in expr.cases
            ),
        )
    return expr


# ---------------------------------------------------------------------------
# Edge-kind lookup: the grammar kind that a child vertex contributes.
# ---------------------------------------------------------------------------


def _arg_edge_kind(expr: LetExprNode) -> str:
    """Return the grammar kind a child vertex registers under.

    BUGS / JAGS `argument_list` / `index_list` discriminate child
    alternatives by vertex kind (the panproto walker reads the child's
    actual kind from the parent's `chose-alt-child-kinds` slot when
    set; otherwise it uses the edge label). The helper labels every
    edge from a list-like parent with the child's grammar kind so the
    parent stays compatible with either resolution path.
    """
    if isinstance(expr, LetExprLiteral):
        return "number"
    if isinstance(expr, LetExprVar):
        return "identifier"
    if isinstance(expr, LetExprBinOp):
        return "binary_expression"
    if isinstance(expr, LetExprUnaryOp):
        return "unary_expression"
    if isinstance(expr, LetExprCall):
        return "function_call"
    if isinstance(expr, LetExprIndex):
        return "indexed_variable"
    if isinstance(expr, (LetExprList, LetExprFactor)):
        # List / factor unroll to `c(...)`, a `function_call`.
        return "function_call"
    # Lambda / method-call / string have no parent edge kind because
    # the helper raises before reaching the parent.
    return "identifier"


def _target(ctx: _BugsLetCtx) -> str:
    """Read the ctx's `target` tag for error messages."""
    return getattr(ctx, "target", "bugs")


# ---------------------------------------------------------------------------
# Shared IR pre-pass: give every empty-plate IRDeterministic whose value
# is axis-carrying the plate it needs to emit as a loop. BUGS and JAGS
# each lack a scalar-to-vector broadcast operator and lift no infix
# operator over an axis, so `let mu = a + b * x_design` and
# `let beta = tau * lambda_local * z_raw` are both rank errors at the
# top level; the only spelling either language has for them is
# ``for (i in 1:N) { mu[i] <- ... }``. Two rules supply the missing
# plate: the plate of the deterministic's first downstream consumer,
# and the plate the expression's own operands already carry.
# ---------------------------------------------------------------------------


def push_scalar_dets_into_loops(ir: IRProgram) -> IRProgram:
    """Lift every empty-plate `IRDeterministic` that denotes an
    axis-carrying value into the plate that value ranges over.

    Two rules supply the plate, in order:

    1. The *consumer* rule covers a deterministic whose expression
       references a plate-less free data input: the plate is that of
       the first `IRObserve` / `IRSample` whose args contain an
       `IRArgRef` to the deterministic's bound name, and the
       referenced free data inputs are retagged with it so subsequent
       emission rebroadcasts them consistently.
    2. The *operand* rule covers a deterministic whose expression
       combines bindings that already carry a plate of their own (the
       horseshoe product ``tau * lambda_local * z_raw`` over a
       coefficient axis): the plate is the one every bare
       positively-ranked operand shares.

    A deterministic whose operands disagree on their axes, or whose
    value the reduction rewrites have already collapsed to a scalar
    (``sum(z * w)`` lowering to ``inprod``), is left alone so the
    emitter reports the shape it cannot spell rather than inventing a
    loop for it.
    """
    det_plate, input_plate_overrides = _consumer_plate_lifts(ir)
    det_plate.update(_operand_plate_lifts(ir, det_plate))
    if not det_plate:
        return ir
    new_inputs = tuple(
        IRDataInput(
            name=inp.name,
            constraint=inp.constraint,
            plate=input_plate_overrides.get(inp.name, inp.plate),
        )
        for inp in ir.inputs
    )
    new_body: list[IRNode] = []
    for node in ir.body:
        if isinstance(node, IRDeterministic) and node.name in det_plate:
            new_body.append(
                IRDeterministic(
                    name=node.name,
                    expr=node.expr,
                    constraint=node.constraint,
                    plate=det_plate[node.name],
                )
            )
        else:
            new_body.append(node)
    return IRProgram(
        name=ir.name,
        inputs=new_inputs,
        body=tuple(new_body),
        cards=ir.cards,
    )


def _consumer_plate_lifts(
    ir: IRProgram,
) -> tuple[dict[str, Plate], dict[str, Plate]]:
    """The consumer rule of
    [`push_scalar_dets_into_loops`][quivers.transpile.renderers._bugs_helpers.push_scalar_dets_into_loops].

    Returns the plate each lifted deterministic acquires alongside the
    plate override each free data input it reads acquires with it.
    """
    free_input_names: set[str] = set()
    for inp in ir.inputs:
        if not inp.plate.batch_dims and not inp.plate.event_dims:
            free_input_names.add(inp.name)
    det_to_free_refs: dict[str, frozenset[str]] = {}
    for node in ir.body:
        if not isinstance(node, IRDeterministic):
            continue
        if node.plate.batch_dims or node.plate.event_dims:
            continue
        free_refs = collect_letexpr_vars(node.expr) & free_input_names
        if free_refs:
            det_to_free_refs[node.name] = frozenset(free_refs)
    if not det_to_free_refs:
        return {}, {}
    det_consumer_plate: dict[str, Plate] = {}
    for node in ir.body:
        if isinstance(node, (IRObserve, IRSample)) and (
            node.plate.batch_dims or node.plate.event_dims
        ):
            referenced = collect_irargref_names(node.args)
            for det_name in det_to_free_refs:
                if det_name in referenced and det_name not in det_consumer_plate:
                    det_consumer_plate[det_name] = node.plate
    input_plate_overrides: dict[str, Plate] = {}
    for det_name, free_refs in det_to_free_refs.items():
        consumer_plate = det_consumer_plate.get(det_name)
        if consumer_plate is None:
            continue
        for free_ref in free_refs:
            input_plate_overrides[free_ref] = consumer_plate
    return det_consumer_plate, input_plate_overrides


def _operand_plate_lifts(
    ir: IRProgram, already_lifted: dict[str, Plate]
) -> dict[str, Plate]:
    """The operand rule of
    [`push_scalar_dets_into_loops`][quivers.transpile.renderers._bugs_helpers.push_scalar_dets_into_loops].

    A deterministic qualifies when its value still carries an axis
    ([`axis_rank`][quivers.transpile.renderers._bugs_helpers.axis_rank]
    is positive) and every bare operand that carries one agrees on a
    single batch-only plate. The map is built in body order and folded
    back into the declared-plate table as it goes, so a chain of
    deterministics each built from the one before lifts as a whole.
    """
    decl = build_decl_plates(ir)
    decl.update(already_lifted)
    out: dict[str, Plate] = {}
    for node in ir.body:
        if not isinstance(node, IRDeterministic):
            continue
        if node.plate.batch_dims or node.plate.event_dims:
            continue
        if node.name in already_lifted:
            continue
        if axis_rank(decl, node.expr) <= 0:
            continue
        plate = _shared_operand_plate(node.expr, decl)
        if plate is None:
            continue
        out[node.name] = plate
        decl[node.name] = plate
    return out


def _shared_operand_plate(
    expr: LetExprNode, decl_plates: dict[str, Plate]
) -> Plate | None:
    """The single batch-only plate every bare axis-carrying operand of
    `expr` shares, or `None` when they disagree or none carries one.

    Only bare `LetExprVar` operands count. A subscripted reference has
    already consumed the axes its indices supply, so the loop variable
    the lift would introduce has no position to occupy in it, and a
    plate carrying event dims needs a slice rather than a loop index.
    Both leave the deterministic where it is.
    """
    plates: list[Plate] = []
    _collect_bare_var_plates(expr, decl_plates, plates)
    if not plates:
        return None
    first = plates[0]
    if first.event_dims:
        return None
    for plate in plates[1:]:
        if plate.batch_dims != first.batch_dims or plate.event_dims:
            return None
    return first


def _collect_bare_var_plates(
    expr: LetExprNode, decl_plates: dict[str, Plate], out: list[Plate]
) -> None:
    """Append the declared plate of every bare axis-carrying variable
    reachable from `expr` without passing through a subscript."""
    if isinstance(expr, LetExprVar):
        plate = decl_plates.get(expr.name)
        if plate is not None and plate.batch_dims:
            out.append(plate)
        return
    if isinstance(expr, LetExprBinOp):
        _collect_bare_var_plates(expr.left, decl_plates, out)
        _collect_bare_var_plates(expr.right, decl_plates, out)
        return
    if isinstance(expr, LetExprUnaryOp):
        _collect_bare_var_plates(expr.operand, decl_plates, out)
        return
    if isinstance(expr, LetExprCall):
        for arg in expr.args:
            _collect_bare_var_plates(arg, decl_plates, out)
        return
    if isinstance(expr, LetExprIndex):
        for index in expr.indices:
            _collect_bare_var_plates(index, decl_plates, out)
        return


def collect_letexpr_vars(expr: LetExprNode) -> frozenset[str]:
    """Collect every bare-variable name in a let-expression tree."""
    if isinstance(expr, LetExprVar):
        return frozenset({expr.name})
    if isinstance(expr, LetExprLiteral):
        return frozenset()
    if isinstance(expr, LetExprBinOp):
        return collect_letexpr_vars(expr.left) | collect_letexpr_vars(expr.right)
    if isinstance(expr, LetExprUnaryOp):
        return collect_letexpr_vars(expr.operand)
    if isinstance(expr, LetExprCall):
        out: frozenset[str] = frozenset()
        for a in expr.args:
            out = out | collect_letexpr_vars(a)
        return out
    if isinstance(expr, LetExprIndex):
        out2: frozenset[str] = collect_letexpr_vars(expr.array)
        for ix in expr.indices:
            out2 = out2 | collect_letexpr_vars(ix)
        return out2
    return frozenset()


def collect_irargref_names(args: tuple[IRArg, ...]) -> frozenset[str]:
    """Collect every `IRArgRef.name` reachable via the arg tuple."""
    out: set[str] = set()
    for a in args:
        _collect_irargref_names_into(a, out)
    return frozenset(out)


def _collect_irargref_names_into(arg: IRArg, out: set[str]) -> None:
    if isinstance(arg, IRArgRef):
        out.add(arg.name)
        for ix in arg.indices:
            _collect_irargref_names_into(ix, out)
        return
    if isinstance(arg, IRArgBroadcast):
        _collect_irargref_names_into(arg.value, out)
        return
    if isinstance(arg, IRArgList):
        for el in arg.elements:
            _collect_irargref_names_into(el, out)
        return
    if isinstance(arg, IRArgMatrix):
        for row in arg.rows:
            for el in row.elements:
                _collect_irargref_names_into(el, out)
        return
    # `IRArgTransform` (renderer-local wrapper) is structurally a
    # nested IRArg; treat it as opaque here -- nothing in the
    # pre-pass examines transform-wrapped args before the renderer
    # injects them downstream.


def build_decl_plates(ir: IRProgram) -> dict[str, Plate]:
    """Build the declared-plate map for every named binding.

    Combines `ir.inputs` and every node in `ir.body` so the let-expr
    re-indexer can look up the plate of any reference encountered.
    """
    out: dict[str, Plate] = {}
    for inp in ir.inputs:
        out[inp.name] = inp.plate
    stack: list[IRNode] = list(ir.body)
    while stack:
        node = stack.pop()
        if isinstance(node, IRSample):
            out[node.name] = node.plate
        elif isinstance(node, IRObserve):
            out[node.name] = node.plate
        elif isinstance(node, IRDeterministic):
            out[node.name] = node.plate
        elif isinstance(node, IRMarginalize):
            out[node.latent] = node.plate
            stack.extend(node.scope)
    return out


def index_letexpr_refs(
    expr: LetExprNode,
    decl_plates: dict[str, Plate],
    enclosing_plate: Plate,
    loop_names: tuple[str, ...],
) -> LetExprNode:
    """Rewrite each `LetExprVar` whose declared plate shares axes with
    ``enclosing_plate`` into a `LetExprIndex` indexed by the matching
    loop variables.

    Axes are matched by name: for each axis in the var's declared
    batch_dims the helper looks up the parallel loop variable in
    ``enclosing_plate`` and emits it as the index expression. Vars
    whose declared plate has no axes in common with the surrounding
    loop stay as bare names so they broadcast as constants per
    iteration; vars whose declared plate has an axis the surrounding
    loop does not iterate are also left bare so the emitter can flag
    the shape mismatch downstream rather than silently picking the
    wrong loop variable.
    """
    if not loop_names:
        return expr
    axis_to_loop: dict[str, str] = {}
    for dim, lname in zip(enclosing_plate.batch_dims, loop_names, strict=True):
        axis_to_loop[dim.name] = lname
    return _index_letexpr_refs_inner(expr, decl_plates, axis_to_loop)


def _index_letexpr_refs_inner(
    expr: LetExprNode,
    decl_plates: dict[str, Plate],
    axis_to_loop: dict[str, str],
) -> LetExprNode:
    if isinstance(expr, LetExprVar):
        plate = decl_plates.get(expr.name)
        if plate is None or not plate.batch_dims:
            return expr
        indices: list[LetExprNode] = []
        for dim in plate.batch_dims:
            lname = axis_to_loop.get(dim.name)
            if lname is None:
                return expr
            indices.append(LetExprVar(name=lname))
        return LetExprIndex(
            array=LetExprVar(name=expr.name),
            indices=tuple(indices),
        )
    if isinstance(expr, LetExprLiteral):
        return expr
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_index_letexpr_refs_inner(expr.left, decl_plates, axis_to_loop),
            right=_index_letexpr_refs_inner(expr.right, decl_plates, axis_to_loop),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            operand=_index_letexpr_refs_inner(expr.operand, decl_plates, axis_to_loop),
        )
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(
                _index_letexpr_refs_inner(a, decl_plates, axis_to_loop)
                for a in expr.args
            ),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=expr.array,
            indices=tuple(
                _index_letexpr_refs_inner(ix, decl_plates, axis_to_loop)
                for ix in expr.indices
            ),
        )
    return expr


__all__ = [
    "axis_rank",
    "build_decl_plates",
    "collect_irargref_names",
    "collect_letexpr_vars",
    "dim_upper_text",
    "factor_axis_sizes",
    "factor_cells",
    "index_letexpr_refs",
    "push_scalar_dets_into_loops",
    "render_let_expr_bugs",
    "residual_event_dims",
    "split_event_dims",
]
