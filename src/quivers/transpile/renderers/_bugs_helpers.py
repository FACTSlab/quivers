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
    IRSample,
    Plate,
)


@runtime_checkable
class _BugsLetCtx(Protocol):
    """Structural protocol for the helper's ctx parameter."""

    target: str
    cards: dict[str, int]

    def fresh(self, prefix: str) -> str: ...
    def v(self, vid: str, kind: str) -> str: ...
    def e(self, src: str, tgt: str, kind: str) -> None: ...
    def lit(self, vid: str, text: str) -> None: ...
    def constraint(self, vid: str, sort: str, value: str) -> None: ...


def render_let_expr_bugs(ctx: _BugsLetCtx, expr: LetExprNode) -> str:
    """Build a BUGS / JAGS expression schema for ``expr`` in ``ctx``.

    Returns the root vertex id. Recurses into nested
    [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode] values.
    Raises [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
    when the construct has no representation in the BUGS / JAGS
    family (``LetExprString``, ``LetExprLambda``, ``LetExprMethodCall``,
    or a [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor]
    whose binders reference an axis of unknown static cardinality).
    """
    if isinstance(expr, LetExprLiteral):
        return _emit_number(ctx, expr.value)
    if isinstance(expr, LetExprVar):
        return _emit_identifier(ctx, expr.name)
    if isinstance(expr, LetExprBinOp):
        return _emit_binop(ctx, expr)
    if isinstance(expr, LetExprUnaryOp):
        return _emit_unary(ctx, expr)
    if isinstance(expr, LetExprCall):
        return _emit_call(
            ctx,
            expr.func,
            tuple(render_let_expr_bugs(ctx, a) for a in expr.args),
            tuple(_arg_edge_kind(a) for a in expr.args),
        )
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
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprMethodCall:{_target(ctx)}: BUGS / "
                f"JAGS have no method-dispatch syntax"
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


def _emit_binop(ctx: _BugsLetCtx, expr: LetExprBinOp) -> str:
    """Emit a `binary_expression` with `left`/`right` field edges.

    BUGS / JAGS `binary_expression` discriminates the operator via the
    grammar's CHOICE alternative; the panproto walker picks the alt
    from the `field:operator` + `chose-alt-fingerprint` pair.
    """
    b = ctx.v(ctx.fresh("be"), "binary_expression")
    ctx.constraint(b, "field:operator", expr.op)
    ctx.constraint(b, "chose-alt-fingerprint", expr.op)
    ctx.e(b, render_let_expr_bugs(ctx, expr.left), "left")
    ctx.e(b, render_let_expr_bugs(ctx, expr.right), "right")
    return b


def _emit_unary(ctx: _BugsLetCtx, expr: LetExprUnaryOp) -> str:
    """Emit a unary-minus `unary_expression` whose single child rides
    the `operand` field."""
    u = ctx.v(ctx.fresh("ue"), "unary_expression")
    ctx.constraint(u, "field:operator", "-")
    ctx.constraint(u, "chose-alt-fingerprint", "-")
    ctx.e(u, render_let_expr_bugs(ctx, expr.operand), "operand")
    return u


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
    alternative).
    """
    c = ctx.v(ctx.fresh("call"), "function_call")
    name_id = _emit_identifier(ctx, func)
    ctx.e(c, name_id, "name")
    if not arg_ids:
        return c
    al = ctx.v(ctx.fresh("al"), "argument_list")
    ctx.e(c, al, "arguments")
    for aid, akind in zip(arg_ids, arg_kinds, strict=True):
        ctx.e(al, aid, akind)
    return c


def _emit_index(ctx: _BugsLetCtx, expr: LetExprIndex) -> str:
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
        cid = render_let_expr_bugs(ctx, idx)
        ctx.e(il, cid, _arg_edge_kind(idx))
    return iv


def _emit_list(
    ctx: _BugsLetCtx, items: tuple[LetExprNode, ...]
) -> str:
    """Render a list literal as the BUGS / JAGS `c(...)` combine call.

    Neither language has an inline list-literal surface form; the
    canonical concatenation idiom is the built-in `c(...)` function
    (S-style combine), which both languages parse as a regular
    `function_call`.
    """
    return _emit_call(
        ctx,
        "c",
        tuple(render_let_expr_bugs(ctx, item) for item in items),
        tuple(_arg_edge_kind(item) for item in items),
    )


def _emit_factor(ctx: _BugsLetCtx, expr: LetExprFactor) -> str:
    """Unroll a `factor` expression over static-size binders into a
    `c(<body[i_1=0, ...]>, <body[i_1=1, ...]>, ...)` combine call.

    The factor binders' product index space is enumerated row-major;
    for each tuple of integer assignments the body is rendered with
    the bound variables substituted by their integer values. The
    single-axis `cases` form labels each case body by integer; the
    helper enumerates `0, ..., |I|-1` and looks up the matching case.
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
        return _emit_factor_cases(
            ctx, expr.binders[0], sizes[0], expr.cases
        )
    if expr.body is None:
        raise UnsupportedConstruct(
            f"qvr-{_target(ctx)}-helper",
            [
                f"let-expr:LetExprFactor:{_target(ctx)}: missing "
                f"body and no cases"
            ],
        )
    elements: list[str] = []
    element_kinds: list[str] = []
    for indices in _enumerate_indices(sizes):
        subst = dict(zip((b.var for b in expr.binders), indices, strict=True))
        substituted = _substitute(expr.body, subst)
        elements.append(render_let_expr_bugs(ctx, substituted))
        element_kinds.append(_arg_edge_kind(substituted))
    return _emit_call(ctx, "c", tuple(elements), tuple(element_kinds))


def _emit_factor_cases(
    ctx: _BugsLetCtx,
    binder: LetFactorBinder,
    size: int,
    cases: tuple[LetFactorCase, ...],
) -> str:
    """Build `c(<case[0].value>, ..., <case[size-1].value>)` from the
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
    elements: list[str] = []
    element_kinds: list[str] = []
    for label in range(size):
        body = by_label[label]
        elements.append(render_let_expr_bugs(ctx, body))
        element_kinds.append(_arg_edge_kind(body))
    return _emit_call(ctx, "c", tuple(elements), tuple(element_kinds))


# ---------------------------------------------------------------------------
# Factor-binder support: axis-size lookup and body substitution.
# ---------------------------------------------------------------------------


def _factor_axis_size(
    ctx: _BugsLetCtx, binder: LetFactorBinder
) -> int:
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


def _object_expr_axis_name(
    ctx: _BugsLetCtx, obj: ObjectExpr
) -> str:
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


def _substitute(
    expr: LetExprNode, env: dict[str, int]
) -> LetExprNode:
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
            body=(
                _substitute(expr.body, inner_env)
                if expr.body is not None
                else None
            ),
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
# Shared IR pre-pass: lift empty-plate IRDeterministic nodes into the
# plate of their first downstream consumer. BUGS and JAGS each lack a
# scalar-to-vector broadcast operator, so a `let mu = a + b * x_design`
# top-level emit becomes invalid the moment ``x_design`` is a vector
# supplied via the data list; this helper rewrites the IR so the
# deterministic and its free-data-input dependencies acquire the
# consumer's plate and emit as ``for (i in 1:N) { mu[i] <- ... }``.
# ---------------------------------------------------------------------------


def push_scalar_dets_into_loops(ir: IRProgram) -> IRProgram:
    """Lift each empty-plate `IRDeterministic` whose expression
    references a plate-less free data input into the plate of the
    first downstream consumer.

    The consumer is the first `IRObserve` / `IRSample` whose args
    contain an `IRArgRef` to the deterministic's bound name. The
    referenced free data inputs are retagged with that consumer's
    plate so subsequent emission rebroadcasts them consistently.
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
        return ir
    det_consumer_plate: dict[str, Plate] = {}
    for node in ir.body:
        if isinstance(node, (IRObserve, IRSample)) and (
            node.plate.batch_dims or node.plate.event_dims
        ):
            referenced = collect_irargref_names(node.args)
            for det_name in det_to_free_refs:
                if det_name in referenced and det_name not in det_consumer_plate:
                    det_consumer_plate[det_name] = node.plate
    if not det_consumer_plate:
        return ir
    input_plate_overrides: dict[str, Plate] = {}
    for det_name, free_refs in det_to_free_refs.items():
        consumer_plate = det_consumer_plate.get(det_name)
        if consumer_plate is None:
            continue
        for free_ref in free_refs:
            input_plate_overrides[free_ref] = consumer_plate
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
        if isinstance(node, IRDeterministic) and node.name in det_consumer_plate:
            new_body.append(
                IRDeterministic(
                    name=node.name,
                    expr=node.expr,
                    constraint=node.constraint,
                    plate=det_consumer_plate[node.name],
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


def collect_letexpr_vars(expr: LetExprNode) -> frozenset[str]:
    """Collect every bare-variable name in a let-expression tree."""
    if isinstance(expr, LetExprVar):
        return frozenset({expr.name})
    if isinstance(expr, LetExprLiteral):
        return frozenset()
    if isinstance(expr, LetExprBinOp):
        return collect_letexpr_vars(expr.left) | collect_letexpr_vars(
            expr.right
        )
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
    for dim, lname in zip(
        enclosing_plate.batch_dims, loop_names, strict=True
    ):
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
            right=_index_letexpr_refs_inner(
                expr.right, decl_plates, axis_to_loop
            ),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            operand=_index_letexpr_refs_inner(
                expr.operand, decl_plates, axis_to_loop
            ),
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
    "render_let_expr_bugs",
    "push_scalar_dets_into_loops",
    "build_decl_plates",
    "index_letexpr_refs",
    "collect_letexpr_vars",
    "collect_irargref_names",
]
