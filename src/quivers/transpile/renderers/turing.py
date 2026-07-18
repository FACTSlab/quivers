"""Turing.jl renderer: consume an [`IRProgram`][quivers.transpile.ir.IRProgram]
and emit a Julia panproto schema for the Turing.jl model idiom.

Output shape (LDA canonical):

```julia
@model function model(alpha, beta, word_idx, w)
    theta ~ filldist(Dirichlet(fill(alpha, 3)), 20)
    phi ~ filldist(Dirichlet(fill(beta, 200)), 3)
    z ~ filldist(Categorical(theta), 20)
    w .~ Categorical.(eachrow(phi)[z[word_idx]])
    return theta
end
```

Per the Turing.jl idiom the renderer:

* declares no separate variable bindings; declarations are the `~` /
  `.~` statements themselves;
* lifts each batch axis into a `filldist(<Family>(<args>), B)` call
  when none of the family's args depend on that batch's index; into
  an `arraydist([<Family>(<args[i]>) for i in 1:B])` call when at
  least one arg has an index expression rooted in that batch;
* drops [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] into an
  explicit [`IRSample`][quivers.transpile.ir.IRSample] plus the
  scoped body inline, via the shared
  [`RendererBase.explicit_latent_scope`][quivers.transpile.renderers._base.RendererBase.explicit_latent_scope]
  helper (Turing samples discrete latents natively, no `log_sum_exp`);
* broadcasts scalars to vector / matrix shapes via Julia's `fill`;
* renders list literals as `[<e0>, <e1>, ...]` (Julia vector_expression)
  and matrix literals as `[<row0>; <row1>; ...]`
  (Julia matrix_expression);
* renders an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
  reference to a `Truncated` wrapper as `truncated(<base>, lower=L,
  upper=U)`.
"""

from __future__ import annotations

import pathlib

import panproto
import torch.distributions.constraints as _torch_constraints

from quivers.dsl.ast_nodes import (
    ExportDecl,
    ExprIdent,
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
    Module,
    ProgramDecl,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile._pipeline import (
    EmitPretty,
    parser_registry,
    target_protocol,
)
from quivers.transpile.renderers._julia_helpers import (
    render_let_expr_julia,
)
from quivers.transpile._resolve import (
    build_let_table,
    build_morphism_table,
)
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
from quivers.transpile.ir import (
    ConstraintSpec,
    Dim,
    DimDynamic,
    DimStatic,
    IRArg,
    IRArgBroadcast,
    IRArgFamilyRef,
    IRArgKernel,
    IRArgList,
    IRArgMatrix,
    IRArgNumber,
    IRArgRef,
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRReturn,
    IRSample,
    IRScore,
    Plate,
)
from quivers.transpile.lower import Lower
from quivers.transpile.renderers._base import (
    BlockKind,
    RendererBase,
    SchemaFragment,
    _RenderCtx,
)


# ---------------------------------------------------------------------------
# Julia tree-sitter helpers: scoped to the Turing renderer; not promoted
# to `_base.py` because no other backend builds Julia source.
# ---------------------------------------------------------------------------


def _vertex(sb: panproto.SchemaBuilder, counter: list[int], kind: str) -> str:
    """Allocate a fresh vertex id and register it under `kind`."""
    counter[0] += 1
    vid = f"v{counter[0]}"
    sb.vertex(vid, kind)
    return vid


def _literal(
    sb: panproto.SchemaBuilder, counter: list[int], kind: str, text: str
) -> str:
    """Register a vertex of `kind` whose `literal-value` constraint
    carries `text`."""
    vid = _vertex(sb, counter, kind)
    sb.constraint(vid, "literal-value", text)
    return vid


def _identifier(
    sb: panproto.SchemaBuilder, counter: list[int], text: str
) -> str:
    return _literal(sb, counter, "identifier", text)


def _operator(
    sb: panproto.SchemaBuilder, counter: list[int], text: str
) -> str:
    return _literal(sb, counter, "operator", text)


def _integer(
    sb: panproto.SchemaBuilder, counter: list[int], value: int
) -> str:
    return _literal(sb, counter, "integer_literal", str(value))


def _float(
    sb: panproto.SchemaBuilder, counter: list[int], value: float
) -> str:
    return _literal(sb, counter, "float_literal", repr(value))


def _number(
    sb: panproto.SchemaBuilder, counter: list[int], value: float
) -> str:
    """Pick `integer_literal` for whole values; `float_literal` for the
    rest. Mirrors the Julia source idiom of writing `3` instead of
    `3.0` for plate sizes and integer arguments."""
    if isinstance(value, int) or (
        isinstance(value, float) and value.is_integer()
    ):
        return _integer(sb, counter, int(value))
    return _float(sb, counter, float(value))


def _argument_list(
    sb: panproto.SchemaBuilder, counter: list[int], children: tuple[str, ...]
) -> str:
    """Build an `argument_list` vertex aliased from `tuple_expression`.

    Julia tree-sitter aliases `argument_list` from `tuple_expression`;
    panproto's emitter uses the `pre-alias-symbol` constraint to route
    the production to the parenthesised form.
    """
    vid = _vertex(sb, counter, "argument_list")
    sb.constraint(vid, "pre-alias-symbol", "tuple_expression")
    for child in children:
        sb.edge(vid, child, "child_of")
    return vid


def _call(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    callee: str,
    children: tuple[str, ...],
) -> str:
    """`<callee>(<children[0]>, <children[1]>, ...)`."""
    vid = _vertex(sb, counter, "call_expression")
    sb.edge(vid, callee, "child_of")
    sb.edge(vid, _argument_list(sb, counter, children), "child_of")
    return vid


def _broadcast_call(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    callee: str,
    children: tuple[str, ...],
) -> str:
    """`<callee>.(<children[0]>, <children[1]>, ...)` (the dot-broadcast
    form of a function call)."""
    vid = _vertex(sb, counter, "broadcast_call_expression")
    sb.edge(vid, callee, "child_of")
    sb.edge(vid, _argument_list(sb, counter, children), "child_of")
    return vid


def _index_expr(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    array: str,
    indices: tuple[str, ...],
) -> str:
    """`<array>[<indices[0]>, <indices[1]>, ...]`.

    Julia's `index_expression` production is `primary [ _array ]` where
    `_array` is one of `vector_expression`, `matrix_expression`, or
    `comprehension_expression`. The index list is wrapped in a
    `vector_expression` so the pretty emitter selects the bracketed
    form.
    """
    vid = _vertex(sb, counter, "index_expression")
    sb.edge(vid, array, "child_of")
    inner = _vertex(sb, counter, "vector_expression")
    for idx in indices:
        sb.edge(inner, idx, "child_of")
    sb.edge(vid, inner, "child_of")
    return vid


def _assignment(
    sb: panproto.SchemaBuilder, counter: list[int], lhs: str, rhs: str
) -> str:
    """`<lhs> = <rhs>` (regular Julia assignment)."""
    vid = _vertex(sb, counter, "assignment")
    sb.edge(vid, lhs, "child_of")
    sb.edge(vid, _operator(sb, counter, "="), "child_of")
    sb.edge(vid, rhs, "child_of")
    return vid


def _tilde(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    lhs: str,
    rhs: str,
    *,
    broadcast: bool = False,
) -> str:
    """`<lhs> ~ <rhs>` or `<lhs> .~ <rhs>` (Turing.jl sampling).

    Encoded as a `compound_assignment_expression` whose operator is the
    literal `~` or `.~` token.
    """
    vid = _vertex(sb, counter, "compound_assignment_expression")
    sb.edge(vid, lhs, "child_of")
    sb.edge(vid, _operator(sb, counter, ".~" if broadcast else "~"), "child_of")
    sb.edge(vid, rhs, "child_of")
    return vid


def _vector_literal(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    elements: tuple[str, ...],
) -> str:
    """`[<e0>, <e1>, ...]` (Julia `vector_expression`)."""
    vid = _vertex(sb, counter, "vector_expression")
    for el in elements:
        sb.edge(vid, el, "child_of")
    return vid


def _matrix_literal(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    rows: tuple[tuple[str, ...], ...],
) -> str:
    """`[<r0[0]> <r0[1]> ...; <r1[0]> <r1[1]> ...; ...]` (Julia
    `matrix_expression`)."""
    vid = _vertex(sb, counter, "matrix_expression")
    for row in rows:
        row_vid = _vertex(sb, counter, "matrix_row")
        for el in row:
            sb.edge(row_vid, el, "child_of")
        sb.edge(vid, row_vid, "child_of")
    return vid


def _range(
    sb: panproto.SchemaBuilder, counter: list[int], lo: str, hi: str
) -> str:
    """`<lo>:<hi>` (Julia `range_expression`, an infix `:`)."""
    vid = _vertex(sb, counter, "range_expression")
    sb.edge(vid, lo, "child_of")
    sb.edge(vid, hi, "child_of")
    return vid


def _comprehension(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    body: str,
    binder_var: str,
    binder_range: str,
) -> str:
    """`[<body> for <binder_var> in <binder_range>]` (Julia comprehension).

    Used by the `arraydist([... for i in 1:B])` index-dependent fallback.
    """
    comp = _vertex(sb, counter, "comprehension_expression")
    sb.edge(comp, body, "child_of")
    fc = _vertex(sb, counter, "for_clause")
    fb = _vertex(sb, counter, "for_binding")
    sb.edge(fb, _identifier(sb, counter, binder_var), "child_of")
    sb.edge(fb, binder_range, "child_of")
    sb.edge(fc, fb, "child_of")
    sb.edge(comp, fc, "child_of")
    return comp


def _return(
    sb: panproto.SchemaBuilder, counter: list[int], value: str
) -> str:
    """`return <value>`."""
    vid = _vertex(sb, counter, "return_statement")
    sb.edge(vid, value, "child_of")
    return vid


def _function_def(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    *,
    name: str,
    params: tuple[str, ...],
    body_vid: str,
) -> str:
    """`function <name>(<params...>) <body> end`."""
    fn = _vertex(sb, counter, "function_definition")
    sig = _vertex(sb, counter, "signature")
    call = _vertex(sb, counter, "call_expression")
    sb.edge(call, _identifier(sb, counter, name), "child_of")
    args = _argument_list(
        sb, counter, tuple(_identifier(sb, counter, p) for p in params)
    )
    sb.edge(call, args, "child_of")
    sb.edge(sig, call, "child_of")
    sb.edge(fn, sig, "child_of")
    sb.edge(fn, body_vid, "child_of")
    return fn


def _macro_call(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    macro_name: str,
    body_vid: str,
) -> str:
    """`@<macro_name> <body>` (no parens; long-form).

    Used for `@model function ... end`.
    """
    mc = _vertex(sb, counter, "macrocall_expression")
    mid = _vertex(sb, counter, "macro_identifier")
    sb.edge(mid, _identifier(sb, counter, macro_name), "child_of")
    margs = _vertex(sb, counter, "macro_argument_list")
    sb.edge(margs, body_vid, "child_of")
    sb.edge(mc, mid, "child_of")
    sb.edge(mc, margs, "child_of")
    return mc


# ---------------------------------------------------------------------------
# `TuringRenderer`
# ---------------------------------------------------------------------------


class TuringRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] as a
    Turing.jl model.

    Overrides [`render`][quivers.transpile.renderers._base.RendererBase.render]
    to wrap the IR-walk with the `@model function model(...) ... end`
    prologue / epilogue Turing.jl requires. The per-node dispatches
    (`declare`, `sample`, `marginalize`, `broadcast`) accumulate Julia
    schema vertices into a single function body block.
    """

    target: str = "turing"

    def __init__(
        self,
        *,
        morphisms: dict | None = None,
        lets: dict | None = None,
    ) -> None:
        """Construct a renderer.

        `morphisms` and `lets` are the resolved tables from the surface
        module; the renderer consults `morphisms` when an
        [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] needs
        the referenced morphism's `~ Family(...)` init clause. Both
        default to empty so callers driving a synthetic
        [`IRProgram`][quivers.transpile.ir.IRProgram] (tests) can
        instantiate the renderer with no surface context.
        """
        self._morphisms: dict = morphisms or {}
        self._lets: dict = lets or {}

    # ----- protocol / context plumbing -----

    def target_protocol(self) -> panproto.Protocol:
        return target_protocol("julia")

    # ----- top-level render -----

    def render(self, ir: IRProgram) -> panproto.Schema:
        proto = self.target_protocol()
        sb = proto.schema()
        counter: list[int] = [0]
        # Source file shell.
        source = _vertex(sb, counter, "source_file")
        body = _vertex(sb, counter, "block")

        # The Turing.jl function signature carries every IRDataInput
        # in alphabetical name order. The in-container probe (which
        # has no access to the IR's lowering order) passes observed
        # values as positional args in `sort(collect(keys(data)))`
        # order; sorting here keeps the two sides in lockstep so
        # callers never have to know the original IR order.
        params = tuple(sorted(inp.name for inp in ir.inputs))

        # Walk the body. Each node emits a `~` / `=` / `return`
        # statement into `body`.
        ctx = _TuringCtx(
            sb=sb,
            morphisms=self._morphisms,
            lets=self._lets,
            counter=counter,
            cards=dict(ir.cards),
            body=body,
            input_plates={inp.name: inp.plate for inp in ir.inputs},
            sample_plates={},
            batch_shaped_names=set(),
        )
        # Pre-populate the sample-plate table by walking the body so
        # observe / marginalize bodies can detect index-dependent args
        # against the originating sample's plate.
        _seed_sample_plates(ctx, ir.body)
        # Pre-populate the batch-shaped-names table: every
        # IRDataInput / IRSample with non-empty batch_dims, plus every
        # IRDeterministic whose RHS transitively references one such
        # name. Drives the broadcast-dot fallback for observes whose
        # `loc=` etc. is a let-bound vector (e.g. `mu = a + b *
        # x_design` in bayes_linear_regression).
        _seed_batch_shaped(ctx, ir.body)
        for node in ir.body:
            self._dispatch(ctx, node)

        fn = _function_def(
            sb, counter, name="model", params=params, body_vid=body
        )
        macro = _macro_call(sb, counter, "model", fn)
        # Turing.jl + Distributions.jl ship a large catalogue of
        # distributions but lack `HalfStudentT` and `ContinuousBernoulli`.
        # When the IR samples or observes from either, graft the helper
        # at [`runtime_turing.jl`][quivers.transpile.runtime_turing] onto
        # the source above the `@model function model` macrocall so the
        # body's `~ HalfStudentT(...)` / `~ ContinuousBernoulli(...)`
        # call sites resolve through normal Julia name lookup.
        if any(
            _ir_uses_family(ir.body, f)
            for f in _TURING_RUNTIME_HELPER_FAMILIES
        ):
            _graft_runtime_turing_helper(sb, counter, source)
        sb.edge(source, macro, "child_of")
        return sb.build()

    # ----- IRNode dispatch (overrides RendererBase._dispatch_node) -----

    def _dispatch(self, ctx: _TuringCtx, node: IRNode) -> None:
        """Route one IR body node to the right Turing.jl emission."""
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(ctx, node)
                return
            self.sample(
                ctx,
                node.name,
                node.family,
                node.args,
                node.arg_names,
                node.constraint,
                node.plate,
                observed=False,
            )
            return
        if isinstance(node, IRObserve):
            self.sample(
                ctx,
                node.name,
                node.family,
                node.args,
                node.arg_names,
                node.constraint,
                node.plate,
                observed=True,
                via=node.via,
            )
            return
        if isinstance(node, IRDeterministic):
            self._emit_deterministic(ctx, node)
            return
        if isinstance(node, IRScore):
            self._emit_score(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._emit_return(ctx, node.names)
            return
        if isinstance(node, IRDataInput):
            return
        raise UnsupportedConstruct(
            "qvr-turing", [f"node:{type(node).__name__}"]
        )

    def _emit_gp_block(
        self,
        ctx: _TuringCtx,
        node: IRSample,
    ) -> None:
        """Emit a Gaussian-process sample as a triple of Julia
        statements inside the ``@model`` body:

            __gp_mean_f = zeros(N)
            __gp_cov_f  = _qvr_rbf_kernel(x, length_scale, jitter)
            f ~ MvNormal(__gp_mean_f, __gp_cov_f)

        The ``_qvr_rbf_kernel`` helper is defined in
        [`runtime_turing.jl`][quivers.transpile.runtime_turing] and
        is grafted onto the emit when GP is in the IR (handled by
        the existing
        [`_graft_runtime_turing_helper`][quivers.transpile.renderers.turing._graft_runtime_turing_helper]
        path with GP added to the helper-family set).
        """
        if len(node.args) != 2 or not isinstance(
            node.args[1], IRArgKernel
        ):
            raise UnsupportedConstruct(
                "qvr-turing",
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                "qvr-turing",
                [
                    f"family:GP:kernel:{kernel_arg.kernel}: only rbf "
                    f"is implemented"
                ],
            )
        sb, counter = ctx.sb, ctx.counter
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{node.name}"
        cov_name = f"__gp_cov_{node.name}"
        # __gp_mean_<name> = zeros(N)
        mean_lhs = _identifier(sb, counter, mean_name)
        mean_rhs = _call(
            sb, counter,
            _identifier(sb, counter, "zeros"),
            (_integer(sb, counter, n),),
        )
        sb.edge(
            ctx.body,
            _assignment(sb, counter, mean_lhs, mean_rhs),
            "child_of",
        )
        # __gp_cov_<name> = _qvr_rbf_kernel(x, ls, jitter)
        cov_lhs = _identifier(sb, counter, cov_name)
        cov_rhs = _call(
            sb, counter,
            _identifier(sb, counter, "_qvr_rbf_kernel"),
            (
                _identifier(sb, counter, x),
                _float(sb, counter, ls),
                _float(sb, counter, jitter),
            ),
        )
        sb.edge(
            ctx.body,
            _assignment(sb, counter, cov_lhs, cov_rhs),
            "child_of",
        )
        # f ~ MvNormal(__gp_mean_f, __gp_cov_f)
        f_lhs = _identifier(sb, counter, node.name)
        mvn_rhs = _call(
            sb, counter,
            _identifier(sb, counter, "MvNormal"),
            (
                _identifier(sb, counter, mean_name),
                _identifier(sb, counter, cov_name),
            ),
        )
        sb.edge(
            ctx.body,
            _tilde(sb, counter, f_lhs, mvn_rhs),
            "child_of",
        )
        ctx.sample_plates[node.name] = node.plate

    # ----- declare: Turing.jl has no separate declaration block -----

    def declare(
        self,
        ctx: _RenderCtx,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """No-op: in Turing.jl, declarations are the `~` statements
        themselves. The function signature is populated from the
        IRDataInput list in [`render`][TuringRenderer.render]."""
        del ctx, name, constraint, plate, block
        return ""

    # ----- sample / observe emission -----

    def sample(
        self,
        ctx: _RenderCtx,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        constraint: ConstraintSpec,
        plate: Plate,
        observed: bool,
        *,
        via: str | None = None,
    ) -> SchemaFragment:
        """Emit one `~` / `.~` statement for a sample or observe step.

        Plate handling:
        * `batch_dims=()` and no `via`: plain `name ~ Family(args)`.
        * `batch_dims=(B,)` and no arg references a batch-plated name
          via index: `name ~ filldist(Family(args), B)`.
        * Otherwise (index-dependent args or a `via` fibration):
          broadcast-dot form `name .~ Family.(idx_args)`, or the
          `arraydist([Family(args[i]) for i in 1:B])` fallback when no
          via is present.
        """
        del constraint  # output support already encoded in the family choice
        assert isinstance(ctx, _TuringCtx)
        ctx.sample_plates[name] = plate
        sb, counter = ctx.sb, ctx.counter

        meta = _meta_or_raise(family)
        target_dist = meta.target_names.get("turing")
        if target_dist is None:
            raise UnsupportedConstruct(
                "qvr-turing", [f"family:{family}: no Turing.jl mapping"]
            )

        # Detect index-dependent arg shapes against the surrounding
        # observe/sample's batch axes. The presence of `via` is the
        # strongest signal: it always rewrites the indexing to thread
        # through the fibration variable.
        index_dep = (
            via is not None
            or _args_have_batch_index(
                args,
                ctx.sample_plates,
                plate,
                ctx.batch_shaped_names,
            )
        )

        lhs = _identifier(sb, counter, name)

        if index_dep and observed and via is not None:
            # Index-dependent observe with a `via` fibration: emit the
            # broadcast-dot form `name .~ Family.(rewritten_args)`.
            rhs_args = tuple(
                _arg_to_julia(ctx, a, via=via, family=family)
                for a in args
            )
            family_callee = _identifier(sb, counter, target_dist)
            rhs = _broadcast_call(sb, counter, family_callee, rhs_args)
            stmt = _tilde(sb, counter, lhs, rhs, broadcast=True)
            sb.edge(ctx.body, stmt, "child_of")
            return ""

        if index_dep and observed and via is None:
            # Index-dependent observe whose arg(s) reference a
            # batch-shaped name (e.g. a let-bound vector) directly,
            # without a `via` fibration. The plain
            # `filldist(Family(args), B)` form requires the args to
            # be scalar; wrap the per-element distribution array in
            # `product_distribution(Family.(args))` and emit the
            # scalar tilde. DynamicPPL no longer accepts `.~` over
            # an array of distributions; `product_distribution` is
            # the supported replacement.
            rhs_args = tuple(
                _arg_to_julia(ctx, a, family=family) for a in args
            )
            family_callee = _identifier(sb, counter, target_dist)
            elemwise = _broadcast_call(sb, counter, family_callee, rhs_args)
            rhs = _call(
                sb,
                counter,
                _identifier(sb, counter, "product_distribution"),
                (elemwise,),
            )
            stmt = _tilde(sb, counter, lhs, rhs)
            sb.edge(ctx.body, stmt, "child_of")
            return ""

        if index_dep and not observed:
            # Index-dependent latent: fall back to `arraydist([Family(
            # args[i]) for i in 1:B])`. Pick the first batch dim as
            # the comprehension axis.
            if not plate.batch_dims:
                # No batch axis to iterate; treat as a scalar call.
                rhs = self._family_call(ctx, target_dist, args, family)
            else:
                rhs = self._arraydist_call(
                    ctx, target_dist, args, plate.batch_dims[0], family
                )
            stmt = _tilde(sb, counter, lhs, rhs)
            sb.edge(ctx.body, stmt, "child_of")
            return ""

        # Plain / batch-wrapped form. Apply filldist for each batch dim.
        # Pre-process args: a scalar IRArgRef against a vector-shaped
        # arg constraint (e.g. Dirichlet `concentration`) is wrapped in
        # `fill(<ref>, <event_size>)` so the emitted call matches the
        # Turing.jl idiom of a fully-shaped vector for the event-dim arg.
        promoted = tuple(
            _promote_scalar_ref(a, name, plate, meta)
            for name, a in zip(arg_names, args, strict=False)
        )
        dist = self._family_call(ctx, target_dist, promoted, family)
        for dim in plate.batch_dims:
            size_vid = _dim_to_size(sb, counter, dim)
            dist = _call(
                sb,
                counter,
                _identifier(sb, counter, "filldist"),
                (dist, size_vid),
            )
        stmt = _tilde(sb, counter, lhs, dist)
        sb.edge(ctx.body, stmt, "child_of")
        return ""

    # ----- marginalize: lower to explicit sample + scope inline -----

    def marginalize(
        self, ctx: _RenderCtx, node: IRMarginalize
    ) -> SchemaFragment:
        """Lower an [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        to an explicit [`IRSample`][quivers.transpile.ir.IRSample] of
        the latent followed by the scoped body inline.

        Turing.jl supports discrete latents natively (NUTS dispatches
        them automatically), so no `log_sum_exp` rewriting is needed;
        the shared
        [`RendererBase.explicit_latent_scope`][quivers.transpile.renderers._base.RendererBase.explicit_latent_scope]
        helper supplies the rewrite.
        """
        assert isinstance(ctx, _TuringCtx)
        rewritten = self.explicit_latent_scope(node)
        for child in rewritten:
            self._dispatch(ctx, child)
        return ""

    # ----- broadcast: Julia's fill(<value>, K) / fill(<value>, R, C) -----

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        assert isinstance(ctx, _TuringCtx)
        sb, counter = ctx.sb, ctx.counter
        value_vid = _arg_to_julia(ctx, value)
        shape_args = tuple(_integer(sb, counter, s) for s in target_shape)
        return _call(
            sb,
            counter,
            _identifier(sb, counter, "fill"),
            (value_vid, *shape_args),
        )

    # ----- helpers: family call construction -----

    def _family_call(
        self,
        ctx: _TuringCtx,
        target_dist: str,
        args: tuple[IRArg, ...],
        family: str,
    ) -> str:
        """Build `<TargetDist>(<args>)` for the no-batch-iteration case.

        Most families lower to a direct `<target_dist>(<args>)` call.
        Three QVR families have no native Turing.jl distribution and
        the renderer composes them out of the `truncated` wrapper:

        * `HalfNormal(sigma)` -> `truncated(Normal(0, sigma), 0, Inf)`
        * `HalfCauchy(gamma)` -> `truncated(Cauchy(0, gamma), 0, Inf)`
        * `TruncatedNormal(loc, scale, low, high)` ->
          `truncated(Normal(loc, scale), low, high)`

        Two QVR families take a rate parameter that the Distributions.jl
        equivalent expects as a scale `theta = 1/rate`; the renderer
        emits the inverse explicitly so the log-density matches:

        * `Exponential(rate)` -> `Exponential(1/rate)`
        * `Gamma(concentration, rate)` -> `Gamma(concentration, 1/rate)`

        The compositions are keyed on the family name (the FAMILY_META
        target_name for the half-truncated and `TruncatedNormal`
        families is `"truncated"`, which is the wrapper callable); a
        per-renderer recipe supplies the inner base distribution and
        argument layout because those choices are per-target lowering
        conventions rather than renderer-level dispatch on the QVR
        family discriminator.
        """
        sb, counter = ctx.sb, ctx.counter
        rhs_args = tuple(
            _arg_to_julia(ctx, a, family=family) for a in args
        )
        recipe = _HALF_TRUNCATED_BASES.get(family)
        if recipe is not None:
            base_name = recipe
            zero = _integer(sb, counter, 0)
            base_call = _call(
                sb,
                counter,
                _identifier(sb, counter, base_name),
                (zero, *rhs_args),
            )
            return _call(
                sb,
                counter,
                _identifier(sb, counter, target_dist),
                (
                    base_call,
                    _integer(sb, counter, 0),
                    _identifier(sb, counter, "Inf"),
                ),
            )
        if family == "TruncatedNormal":
            # args order: (loc, scale, low, high).
            if len(rhs_args) != 4:
                raise UnsupportedConstruct(
                    "qvr-turing",
                    [
                        f"family:TruncatedNormal: expected 4 args "
                        f"(loc, scale, low, high), got {len(rhs_args)}"
                    ],
                )
            base_call = _call(
                sb,
                counter,
                _identifier(sb, counter, "Normal"),
                (rhs_args[0], rhs_args[1]),
            )
            return _call(
                sb,
                counter,
                _identifier(sb, counter, target_dist),
                (base_call, rhs_args[2], rhs_args[3]),
            )
        if family in _RATE_TO_SCALE_INVERT_POSITIONS:
            rhs_args = _invert_rate_arg(
                sb, counter, rhs_args, _RATE_TO_SCALE_INVERT_POSITIONS[family]
            )
        return _call(
            sb,
            counter,
            _identifier(sb, counter, target_dist),
            rhs_args,
        )

    def _arraydist_call(
        self,
        ctx: _TuringCtx,
        target_dist: str,
        args: tuple[IRArg, ...],
        batch_dim: Dim,
        family: str,
    ) -> str:
        """Build `arraydist([<TargetDist>(<args[i]>) for i in 1:B])`."""
        sb, counter = ctx.sb, ctx.counter
        binder = _ARRAYDIST_BINDER
        # Rewrite arg references so a `name[k]` index becomes
        # `name[<binder>]` (when the existing index is itself a
        # reference into the batch axis).
        rewritten = tuple(
            _replace_first_index(a, binder) for a in args
        )
        body_call = _call(
            sb,
            counter,
            _identifier(sb, counter, target_dist),
            tuple(_arg_to_julia(ctx, a, family=family) for a in rewritten),
        )
        size_vid = _dim_to_size(sb, counter, batch_dim)
        rng = _range(sb, counter, _integer(sb, counter, 1), size_vid)
        comp = _comprehension(sb, counter, body_call, binder, rng)
        return _call(
            sb,
            counter,
            _identifier(sb, counter, "arraydist"),
            (comp,),
        )

    # ----- deterministic / score / return -----

    def _emit_deterministic(
        self, ctx: _TuringCtx, node: IRDeterministic
    ) -> None:
        """Emit one `<name> = <rhs>` or `<name> = @. <rhs>` assignment.

        Wraps the RHS in Julia's `@.` macro when the deterministic is
        batch-shaped (its `name` was pre-seeded into
        [`ctx.batch_shaped_names`][_TuringCtx]); this turns every
        arithmetic operator and function call in the body into its
        broadcasting variant so a `mu = a + b * x_design` against a
        vector `x_design` produces a Vector{Float64} mu rather than
        raising "no method matching +(::Int64, ::Vector)".
        """
        sb, counter = ctx.sb, ctx.counter
        lhs = _identifier(sb, counter, node.name)
        rhs = render_let_expr_julia(
            _JlCtxShim(sb, counter, ctx.cards, "turing"), node.expr
        )
        if node.name in ctx.batch_shaped_names:
            rhs = _macro_call(sb, counter, ".", rhs)
        stmt = _assignment(sb, counter, lhs, rhs)
        sb.edge(ctx.body, stmt, "child_of")

    def _emit_score(self, ctx: _RenderCtx, node: IRScore) -> None:
        """Bind the score expression then add it to the log-joint via
        `Turing.@addlogprob!`."""
        assert isinstance(ctx, _TuringCtx)
        sb, counter = ctx.sb, ctx.counter
        lhs = _identifier(sb, counter, node.name)
        rhs = render_let_expr_julia(
            _JlCtxShim(sb, counter, ctx.cards, "turing"), node.expr
        )
        stmt = _assignment(sb, counter, lhs, rhs)
        sb.edge(ctx.body, stmt, "child_of")
        mac = _macro_call(
            sb, counter, "addlogprob!", _identifier(sb, counter, node.name)
        )
        sb.edge(ctx.body, mac, "child_of")

    def _emit_return(
        self, ctx: _RenderCtx, names: tuple[str, ...]
    ) -> None:
        assert isinstance(ctx, _TuringCtx)
        sb, counter = ctx.sb, ctx.counter
        if not names:
            return
        if len(names) == 1:
            value = _identifier(sb, counter, names[0])
        else:
            value = _vertex(sb, counter, "tuple_expression")
            for n in names:
                sb.edge(value, _identifier(sb, counter, n), "child_of")
        ret = _return(sb, counter, value)
        sb.edge(ctx.body, ret, "child_of")


# Comprehension binder reserved for `arraydist([... for i in 1:B])`.
_ARRAYDIST_BINDER = "i"


# Recipe table: QVR families with no native Turing.jl distribution
# whose canonical Turing.jl encoding is `truncated(<base>(0, scale),
# 0, Inf)`. The mapped value is the inner base-distribution name.
# Keyed on the QVR family name; consulted from `_family_call`.
_HALF_TRUNCATED_BASES: dict[str, str] = {
    "HalfNormal": "Normal",
    "HalfCauchy": "Cauchy",
}


# Recipe table: QVR families whose `rate` arg corresponds to a
# `scale = 1/rate` argument slot in Distributions.jl. The mapped
# value is the 0-based positional index of the rate arg in the QVR
# call (per the family's arg-name tuple).
#
# * Exponential(rate) -> Exponential(1/rate), rate at position 0.
# * Gamma(concentration, rate) -> Gamma(concentration, 1/rate),
#   rate at position 1.
_RATE_TO_SCALE_INVERT_POSITIONS: dict[str, int] = {
    "Exponential": 0,
    "Gamma": 1,
}


def _invert_rate_arg(
    sb: panproto.SchemaBuilder,
    counter: list[int],
    rhs_args: tuple[str, ...],
    position: int,
) -> tuple[str, ...]:
    """Replace `rhs_args[position]` with `inv(<rhs_args[position]>)`.

    Julia's `Base.inv` returns the multiplicative inverse for any
    numeric type; emitting `inv(rate)` rather than the literal
    `1/rate` avoids encoding a `binary_expression` whose right
    child kind would need to be inferred from the arg's emitter
    output. The reciprocal is the canonical Distributions.jl
    `theta = 1/rate` substitution for the `Gamma` and `Exponential`
    families.
    """
    if position >= len(rhs_args):
        raise UnsupportedConstruct(
            "qvr-turing",
            [
                f"rate-arg-invert: position {position} out of range for "
                f"{len(rhs_args)} args"
            ],
        )
    inv_call = _call(
        sb,
        counter,
        _identifier(sb, counter, "inv"),
        (rhs_args[position],),
    )
    return rhs_args[:position] + (inv_call,) + rhs_args[position + 1:]


# ---------------------------------------------------------------------------
# Renderer context: extends _RenderCtx with Turing-specific carriers.
# ---------------------------------------------------------------------------


class _TuringCtx(_RenderCtx):
    """Turing-renderer-internal context. Adds the function-body block
    vid and the plate tables a `~` emission needs to detect index
    dependence between args and surrounding batch axes."""

    def __init__(
        self,
        *,
        sb: panproto.SchemaBuilder,
        morphisms: dict,
        lets: dict,
        counter: list[int],
        cards: dict[str, int],
        body: str,
        input_plates: dict[str, Plate],
        sample_plates: dict[str, Plate],
        batch_shaped_names: set[str],
    ) -> None:
        super().__init__(sb=sb, morphisms=morphisms, defines=lets)
        self.counter = counter
        self.cards = cards
        self.body = body
        self.input_plates = input_plates
        self.sample_plates = sample_plates
        self.batch_shaped_names = batch_shaped_names


# `_JlCtxShim` lets us reuse
# [`render_let_expr_julia`][quivers.transpile.renderers._julia_helpers.render_let_expr_julia]
# (which expects a `JlCtx` exposing `v`, `e`, `lit`, `fresh`,
# `constraint`, `cards`, `target`) without pulling in the legacy
# backend's whole helper module.
class _JlCtxShim:
    """Minimal adapter exposing the methods
    [`render_let_expr_julia`][quivers.transpile.renderers._julia_helpers.render_let_expr_julia]
    reads off its ctx parameter.

    Carries the static-axis-size table `cards` so
    [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor] unrolling
    can resolve binder cardinalities, and the `target` tag so error
    messages identify the backend ("turing").
    """

    target: str
    cards: dict[str, int]

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        counter: list[int],
        cards: dict[str, int],
        target: str,
    ) -> None:
        self._sb = sb
        self._counter = counter
        self.cards = cards
        self.target = target

    def fresh(self, prefix: str) -> str:
        self._counter[0] += 1
        return f"{prefix}_{self._counter[0]}"

    def v(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)


# ---------------------------------------------------------------------------
# Internal helpers: arg conversion, dim sizing, index-dependence detection.
# ---------------------------------------------------------------------------


def _meta_or_raise(family: str) -> FamilyMeta:
    meta = FAMILY_META.get(family)
    if meta is None:
        raise UnsupportedConstruct(
            "qvr-turing", [f"family:{family}: not in FAMILY_META"]
        )
    return meta


def _dim_to_size(
    sb: panproto.SchemaBuilder, counter: list[int], dim: Dim
) -> str:
    """Render a [`Dim`][quivers.transpile.ir.Dim] as a Julia size
    expression. Static dims become integer literals; dynamic dims
    become identifiers referencing the runtime-supplied size."""
    if isinstance(dim, DimStatic):
        return _integer(sb, counter, dim.size)
    if isinstance(dim, DimDynamic):
        return _identifier(sb, counter, dim.size_name)
    raise UnsupportedConstruct("qvr-turing", [f"dim:{type(dim).__name__}"])


def _arg_to_julia(
    ctx: _TuringCtx,
    arg: IRArg,
    *,
    via: str | None = None,
    family: str | None = None,
) -> str:
    """Convert one IR arg into a Julia schema vertex.

    `via` rewrites any one-level `IRArgRef(name, indices=(IRArgRef(z),))`
    into `eachrow(name)[z[via]]` form when the indexed parent has
    event_dim > 0 (a matrix-shaped distribution result); plain
    `name[z[via]]` when it is a vector. The detection consults
    [`ctx.sample_plates`][TuringRenderer] for the parent's event_dims.

    `family` reserved for future per-family arg rewriting (none today).
    """
    del family
    sb, counter = ctx.sb, ctx.counter
    if isinstance(arg, IRArgNumber):
        return _number(sb, counter, arg.value)
    if isinstance(arg, IRArgRef):
        return _ref_to_julia(ctx, arg, via=via)
    if isinstance(arg, IRArgBroadcast):
        return _broadcast_arg(ctx, arg)
    if isinstance(arg, IRArgList):
        return _vector_literal(
            sb,
            counter,
            tuple(_arg_to_julia(ctx, e) for e in arg.elements),
        )
    if isinstance(arg, IRArgMatrix):
        return _matrix_literal(
            sb,
            counter,
            tuple(
                tuple(_arg_to_julia(ctx, e) for e in row.elements)
                for row in arg.rows
            ),
        )
    if isinstance(arg, IRArgFamilyRef):
        return _family_ref_to_julia(ctx, arg)
    raise UnsupportedConstruct(
        "qvr-turing", [f"arg:{type(arg).__name__}"]
    )


def _ref_to_julia(
    ctx: _TuringCtx, ref: IRArgRef, *, via: str | None
) -> str:
    """Render an [`IRArgRef`][quivers.transpile.ir.IRArgRef].

    Specialises on the LDA-style case: `phi[z]` with a surrounding
    observe whose `via` fibration is `word_idx` and whose indexed
    parent `phi` has event_dim > 0 becomes
    `eachrow(phi)[z[word_idx]]`. The vector-parent case becomes
    `phi[z[word_idx]]` (no `eachrow`).
    """
    sb, counter = ctx.sb, ctx.counter
    base = _identifier(sb, counter, ref.name)
    if not ref.indices:
        return base
    parent_plate = ctx.sample_plates.get(ref.name)
    parent_event_dim = (
        len(parent_plate.event_dims) if parent_plate is not None else 0
    )
    # Rewrite each index: a name-ref index may be threaded through the
    # via fibration.
    rendered_indices: list[str] = []
    for idx in ref.indices:
        if (
            via is not None
            and isinstance(idx, IRArgRef)
            and not idx.indices
        ):
            # `z` becomes `z[<via>]`.
            inner = _identifier(sb, counter, idx.name)
            via_id = _identifier(sb, counter, via)
            rendered_indices.append(
                _index_expr(sb, counter, inner, (via_id,))
            )
        else:
            rendered_indices.append(_arg_to_julia(ctx, idx))
    if parent_event_dim > 0:
        # `phi` is a matrix-shaped distribution sample; index its rows.
        eachrow = _call(
            sb,
            counter,
            _identifier(sb, counter, "eachrow"),
            (base,),
        )
        return _index_expr(sb, counter, eachrow, tuple(rendered_indices))
    return _index_expr(sb, counter, base, tuple(rendered_indices))


def _broadcast_arg(ctx: _TuringCtx, arg: IRArgBroadcast) -> str:
    """Render an [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast]
    as `fill(<value>, <shape...>)`."""
    sb, counter = ctx.sb, ctx.counter
    value_vid = _arg_to_julia(ctx, arg.value)
    shape_args = tuple(_integer(sb, counter, s) for s in arg.target_shape)
    return _call(
        sb,
        counter,
        _identifier(sb, counter, "fill"),
        (value_vid, *shape_args),
    )


def _family_ref_to_julia(
    ctx: _TuringCtx, ref: IRArgFamilyRef
) -> str:
    """Render an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef].

    Looks up the referenced morphism's `~ Family(...)` init clause and
    emits the inner distribution call directly. For wrapper families
    like Truncated the caller's `_render_truncated_wrapper` builds the
    enclosing `truncated(<base>, lower=L, upper=U)` form.
    """
    sb, counter = ctx.sb, ctx.counter
    decl = ctx.morphisms.get(ref.name)
    if decl is None or decl.init_family is None:
        raise UnsupportedConstruct(
            "qvr-turing",
            [
                f"arg:family-ref:{ref.name}: morphism not declared with "
                f"`~ Family(...)` init"
            ],
        )
    inner_meta = _meta_or_raise(decl.init_family.family)
    inner_target = inner_meta.target_names.get("turing")
    if inner_target is None:
        raise UnsupportedConstruct(
            "qvr-turing",
            [
                f"family:{decl.init_family.family}: no Turing.jl mapping "
                f"for wrapper-referenced inner family"
            ],
        )
    callee = _identifier(sb, counter, inner_target)
    inner_args = tuple(
        _raw_init_arg_to_julia(ctx, a) for a in decl.init_family.args
    )
    return _call(sb, counter, callee, inner_args)


def _raw_init_arg_to_julia(ctx: _TuringCtx, raw: object) -> str:
    """Render a parser-form `init_family` arg. The parser uses bare
    `str` for identifiers and `float` for literals; the wrapper
    `IRArgFamilyRef` rendering pre-dates the structured DrawArg
    encoding for nested family calls."""
    sb, counter = ctx.sb, ctx.counter
    if isinstance(raw, (int, float)):
        return _number(sb, counter, float(raw))
    if isinstance(raw, str):
        try:
            return _number(sb, counter, float(raw))
        except ValueError:
            return _identifier(sb, counter, raw)
    raise UnsupportedConstruct(
        "qvr-turing", [f"arg:family-ref-init:{type(raw).__name__}"]
    )


def _args_have_batch_index(
    args: tuple[IRArg, ...],
    sample_plates: dict[str, Plate],
    plate: Plate,
    batch_shaped_names: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """Detect whether any arg references a name whose plate has a
    nonempty batch axis (i.e. an index-dependent call site for the
    surrounding step's plate).

    Used by [`sample`][TuringRenderer.sample] to choose between the
    `filldist(...)` form (no dependence) and the `arraydist(...)` /
    broadcast-dot form (dependence). The `batch_shaped_names` set
    additionally flags direct (un-indexed) references to a
    batch-shaped IRDataInput or to a let-bound deterministic whose
    RHS transitively references one, so a vector-valued `loc` like
    `mu = a + b * x_design` in bayes_linear_regression is recognised
    even though its IRDeterministic carries an empty plate.
    """
    del plate
    return any(
        _arg_indexes_plated(a, sample_plates, batch_shaped_names)
        for a in args
    )


def _arg_indexes_plated(
    arg: IRArg,
    sample_plates: dict[str, Plate],
    batch_shaped_names: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """Recursive helper: True iff `arg` carries a bracket index whose
    inner reference is itself a previously-bound sample with a
    nonempty plate, OR `arg` (or a sub-arg) references a name in
    `batch_shaped_names` directly (un-indexed)."""
    if isinstance(arg, IRArgRef):
        if not arg.indices and arg.name in batch_shaped_names:
            return True
        if arg.indices:
            for idx in arg.indices:
                if isinstance(idx, IRArgRef) and idx.name in sample_plates:
                    return True
        for idx in arg.indices:
            if _arg_indexes_plated(idx, sample_plates, batch_shaped_names):
                return True
        return False
    if isinstance(arg, IRArgBroadcast):
        return _arg_indexes_plated(arg.value, sample_plates, batch_shaped_names)
    if isinstance(arg, IRArgList):
        return any(
            _arg_indexes_plated(e, sample_plates, batch_shaped_names)
            for e in arg.elements
        )
    if isinstance(arg, IRArgMatrix):
        for row in arg.rows:
            for e in row.elements:
                if _arg_indexes_plated(e, sample_plates, batch_shaped_names):
                    return True
    return False


def _promote_scalar_ref(
    arg: IRArg,
    arg_name: str,
    plate: Plate,
    meta: FamilyMeta,
) -> IRArg:
    """Promote an [`IRArgRef`][quivers.transpile.ir.IRArgRef] that
    targets a vector-shaped arg constraint into an
    [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast].

    Lower leaves scalar references unwrapped (a tensor reference may
    already carry the right shape at runtime), but the Turing.jl
    idiom for a Dirichlet's `concentration` and similar vector
    parameters writes `fill(alpha, K)` for the scalar-broadcast case.
    The promotion rule: if the arg has no bracket indices and the
    family's `arg_constraints[arg_name]` is an
    `IndependentConstraint` of rank `n >= 1`, wrap in `IRArgBroadcast`
    whose `target_shape` is read from the surrounding step's
    `event_dims`.
    """
    if not isinstance(arg, IRArgRef) or arg.indices:
        return arg
    cls_attr = meta.distribution_class.arg_constraints
    if not isinstance(cls_attr, dict):
        return arg
    expected = cls_attr.get(arg_name)
    if expected is None or not isinstance(
        expected, _torch_constraints._IndependentConstraint
    ):
        return arg
    if len(plate.event_dims) < expected.event_dim:
        return arg
    sizes: tuple[int, ...] = ()
    for dim in plate.event_dims[: expected.event_dim]:
        if isinstance(dim, DimStatic):
            sizes = (*sizes, dim.size)
        else:
            return arg
    if not sizes:
        return arg
    return IRArgBroadcast(value=arg, target_shape=sizes)


def _replace_first_index(arg: IRArg, binder: str) -> IRArg:
    """For the `arraydist([... for i in 1:B])` fallback: replace any
    bracket-index `arg[k]` with `arg[<binder>]` so the comprehension
    walks the batch axis. Numeric / literal args pass through."""
    if isinstance(arg, IRArgRef) and arg.indices:
        new_indices = tuple(
            IRArgRef(name=binder) if isinstance(idx, IRArgRef) else idx
            for idx in arg.indices
        )
        return IRArgRef(name=arg.name, indices=new_indices)
    return arg


def _seed_sample_plates(ctx: _TuringCtx, body: tuple[IRNode, ...]) -> None:
    """Populate `ctx.sample_plates` from every IRSample / IRObserve /
    IRMarginalize in `body` (descending into marginalize scopes).

    Used by [`_args_have_batch_index`][_args_have_batch_index] and
    [`_ref_to_julia`][_ref_to_julia] to look up the plate of an
    indexed reference's parent so the renderer can dispatch on the
    parent's event_dim (eachrow vs. plain index).
    """
    for node in body:
        if isinstance(node, IRSample):
            ctx.sample_plates[node.name] = node.plate
        elif isinstance(node, IRObserve):
            ctx.sample_plates[node.name] = node.plate
        elif isinstance(node, IRMarginalize):
            ctx.sample_plates[node.latent] = node.plate
            _seed_sample_plates(ctx, node.scope)


def _seed_batch_shaped(ctx: _TuringCtx, body: tuple[IRNode, ...]) -> None:
    """Populate `ctx.batch_shaped_names` with every name that carries a
    batch dimension at runtime.

    A name is batch-shaped when:

    * it is an [`IRDataInput`][quivers.transpile.ir.IRDataInput] whose
      plate has any `batch_dims`;
    * it is an [`IRSample`][quivers.transpile.ir.IRSample] or
      [`IRObserve`][quivers.transpile.ir.IRObserve] whose plate has any
      `batch_dims`;
    * it is an [`IRDataInput`][quivers.transpile.ir.IRDataInput] with
      empty plate that is referenced (transitively, through let
      bindings) by a plated observe's arg (e.g. `x_design` in
      `let mu = a + b * x_design` followed by
      `observe y : Obs <- Normal(mu, 0.3)`; the IR carries no shape
      annotation for `x_design` so the implicit shape is recovered
      from its use site);
    * it is an [`IRDeterministic`][quivers.transpile.ir.IRDeterministic]
      whose RHS [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode]
      references at least one batch-shaped name (closure under
      `let`-binding).

    The Turing renderer uses this set to pick the broadcast-dot form
    `y .~ Family.(...)` for observes whose plain-call form would
    fail at the boundary between a scalar Julia distribution
    constructor and a vector argument, and to wrap each
    batch-shaped deterministic's RHS in `@.` so its arithmetic
    broadcasts elementwise.
    """
    # Seed inputs first (no dependencies).
    for name, plate in ctx.input_plates.items():
        if plate.batch_dims:
            ctx.batch_shaped_names.add(name)
    # Pre-pass: every sample / observe with a non-empty batch_dims is
    # batch-shaped. (Marginalize scopes recurse.)
    _collect_sample_batch_shaped(ctx, body)
    # Build a name -> IRDeterministic table so closure propagation
    # below can dereference let bindings in any order.
    dets: dict[str, IRDeterministic] = {
        n.name: n for n in body if isinstance(n, IRDeterministic)
    }
    # Implicit-shape propagation: for every plated IRObserve / IRSample
    # whose args reference a let-bound deterministic or an empty-plate
    # input, mark every transitively-referenced name (including the
    # original input) as batch-shaped. This recovers the implicit
    # vector shape of inputs like `x_design` that have no explicit
    # plate annotation but whose use site (a plated observe's `loc`)
    # demands a per-element value.
    for node in body:
        if isinstance(node, (IRSample, IRObserve)) and node.plate.batch_dims:
            for arg in node.args:
                _mark_arg_refs_batch_shaped(arg, ctx, dets)
    # Fixpoint over IRDeterministic let bindings: a deterministic is
    # batch-shaped iff any name its RHS references is batch-shaped.
    # Repeat until no new name is added (LetExpr graphs are acyclic by
    # construction, so this converges in O(|body|) iterations).
    changed = True
    while changed:
        changed = False
        for node in body:
            if isinstance(node, IRDeterministic):
                if node.name in ctx.batch_shaped_names:
                    continue
                refs = _let_expr_var_refs(node.expr)
                if any(r in ctx.batch_shaped_names for r in refs):
                    ctx.batch_shaped_names.add(node.name)
                    changed = True


def _mark_arg_refs_batch_shaped(
    arg: IRArg,
    ctx: _TuringCtx,
    dets: dict[str, IRDeterministic],
) -> None:
    """Walk `arg`'s referenced names and add each let-bound
    deterministic or empty-plate IRDataInput to
    `ctx.batch_shaped_names`, recursively descending into the
    deterministic's RHS so a chain
    ``observe y <- Normal(mu, 0.3); let mu = a + b * x_design``
    marks `mu` AND `x_design` (the un-plated input whose use under a
    plated observe's `loc` implies a per-element value)."""
    if isinstance(arg, IRArgRef):
        name = arg.name
        if name in dets and name not in ctx.batch_shaped_names:
            ctx.batch_shaped_names.add(name)
            for ref in _let_expr_var_refs(dets[name].expr):
                _mark_name_batch_shaped(ref, ctx, dets)
        elif name in ctx.input_plates and name not in ctx.batch_shaped_names:
            ctx.batch_shaped_names.add(name)
        for idx in arg.indices:
            _mark_arg_refs_batch_shaped(idx, ctx, dets)
    elif isinstance(arg, IRArgBroadcast):
        _mark_arg_refs_batch_shaped(arg.value, ctx, dets)
    elif isinstance(arg, IRArgList):
        for el in arg.elements:
            _mark_arg_refs_batch_shaped(el, ctx, dets)
    elif isinstance(arg, IRArgMatrix):
        for row in arg.rows:
            for el in row.elements:
                _mark_arg_refs_batch_shaped(el, ctx, dets)


def _mark_name_batch_shaped(
    name: str,
    ctx: _TuringCtx,
    dets: dict[str, IRDeterministic],
) -> None:
    """Add `name` (a bare identifier from a let-expression) to
    `ctx.batch_shaped_names` if it resolves to an
    [`IRDeterministic`][quivers.transpile.ir.IRDeterministic] or an
    [`IRDataInput`][quivers.transpile.ir.IRDataInput], descending
    through any let bindings recursively."""
    if name in ctx.batch_shaped_names:
        return
    if name in dets:
        ctx.batch_shaped_names.add(name)
        for ref in _let_expr_var_refs(dets[name].expr):
            _mark_name_batch_shaped(ref, ctx, dets)
    elif name in ctx.input_plates:
        ctx.batch_shaped_names.add(name)


def _collect_sample_batch_shaped(
    ctx: _TuringCtx, body: tuple[IRNode, ...]
) -> None:
    """Add every IRSample / IRObserve / IRMarginalize name with
    non-empty batch_dims to `ctx.batch_shaped_names`."""
    for node in body:
        if isinstance(node, IRSample) and node.plate.batch_dims:
            ctx.batch_shaped_names.add(node.name)
        elif isinstance(node, IRObserve) and node.plate.batch_dims:
            ctx.batch_shaped_names.add(node.name)
        elif isinstance(node, IRMarginalize):
            if node.plate.batch_dims:
                ctx.batch_shaped_names.add(node.latent)
            _collect_sample_batch_shaped(ctx, node.scope)


def _let_expr_var_refs(expr: LetExprNode) -> set[str]:
    """Collect the set of variable names a let-expression references.

    Walks the [`LetExprNode`][quivers.dsl.ast_nodes.LetExprNode]
    discriminator union exhaustively; unknown discriminators raise
    [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
    so a new let-expression kind announces itself rather than silently
    contributing an empty ref set (which would mis-classify a
    deterministic as scalar and break the broadcast-dot dispatch).
    """
    if isinstance(expr, LetExprVar):
        return {expr.name}
    if isinstance(expr, LetExprLiteral):
        return set()
    if isinstance(expr, LetExprString):
        return set()
    if isinstance(expr, LetExprBinOp):
        return _let_expr_var_refs(expr.left) | _let_expr_var_refs(expr.right)
    if isinstance(expr, LetExprUnaryOp):
        return _let_expr_var_refs(expr.operand)
    if isinstance(expr, LetExprCall):
        refs: set[str] = set()
        for a in expr.args:
            refs |= _let_expr_var_refs(a)
        return refs
    if isinstance(expr, LetExprIndex):
        out = _let_expr_var_refs(expr.array)
        for i in expr.indices:
            out |= _let_expr_var_refs(i)
        return out
    if isinstance(expr, LetExprList):
        out2: set[str] = set()
        for item in expr.items:
            out2 |= _let_expr_var_refs(item)
        return out2
    if isinstance(expr, LetExprLambda):
        body_refs = _let_expr_var_refs(expr.body)
        return body_refs - {expr.param}
    if isinstance(expr, LetExprMethodCall):
        out3 = _let_expr_var_refs(expr.receiver)
        for a in expr.args:
            out3 |= _let_expr_var_refs(a)
        return out3
    if isinstance(expr, LetExprFactor):
        body_refs2: set[str] = (
            _let_expr_var_refs(expr.body) if expr.body is not None else set()
        )
        for case in expr.cases:
            body_refs2 |= _let_expr_var_refs(case.value)
        bound2 = {b.var for b in expr.binders}
        return body_refs2 - bound2
    raise UnsupportedConstruct(
        "qvr-turing",
        [f"let-expr:{type(expr).__name__}: unhandled for batch-shape inference"],
    )


# ---------------------------------------------------------------------------
# Public convenience entry point: parse a Module, lower, render, pretty.
# ---------------------------------------------------------------------------


def render_module(module: Module) -> bytes:
    """Lower a parsed [`Module`][quivers.dsl.ast_nodes.Module] and emit
    Turing.jl source bytes.

    Convenience wrapper around the `Module -> IRProgram -> Schema ->
    bytes` pipeline. Tests and the CLI driver call this; the renderer
    is composable directly via [`TuringRenderer.render`][TuringRenderer.render]
    for callers that already hold an IRProgram.
    """
    # `target="stan"` keeps `MarginalizeStep` intact through composite
    # expansion; the Turing renderer's `marginalize` handler lowers it
    # to an explicit `IRSample(latent)` plus the scoped body inline.
    # Driving the lowering through the renderer (not through
    # `expand_composite_lets`) preserves the marginalize step's
    # `over=` axis spec, which the in-place flatten path drops.
    expanded = expand_composite_lets(module, target="stan")
    morphisms = build_morphism_table(expanded)
    lets = build_let_table(expanded)
    # `_pick_program` raises early when the module has no
    # ProgramDecl, surfacing the failure here rather than letting
    # `Lower` raise an opaquer error downstream.
    _pick_program(expanded)
    ir = Lower()(expanded)
    schema = TuringRenderer(morphisms=morphisms, lets=lets).render(ir)
    return bytes(EmitPretty("julia")(schema))


def _pick_program(module: Module) -> ProgramDecl:
    """Pick the program decl to render, preferring the exported one."""
    programs: list[ProgramDecl] = []
    exported: set[str] = set()
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            programs.append(stmt)
        elif isinstance(stmt, ExportDecl) and isinstance(stmt.expr, ExprIdent):
            exported.add(stmt.expr.name)
    if not programs:
        raise UnsupportedConstruct(
            "qvr-turing", ["no program_decl: nothing to render"]
        )
    return next((p for p in programs if p.name in exported), programs[-1])


# ---------------------------------------------------------------------------
# Runtime-helper graft: `HalfStudentT`, `ContinuousBernoulli` as
# Distributions.ContinuousUnivariateDistribution subclasses.
#
# Distributions.jl ships `Normal`, `Beta`, `TDist`, `Kumaraswamy`, ... as
# built-in distributions but lacks `HalfStudentT` and `ContinuousBernoulli`.
# The transpile-time graft parses the hand-written helper at
# [`runtime_turing.jl`][quivers.transpile.runtime_turing] once at module
# load through panproto's Julia tree-sitter grammar; per-render, it
# copies every grafted vertex / constraint / edge into the per-render
# schema (with fresh vertex ids) and attaches the runtime's top-level
# statements as `child_of` of the emitted `source_file` above the
# `@model function model` macrocall.
#
# The emit is structurally a normal Julia source file: `using Distributions`,
# `using Random`, `using SpecialFunctions`, the `HalfStudentT` struct, the
# `Distributions.logpdf` / `Distributions.rand` / support methods, and the
# `ContinuousBernoulli` struct with the same method set. Subsequent
# `~ HalfStudentT(df, scale)` and `~ ContinuousBernoulli(probs)` call sites
# in the model body then resolve to the grafted types via normal Julia
# name lookup.
# ---------------------------------------------------------------------------


_RUNTIME_TURING_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "runtime_turing.jl"
)


#: Families whose Turing.jl emit relies on the
#: [`runtime_turing.jl`][quivers.transpile.runtime_turing] helper subtree.
#: Distributions.jl ships `Normal`, `Beta`, `TDist`, `Kumaraswamy`, etc.
#: as built-in distributions but lacks these; the renderer grafts the
#: helper when the IR samples or observes from any of them.
_TURING_RUNTIME_HELPER_FAMILIES: frozenset[str] = frozenset({
    "HalfStudentT",
    "ContinuousBernoulli",
    "GP",
})


def _load_runtime_turing_schema() -> tuple[
    panproto.Schema, str, tuple[str, ...]
]:
    """Parse [`runtime_turing.jl`][quivers.transpile.runtime_turing] through
    panproto's Julia tree-sitter grammar at module-load time.

    Returns the parsed schema, the parsed `source_file` vertex id, and
    the tuple of top-level child ids in source order (sorted by
    `start-byte`). The graft replays these children in order beneath
    the per-render `source_file` so the emit's top-level statements
    appear in the original file's layout.
    """
    schema = parser_registry().parse_with_protocol(
        "julia",
        _RUNTIME_TURING_PATH.read_bytes(),
        str(_RUNTIME_TURING_PATH),
    )
    src_id = next(
        (v.id for v in schema.vertices if v.kind == "source_file"),
        None,
    )
    if src_id is None:
        raise RuntimeError(
            f"`source_file` not found in parse of {_RUNTIME_TURING_PATH}"
        )
    children_with_sb: list[tuple[int, str]] = []
    for edge in schema.edges:
        if edge.src != src_id:
            continue
        sb_val = next(
            (
                int(c.value)
                for c in schema.constraints_for(edge.tgt)
                if c.sort == "start-byte"
            ),
            0,
        )
        children_with_sb.append((sb_val, edge.tgt))
    children_with_sb.sort()
    return schema, src_id, tuple(child for _, child in children_with_sb)


_RUNTIME_TURING_SCHEMA, _RUNTIME_TURING_SOURCE_ID, _RUNTIME_TURING_TOP_LEVEL = (
    _load_runtime_turing_schema()
)


def _subtree_vertex_ids(
    schema: panproto.Schema, roots: tuple[str, ...]
) -> set[str]:
    """Return every vertex id reachable from `roots` via outgoing edges."""
    seen: set[str] = set(roots)
    frontier: list[str] = list(roots)
    while frontier:
        src = frontier.pop()
        for edge in schema.edges:
            if edge.src == src and edge.tgt not in seen:
                seen.add(edge.tgt)
                frontier.append(edge.tgt)
    return seen


_RUNTIME_TURING_SUBTREE = _subtree_vertex_ids(
    _RUNTIME_TURING_SCHEMA, _RUNTIME_TURING_TOP_LEVEL
)


def _ir_uses_family(body: tuple[IRNode, ...], family: str) -> bool:
    """True iff any [`IRSample`][quivers.transpile.ir.IRSample] or
    [`IRObserve`][quivers.transpile.ir.IRObserve] in `body` (including
    nested [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] scopes)
    samples from `family`."""
    for node in body:
        if (
            isinstance(node, (IRSample, IRObserve))
            and node.family == family
        ):
            return True
        if isinstance(node, IRMarginalize) and _ir_uses_family(
            node.scope, family
        ):
            return True
    return False


def _graft_runtime_turing_helper(
    sb: panproto.SchemaBuilder, counter: list[int], source_vid: str
) -> None:
    """Graft the runtime-helper subtree onto the per-render schema.

    Copies every vertex, every constraint, and every internal edge of
    the parsed `runtime_turing.jl` subtree into the per-render
    `SchemaBuilder` with fresh vertex ids, then attaches each
    top-level child as a `child_of` of `source_vid` in source order.
    The grafted top-level children appear above the `@model function
    model` macrocall in the emit.
    """
    src_schema = _RUNTIME_TURING_SCHEMA
    subtree = _RUNTIME_TURING_SUBTREE
    id_map: dict[str, str] = {}

    for old in subtree:
        counter[0] += 1
        new = f"rt{counter[0]}"
        id_map[old] = new
        kind = next(
            v.kind for v in src_schema.vertices if v.id == old
        )
        sb.vertex(new, kind)
        for cstr in src_schema.constraints_for(old):
            sb.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            sb.edge(id_map[edge.src], id_map[edge.tgt], edge.kind)
    for child_old in _RUNTIME_TURING_TOP_LEVEL:
        sb.edge(source_vid, id_map[child_old], "child_of")


__all__ = [
    "TuringRenderer",
    "render_module",
]
