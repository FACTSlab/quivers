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

import panproto
import torch.distributions.constraints as _torch_constraints

from quivers.dsl.ast_nodes import (
    ExportDecl,
    ExprIdent,
    Module,
    ProgramDecl,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile._pipeline import (
    EmitPretty,
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
        # name in lowering order (scalar params, observed vars, free
        # identifiers).
        params = tuple(inp.name for inp in ir.inputs)

        # Walk the body. Each node emits a `~` / `=` / `return`
        # statement into `body`.
        ctx = _TuringCtx(
            sb=sb,
            morphisms=self._morphisms,
            lets=self._lets,
            counter=counter,
            cards={},
            body=body,
            input_plates={inp.name: inp.plate for inp in ir.inputs},
            sample_plates={},
        )
        # Pre-populate the sample-plate table by walking the body so
        # observe / marginalize bodies can detect index-dependent args
        # against the originating sample's plate.
        _seed_sample_plates(ctx, ir.body)
        for node in ir.body:
            self._dispatch(ctx, node)

        fn = _function_def(
            sb, counter, name="model", params=params, body_vid=body
        )
        macro = _macro_call(sb, counter, "model", fn)
        sb.edge(source, macro, "child_of")
        return sb.build()

    # ----- IRNode dispatch (overrides RendererBase._dispatch_node) -----

    def _dispatch(self, ctx: _TuringCtx, node: IRNode) -> None:
        """Route one IR body node to the right Turing.jl emission."""
        if isinstance(node, IRSample):
            self.sample(
                ctx,  # type: ignore[arg-type]
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
                ctx,  # type: ignore[arg-type]
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
            self._emit_score(ctx, node)  # type: ignore[arg-type]
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node)  # type: ignore[arg-type]
            return
        if isinstance(node, IRReturn):
            self._emit_return(ctx, node.names)  # type: ignore[arg-type]
            return
        if isinstance(node, IRDataInput):
            return
        raise UnsupportedConstruct(
            "qvr-turing", [f"node:{type(node).__name__}"]
        )

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

    def sample(  # type: ignore[override]
        self,
        ctx: _TuringCtx,
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
            or _args_have_batch_index(args, ctx.sample_plates, plate)
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

    def marginalize(  # type: ignore[override]
        self, ctx: _TuringCtx, node: IRMarginalize
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
        rewritten = self.explicit_latent_scope(node)
        for child in rewritten:
            self._dispatch(ctx, child)
        return ""

    # ----- broadcast: Julia's fill(<value>, K) / fill(<value>, R, C) -----

    def broadcast(  # type: ignore[override]
        self,
        ctx: _TuringCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
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
        Two QVR families have no native Turing.jl distribution and the
        renderer composes them out of the `truncated` wrapper:

        * `HalfNormal(sigma)` -> `truncated(Normal(0, sigma), 0, Inf)`
        * `HalfCauchy(gamma)` -> `truncated(Cauchy(0, gamma), 0, Inf)`

        The composition is keyed on the family name (the FAMILY_META
        target_name for both is `"truncated"`, which is the wrapper
        callable); a per-renderer recipe table supplies the inner
        base distribution because that choice is the family's own
        per-target lowering convention rather than a renderer-level
        dispatch on the QVR family discriminator.
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
        sb, counter = ctx.sb, ctx.counter
        lhs = _identifier(sb, counter, node.name)
        rhs = render_let_expr_julia(
            _JlCtxShim(sb, counter), node.expr
        )
        stmt = _assignment(sb, counter, lhs, rhs)
        sb.edge(ctx.body, stmt, "child_of")

    def _emit_score(self, ctx: _TuringCtx, node: IRScore) -> None:  # type: ignore[override]
        """Bind the score expression then add it to the log-joint via
        `Turing.@addlogprob!`."""
        sb, counter = ctx.sb, ctx.counter
        lhs = _identifier(sb, counter, node.name)
        rhs = render_let_expr_julia(_JlCtxShim(sb, counter), node.expr)
        stmt = _assignment(sb, counter, lhs, rhs)
        sb.edge(ctx.body, stmt, "child_of")
        mac = _macro_call(
            sb, counter, "addlogprob!", _identifier(sb, counter, node.name)
        )
        sb.edge(ctx.body, mac, "child_of")

    def _emit_return(self, ctx: _TuringCtx, names: tuple[str, ...]) -> None:  # type: ignore[override]
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
    ) -> None:
        super().__init__(sb=sb, morphisms=morphisms, lets=lets)
        self.counter = counter
        self.cards = cards
        self.body = body
        self.input_plates = input_plates
        self.sample_plates = sample_plates


# `_JlCtxShim` lets us reuse [`render_let_expr_julia`][quivers.transpile.renderers._julia_helpers.render_let_expr_julia]
# (which expects a `JlCtx` with `v`, `e`, `lit`, `fresh`) without
# pulling in the legacy backend's whole helper module.
class _JlCtxShim:
    """Minimal adapter exposing the four methods
    [`render_let_expr_julia`][quivers.transpile.renderers._julia_helpers.render_let_expr_julia]
    reads off its ctx parameter."""

    def __init__(
        self, sb: panproto.SchemaBuilder, counter: list[int]
    ) -> None:
        self._sb = sb
        self._counter = counter

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
) -> bool:
    """Detect whether any arg references a name whose plate has a
    nonempty batch axis (i.e. an index-dependent call site for the
    surrounding step's plate).

    Used by [`sample`][TuringRenderer.sample] to choose between the
    `filldist(...)` form (no dependence) and the `arraydist(...)` /
    broadcast-dot form (dependence).
    """
    del plate
    return any(_arg_indexes_plated(a, sample_plates) for a in args)


def _arg_indexes_plated(
    arg: IRArg, sample_plates: dict[str, Plate]
) -> bool:
    """Recursive helper: True iff `arg` carries a bracket index whose
    inner reference is itself a previously-bound sample with a
    nonempty plate."""
    if isinstance(arg, IRArgRef):
        if arg.indices:
            for idx in arg.indices:
                if isinstance(idx, IRArgRef) and idx.name in sample_plates:
                    return True
        for idx in arg.indices:
            if _arg_indexes_plated(idx, sample_plates):
                return True
        return False
    if isinstance(arg, IRArgBroadcast):
        return _arg_indexes_plated(arg.value, sample_plates)
    if isinstance(arg, IRArgList):
        return any(_arg_indexes_plated(e, sample_plates) for e in arg.elements)
    if isinstance(arg, IRArgMatrix):
        for row in arg.rows:
            for e in row.elements:
                if _arg_indexes_plated(e, sample_plates):
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


__all__ = [
    "TuringRenderer",
    "render_module",
]
