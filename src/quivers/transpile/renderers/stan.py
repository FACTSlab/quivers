"""[`StanRenderer`][quivers.transpile.renderers.stan.StanRenderer]: IR to Stan source.

The renderer subclasses
[`RendererBase`][quivers.transpile.renderers._base.RendererBase] and
implements the four dispatch points (`declare`, `sample`,
`marginalize`, `broadcast`) plus the two arg helpers (`render_list`,
`render_matrix`). Distribution names live in
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META]'s
`target_names["stan"]`; no per-renderer family table. Support
classification dispatches on the predicates exported from
[`ir.py`][quivers.transpile.ir]; the renderer never branches on the
family-name string.

The Stan-specific layout decisions:

* Program block: top-level `program` vertex carrying the five
  standard child blocks (`data`, `parameters`,
  `transformed_parameters`, `model`, `generated_quantities`).
  Blocks are materialised lazily when first written to, so a
  program with no `transformed_parameters` declarations emits no
  empty block.
* Sample-step plate loops: every batch dimension on a sample's
  plate becomes a nested `for (m_<axis> in 1:<size>)` loop, with the
  LHS and every same-axis-indexed arg rewritten through
  [`substitute_indices`][quivers.transpile.renderers._base.RendererBase.substitute_indices].
* Marginalize: per spec, Stan emits
  `log_sum_exp` per-group enumeration. The per-call
  [`finite_enumerable_at_call_site`][quivers.transpile.family_meta.finite_enumerable_at_call_site]
  predicate guards the construct;
  non-finite-support families raise `UnsupportedConstruct` with
  kind `marginalize:non-finite-support:<family>`.
* Broadcast: `rep_vector(<value>, K)` for 1D,
  `rep_matrix(<value>, R, C)` for 2D. The Stan grammar's
  `function_expression` carries the call.
* List / matrix args: `[<e0>, <e1>, ...]'` (with transpose) for
  list literals; `to_matrix({{<row0>}, {<row1>}, ...})` for matrix
  literals.
* IRArgFamilyRef: the referenced morphism's `init_family` is read
  from `_RenderCtx.morphisms`; Stan-side wrappers emit the inner
  call inline followed by Stan's truncation / wrapper syntax (e.g.
  `normal(0, 1) T[-2, 2]` for `Truncated(base, -2, 2)`).
"""

from __future__ import annotations

import pathlib
from typing import Callable

import panproto
import torch.distributions.constraints as _torch_constraints
from torch.distributions.constraints import Constraint

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgName,
    DrawArgScalar,
    Expr,
    LetDecl,
    Module,
    MorphismDecl,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprFactor,
    LetExprIndex,
    LetExprList,
    LetExprNode,
    LetExprVar,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import parser_registry, target_protocol
from quivers.transpile.lower import _collect_let_expr_var_names
from quivers.transpile.family_meta import (
    FAMILY_META,
    FamilyMeta,
    finite_enumerable_at_call_site,
)
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
    is_int_bit,
    is_int_category,
    is_int_count,
    is_real_corr_chol,
    is_real_cov_matrix,
    is_real_matrix,
    is_real_one_hot,
    is_real_bounded_interval,
    is_real_positive,
    is_real_scalar,
    is_real_simplex,
    is_real_unit_interval,
    is_real_vector,
    real_interval_bounds,
)
from quivers.transpile.renderers._base import (
    BlockKind,
    RendererBase,
    SchemaFragment,
    _RenderCtx,
    assert_no_dangling_refs,
)
from quivers.transpile.renderers._stan_helpers import (
    _substitute_let_expr,
    render_let_expr_stan,
)


class _StanLetCtx:
    """Bridge ``_RenderCtx.sb`` to the
    [`render_let_expr_stan`][quivers.transpile.renderers._stan_helpers.render_let_expr_stan]
    helper interface (`vertex`, `edge`, `literal`, `constraint`,
    `fresh`) and carry the object-name -> static-cardinality map
    consulted when unrolling
    [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor].

    The Stan IR-walk operates over `_RenderCtx`; the let-expression
    helper expects a small carrier that exposes
    [`panproto.SchemaBuilder`][panproto.SchemaBuilder] operations
    under terse method names. The shim keeps the helper independent
    of any specific renderer class.
    """

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        fresh: Callable[[str], str],
        cards: dict[str, int],
    ) -> None:
        self._sb = sb
        self._fresh_fn = fresh
        self.cards = cards

    def fresh(self, prefix: str) -> str:
        return self._fresh_fn(prefix)

    def vertex(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def edge(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._sb.edge(src, tgt, kind)

    def literal(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)


class StanRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
    Stan [`panproto.Schema`][panproto.Schema].

    Subclasses
    [`RendererBase`][quivers.transpile.renderers._base.RendererBase]
    and overrides the four dispatch points
    (`declare`, `sample`, `marginalize`, `broadcast`) plus the two
    list / matrix arg helpers per the spec.
    """

    target: str = "stan"

    def __init__(self, *, source_module: Module | None = None) -> None:
        """Initialise the renderer.

        Parameters
        ----------
        source_module
            Optional original [`Module`][quivers.dsl.ast_nodes.Module]
            the IR came from. Carries `MorphismDecl` / `LetDecl`
            entries the renderer reads when resolving
            [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] args
            (`Truncated(base, ...)` style wrappers). When omitted, the
            renderer raises on `IRArgFamilyRef` rather than guess.
        """
        self._source_module = source_module
        # Block roots are populated lazily during the IR walk.
        self._blocks: dict[BlockKind, str] = {}
        # Marginalize state for nested arg-substitution.
        self._marginalize_stack: tuple[Plate, ...] = ()
        # `lps_<latent>` accumulators per marginalize call site,
        # used to thread per-group log-sums through scope observes.
        self._marginalize_var: str | None = None
        self._marginalize_latent_card: int | None = None
        self._marginalize_group_idx: tuple[str, ...] = ()
        # Scope-local let bindings inside the current marginalize
        # block; the per-k observe expands these inline since the
        # let target is never declared as a Stan parameter.
        self._marginalize_let_subs: dict[str, LetExprNode] = {}
        # Bookkeeping for redeclaration avoidance.
        self._declared: dict[BlockKind, set[str]] = {}
        # Fresh counter is renderer-internal; the base's _RenderCtx
        # counter remains untouched so per-walk node IDs stay stable.
        self._fresh_n = 0
        # Per-render lookup caches; `render()` clears them on every
        # entry so repeated render calls on the same renderer produce
        # identical schemas.
        self._simplex_cards_state: dict[str, int] = {}
        self._declared_shapes_state: dict[
            str, tuple[Constraint, Plate]
        ] = {}
        # name -> K for chain variables whose downstream consumer
        # binds the value to a slot whose constraint has
        # `event_dim >= 1` (e.g. Categorical's `probs`). The
        # declaration path promotes the scalar real type to
        # `vector[K]`, and the deterministic / sample emit paths
        # wrap any scalar RHS in `rep_vector(rhs, K)`. See
        # `_compute_vector_promotions`.
        self._vector_promotions_state: dict[str, int] = {}

    # ------------------------------------------------------------------
    # abstract overrides
    # ------------------------------------------------------------------

    def target_protocol(self) -> panproto.Protocol:
        return target_protocol("stan")

    # ----- the full render override -----

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and emit a Stan schema.

        Override of the base `render` so the program-level layout
        (the `program` vertex with five child blocks) is built once
        per call.
        """
        assert_no_dangling_refs(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        morphisms, lets = self._resolve_morphisms_and_lets()
        ctx = _RenderCtx(sb=sb, morphisms=morphisms, lets=lets)
        # Reset per-render state.
        self._blocks = {}
        self._declared = {
            "data": set(),
            "parameters": set(),
            "transformed_parameters": set(),
            "model": set(),
            "generated_quantities": set(),
            "function_body": set(),
        }
        self._marginalize_stack = ()
        self._marginalize_var = None
        self._marginalize_latent_card = None
        self._marginalize_group_idx = ()
        self._fresh_n = 0
        self._cards = dict(ir.cards)
        # Reset the per-render lookup caches so repeated render
        # calls on the same renderer produce identical schemas.
        self._simplex_cards_state.clear()
        self._declared_shapes_state.clear()
        self._vector_promotions_state.clear()
        self._vector_promotions_state.update(
            self._compute_vector_promotions(ir)
        )
        # Program root.
        ctx.sb.vertex("prog", "program")
        # Stan ships `normal`, `beta`, `gamma`, ... as built-in
        # densities but lacks `kumaraswamy`. When the IR samples or
        # observes from a family whose Stan emit relies on a user-
        # defined `<family>_lpdf` / `<family>_rng`, graft the
        # hand-written helper at
        # [`runtime_stan_functions.stan`][quivers.transpile.runtime_stan_functions]
        # into the program above the data block so the sampling
        # statement `y ~ kumaraswamy(a, b);` resolves through Stan's
        # `<family>_lpdf` lookup convention.
        if any(
            _ir_uses_family(ir.body, f)
            for f in _STAN_RUNTIME_HELPER_FAMILIES
        ):
            _graft_runtime_stan_helper(ctx.sb, self, "prog")
        # Pre-create blocks in canonical Stan order so emit_pretty
        # honours the `data? transformed_data? parameters?
        # transformed_parameters? model? generated_quantities?`
        # grammar production. Without this the IR-walk's lazy
        # `_ensure_block` connects blocks in IR-encounter order
        # (`data` -> `parameters` -> `model` -> `transformed_parameters`),
        # which puts `transformed_parameters` past `model` and
        # makes the pretty-printer drop it.
        for kind in self._needed_blocks(ir):
            self._ensure_block(ctx, kind)
        # IRDataInput entries land in `data`.
        for inp in ir.inputs:
            self.declare(
                ctx,
                inp.name,
                inp.constraint,
                inp.plate,
                block="data",
            )
        # Walk the body.
        for node in ir.body:
            self._dispatch_node(ctx, node)
        return ctx.sb.build()

    @property
    def _vector_promotions(self) -> dict[str, int]:
        """Per-render map from chain-tail name to the K required by
        its downstream Categorical (or other event-dim>=1) consumer.

        Populated by `_compute_vector_promotions` at the top of
        `render()`. The
        [`declare`][quivers.transpile.renderers.stan.StanRenderer.declare]
        and
        [`_emit_deterministic`][quivers.transpile.renderers.stan.StanRenderer._emit_deterministic]
        paths consult this map to promote scalar real declarations to
        `vector[K]` and to wrap scalar RHS expressions in
        `rep_vector(rhs, K)`.
        """
        return self._vector_promotions_state

    def _compute_vector_promotions(
        self, ir: IRProgram
    ) -> dict[str, int]:
        """Scan `ir.body` for consumers whose argument slot has
        `event_dim >= 1` and record the producer name plus required K.

        For each [`IRObserve`][quivers.transpile.ir.IRObserve] or
        [`IRSample`][quivers.transpile.ir.IRSample] node whose
        family's positional arg constraint is an
        `IndependentConstraint(base, n>=1)`, walk its `args`. When the
        matching arg is an [`IRArgRef`][quivers.transpile.ir.IRArgRef]
        to a name that the program declares with scalar real support
        (no event dims), record `{name: K}` where K is the consumer's
        event-axis cardinality.

        K is sourced from the consumer node's plate's last batch dim
        (the Categorical's output type axis, per the language-model
        gallery's `morphism lm_head : Hidden -> Token ~ Categorical`
        idiom). When the consumer's batch dim does not exist or its
        size is dynamic, the promotion is skipped and the original
        declaration shape stands.
        """
        producers: dict[str, ConstraintSpec] = {}
        producer_event_dims: dict[str, tuple[Dim, ...]] = {}
        for inp in ir.inputs:
            producers[inp.name] = inp.constraint
            producer_event_dims[inp.name] = inp.plate.event_dims
        promotions: dict[str, int] = {}
        for node in ir.body:
            if isinstance(node, (IRSample, IRObserve)):
                meta = FAMILY_META.get(node.family)
                if meta is not None:
                    self._record_vector_promotions(
                        promotions, node, meta, producers,
                        producer_event_dims,
                    )
            if isinstance(node, (IRSample, IRDeterministic, IRObserve)):
                producers[node.name] = node.constraint
                producer_event_dims[node.name] = node.plate.event_dims
        return promotions

    def _record_vector_promotions(
        self,
        promotions: dict[str, int],
        node: IRSample | IRObserve,
        meta: FamilyMeta,
        producers: dict[str, ConstraintSpec],
        producer_event_dims: dict[str, tuple[Dim, ...]],
    ) -> None:
        """For each arg-position whose family constraint has
        `event_dim >= 1` and whose arg is a ref to a scalar producer,
        record the producer name with the required K."""
        cls_attr = meta.distribution_class.arg_constraints
        if not isinstance(cls_attr, dict):
            return
        constraints = tuple(cls_attr.values())
        K = self._consumer_event_K(node)
        if K is None:
            return
        for i, arg in enumerate(node.args):
            if i >= len(constraints):
                continue
            expected = constraints[i]
            event_dim = int(getattr(expected, "event_dim", 0))
            if event_dim < 1:
                continue
            if not isinstance(arg, IRArgRef) or arg.indices:
                continue
            producer_cs = producers.get(arg.name)
            if producer_cs is None:
                continue
            producer_event = producer_event_dims.get(arg.name, ())
            if producer_event:
                # Producer already carries event dims; no promotion.
                continue
            producer_sup = producer_cs.to_constraint()
            if not (
                is_real_scalar(producer_sup)
                or is_real_positive(producer_sup)
                or is_real_unit_interval(producer_sup)
                or is_real_bounded_interval(producer_sup)
            ):
                continue
            existing = promotions.get(arg.name)
            if existing is not None and existing != K:
                # Conflicting K from two different consumers; skip
                # promotion rather than guess.
                continue
            promotions[arg.name] = K

    def _consumer_event_K(
        self, node: IRSample | IRObserve
    ) -> int | None:
        """Derive the event-axis cardinality K for a consumer whose
        slot expects an event_dim>=1 arg.

        For the language-model idiom `observe target : Token <-
        Categorical(h)`, the Token axis is the consumer node's plate
        batch_dim. The cardinality is the static size of that batch
        dim. When the batch dim is dynamic or absent, returns None
        and the promotion path skips this consumer.
        """
        if not node.plate.batch_dims:
            return None
        last = node.plate.batch_dims[-1]
        if isinstance(last, DimStatic):
            return last.size
        return None

    _CANONICAL_BLOCK_ORDER: tuple[BlockKind, ...] = (
        "data",
        "parameters",
        "transformed_parameters",
        "model",
        "generated_quantities",
    )

    def _needed_blocks(self, ir: IRProgram) -> list[BlockKind]:
        """Return the Stan blocks the renderer must materialise for
        `ir`, in canonical grammar order. A block is needed iff some
        IR node will route emit into it; pre-allocating in this order
        ensures the pretty-printer never sees a block out of grammar
        position."""
        needed: set[BlockKind] = set()
        if ir.inputs:
            needed.add("data")
        for node in ir.body:
            if isinstance(node, IRSample):
                needed.add("parameters")
                needed.add("model")
            elif isinstance(node, IRObserve):
                needed.add("data")
                needed.add("model")
            elif isinstance(node, IRDataInput):
                needed.add("data")
            elif isinstance(node, IRDeterministic):
                needed.add("transformed_parameters")
            elif isinstance(node, IRScore):
                needed.add("model")
            elif isinstance(node, IRMarginalize):
                needed.add("model")
            elif isinstance(node, IRReturn):
                needed.add("generated_quantities")
        return [k for k in self._CANONICAL_BLOCK_ORDER if k in needed]

    # ----- declare dispatch -----

    def declare(
        self,
        ctx: _RenderCtx,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """Emit a declaration for `name` in `block`.

        Dispatches purely on the support predicates of §2.2 of the
        spec, threaded through
        [`ConstraintSpec.to_constraint`][quivers.transpile.ir.ConstraintSpec.to_constraint].
        The table is the §5 Stan table.
        """
        if name in self._declared[block]:
            return ""
        self._declared[block].add(name)
        sup = constraint.to_constraint()
        # Track for later GQ aliasing emission.
        self._declared_shapes[name] = (sup, plate)
        parent = self._ensure_block(ctx, block)
        decl = self._fresh(ctx, "decl")
        ctx.sb.vertex(decl, "top_var_decl_no_assign")
        ctx.sb.edge(parent, decl, "child_of")
        # arr_dims when there are batch dims.
        if plate.batch_dims:
            arr = self._fresh(ctx, "arr")
            ctx.sb.vertex(arr, "arr_dims")
            for dim in plate.batch_dims:
                size_vid = self._dim_size_vertex(ctx, dim)
                ctx.sb.edge(arr, size_vid, "child_of")
            ctx.sb.edge(decl, arr, "child_of")
        # top_var_type, then the appropriate inner type per predicate.
        tvt = self._fresh(ctx, "tvt")
        ctx.sb.vertex(tvt, "top_var_type")
        promoted_k = self._vector_promotions.get(name)
        if promoted_k is not None:
            self._emit_vector_type_of_size(ctx, tvt, promoted_k)
        else:
            self._emit_type(ctx, tvt, sup, plate.event_dims)
        ctx.sb.edge(decl, tvt, "child_of")
        # name field.
        nm = self._fresh(ctx, "ident")
        ctx.sb.vertex(nm, "identifier")
        ctx.sb.constraint(nm, "literal-value", name)
        ctx.sb.edge(decl, nm, "name")
        return decl

    def _emit_type(
        self,
        ctx: _RenderCtx,
        tvt_vid: str,
        sup: Constraint,
        event_dims: tuple[Dim, ...],
    ) -> None:
        """Materialise the appropriate top_var_type child per the
        renderer's support dispatch table."""
        # A real-valued family with non-empty event dims is logically
        # a vector / matrix even though the per-element support is
        # `CSReal`; defer to the event-shape-aware emitters so the
        # declaration is `vector[K]` rather than scalar `real`.
        if event_dims and (
            is_real_scalar(sup)
            or is_real_positive(sup)
            or is_real_unit_interval(sup)
            or is_real_bounded_interval(sup)
        ):
            if len(event_dims) == 1:
                self._emit_vector_type(ctx, tvt_vid, event_dims)
                return
            if len(event_dims) == 2:
                self._emit_matrix_type(ctx, tvt_vid, event_dims)
                return
        # Scalar real family.
        if is_real_scalar(sup):
            self._emit_real_type(ctx, tvt_vid, lower=None, upper=None)
            return
        if is_real_positive(sup):
            self._emit_real_type(ctx, tvt_vid, lower=0, upper=None)
            return
        if is_real_unit_interval(sup):
            self._emit_real_type(ctx, tvt_vid, lower=0, upper=1)
            return
        if is_real_bounded_interval(sup):
            lo, hi = real_interval_bounds(sup)
            self._emit_real_type(ctx, tvt_vid, lower=lo, upper=hi)
            return
        if is_real_vector(sup):
            self._emit_vector_type(ctx, tvt_vid, event_dims)
            return
        if is_real_simplex(sup) or is_real_one_hot(sup):
            self._emit_simplex_type(ctx, tvt_vid, event_dims)
            return
        if is_real_cov_matrix(sup):
            self._emit_cov_matrix_type(ctx, tvt_vid, event_dims)
            return
        if is_real_corr_chol(sup):
            self._emit_corr_chol_type(ctx, tvt_vid, event_dims)
            return
        if is_real_matrix(sup):
            self._emit_matrix_type(ctx, tvt_vid, event_dims)
            return
        if is_int_bit(sup):
            self._emit_int_type(ctx, tvt_vid, lower=0, upper=1)
            return
        if is_int_category(sup):
            upper = self._category_upper(sup)
            self._emit_int_type(ctx, tvt_vid, lower=1, upper=upper)
            return
        if is_int_count(sup):
            self._emit_int_type(ctx, tvt_vid, lower=0, upper=None)
            return
        raise UnsupportedConstruct(
            "qvr-stan",
            [f"declare:unsupported-support:{type(sup).__name__}"],
        )

    def _emit_real_type(
        self,
        ctx: _RenderCtx,
        tvt_vid: str,
        *,
        lower: int | float | None,
        upper: int | float | None,
    ) -> None:
        rt = self._fresh(ctx, "rt")
        ctx.sb.vertex(rt, "real_type")
        ctx.sb.constraint(rt, "literal-value", "real")
        self._maybe_emit_range_constraint(ctx, rt, lower, upper)
        ctx.sb.edge(tvt_vid, rt, "child_of")

    def _emit_int_type(
        self,
        ctx: _RenderCtx,
        tvt_vid: str,
        *,
        lower: int | float | None,
        upper: int | float | None,
    ) -> None:
        it = self._fresh(ctx, "it")
        ctx.sb.vertex(it, "int_type")
        ctx.sb.constraint(it, "literal-value", "int")
        self._maybe_emit_int_range(ctx, it, lower, upper)
        ctx.sb.edge(tvt_vid, it, "child_of")

    def _emit_vector_type(
        self,
        ctx: _RenderCtx,
        tvt_vid: str,
        event_dims: tuple[Dim, ...],
    ) -> None:
        if len(event_dims) != 1:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"declare:vector:event-rank:{len(event_dims)}: "
                    f"Stan vector requires exactly one event dim"
                ],
            )
        vt = self._fresh(ctx, "vt")
        ctx.sb.vertex(vt, "vector_type")
        size_vid = self._dim_size_vertex(ctx, event_dims[0])
        ctx.sb.edge(vt, size_vid, "child_of")
        ctx.sb.edge(tvt_vid, vt, "child_of")

    def _emit_vector_type_of_size(
        self,
        ctx: _RenderCtx,
        tvt_vid: str,
        size: int,
    ) -> None:
        """Emit a `vector[K]` type child under `tvt_vid` with a
        literal integer size.

        Used by the vector-promotion path: chain-tail variables whose
        downstream consumer binds them to an event_dim>=1 slot are
        declared as `vector[K]` (wrapped in `array[batch]` when the
        producer had batch dims) rather than scalar `real`.
        """
        vt = self._fresh(ctx, "vt")
        ctx.sb.vertex(vt, "vector_type")
        size_vid = self._int_literal(ctx, size)
        ctx.sb.edge(vt, size_vid, "child_of")
        ctx.sb.edge(tvt_vid, vt, "child_of")

    def _emit_simplex_type(
        self,
        ctx: _RenderCtx,
        tvt_vid: str,
        event_dims: tuple[Dim, ...],
    ) -> None:
        if len(event_dims) != 1:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"declare:simplex:event-rank:{len(event_dims)}: "
                    f"Stan simplex requires exactly one event dim"
                ],
            )
        st = self._fresh(ctx, "st")
        ctx.sb.vertex(st, "simplex_type")
        size_vid = self._dim_size_vertex(ctx, event_dims[0])
        ctx.sb.edge(st, size_vid, "child_of")
        ctx.sb.edge(tvt_vid, st, "child_of")

    def _emit_cov_matrix_type(
        self,
        ctx: _RenderCtx,
        tvt_vid: str,
        event_dims: tuple[Dim, ...],
    ) -> None:
        # Accept either a single event dim (e.g. `(Dim,)` from an
        # explicit `over=[Dim]`) or two equal-size event dims (e.g.
        # `(Dim, Dim)` from a matrix-valued morphism with two
        # identical axes). Stan's `cov_matrix[N]` is square by
        # definition; both representations carry the same `N`.
        if len(event_dims) == 2:
            d0, d1 = event_dims
            size0 = getattr(d0, "size", None)
            size1 = getattr(d1, "size", None)
            if size0 is None or size1 is None or size0 != size1:
                raise UnsupportedConstruct(
                    "qvr-stan",
                    [
                        f"declare:cov_matrix:event-rank:2-non-square: "
                        f"sizes {size0} vs {size1}"
                    ],
                )
            chosen = d0
        elif len(event_dims) == 1:
            chosen = event_dims[0]
        else:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"declare:cov_matrix:event-rank:{len(event_dims)}: "
                    f"Stan cov_matrix requires a square event dim"
                ],
            )
        ct = self._fresh(ctx, "ct")
        ctx.sb.vertex(ct, "cov_matrix_type")
        size_vid = self._dim_size_vertex(ctx, chosen)
        ctx.sb.edge(ct, size_vid, "child_of")
        ctx.sb.edge(tvt_vid, ct, "child_of")

    def _emit_corr_chol_type(
        self,
        ctx: _RenderCtx,
        tvt_vid: str,
        event_dims: tuple[Dim, ...],
    ) -> None:
        if len(event_dims) != 1:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"declare:cholesky:event-rank:{len(event_dims)}: "
                    f"Stan cholesky_factor_corr requires one event dim"
                ],
            )
        ct = self._fresh(ctx, "cct")
        ctx.sb.vertex(ct, "cholesky_factor_corr_type")
        size_vid = self._dim_size_vertex(ctx, event_dims[0])
        ctx.sb.edge(ct, size_vid, "child_of")
        ctx.sb.edge(tvt_vid, ct, "child_of")

    def _emit_matrix_type(
        self,
        ctx: _RenderCtx,
        tvt_vid: str,
        event_dims: tuple[Dim, ...],
    ) -> None:
        if len(event_dims) != 2:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"declare:matrix:event-rank:{len(event_dims)}: "
                    f"Stan matrix requires two event dims"
                ],
            )
        mt = self._fresh(ctx, "mt")
        ctx.sb.vertex(mt, "matrix_type")
        row_vid = self._dim_size_vertex(ctx, event_dims[0])
        col_vid = self._dim_size_vertex(ctx, event_dims[1])
        ctx.sb.edge(mt, row_vid, "child_of")
        ctx.sb.edge(mt, col_vid, "child_of")
        ctx.sb.edge(tvt_vid, mt, "child_of")

    def _maybe_emit_range_constraint(
        self,
        ctx: _RenderCtx,
        parent: str,
        lower: int | float | None,
        upper: int | float | None,
    ) -> None:
        """Attach a `type_constraint` with `<lower=..., upper=...>` to
        a real-typed declaration."""
        if lower is None and upper is None:
            return
        tc = self._fresh(ctx, "tc")
        ctx.sb.vertex(tc, "type_constraint")
        range_vid = self._build_range_node(ctx, lower, upper)
        ctx.sb.edge(tc, range_vid, "child_of")
        ctx.sb.edge(parent, tc, "child_of")

    def _maybe_emit_int_range(
        self,
        ctx: _RenderCtx,
        int_type_vid: str,
        lower: int | float | None,
        upper: int | float | None,
    ) -> None:
        """Attach a `range_*` directly to an `int_type`. Stan's int
        type accepts the range as a direct child."""
        if lower is None and upper is None:
            return
        range_vid = self._build_range_node(ctx, lower, upper)
        ctx.sb.edge(int_type_vid, range_vid, "child_of")

    def _build_range_node(
        self,
        ctx: _RenderCtx,
        lower: int | float | None,
        upper: int | float | None,
    ) -> str:
        if lower is not None and upper is not None:
            rng = self._fresh(ctx, "rng")
            ctx.sb.vertex(rng, "range_lower_upper")
            lo = self._int_literal(ctx, lower)
            hi = self._int_literal(ctx, upper)
            ctx.sb.edge(rng, lo, "child_of")
            ctx.sb.edge(rng, hi, "child_of")
            return rng
        if lower is not None:
            rng = self._fresh(ctx, "rng")
            ctx.sb.vertex(rng, "range_lower")
            lo = self._int_literal(ctx, lower)
            ctx.sb.edge(rng, lo, "child_of")
            return rng
        rng = self._fresh(ctx, "rng")
        ctx.sb.vertex(rng, "range_upper")
        hi = self._int_literal(ctx, upper if upper is not None else 0)
        ctx.sb.edge(rng, hi, "child_of")
        return rng

    def _category_upper(self, sup: Constraint) -> int:
        """For an `is_int_category` support, return the upper category
        index (Stan's `int<lower=1, upper=K>` requires K)."""
        lower = int(getattr(sup, "lower_bound", 0))
        upper = int(getattr(sup, "upper_bound", 1))
        if lower == 0:
            return upper + 1
        return upper

    # ----- sample / observe dispatch -----

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
    ) -> SchemaFragment:
        """Emit a `~` statement for a sample / observe step.

        Wraps the `~` in nested `for (m_<axis> in 1:<size>)` loops for
        each batch dim. Indexes the LHS and every arg whose ref-name
        sits on the plate.
        """
        del observed  # The sample emission shape does not differ.
        del constraint  # The constraint shaped the declaration; the
        # tilde uses the family's name from FAMILY_META.
        # Resolve the per-target family name.
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:unknown:{family}"],
            )
        stan_name = meta.target_names.get("stan")
        if stan_name is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:no-stan-target:{family}"],
            )
        del arg_names  # Stan is positional; arg_names are unused.
        parent = self._ensure_block(ctx, "model")
        # Build loop indices for the batch dims.
        loop_names = self.index_for(ctx, plate)
        # The innermost block we'll attach the sampling_statement to.
        innermost_parent = self._wrap_in_for_loops(
            ctx, parent, plate.batch_dims, loop_names
        )
        # Emit the sampling statement.
        ss = self._fresh(ctx, "ss")
        ctx.sb.vertex(ss, "sampling_statement")
        # LHS: name indexed by loop vars.
        lhs_vid = self._build_lhs(ctx, name, loop_names)
        ctx.sb.edge(ss, lhs_vid, "child_of")
        # Distribution name on the `name` field.
        dn = self._fresh(ctx, "dnm")
        ctx.sb.vertex(dn, "identifier")
        ctx.sb.constraint(dn, "literal-value", stan_name)
        ctx.sb.edge(ss, dn, "name")
        # Args + optional truncation suffix.
        # TruncatedNormal: split (loc, scale, low, high) -> two args
        # to `normal()` plus a `T[low, high]` suffix.
        if family == "TruncatedNormal":
            if len(args) != 4:
                raise UnsupportedConstruct(
                    "qvr-stan",
                    [
                        f"family:TruncatedNormal: expected 4 args "
                        f"(loc, scale, low, high), got {len(args)}"
                    ],
                )
            family_args = args[:2]
            truncation_args: tuple[IRArg, ...] | None = args[2:]
        else:
            family_args = args
            truncation_args = None
        injected = self._inject_stan_specific_args(family, family_args)
        rewritten = self._broadcast_scalar_refs(injected, meta, plate)
        for arg in rewritten:
            substituted = self._substitute_for_loops(
                arg, plate, loop_names
            )
            arg_vid = self._render_arg(ctx, substituted)
            ctx.sb.edge(ss, arg_vid, "child_of")
        if truncation_args is not None:
            trunc_vid = self._build_truncation_suffix(
                ctx, truncation_args, plate, loop_names
            )
            ctx.sb.edge(ss, trunc_vid, "child_of")
        ctx.sb.edge(innermost_parent, ss, "child_of")
        return ss

    def _build_truncation_suffix(
        self,
        ctx: _RenderCtx,
        bounds: tuple[IRArg, ...],
        plate: Plate,
        loop_names: tuple[str, ...],
    ) -> str:
        """Emit a Stan `lower_upper_truncation` vertex (the `T[low,
        high]` suffix on a sampling statement) carrying two rendered
        bound expressions.

        Stan's grammar slots this vertex as a `child_of` of the
        enclosing `sampling_statement`; the pretty printer recovers
        the canonical `T[lower, upper];` layout from the
        `chose-alt-fingerprint` constraint.
        """
        trunc = self._fresh(ctx, "tnc")
        ctx.sb.vertex(trunc, "lower_upper_truncation")
        rendered_kinds: list[str] = []
        for bound in bounds:
            substituted = self._substitute_for_loops(
                bound, plate, loop_names
            )
            bound_vid = self._render_arg(ctx, substituted)
            ctx.sb.edge(trunc, bound_vid, "child_of")
            rendered_kinds.append(
                ctx.sb.kind_of(bound_vid)
                if hasattr(ctx.sb, "kind_of")
                else "real_literal"
            )
        ctx.sb.constraint(trunc, "chose-alt-fingerprint", "T[ , ]")
        ctx.sb.constraint(
            trunc, "chose-alt-child-kinds",
            " ".join(rendered_kinds),
        )
        return trunc

    def _inject_stan_specific_args(
        self,
        family: str,
        args: tuple[IRArg, ...],
    ) -> tuple[IRArg, ...]:
        """Prepend Stan-side parameter placeholders for QVR families
        whose torch distribution carries fewer parameters than Stan's
        same-named distribution.

        `HalfNormal(scale)` maps to Stan's `normal(0, scale)`; the
        renderer prepends `IRArgNumber(0)` before emission. The
        `<lower=0>` parameter constraint on the LHS comes from the
        family's `support`, dispatched by `is_real_positive` in
        `declare`.
        """
        if family in _PREPEND_ZERO:
            return (IRArgNumber(value=0.0), *args)
        pos = _INSERT_ZERO_AT.get(family)
        if pos is not None:
            return (
                *args[:pos],
                IRArgNumber(value=0.0),
                *args[pos:],
            )
        return args

    def _broadcast_scalar_refs(
        self,
        args: tuple[IRArg, ...],
        meta: FamilyMeta,
        plate: Plate,
    ) -> tuple[IRArg, ...]:
        """Per-renderer broadcast normalisation.

        Lower leaves [`IRArgRef`][quivers.transpile.ir.IRArgRef] args
        unchanged when the user supplied a name; the family's
        `arg_constraints` may still require a vector / matrix at that
        position. When a ref's declared shape is scalar but the
        constraint is an `IndependentConstraint(base, n>=1)`, wrap it
        in [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast] so
        the Stan emit becomes `rep_vector(<name>, K)`.

        Target shape is derived from the family's event-axis size on
        the sample step (the first event_dim, when present).
        """
        cls_attr = meta.distribution_class.arg_constraints
        if not isinstance(cls_attr, dict):
            return args
        constraints = tuple(cls_attr.values())
        out: list[IRArg] = []
        for i, arg in enumerate(args):
            if i >= len(constraints):
                out.append(arg)
                continue
            expected = constraints[i]
            if not isinstance(
                arg, IRArgRef
            ) or not isinstance(
                expected, _torch_constraints._IndependentConstraint
            ):
                out.append(arg)
                continue
            declared = self._declared_shapes.get(arg.name)
            if declared is None:
                out.append(arg)
                continue
            declared_sup, declared_plate = declared
            if (
                declared_plate.event_dims
                or not is_real_scalar(declared_sup)
                and not is_real_positive(declared_sup)
                and not is_real_unit_interval(declared_sup)
            ):
                out.append(arg)
                continue
            # Derive the target shape: take the first n event dims off
            # the sample's plate (event_dim of the constraint).
            n = int(expected.event_dim)
            if len(plate.event_dims) < n:
                out.append(arg)
                continue
            sizes: list[int] = []
            ok = True
            for dim in plate.event_dims[:n]:
                if isinstance(dim, DimStatic):
                    sizes.append(dim.size)
                else:
                    ok = False
                    break
            if not ok:
                out.append(arg)
                continue
            out.append(
                IRArgBroadcast(
                    value=arg, target_shape=tuple(sizes)
                )
            )
        return tuple(out)

    def _wrap_in_for_loops(
        self,
        ctx: _RenderCtx,
        parent: str,
        batch_dims: tuple[Dim, ...],
        loop_names: tuple[str, ...],
    ) -> str:
        """Emit nested `for (m_<name> in 1:<size>) { ... }` loops, one
        per batch dim. Returns the innermost block vertex id."""
        current = parent
        for dim, lv_name in zip(batch_dims, loop_names, strict=True):
            fs = self._fresh(ctx, "fs")
            ctx.sb.vertex(fs, "for_statement")
            ctx.sb.edge(current, fs, "child_of")
            # loopvar identifier
            lv = self._fresh(ctx, "lv")
            ctx.sb.vertex(lv, "identifier")
            ctx.sb.constraint(lv, "literal-value", lv_name)
            ctx.sb.edge(fs, lv, "loopvar")
            # lower = 1
            lo = self._int_literal(ctx, 1)
            ctx.sb.edge(fs, lo, "child_of")
            # upper = batch_dim size
            hi = self._dim_size_vertex(ctx, dim)
            ctx.sb.edge(fs, hi, "child_of")
            # body block
            bs = self._fresh(ctx, "bs")
            ctx.sb.vertex(bs, "block_statement")
            ctx.sb.edge(fs, bs, "child_of")
            current = bs
        return current

    def _build_lhs(
        self,
        ctx: _RenderCtx,
        name: str,
        loop_names: tuple[str, ...],
    ) -> str:
        """Build the LHS of a `~` statement: either a bare
        `variable_expression` or an `indexed_expression`. Lower's
        sample emission for a plated draw lands as
        `<name>[m_0, m_1, ...]` when loop vars exist."""
        base_ve = self._variable_expression(ctx, name)
        if not loop_names:
            return base_ve
        ie = self._fresh(ctx, "ie")
        ctx.sb.vertex(ie, "indexed_expression")
        ctx.sb.edge(ie, base_ve, "child_of")
        for lv_name in loop_names:
            idx = self._fresh(ctx, "idx")
            ctx.sb.vertex(idx, "index")
            idx_ve = self._variable_expression(ctx, lv_name)
            ctx.sb.edge(idx, idx_ve, "child_of")
            ctx.sb.edge(ie, idx, "child_of")
        return ie

    def _substitute_for_loops(
        self,
        arg: IRArg,
        plate: Plate,
        loop_names: tuple[str, ...],
    ) -> IRArg:
        """For an [`IRArgRef`][quivers.transpile.ir.IRArgRef] whose
        referenced name itself sits on the surrounding plate, prepend
        the loop indices.

        Two cases handled:

        * User-written index already present (`phi[z]`, `mu[cls]`):
          left alone; the IR already encodes the per-element lookup.
        * Unindexed ref whose declared plate's batch_dims align (as a
          tuple-equality on `Dim`s) with `plate.batch_dims`: prepend
          one [`IRArgRef`][quivers.transpile.ir.IRArgRef] index per
          aligned batch dim, in declaration order. The loop indices
          are 1-based identifiers; the Stan
          [`indexed_expression`][quivers.transpile.renderers._stan_helpers]
          emit takes them verbatim.

        This is what
        [`_propagate_let_plates`][quivers.transpile.lower._propagate_let_plates]
        relies on: it promotes a let-bound `mu` from scalar to
        `array[Obs] real`, and this hook makes the surrounding
        `observe y : Obs <- Normal(mu, ...)` index `mu` per-element
        rather than reading the whole array into the scalar Normal
        slot (which Stan rejects with a dimension mismatch at runtime
        data binding).
        """
        if not isinstance(arg, IRArgRef):
            return arg
        if arg.indices:
            return arg
        declared = self._declared_shapes.get(arg.name)
        if declared is None:
            return arg
        decl_plate = declared[1]
        if decl_plate.batch_dims != plate.batch_dims:
            return arg
        if not loop_names:
            return arg
        return IRArgRef(
            name=arg.name,
            indices=tuple(
                IRArgRef(name=lv) for lv in loop_names
            ),
        )

    # ----- marginalize dispatch -----

    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Emit Stan's `log_sum_exp` enumeration for an
        [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] node.

        Raises
        [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
        with kind `marginalize:non-finite-support:<family>` when
        [`finite_enumerable_at_call_site`][quivers.transpile.family_meta.finite_enumerable_at_call_site]
        returns False.
        """
        meta = FAMILY_META.get(node.family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:unknown:{node.family}"],
            )
        if not finite_enumerable_at_call_site(meta, node.args):
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"marginalize:non-finite-support:{node.family}"],
            )
        stan_name = meta.target_names.get("stan")
        if stan_name is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:no-stan-target:{node.family}"],
            )
        # Eligibility checks satisfied; compute the latent cardinality.
        latent_card = self._latent_cardinality(meta, node, ctx)
        if latent_card is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"marginalize:unknown-cardinality:{node.family}: "
                    f"cannot determine support size at compile time"
                ],
            )
        parent = self._ensure_block(ctx, "model")
        # Open a fresh `{ ... }` block scope for the lps_<latent> array.
        outer_block = self._fresh(ctx, "mbs")
        ctx.sb.vertex(outer_block, "block_statement")
        ctx.sb.edge(parent, outer_block, "child_of")
        # 1. Declare `array[batch_dims] vector[K] lps_<latent>;`.
        lps_name = f"lps_{node.latent}"
        self._declare_lps_array(
            ctx, outer_block, lps_name, node.plate.batch_dims, latent_card
        )
        # 2. Initialise per-group lps with the latent log-pmf:
        #    for each group index, for k in 1:K, lps[..., k] = lpmf(k | args).
        group_idx_names = self._marginalize_loop_names(node.plate.batch_dims)
        self._emit_lps_init(
            ctx,
            outer_block,
            lps_name,
            node.plate.batch_dims,
            group_idx_names,
            latent_card,
            stan_name,
            node.args,
            meta,
        )
        # 3. Walk the scope body: each scope IRObserve emits an
        #    inner-loop that accumulates per-k contributions into
        #    `lps[group, k]`.
        prev_marg_var = self._marginalize_var
        prev_marg_card = self._marginalize_latent_card
        prev_group_idx = self._marginalize_group_idx
        prev_stack = self._marginalize_stack
        self._marginalize_var = lps_name
        self._marginalize_latent_card = latent_card
        self._marginalize_group_idx = group_idx_names
        self._marginalize_stack = (*self._marginalize_stack, node.plate)
        try:
            for scope_node in node.scope:
                self._dispatch_marginalize_scope(
                    ctx, outer_block, scope_node, node
                )
        finally:
            self._marginalize_var = prev_marg_var
            self._marginalize_latent_card = prev_marg_card
            self._marginalize_group_idx = prev_group_idx
            self._marginalize_stack = prev_stack
        # 4. Accumulate the group log-sums into `target`.
        self._emit_lps_accumulate(
            ctx,
            outer_block,
            lps_name,
            node.plate.batch_dims,
            group_idx_names,
        )
        return outer_block

    def _dispatch_marginalize_scope(
        self,
        ctx: _RenderCtx,
        scope_block: str,
        node: IRNode,
        parent: IRMarginalize,
    ) -> None:
        """Handle one node inside a [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        scope. Inner observe steps contribute to the per-group `lps`
        accumulator; other constructs raise."""
        if isinstance(node, IRObserve):
            self._emit_marginalize_scope_observe(
                ctx, scope_block, node, parent
            )
            return
        if isinstance(node, IRSample):
            self._emit_marginalize_scope_sample(
                ctx, scope_block, node, parent
            )
            return
        if isinstance(node, IRDeterministic):
            # A scope-local let binding (e.g. `let gated_rate = z *
            # rate; observe y <- Poisson(gated_rate)` inside a
            # marginalize over z): inlined into downstream observe
            # args via _marginalize_let_subs rather than declared as
            # a Stan parameter, since `z` is the loop variable `k`
            # and the let is meaningful only per-k.
            self._marginalize_let_subs[node.name] = node.expr
            return
        raise UnsupportedConstruct(
            "qvr-stan",
            [
                f"marginalize:scope:unsupported:"
                f"{type(node).__name__}"
            ],
        )

    def _emit_marginalize_scope_observe(
        self,
        ctx: _RenderCtx,
        scope_block: str,
        node: IRObserve,
        parent: IRMarginalize,
    ) -> None:
        """Emit the per-k log-pmf accumulation for a scope observe.

        For an observe `r ~ Normal(mu[cls], sigma[cls])` inside a
        `marginalize cls : Component` scope with parent plate `[Item]`,
        Stan emits:

            for (n in 1:N_r) {
              for (k in 1:K) {
                lps_cls[<via(n)>, k] += normal_lpdf(r[n] | mu[k], sigma[k]);
              }
            }
        """
        meta = FAMILY_META.get(node.family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:unknown:{node.family}"],
            )
        stan_name = meta.target_names.get("stan")
        if stan_name is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:no-stan-target:{node.family}"],
            )
        lps_name = self._marginalize_var or ""
        if not lps_name:
            raise UnsupportedConstruct(
                "qvr-stan",
                ["marginalize:scope:observe:no-lps-context"],
            )
        # Outer loop over the observe's plate batch dims.
        loop_names = self._observe_scope_loop_names(node.plate.batch_dims)
        current = self._wrap_in_for_loops(
            ctx, scope_block, node.plate.batch_dims, loop_names
        )
        # Inner loop over k in 1:K.
        k_name = "k"
        latent_card = self._marginalize_latent_card or 0
        k_loop = self._fresh(ctx, "kfs")
        ctx.sb.vertex(k_loop, "for_statement")
        ctx.sb.edge(current, k_loop, "child_of")
        lv = self._fresh(ctx, "klv")
        ctx.sb.vertex(lv, "identifier")
        ctx.sb.constraint(lv, "literal-value", k_name)
        ctx.sb.edge(k_loop, lv, "loopvar")
        ctx.sb.edge(k_loop, self._int_literal(ctx, 1), "child_of")
        ctx.sb.edge(
            k_loop, self._int_literal(ctx, latent_card), "child_of"
        )
        kbs = self._fresh(ctx, "kbs")
        ctx.sb.vertex(kbs, "block_statement")
        ctx.sb.edge(k_loop, kbs, "child_of")
        # assignment_statement: lps_<latent>[<group_indices>, k] += lpdf(...)
        asn = self._fresh(ctx, "asn")
        ctx.sb.vertex(asn, "assignment_statement")
        ctx.sb.edge(kbs, asn, "child_of")
        # LHS: indexed_lhs lps[group_idx..., k]
        group_idx_exprs = self._marginalize_group_index_exprs(
            node, parent, loop_names
        )
        lhs_vid = self._build_indexed_lhs(
            ctx, lps_name, (*group_idx_exprs, k_name)
        )
        ctx.sb.edge(asn, lhs_vid, "child_of")
        # op: +=
        op = self._fresh(ctx, "aop")
        ctx.sb.vertex(op, "assignment_op")
        ctx.sb.constraint(op, "literal-value", "+=")
        ctx.sb.edge(asn, op, "child_of")
        # RHS: <family>_lpdf/_lpmf(observed_var[n] | substituted args)
        lpdf_name = self._lpdf_name(stan_name, node.constraint)
        lpdf_call = self._build_lpdf_call(
            ctx,
            lpdf_name,
            node.name,
            node.plate.batch_dims,
            loop_names,
            node.args,
            k_name,
        )
        ctx.sb.edge(asn, lpdf_call, "child_of")

    def _emit_marginalize_scope_sample(
        self,
        ctx: _RenderCtx,
        scope_block: str,
        node: IRSample,
        parent: IRMarginalize,
    ) -> None:
        """A nested sample inside a marginalize scope.

        Stan emits per-k conditional accumulation; an inner sample
        of a continuous latent is uncommon in canonical
        log_sum_exp idiom (the spec's gallery examples don't
        exercise this), so raise rather than emit something
        ambiguous.
        """
        del ctx, scope_block, parent
        raise UnsupportedConstruct(
            "qvr-stan",
            [
                f"marginalize:scope:IRSample:{node.name}: nested "
                f"sample inside marginalize scope not supported by "
                f"the Stan log_sum_exp emit"
            ],
        )

    def _marginalize_group_index_exprs(
        self,
        observe: IRObserve,
        parent: IRMarginalize,
        loop_names: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Map the outer marginalize plate's batch dims to indexing
        expressions usable inside this observe's loop body.

        When the observe's `via` fibration names a per-row indexer
        into a group axis, the group index is `<via>[<row-loop>]`;
        otherwise the surrounding marginalize loop variables apply
        directly. When the marginalize has no batch dims, returns ().
        """
        del parent  # parent plate already threaded through
        # _marginalize_group_idx by the marginalize() caller.
        group_count = len(self._marginalize_group_idx)
        if group_count == 0:
            return ()
        if observe.via is not None and loop_names:
            # `<via>[<observe-loop-0>]` per group dim. For
            # single-group marginalize this is exactly the spec's
            # `lps[word_idx[n], k]` form.
            return tuple(
                f"{observe.via}[{loop_names[0]}]"
                for _ in range(group_count)
            )
        # Fall through: use the marginalize group's loop names.
        return self._marginalize_group_idx

    def _build_indexed_lhs(
        self,
        ctx: _RenderCtx,
        base_name: str,
        index_exprs: tuple[str, ...],
    ) -> str:
        """Build a lhs node for `<base_name>[<idx0>, <idx1>, ...]`.

        Index expressions are bare strings (loop var names like `k`
        or call forms like `word_idx[n]`); they're materialised via
        the Stan parser-facing identifier or indexed_expression form.
        """
        lhs = self._fresh(ctx, "lhs")
        ctx.sb.vertex(lhs, "lhs")
        if not index_exprs:
            ve = self._variable_expression(ctx, base_name)
            ctx.sb.edge(lhs, ve, "child_of")
            return lhs
        ilhs = self._fresh(ctx, "ilhs")
        ctx.sb.vertex(ilhs, "indexed_lhs")
        # Inner lhs is the bare-name variable_expression.
        inner_lhs = self._fresh(ctx, "ilhs_inner")
        ctx.sb.vertex(inner_lhs, "lhs")
        base_ve = self._variable_expression(ctx, base_name)
        ctx.sb.edge(inner_lhs, base_ve, "child_of")
        ctx.sb.edge(ilhs, inner_lhs, "child_of")
        for idx_expr in index_exprs:
            idx_vid = self._build_index_node(ctx, idx_expr)
            ctx.sb.edge(ilhs, idx_vid, "child_of")
        ctx.sb.edge(lhs, ilhs, "child_of")
        return lhs

    def _build_index_node(self, ctx: _RenderCtx, expr_text: str) -> str:
        """Build an `index` node carrying a parsed expression.

        Recognises the `<name>[<inner>]` shape and emits an
        `indexed_expression`; otherwise treats `expr_text` as a bare
        name and emits a `variable_expression`.
        """
        idx = self._fresh(ctx, "idx")
        ctx.sb.vertex(idx, "index")
        inner = self._build_expr_from_text(ctx, expr_text)
        ctx.sb.edge(idx, inner, "child_of")
        return idx

    def _build_expr_from_text(
        self, ctx: _RenderCtx, expr_text: str
    ) -> str:
        """Build an expression vertex for a small text form.

        Supports bare identifiers, integer literals, and a single
        level of `<name>[<inner>]` indexing.
        """
        text = expr_text.strip()
        if text.lstrip("-").isdigit():
            return self._int_literal(ctx, int(text))
        if "[" in text and text.endswith("]"):
            bracket = text.index("[")
            base = text[:bracket]
            inner = text[bracket + 1 : -1]
            ie = self._fresh(ctx, "ie")
            ctx.sb.vertex(ie, "indexed_expression")
            base_ve = self._variable_expression(ctx, base)
            ctx.sb.edge(ie, base_ve, "child_of")
            inner_idx = self._build_index_node(ctx, inner)
            ctx.sb.edge(ie, inner_idx, "child_of")
            return ie
        return self._variable_expression(ctx, text)

    def _observe_scope_loop_names(
        self, batch_dims: tuple[Dim, ...]
    ) -> tuple[str, ...]:
        """Generate per-observe loop-variable names. Distinct from
        the outer marginalize group's loop names to keep nesting
        unambiguous."""
        return tuple(f"n_{dim.name}" for dim in batch_dims)

    def _marginalize_loop_names(
        self, batch_dims: tuple[Dim, ...]
    ) -> tuple[str, ...]:
        """Generate group-loop names for the outer marginalize."""
        return tuple(f"g_{dim.name}" for dim in batch_dims)

    def _build_lpdf_call(
        self,
        ctx: _RenderCtx,
        lpdf_name: str,
        observed_var: str,
        batch_dims: tuple[Dim, ...],
        loop_names: tuple[str, ...],
        args: tuple[IRArg, ...],
        k_name: str,
    ) -> str:
        """Build `<lpdf_name>(<observed>[<loop_idx>] | <args with z=k>)`.

        Stan requires the `|` syntax for `_lpdf` / `_lpmf` calls:
        emit a `distr_expression` whose `distr_argument_list` carries
        the observed value first, then a `|`, then the remaining
        positional args.
        """
        del batch_dims
        de = self._fresh(ctx, "de")
        ctx.sb.vertex(de, "distr_expression")
        fnid = self._fresh(ctx, "defnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", lpdf_name)
        ctx.sb.edge(de, fnid, "name")
        dal = self._fresh(ctx, "dal")
        ctx.sb.vertex(dal, "distr_argument_list")
        # First arg: observed[loop_idx] or bare observed.
        if loop_names:
            obs_vid = self._indexed_expression_text(
                ctx, observed_var, loop_names[0]
            )
        else:
            obs_vid = self._variable_expression(ctx, observed_var)
        ctx.sb.edge(dal, obs_vid, "child_of")
        # Subsequent args: every IRArg substituted: latent ref becomes
        # the k loop var.
        for arg in args:
            substituted = self._substitute_latent_for_k(arg, k_name)
            arg_vid = self._render_arg(ctx, substituted)
            ctx.sb.edge(dal, arg_vid, "child_of")
        ctx.sb.edge(de, dal, "child_of")
        return de

    def _substitute_latent_for_k(
        self, arg: IRArg, k_name: str
    ) -> IRArg:
        """Rewrite any IR ref whose name is the marginalize latent
        into `IRArgRef(name=k_name)` so the per-k iteration uses the
        loop variable in place of the latent."""
        latent = self._current_latent_name()
        if latent is None:
            return arg
        if isinstance(arg, IRArgRef):
            new_indices = tuple(
                self._substitute_latent_for_k(idx, k_name)
                for idx in arg.indices
            )
            if arg.name == latent:
                return IRArgRef(name=k_name, indices=new_indices)
            return IRArgRef(name=arg.name, indices=new_indices)
        if isinstance(arg, IRArgBroadcast):
            return IRArgBroadcast(
                value=self._substitute_latent_for_k(arg.value, k_name),
                target_shape=arg.target_shape,
            )
        if isinstance(arg, IRArgList):
            return IRArgList(
                elements=tuple(
                    self._substitute_latent_for_k(e, k_name)
                    for e in arg.elements
                )
            )
        if isinstance(arg, IRArgMatrix):
            new_rows: list[IRArgList] = []
            for row in arg.rows:
                new_rows.append(
                    IRArgList(
                        elements=tuple(
                            self._substitute_latent_for_k(e, k_name)
                            for e in row.elements
                        )
                    )
                )
            return IRArgMatrix(rows=tuple(new_rows))
        return arg

    def _current_latent_name(self) -> str | None:
        """The latent name of the innermost active marginalize scope.

        Encoded into `_marginalize_var` as `lps_<latent>`; strip the
        prefix.
        """
        if self._marginalize_var is None:
            return None
        if self._marginalize_var.startswith("lps_"):
            return self._marginalize_var[4:]
        return None

    def _indexed_expression_text(
        self,
        ctx: _RenderCtx,
        base_name: str,
        index_text: str,
    ) -> str:
        ie = self._fresh(ctx, "ie")
        ctx.sb.vertex(ie, "indexed_expression")
        base_ve = self._variable_expression(ctx, base_name)
        ctx.sb.edge(ie, base_ve, "child_of")
        idx_node = self._build_index_node(ctx, index_text)
        ctx.sb.edge(ie, idx_node, "child_of")
        return ie

    def _lpdf_name(
        self,
        stan_family_name: str,
        constraint: ConstraintSpec,
    ) -> str:
        """Pick the right `<family>_lpmf` vs `<family>_lpdf` suffix
        from the observe's output constraint.

        Per Stan: discrete outputs use `lpmf`; continuous outputs use
        `lpdf`.
        """
        sup = constraint.to_constraint()
        if (
            is_int_bit(sup)
            or is_int_category(sup)
            or is_int_count(sup)
        ):
            return f"{stan_family_name}_lpmf"
        return f"{stan_family_name}_lpdf"

    def _latent_cardinality(
        self,
        meta: FamilyMeta,
        node: IRMarginalize,
        ctx: _RenderCtx,
    ) -> int | None:
        """Compute the finite-support cardinality K for the latent.

        For Categorical / Bernoulli / OrderedLogistic / OrderedProbit
        the cardinality is data-dependent on the `probs` /
        `cutpoints` argument shape. The IR's
        [`CSIntegerInterval`][quivers.transpile.ir.CSIntegerInterval]
        on `node.constraint` carries the sentinel-derived bounds,
        which for a Categorical built from a length-2 placeholder is
        `[0, 1]`; the true K comes from the actual call-site arg
        shape.
        """
        del ctx, meta
        # First, inspect the args for a definitive cardinality.
        if node.args:
            first = node.args[0]
            if isinstance(first, IRArgRef):
                size = self._lookup_simplex_card(first.name)
                if size is not None:
                    return size
            if isinstance(first, IRArgList):
                return len(first.elements)
            if isinstance(first, IRArgBroadcast) and first.target_shape:
                return first.target_shape[0]
        # Fall back to the IR's encoded interval.
        sup = node.constraint.to_constraint()
        if is_int_bit(sup):
            return 2
        if is_int_category(sup):
            lower = int(getattr(sup, "lower_bound", 0))
            upper = int(getattr(sup, "upper_bound", 1))
            if lower == 0:
                return upper + 1
            return upper
        return None

    def _lookup_simplex_card(self, name: str) -> int | None:
        """Try to find the event_dim cardinality of a previously-
        declared simplex named `name` from the program-level
        IRSample / IRDataInput entries.

        The render walk records every plate seen via
        `_simplex_cards`. The renderer populates this incrementally
        as it walks [`IRSample`][quivers.transpile.ir.IRSample]
        nodes; for marginalize over a Categorical the `probs` arg
        is typically a ref to a previously-declared simplex, and
        the cardinality is the simplex's `event_dim`.
        """
        return self._simplex_cards.get(name)

    @property
    def _simplex_cards(self) -> dict[str, int]:
        """Per-render map from simplex-typed name to its event_dim."""
        return self._simplex_cards_state

    def _declare_lps_array(
        self,
        ctx: _RenderCtx,
        scope_block: str,
        lps_name: str,
        batch_dims: tuple[Dim, ...],
        latent_card: int,
    ) -> None:
        """Emit `array[<batch>] vector[K] lps_<latent>;` inside the
        marginalize block. Initialised to zero implicitly by the
        Stan runtime."""
        vd = self._fresh(ctx, "lpsvd")
        ctx.sb.vertex(vd, "var_decl")
        if batch_dims:
            arr = self._fresh(ctx, "lpsarr")
            ctx.sb.vertex(arr, "arr_dims")
            for dim in batch_dims:
                size_vid = self._dim_size_vertex(ctx, dim)
                ctx.sb.edge(arr, size_vid, "child_of")
            ctx.sb.edge(vd, arr, "child_of")
        sbt = self._fresh(ctx, "lpssbt")
        ctx.sb.vertex(sbt, "sized_basic_type")
        # vector[K]
        kv = self._int_literal(ctx, latent_card)
        ctx.sb.edge(sbt, kv, "child_of")
        ctx.sb.edge(vd, sbt, "child_of")
        nm = self._fresh(ctx, "lpsnm")
        ctx.sb.vertex(nm, "identifier")
        ctx.sb.constraint(nm, "literal-value", lps_name)
        ctx.sb.edge(vd, nm, "name")
        # Initializer: rep_vector(0, K) for a scalar accumulator (no
        # batch_dims) or rep_array(rep_vector(0, K), B0, B1, ...) for
        # the per-group array. The two-argument variant Stan offers
        # for ``rep_array`` requires at least one size, so the scalar
        # case must call ``rep_vector`` directly.
        rep_v = self._fresh(ctx, "lpsrv")
        ctx.sb.vertex(rep_v, "function_expression")
        rep_v_fn = self._fresh(ctx, "lpsrvf")
        ctx.sb.vertex(rep_v_fn, "identifier")
        ctx.sb.constraint(rep_v_fn, "literal-value", "rep_vector")
        ctx.sb.edge(rep_v, rep_v_fn, "name")
        rep_v_al = self._fresh(ctx, "lpsrval")
        ctx.sb.vertex(rep_v_al, "argument_list")
        ctx.sb.edge(rep_v_al, self._int_literal(ctx, 0), "child_of")
        ctx.sb.edge(
            rep_v_al, self._int_literal(ctx, latent_card), "child_of"
        )
        ctx.sb.edge(rep_v, rep_v_al, "child_of")
        if batch_dims:
            init_call = self._fresh(ctx, "lpsinit")
            ctx.sb.vertex(init_call, "function_expression")
            init_fn = self._fresh(ctx, "lpsfn")
            ctx.sb.vertex(init_fn, "identifier")
            ctx.sb.constraint(init_fn, "literal-value", "rep_array")
            ctx.sb.edge(init_call, init_fn, "name")
            init_al = self._fresh(ctx, "lpsal")
            ctx.sb.vertex(init_al, "argument_list")
            ctx.sb.edge(init_al, rep_v, "child_of")
            for dim in batch_dims:
                ctx.sb.edge(
                    init_al, self._dim_size_vertex(ctx, dim), "child_of"
                )
            ctx.sb.edge(init_call, init_al, "child_of")
            ctx.sb.edge(vd, init_call, "child_of")
        else:
            ctx.sb.edge(vd, rep_v, "child_of")
        ctx.sb.edge(scope_block, vd, "child_of")

    def _emit_lps_init(
        self,
        ctx: _RenderCtx,
        scope_block: str,
        lps_name: str,
        batch_dims: tuple[Dim, ...],
        group_idx_names: tuple[str, ...],
        latent_card: int,
        stan_name: str,
        latent_args: tuple[IRArg, ...],
        latent_meta: FamilyMeta,
    ) -> None:
        """Emit `for g in batch_dims, for k in 1:K, lps[g, k] +=
        <family>_lpmf(k | <latent_args>);` to seed the per-group
        marginal with the latent's own log-pmf."""
        current = self._wrap_in_for_loops(
            ctx, scope_block, batch_dims, group_idx_names
        )
        k_loop = self._fresh(ctx, "ifs")
        ctx.sb.vertex(k_loop, "for_statement")
        ctx.sb.edge(current, k_loop, "child_of")
        klv = self._fresh(ctx, "ilv")
        ctx.sb.vertex(klv, "identifier")
        ctx.sb.constraint(klv, "literal-value", "k")
        ctx.sb.edge(k_loop, klv, "loopvar")
        ctx.sb.edge(k_loop, self._int_literal(ctx, 1), "child_of")
        ctx.sb.edge(
            k_loop, self._int_literal(ctx, latent_card), "child_of"
        )
        kbs = self._fresh(ctx, "ibs")
        ctx.sb.vertex(kbs, "block_statement")
        ctx.sb.edge(k_loop, kbs, "child_of")
        asn = self._fresh(ctx, "iasn")
        ctx.sb.vertex(asn, "assignment_statement")
        ctx.sb.edge(kbs, asn, "child_of")
        # LHS lps[group..., k]
        lhs_vid = self._build_indexed_lhs(
            ctx, lps_name, (*group_idx_names, "k")
        )
        ctx.sb.edge(asn, lhs_vid, "child_of")
        op = self._fresh(ctx, "iop")
        ctx.sb.vertex(op, "assignment_op")
        ctx.sb.constraint(op, "literal-value", "=")
        ctx.sb.edge(asn, op, "child_of")
        # RHS: <stan_name>_lpmf(k | latent_args with z->k)
        de = self._fresh(ctx, "ide")
        ctx.sb.vertex(de, "distr_expression")
        fn_id = self._fresh(ctx, "ifid")
        ctx.sb.vertex(fn_id, "identifier")
        ctx.sb.constraint(fn_id, "literal-value", f"{stan_name}_lpmf")
        ctx.sb.edge(de, fn_id, "name")
        dal = self._fresh(ctx, "idal")
        ctx.sb.vertex(dal, "distr_argument_list")
        # First arg: k
        k_ve = self._variable_expression(ctx, "k")
        ctx.sb.edge(dal, k_ve, "child_of")
        # Subsequent: the latent's args, with refs to per-group
        # parameter arrays indexed by the group loop variable.
        rewritten = self._index_groupplated_refs(
            latent_args, latent_meta, group_idx_names
        )
        for arg in rewritten:
            arg_vid = self._render_arg(ctx, arg)
            ctx.sb.edge(dal, arg_vid, "child_of")
        ctx.sb.edge(de, dal, "child_of")
        ctx.sb.edge(asn, de, "child_of")

    def _index_groupplated_refs(
        self,
        args: tuple[IRArg, ...],
        meta: FamilyMeta,
        group_idx_names: tuple[str, ...],
    ) -> tuple[IRArg, ...]:
        """For each arg that's an [`IRArgRef`][quivers.transpile.ir.IRArgRef]
        to a previously-declared name whose declaration plate matches
        the surrounding group's batch_dims, prepend the group loop
        variables as indices.

        For LDA's `Categorical(theta)` inside a `marginalize ... [over=Doc]`
        scope where `theta : array[20] simplex[3]`, the ref to `theta`
        becomes `theta[g_Doc]`.
        """
        del meta
        if not group_idx_names:
            return args
        out: list[IRArg] = []
        for arg in args:
            if not isinstance(arg, IRArgRef) or arg.indices:
                out.append(arg)
                continue
            declared = self._declared_shapes.get(arg.name)
            if declared is None:
                out.append(arg)
                continue
            _, declared_plate = declared
            if len(declared_plate.batch_dims) < len(group_idx_names):
                out.append(arg)
                continue
            # Prepend the group loop variables as IRArgRef indices.
            new_indices = tuple(
                IRArgRef(name=lv_name)
                for lv_name in group_idx_names
            )
            out.append(IRArgRef(name=arg.name, indices=new_indices))
        return tuple(out)

    def _emit_lps_accumulate(
        self,
        ctx: _RenderCtx,
        scope_block: str,
        lps_name: str,
        batch_dims: tuple[Dim, ...],
        group_idx_names: tuple[str, ...],
    ) -> None:
        """Emit `for g in batch_dims, target += log_sum_exp(lps[g]);`."""
        current = self._wrap_in_for_loops(
            ctx, scope_block, batch_dims, group_idx_names
        )
        ts = self._fresh(ctx, "ats")
        ctx.sb.vertex(ts, "target_statement")
        ctx.sb.edge(current, ts, "child_of")
        # log_sum_exp(lps[group_idx])
        fn = self._fresh(ctx, "lsefn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "lsefnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", "log_sum_exp")
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "lseal")
        ctx.sb.vertex(al, "argument_list")
        lps_idx = self._build_indexed_arg_expression(
            ctx, lps_name, group_idx_names
        )
        ctx.sb.edge(al, lps_idx, "child_of")
        ctx.sb.edge(fn, al, "child_of")
        ctx.sb.edge(ts, fn, "child_of")

    def _build_indexed_arg_expression(
        self,
        ctx: _RenderCtx,
        base_name: str,
        index_names: tuple[str, ...],
    ) -> str:
        """Build `<base>[<idx0>, <idx1>, ...]` as an expression node."""
        if not index_names:
            return self._variable_expression(ctx, base_name)
        ie = self._fresh(ctx, "ie")
        ctx.sb.vertex(ie, "indexed_expression")
        ctx.sb.edge(ie, self._variable_expression(ctx, base_name), "child_of")
        for idx_name in index_names:
            idx_vid = self._build_index_node(ctx, idx_name)
            ctx.sb.edge(ie, idx_vid, "child_of")
        return ie

    # ----- broadcast dispatch -----

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit a Stan broadcast call:
        `rep_vector(<value>, K)` for 1D, `rep_matrix(<value>, R, C)`
        for 2D."""
        if len(target_shape) == 0:
            return self._render_arg(ctx, value)
        if len(target_shape) == 1:
            fn_name = "rep_vector"
        elif len(target_shape) == 2:
            fn_name = "rep_matrix"
        else:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"broadcast:rank:{len(target_shape)}: Stan "
                    f"broadcast supports up to rank 2"
                ],
            )
        fn = self._fresh(ctx, "bfn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "bfnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", fn_name)
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "bal")
        ctx.sb.vertex(al, "argument_list")
        val_vid = self._render_arg(ctx, value)
        ctx.sb.edge(al, val_vid, "child_of")
        for size in target_shape:
            ctx.sb.edge(al, self._int_literal(ctx, size), "child_of")
        ctx.sb.edge(fn, al, "child_of")
        return fn

    # ----- arg rendering helpers -----

    def render_list(
        self,
        ctx: _RenderCtx,
        arg: IRArgList,
    ) -> SchemaFragment:
        """Per §10.9 of the spec, Stan list args render as a vector
        literal `[<e0>, <e1>, ...]'`.

        The Stan grammar's `vector_expression` accepts the bracketed
        list; the transpose `'` ensures the literal is a column vector
        when used as an arg to a distribution expecting `vector[N]`.
        """
        ve = self._fresh(ctx, "vex")
        ctx.sb.vertex(ve, "vector_expression")
        for element in arg.elements:
            child_vid = self._render_arg(ctx, element)
            ctx.sb.edge(ve, child_vid, "child_of")
        # Wrap in a postfix_op_expression to render the transpose.
        post = self._fresh(ctx, "poe")
        ctx.sb.vertex(post, "postfix_op_expression")
        ctx.sb.constraint(post, "chose-alt-fingerprint", "'")
        ctx.sb.edge(post, ve, "child_of")
        return post

    def render_matrix(
        self,
        ctx: _RenderCtx,
        arg: IRArgMatrix,
    ) -> SchemaFragment:
        """Per §10.9 of the spec, Stan matrix args render as
        `to_matrix({{<row0>}, {<row1>}, ...})`."""
        fn = self._fresh(ctx, "tmfn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "tmfnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", "to_matrix")
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "tmal")
        ctx.sb.vertex(al, "argument_list")
        # The outer `{ ... }` is a vector_expression of vector_expressions.
        outer_ve = self._fresh(ctx, "tmve")
        ctx.sb.vertex(outer_ve, "vector_expression")
        for row in arg.rows:
            row_ve = self._fresh(ctx, "tmrow")
            ctx.sb.vertex(row_ve, "vector_expression")
            for element in row.elements:
                el_vid = self._render_arg(ctx, element)
                ctx.sb.edge(row_ve, el_vid, "child_of")
            ctx.sb.edge(outer_ve, row_ve, "child_of")
        ctx.sb.edge(al, outer_ve, "child_of")
        ctx.sb.edge(fn, al, "child_of")
        return fn

    def _render_arg(
        self,
        ctx: _RenderCtx,
        arg: IRArg,
    ) -> SchemaFragment:
        """Render any [`IRArg`][quivers.transpile.ir.IRArg] to a Stan
        expression vertex."""
        if isinstance(arg, IRArgNumber):
            return self._render_number(ctx, arg.value)
        if isinstance(arg, IRArgRef):
            return self._render_ref(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self.broadcast(ctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self.render_list(ctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self.render_matrix(ctx, arg)
        if isinstance(arg, IRArgFamilyRef):
            return self._render_family_ref(ctx, arg)
        raise UnsupportedConstruct(
            "qvr-stan",
            [f"arg:unknown:{type(arg).__name__}"],
        )

    def _render_number(self, ctx: _RenderCtx, value: float) -> str:
        if float(value).is_integer():
            return self._int_literal(ctx, int(value))
        v = self._fresh(ctx, "rl")
        ctx.sb.vertex(v, "real_literal")
        ctx.sb.constraint(v, "literal-value", repr(float(value)))
        return v

    def _render_ref(
        self, ctx: _RenderCtx, arg: IRArgRef
    ) -> SchemaFragment:
        """Render an IRArgRef. Bare-name refs become a
        `variable_expression`; indexed refs nest `indexed_expression`
        nodes. When the ref name resolves to a scope-local
        marginalize let-binding, inline the let-RHS expression so the
        per-k observe consumes the substituted form (the let target
        is never declared as a Stan parameter)."""
        if not arg.indices and arg.name in self._marginalize_let_subs:
            expr = self._marginalize_let_subs[arg.name]
            latent = self._current_latent_name()
            if latent is not None:
                k_ref = LetExprVar(name="k")
                expr = _substitute_let_expr(
                    expr,
                    latent,
                    index_value=k_ref,
                    scalar_value=k_ref,
                )
            return render_let_expr_stan(
                _StanLetCtx(
                    ctx.sb, lambda p: self._fresh(ctx, p), self._cards
                ),
                expr,
            )
        base = self._variable_expression(ctx, arg.name)
        if not arg.indices:
            return base
        ie = self._fresh(ctx, "ie")
        ctx.sb.vertex(ie, "indexed_expression")
        ctx.sb.edge(ie, base, "child_of")
        for idx in arg.indices:
            idx_vid = self._fresh(ctx, "idx")
            ctx.sb.vertex(idx_vid, "index")
            inner_vid = self._render_arg(ctx, idx)
            ctx.sb.edge(idx_vid, inner_vid, "child_of")
            ctx.sb.edge(ie, idx_vid, "child_of")
        return ie

    def _render_family_ref(
        self,
        ctx: _RenderCtx,
        arg: IRArgFamilyRef,
    ) -> SchemaFragment:
        """Render an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]:
        look up the morphism's `init_family` clause and emit the
        Stan inline call form."""
        decl = ctx.morphisms.get(arg.name)
        if decl is None or decl.init_family is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [
                    f"arg:family-ref:{arg.name}: no morphism with "
                    f"`~ Family(...)` init clause in scope"
                ],
            )
        init = decl.init_family
        inner_meta = FAMILY_META.get(init.family)
        if inner_meta is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:unknown:{init.family}"],
            )
        inner_stan_name = inner_meta.target_names.get("stan")
        if inner_stan_name is None:
            raise UnsupportedConstruct(
                "qvr-stan",
                [f"family:no-stan-target:{init.family}"],
            )
        fn = self._fresh(ctx, "frfn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "frfnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", inner_stan_name)
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "fral")
        ctx.sb.vertex(al, "argument_list")
        for raw in init.args or ():
            arg_vid = self._render_init_family_arg(ctx, raw)
            ctx.sb.edge(al, arg_vid, "child_of")
        ctx.sb.edge(fn, al, "child_of")
        return fn

    def _render_init_family_arg(
        self,
        ctx: _RenderCtx,
        raw: DrawArg,
    ) -> SchemaFragment:
        """Render an `init_family` raw arg.

        Wrapper morphisms (`Truncated(base, -2, 2)`) carry their
        constants here as atomic `DrawArgScalar` / `DrawArgName`
        variants. Structured list / matrix variants are not used in
        `init_family` clauses on any current example; an
        `UnsupportedConstruct` here flags the gap rather than guess.
        """
        if isinstance(raw, DrawArgScalar):
            return self._render_number(ctx, raw.value)
        if isinstance(raw, DrawArgName):
            stripped = raw.text.strip()
            try:
                value = float(stripped)
            except ValueError:
                return self._variable_expression(ctx, stripped)
            return self._render_number(ctx, value)
        raise UnsupportedConstruct(
            "qvr-stan",
            [f"init_family-arg:unsupported-variant:{type(raw).__name__}"],
        )

    # ----- shared utilities -----

    def _variable_expression(
        self, ctx: _RenderCtx, name: str
    ) -> str:
        ve = self._fresh(ctx, "ve")
        ctx.sb.vertex(ve, "variable_expression")
        ident = self._fresh(ctx, "vid")
        ctx.sb.vertex(ident, "identifier")
        ctx.sb.constraint(ident, "literal-value", name)
        ctx.sb.edge(ve, ident, "child_of")
        return ve

    def _int_literal(self, ctx: _RenderCtx, value: int | float) -> str:
        v = self._fresh(ctx, "il")
        ctx.sb.vertex(v, "integer_literal")
        ctx.sb.constraint(v, "literal-value", str(int(value)))
        return v

    def _ensure_block(
        self, ctx: _RenderCtx, kind: BlockKind
    ) -> str:
        """Lazily emit a top-level Stan block of the given kind.

        Blocks are children of the top `program` vertex. Each emit
        happens at most once per render call.
        """
        if kind in self._blocks:
            return self._blocks[kind]
        vid = self._fresh(ctx, kind)
        ctx.sb.vertex(vid, _BLOCK_KIND_MAP[kind])
        ctx.sb.edge("prog", vid, "child_of")
        self._blocks[kind] = vid
        return vid

    def _dim_size_vertex(self, ctx: _RenderCtx, dim: Dim) -> str:
        """Materialise an integer literal (for `DimStatic`) or a
        variable reference (for `DimDynamic`) representing the dim's
        size."""
        if isinstance(dim, DimStatic):
            return self._int_literal(ctx, dim.size)
        if isinstance(dim, DimDynamic):
            return self._variable_expression(ctx, dim.size_name)
        raise UnsupportedConstruct(
            "qvr-stan",
            [f"dim:unknown:{type(dim).__name__}"],
        )

    def _fresh(self, ctx: _RenderCtx, prefix: str) -> str:
        """Return a fresh vertex id with `prefix`. Renderer-internal
        counter; doesn't disturb the base's `_RenderCtx.fresh_counter`."""
        del ctx
        self._fresh_n += 1
        return f"{prefix}_{self._fresh_n}"

    # ----- IR-walk overrides -----

    def _dispatch_node(self, ctx: _RenderCtx, node: IRNode) -> None:
        """Stan-specific dispatch. Records simplex cardinalities along
        the walk for marginalize's K-inference; otherwise defers to
        the base behaviour for layout."""
        if isinstance(node, IRSample):
            # Track simplex declarations for later marginalize K-lookup.
            sup = node.constraint.to_constraint()
            if is_real_simplex(sup) and node.plate.event_dims:
                ed = node.plate.event_dims[0]
                if isinstance(ed, DimStatic):
                    self._simplex_cards[node.name] = ed.size
            self.declare(
                ctx,
                node.name,
                node.constraint,
                node.plate,
                block="parameters",
            )
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
            self.declare(
                ctx,
                node.name,
                node.constraint,
                node.plate,
                block="data",
            )
            self.sample(
                ctx,
                node.name,
                node.family,
                node.args,
                node.arg_names,
                node.constraint,
                node.plate,
                observed=True,
            )
            return
        if isinstance(node, IRDataInput):
            self.declare(
                ctx,
                node.name,
                node.constraint,
                node.plate,
                block="data",
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
        raise UnsupportedConstruct(
            "qvr-stan",
            [f"node:{type(node).__name__}"],
        )

    def _emit_deterministic(
        self,
        ctx: _RenderCtx,
        node: IRDeterministic,
    ) -> None:
        """Emit a deterministic let-binding into the
        ``transformed_parameters`` block.

        Scalar plate (``batch_dims=()``): the binding is a single
        ``<type> <name> = <expr>;`` declaration whose RHS is
        [`node.expr`][quivers.transpile.ir.IRDeterministic.expr]
        rendered through
        [`render_let_expr_stan`][quivers.transpile.renderers._stan_helpers.render_let_expr_stan].

        Non-scalar plate (``batch_dims != ()``): the binding splits
        into a declaration of the array type and a ``for`` loop that
        assigns elementwise. Stan's elementwise arithmetic is defined
        on ``vector``/``matrix`` types but not on
        ``array[N] real``; using a loop sidesteps the type mismatch
        and works for arbitrary expression shapes, including those
        mixing scalar parameters with plated data inputs (the canonical
        ``a + b * x_design`` regression form).

        Plated references inside the let-expression are auto-indexed
        by the loop var. Any free
        [`LetExprVar.name`][quivers.dsl.ast_nodes.LetExprVar.name]
        whose declared plate has the same ``batch_dims`` as the
        surrounding let gets substituted with
        ``<name>[<loop_var>]`` per
        [`_substitute_let_expr`][quivers.transpile.renderers._stan_helpers._substitute_let_expr];
        scalar refs (``a``, ``b``) pass through unchanged.
        """
        parent = self._ensure_block(ctx, "transformed_parameters")
        if node.name in self._declared["transformed_parameters"]:
            return
        self._declared["transformed_parameters"].add(node.name)
        self._declared_shapes[node.name] = (
            node.constraint.to_constraint(), node.plate,
        )
        # Normalise the RHS shape for fan / parallel-branch literals.
        # The lower phase aggregates parallel-branch outputs as
        # `LetExprList` regardless of length; for the Stan declaration
        # the literal's length dictates the declared type:
        #   * length 1 -> unwrap to the inner scalar (the wrapping
        #     list serves no Stan purpose and breaks `real X = {a};`).
        #   * length >1 -> declare `array[len] real` so the
        #     `{a, b, ...}` initialiser type-checks; downstream
        #     consumers vectorise across the array per Stan's
        #     broadcasting rules.
        body_expr = node.expr
        list_array_size: int | None = None
        if (
            isinstance(body_expr, LetExprList)
            and not node.plate.batch_dims
        ):
            if len(body_expr.items) == 1:
                body_expr = body_expr.items[0]
            elif len(body_expr.items) > 1:
                list_array_size = len(body_expr.items)
        decl = self._fresh(ctx, "tpvd")
        ctx.sb.vertex(decl, "top_var_decl")
        ctx.sb.edge(parent, decl, "child_of")
        promoted_k = self._vector_promotions.get(node.name)
        if list_array_size is not None and promoted_k is None:
            # `array[N] real` declaration, no batch dims.
            arr = self._fresh(ctx, "tparr")
            ctx.sb.vertex(arr, "arr_dims")
            ctx.sb.edge(
                arr,
                self._int_literal(ctx, list_array_size),
                "child_of",
            )
            ctx.sb.edge(decl, arr, "child_of")
        elif node.plate.batch_dims:
            arr = self._fresh(ctx, "tparr")
            ctx.sb.vertex(arr, "arr_dims")
            for dim in node.plate.batch_dims:
                size_vid = self._dim_size_vertex(ctx, dim)
                ctx.sb.edge(arr, size_vid, "child_of")
            ctx.sb.edge(decl, arr, "child_of")
        tvt = self._fresh(ctx, "tpvt")
        ctx.sb.vertex(tvt, "top_var_type")
        if promoted_k is not None:
            self._emit_vector_type_of_size(ctx, tvt, promoted_k)
        else:
            self._emit_type(
                ctx, tvt, node.constraint.to_constraint(),
                node.plate.event_dims,
            )
        ctx.sb.edge(decl, tvt, "child_of")
        nm = self._fresh(ctx, "tpnm")
        ctx.sb.vertex(nm, "identifier")
        ctx.sb.constraint(nm, "literal-value", node.name)
        ctx.sb.edge(decl, nm, "name")
        # Inline-init path: emit `<type> <name> = <expr>;` as a single
        # `top_var_decl` with an init child. Correct when either:
        #   - the let is scalar (no batch dims); OR
        #   - the let's expression is already shape-aware (a
        #     [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor]
        #     or [`LetExprList`][quivers.dsl.ast_nodes.LetExprList]
        #     literal produces the full array on the RHS, so no
        #     broadcasting / per-element substitution is needed and
        #     Stan accepts the array-to-array assignment directly).
        if not node.plate.batch_dims or isinstance(
            body_expr, (LetExprFactor, LetExprList)
        ):
            rhs = render_let_expr_stan(
                _StanLetCtx(
                    ctx.sb, lambda p: self._fresh(ctx, p), self._cards
                ),
                body_expr,
            )
            if promoted_k is not None:
                rhs = self._wrap_rep_vector(ctx, rhs, promoted_k)
            ctx.sb.edge(decl, rhs, "child_of")
            return

        # Plated case (non-literal RHS): emit
        # `for (lv0 in 1:N0) ... { name[lv0]...[lvN-1] = expr; }`
        # with plated refs in expr substituted to <ref>[lv]. The
        # for-loop is required because Stan's elementwise arithmetic
        # is undefined on `array[N] real`; the loop unrolls to scalar
        # assignments that Stan does accept.
        loop_names = self.index_for(ctx, node.plate)
        inner_parent = self._wrap_in_for_loops(
            ctx, parent, node.plate.batch_dims, loop_names
        )
        subbed = self._index_plated_let_refs(
            node.expr, node.plate.batch_dims, loop_names
        )
        # LHS: indexed_lhs <name>[lv0, lv1, ...]
        asn = self._fresh(ctx, "asn")
        ctx.sb.vertex(asn, "assignment_statement")
        ctx.sb.edge(inner_parent, asn, "child_of")
        lhs_vid = self._build_indexed_lhs(ctx, node.name, loop_names)
        ctx.sb.edge(asn, lhs_vid, "child_of")
        op = self._fresh(ctx, "aop")
        ctx.sb.vertex(op, "assignment_op")
        ctx.sb.constraint(op, "literal-value", "=")
        ctx.sb.edge(asn, op, "child_of")
        rhs = render_let_expr_stan(
            _StanLetCtx(
                ctx.sb, lambda p: self._fresh(ctx, p), self._cards
            ),
            subbed,
        )
        if promoted_k is not None:
            rhs = self._wrap_rep_vector(ctx, rhs, promoted_k)
        ctx.sb.edge(asn, rhs, "child_of")

    def _wrap_rep_vector(
        self,
        ctx: _RenderCtx,
        inner_vid: str,
        size: int,
    ) -> str:
        """Wrap an existing expression vertex in a
        `rep_vector(<inner>, K)` call.

        Used by the vector-promotion path: when a scalar deterministic
        binding produces a name that is consumed by an event_dim>=1
        slot, the per-element RHS is broadcast to a length-K vector
        so the downstream consumer's shape contract is satisfied.
        """
        fn = self._fresh(ctx, "rvfn")
        ctx.sb.vertex(fn, "function_expression")
        fnid = self._fresh(ctx, "rvfnid")
        ctx.sb.vertex(fnid, "identifier")
        ctx.sb.constraint(fnid, "literal-value", "rep_vector")
        ctx.sb.edge(fn, fnid, "name")
        al = self._fresh(ctx, "rval")
        ctx.sb.vertex(al, "argument_list")
        ctx.sb.edge(al, inner_vid, "child_of")
        ctx.sb.edge(al, self._int_literal(ctx, size), "child_of")
        ctx.sb.edge(fn, al, "child_of")
        return fn

    def _index_plated_let_refs(
        self,
        expr: LetExprNode,
        batch_dims: tuple[Dim, ...],
        loop_names: tuple[str, ...],
    ) -> LetExprNode:
        """Substitute every free
        [`LetExprVar`][quivers.dsl.ast_nodes.LetExprVar] whose
        declared plate's ``batch_dims`` matches ``batch_dims`` with
        ``<name>[<loop_name>]`` (or nested for multi-batch).

        Walks the expression collecting candidate names, then calls
        [`_substitute_let_expr`][quivers.transpile.renderers._stan_helpers._substitute_let_expr]
        once per name so the substitution preserves the
        single-pass semantics the helper expects.
        """
        seen: set[str] = set()
        _collect_let_expr_var_names(expr, seen)
        out = expr
        for name in seen:
            declared = self._declared_shapes.get(name)
            if declared is None:
                continue
            plate = declared[1]
            if plate.batch_dims != batch_dims:
                continue
            indexed = LetExprIndex(
                array=LetExprVar(name=name),
                indices=tuple(
                    LetExprVar(name=lv) for lv in loop_names
                ),
            )
            out = _substitute_let_expr(
                out,
                name,
                index_value=indexed,
                scalar_value=indexed,
            )
        return out

    def _emit_score(self, ctx: _RenderCtx, node: IRScore) -> None:
        """Emit ``target += <expr>;`` to the ``model`` block.

        Renders
        [`node.expr`][quivers.transpile.ir.IRScore.expr] through
        [`render_let_expr_stan`][quivers.transpile.renderers._stan_helpers.render_let_expr_stan]
        so the increment is a real Stan expression rather than a
        bare-name reference to an undeclared variable.
        """
        parent = self._ensure_block(ctx, "model")
        ts = self._fresh(ctx, "scts")
        ctx.sb.vertex(ts, "target_statement")
        ctx.sb.edge(parent, ts, "child_of")
        rhs = render_let_expr_stan(
            _StanLetCtx(ctx.sb, lambda p: self._fresh(ctx, p), self._cards),
            node.expr,
        )
        ctx.sb.edge(ts, rhs, "child_of")

    def _emit_return(
        self,
        ctx: _RenderCtx,
        names: tuple[str, ...],
    ) -> None:
        """Emit `generated quantities { <type> <name>_value = <name>;
        ... }` for each return variable.

        Stan has no program-level return; the idiom is to expose
        sampled / let-bound variables in `generated quantities`.
        Re-declaring an existing name is illegal in Stan, so each
        return-var emerges as `<var>_value`.
        """
        for var in names:
            self._emit_gq_alias(ctx, var)

    def _emit_gq_alias(self, ctx: _RenderCtx, var: str) -> None:
        decl_name = f"{var}_value"
        if decl_name in self._declared["generated_quantities"]:
            return
        self._declared["generated_quantities"].add(decl_name)
        parent = self._ensure_block(ctx, "generated_quantities")
        decl = self._fresh(ctx, "gqvd")
        ctx.sb.vertex(decl, "top_var_decl")
        ctx.sb.edge(parent, decl, "child_of")
        # Reuse the declared support / plate by looking up the
        # parameter / data block entry. For LDA the `theta` is an
        # `array[20] simplex[3]`; emit the same type here.
        sup, plate = self._declared_shape(var)
        if plate.batch_dims:
            arr = self._fresh(ctx, "gqarr")
            ctx.sb.vertex(arr, "arr_dims")
            for dim in plate.batch_dims:
                size_vid = self._dim_size_vertex(ctx, dim)
                ctx.sb.edge(arr, size_vid, "child_of")
            ctx.sb.edge(decl, arr, "child_of")
        tvt = self._fresh(ctx, "gqvt")
        ctx.sb.vertex(tvt, "top_var_type")
        self._emit_type(ctx, tvt, sup, plate.event_dims)
        ctx.sb.edge(decl, tvt, "child_of")
        nm = self._fresh(ctx, "gqnm")
        ctx.sb.vertex(nm, "identifier")
        ctx.sb.constraint(nm, "literal-value", decl_name)
        ctx.sb.edge(decl, nm, "name")
        rhs = self._variable_expression(ctx, var)
        ctx.sb.edge(decl, rhs, "child_of")

    def _declared_shape(
        self, var: str
    ) -> tuple[Constraint, Plate]:
        """Look up a previously-declared name's (support, plate) for
        the generated-quantities aliasing emission.

        Raises [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
        when the return-var was never declared.
        """
        info = self._declared_shapes.get(var)
        if info is not None:
            return info
        raise UnsupportedConstruct(
            "qvr-stan",
            [
                f"return:undeclared:{var}: return variable was not "
                f"declared in any earlier IR node; cannot determine "
                f"shape for generated-quantities aliasing"
            ],
        )

    @property
    def _declared_shapes(self) -> dict[str, tuple[Constraint, Plate]]:
        """Per-render map from declared name to (support, plate).

        Reset at the top of every `render()` call so repeated render
        invocations on the same renderer produce identical schemas.
        """
        return self._declared_shapes_state

    # ----- morphism / let resolution -----

    def _resolve_morphisms_and_lets(
        self,
    ) -> tuple[dict[str, MorphismDecl], dict[str, Expr]]:
        """Build the morphism / let tables from the source module.

        Used by [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        resolution. Empty when the renderer was constructed without a
        source module; in that case `IRArgFamilyRef` rendering raises.
        """
        if self._source_module is None:
            return {}, {}
        morphisms: dict[str, MorphismDecl] = {}
        lets: dict[str, Expr] = {}
        for stmt in self._source_module.statements:
            if isinstance(stmt, MorphismDecl):
                morphisms[stmt.name] = stmt
            elif isinstance(stmt, LetDecl):
                lets[stmt.name] = stmt.expr
        return morphisms, lets

# Stan-side argument injection for QVR families whose underlying
# torch distribution carries fewer parameters than Stan's same-named
# distribution. `HalfNormal(scale)` maps to Stan's `normal(0, scale)`;
# the renderer prepends `IRArgNumber(0)` before emission.
_PREPEND_ZERO: frozenset[str] = frozenset({"HalfNormal", "HalfCauchy"})

# Per-family insertion of `IRArgNumber(0)` at an intermediate
# position. `HalfStudentT(df, scale)` maps to Stan's
# `student_t(df, 0, scale)`; the renderer injects `mu = 0` between
# the first and second user-supplied args.
_INSERT_ZERO_AT: dict[str, int] = {
    "HalfStudentT": 1,
}


_BLOCK_KIND_MAP: dict[BlockKind, str] = {
    "data": "data",
    "parameters": "parameters",
    "transformed_parameters": "transformed_parameters",
    "model": "model",
    "generated_quantities": "generated_quantities",
    "function_body": "functions",
}


def format_stan(source: bytes) -> bytes:
    """Whitespace-only formatter for Stan bytes.

    Invariants enforced:

    * One statement per line.
    * Two-space indentation.
    * Block headers at column 0.
    * No trailing whitespace.
    * No space-before-comma.
    * Space-after-comma.

    The formatter only touches whitespace and idiomatic spacing
    around `[`, `<`, and `=` inside type constraints; every
    non-whitespace token from the input survives.
    """
    text = source.decode("utf-8")
    # Stage 1: tighten Stan-specific spacing on the raw pretty-print
    # output so subsequent line-splitting matches Stan idioms.
    text = _tighten_stan_spacing(text)
    # Stage 2: split on statement boundaries and brace pairs and
    # re-indent.
    lines = _split_and_indent(text)
    # Stage 3: final whitespace pass.
    out: list[str] = []
    prev_blank = False
    for line in lines:
        s = line.rstrip()
        if not s:
            if not prev_blank and out:
                out.append("")
            prev_blank = True
            continue
        prev_blank = False
        out.append(s)
    return ("\n".join(out) + "\n").encode("utf-8")


def _tighten_stan_spacing(text: str) -> str:
    """Apply Stan idiomatic spacing fixes: tight brackets / angle
    constraints / commas. Whitespace-only inside identifiers and
    literals."""
    out = text
    # Remove space before comma.
    while " ," in out:
        out = out.replace(" ,", ",")
    # No space inside `<...>` type constraints.
    out = (
        out.replace("<lower = ", "<lower=")
        .replace("<upper = ", "<upper=")
        .replace(", upper = ", ", upper=")
        .replace(", lower = ", ", lower=")
        .replace("< lower", "<lower")
        .replace("< upper", "<upper")
    )
    # `<keyword>` + space + `[` -> remove the space (Stan idiom:
    # `vector[N]`, not `vector [N]`).
    for kw in (
        "array",
        "vector",
        "simplex",
        "cov_matrix",
        "cholesky_factor_corr",
        "matrix",
        "row_vector",
    ):
        out = out.replace(f"{kw} [", f"{kw}[")
    # `int <` -> `int<`; `real <` -> `real<`.
    out = out.replace("int <", "int<").replace("real <", "real<")
    # Spacing around `:` in `1 : 20` -> `1:20` (Stan idiom for
    # iteration ranges).
    out = _tighten_colon_ranges(out)
    # Spacing in `target +=` (the pretty printer omits the space
    # after `+=`).
    out = out.replace("target +=", "target += ")
    # Ensure space after comma.
    out = _ensure_space_after_comma(out)
    return out


def _tighten_colon_ranges(text: str) -> str:
    """Replace ` : ` with `:` only when both sides are bare numbers /
    identifiers (the Stan iteration range form). Leaves declarations
    like `complex : x` untouched if any ever arise (Stan has no such
    form today, but the guard keeps the formatter safe)."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if (
            i + 2 < n
            and text[i] == " "
            and text[i + 1] == ":"
            and text[i + 2] == " "
        ):
            # Look at neighbouring tokens: previous non-space char and
            # next non-space char both alphanumeric / underscore.
            left = out[-1] if out else ""
            right = text[i + 3] if i + 3 < n else ""
            if (
                (left.isalnum() or left == "_")
                and (right.isalnum() or right == "_")
            ):
                out.append(":")
                i += 3
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _ensure_space_after_comma(text: str) -> str:
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        out.append(c)
        if (
            c == ","
            and i + 1 < n
            and text[i + 1] not in {" ", "\n", "\t", "}", ")", "]"}
        ):
            out.append(" ")
        i += 1
    return "".join(out)


def _split_and_indent(text: str) -> list[str]:
    """Split the tightened source on `{`, `}`, and `;` boundaries and
    re-indent with two-space steps.

    Strings / comments are not present in the renderer's output, so
    naive tokenisation suffices.
    """
    # Insert newlines around block boundaries.
    spaced: list[str] = []
    for ch in text:
        if ch == "{":
            spaced.append(" {\n")
        elif ch == "}":
            spaced.append("\n}\n")
        elif ch == ";":
            spaced.append(";\n")
        elif ch == "\n":
            spaced.append(" ")
        else:
            spaced.append(ch)
    joined = "".join(spaced)
    # Normalise multiple spaces inside lines.
    raw_lines = joined.split("\n")
    # Re-indent.
    out: list[str] = []
    depth = 0
    for raw in raw_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        # Collapse internal runs of spaces.
        compact = _collapse_spaces(stripped)
        # Closing brace decrements depth before printing.
        if compact.startswith("}"):
            depth = max(0, depth - 1)
            out.append("  " * depth + compact)
            # Keep the leading `}` then continue with whatever
            # follows on its line.
        else:
            out.append("  " * depth + compact)
        # Increment depth after an opening brace.
        if compact.endswith("{"):
            depth += 1
    return out


def _collapse_spaces(text: str) -> str:
    """Collapse runs of spaces into single spaces, preserving content."""
    out: list[str] = []
    prev_space = False
    for ch in text:
        if ch == " ":
            if not prev_space:
                out.append(ch)
            prev_space = True
        else:
            out.append(ch)
            prev_space = False
    return "".join(out)


# ---------------------------------------------------------------------------
# Runtime-helper graft: Stan user-defined `<family>_lpdf` / `_rng`.
#
# Stan ships `normal`, `beta`, `gamma`, ... as built-in densities but
# lacks `kumaraswamy`. The transpile-time graft parses the hand-written
# helper at
# [`runtime_stan_functions.stan`][quivers.transpile.runtime_stan_functions]
# once at module-load through panproto's Stan tree-sitter grammar;
# per-render, it copies the parsed `functions { ... }` block into the
# per-render schema (with fresh vertex ids) and attaches it as a
# `child_of` of the program above the data block. Subsequent
# `~ kumaraswamy(a, b)` sampling statements then resolve through Stan's
# `<family>_lpdf` lookup convention.
# ---------------------------------------------------------------------------


_RUNTIME_STAN_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "runtime_stan_functions.stan"
)


#: Families whose Stan emit relies on the
#: [`runtime_stan_functions.stan`][quivers.transpile.runtime_stan_functions]
#: helper subtree. Stan ships `normal`, `beta`, `gamma`, etc. as
#: built-in densities but lacks `kumaraswamy`; the renderer grafts the
#: helper when the IR samples or observes from any of them.
_STAN_RUNTIME_HELPER_FAMILIES: frozenset[str] = frozenset({
    "Kumaraswamy",
})


def _load_runtime_stan_schema() -> tuple[
    panproto.Schema, str, tuple[str, ...]
]:
    """Parse
    [`runtime_stan_functions.stan`][quivers.transpile.runtime_stan_functions]
    through panproto's Stan tree-sitter grammar at module-load time.

    Returns the parsed schema, the parsed `program` vertex id, and
    the tuple of top-level child ids in source order (sorted by
    `start-byte`). The graft replays these children in order beneath
    the per-render `program` so the emit's top-level statements
    appear in the original file's layout.
    """
    schema = parser_registry().parse_with_protocol(
        "stan",
        _RUNTIME_STAN_PATH.read_bytes(),
        str(_RUNTIME_STAN_PATH),
    )
    src_id = next(
        (v.id for v in schema.vertices if v.kind == "program"),
        None,
    )
    if src_id is None:
        raise RuntimeError(
            f"`program` not found in parse of {_RUNTIME_STAN_PATH}"
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


_RUNTIME_STAN_SCHEMA, _RUNTIME_STAN_PROGRAM_ID, _RUNTIME_STAN_TOP_LEVEL = (
    _load_runtime_stan_schema()
)


def _stan_subtree_vertex_ids(
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


_RUNTIME_STAN_SUBTREE = _stan_subtree_vertex_ids(
    _RUNTIME_STAN_SCHEMA, _RUNTIME_STAN_TOP_LEVEL
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


def _graft_runtime_stan_helper(
    sb: panproto.SchemaBuilder,
    renderer: StanRenderer,
    program_vid: str,
) -> None:
    """Graft the runtime-helper subtree onto the per-render schema.

    Copies every vertex, every constraint, and every internal edge of
    the parsed `runtime_stan_functions.stan` subtree into the per-render
    `SchemaBuilder` with fresh vertex ids, then attaches each top-level
    child as a `child_of` of `program_vid` in source order. The grafted
    `functions { ... }` block appears above the data block in the
    emit, satisfying Stan's `functions? data? ...` grammar production.
    """
    src_schema = _RUNTIME_STAN_SCHEMA
    subtree = _RUNTIME_STAN_SUBTREE
    id_map: dict[str, str] = {}

    for old in subtree:
        renderer._fresh_n += 1
        new = f"rs_{renderer._fresh_n}"
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
    for child_old in _RUNTIME_STAN_TOP_LEVEL:
        sb.edge(program_vid, id_map[child_old], "child_of")


__all__ = ["StanRenderer", "format_stan"]
