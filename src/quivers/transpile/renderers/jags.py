"""JAGS renderer: [`IRProgram`][quivers.transpile.ir.IRProgram] to JAGS
source under the ``jags`` tree-sitter grammar.

The JAGS surface mirrors BUGS for the probabilistic-core subset QVR
targets: a single top-level ``model { ... }`` block whose children are
``~`` stochastic relations and ``<-`` deterministic relations, nested
under ``for (m_<axis> in 1:N_<axis>) { ... }`` loops to express plate
structure. JAGS-specific family names (``ddirich`` for Dirichlet,
``dgen.gamma`` for generalised Gamma, etc.) and arithmetic-converting
parameterisation renames (Normal ``scale`` -> ``tau = 1/(scale*scale)``
via the [`IRArgTransform`][quivers.transpile.renderers._base.IRArgTransform]
mechanism) come from
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META].

The renderer:

* Treats every IR declaration as a no-op (JAGS variables are declared
  implicitly by their first ``~`` / ``<-`` binding; data inputs ride
  on an external ``.data`` file the host supplies).
* Lowers [`IRSample`][quivers.transpile.ir.IRSample] and
  [`IRObserve`][quivers.transpile.ir.IRObserve] to per-batch-axis
  ``for (m_<axis> in 1:N_<axis>) { <name>[m_<axis>] ~ d<family>(args) }``
  nests.
* Lowers [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] inline
  via [`RendererBase.explicit_latent_scope`][quivers.transpile.renderers._base.RendererBase.explicit_latent_scope]
  (JAGS samples discrete latents natively).
* Refuses to emit a scalar broadcast to a vector / matrix arg or a
  bare list / matrix literal, raising
  [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
  with a typed kind tag (callers pre-bind vector data via a let-decl
  before transpiling).
* Appends ``T(L, U)`` truncation idiom when an
  [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] resolves to
  a ``Truncated(...)`` wrapper.
"""

from __future__ import annotations

from typing import Literal

import panproto

from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import target_protocol
from quivers.transpile.family_meta import FAMILY_META
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
from quivers.transpile.renderers._bugs_helpers import (
    build_decl_plates,
    index_letexpr_refs,
    push_scalar_dets_into_loops,
    render_let_expr_bugs,
)
from quivers.transpile.renderers._base import (
    BlockKind,
    IRArgTransform,
    RendererBase,
    SchemaFragment,
    _RenderCtx,
)


#: The backend key consulted in
#: [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] for
#: ``target_names`` and ``arg_aliases`` lookups.
_BACKEND: str = "jags"

#: Renderer-internal table: when an
#: [`arg_aliases`][quivers.transpile.family_meta.FamilyMeta.arg_aliases]
#: entry renames an arg to one of these target names, wrap the
#: source value in the matching arithmetic transform before emitting.
#: Normal's ``scale`` -> ``tau`` is the canonical case: JAGS
#: parameterises Normal by precision, so the renderer emits
#: ``1/(scale*scale)``.
_TransformKind = Literal["inv_square", "inv", "neg", "log", "exp"]

_ALIAS_TRANSFORMS: dict[str, _TransformKind] = {
    "tau": "inv_square",
}


#: JAGS-side argument injection for QVR families whose underlying
#: torch distribution carries fewer parameters than the JAGS
#: distribution it maps to. ``HalfNormal(scale)`` maps to JAGS'
#: ``dnorm(0, tau)``; the renderer prepends ``IRArgNumber(0)`` under
#: the loc-position arg name so the alias-transform pipeline still
#: rewrites the scale into ``tau = 1/(scale*scale)``.
_PREPEND_ZERO: frozenset[str] = frozenset({"HalfNormal", "HalfCauchy"})

#: Sentinel name prefix used by the JAGS renderer to encode a `1:N`
#: range as an IRArgRef. The arg-rendering path inspects the name
#: prefix and emits a `range` vertex instead of an
#: `indexed_variable`.
_RANGE_SENTINEL_PREFIX: str = "__jags_range__:"


class JAGSRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] as a
    JAGS model source.

    Subclasses [`RendererBase`][quivers.transpile.renderers._base.RendererBase]:
    overrides `render` to wrap the IR walk in a single top-level
    ``model { ... }`` block. The four dispatch points (`declare`,
    `sample`, `marginalize`, `broadcast`) follow the JAGS surface
    conventions described in section 5.1 of the design spec.
    """

    target: str = _BACKEND

    # ------------------------------------------------------------------
    # `RendererBase` overrides
    # ------------------------------------------------------------------

    def target_protocol(self) -> panproto.Protocol:
        """Use the auto-derived ``jags`` tree-sitter protocol."""
        return target_protocol("jags")

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and emit a complete JAGS module.

        Overrides the base IR walk because JAGS programs need a
        ``model { ... }`` wrapper around every body statement; the
        wrapper is a single ``model_block`` vertex whose children are
        the statements emitted by `_dispatch_jags_node`.
        """
        # JAGS has no scalar-to-vector broadcast; lift empty-plate
        # IRDeterministic nodes whose expressions reference plate-less
        # free data inputs into the plate of the first downstream
        # consumer, then re-index those references at emit time.
        ir = push_scalar_dets_into_loops(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        jctx = _JAGSCtx(sb=sb, morphisms={}, lets={})
        self._cards = dict(ir.cards)
        # Cache decl_plates so the deterministic emitter can re-index
        # let-expression refs by their declared batch_dims.
        jctx.decl_plates = build_decl_plates(ir)

        _vertex(jctx, "src", "source_file")
        jctx.sb.constraint("src", "ptrace-0", "Cmodel_block")
        jctx.sb.constraint("src", "chose-alt-child-kinds", "model_block")

        mb = _fresh(jctx, "mb", "model_block")
        jctx.sb.edge("src", mb, "child_of")
        jctx.current_block = mb
        jctx.model_block = mb

        for node in ir.body:
            self._dispatch_jags_node(jctx, node)

        self._finalise_model_block(jctx)
        return sb.build()

    def declare(
        self,
        ctx: _RenderCtx,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """JAGS variables are declared implicitly by their first ``~``
        or ``<-`` binding; data inputs ride on the host's external
        ``.data`` file. This dispatch is a no-op for every block."""
        del ctx, name, constraint, plate, block
        return ""

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
        """Emit a per-batch-axis ``for (m_<axis> in 1:N_<axis>) { <lhs>
        ~ d<family>(args) }`` form."""
        del constraint, observed
        jctx = _as_jags_ctx(ctx)
        return self._emit_sample(
            jctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            plate=plate,
        )

    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Lower an [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        scope inline to its latent ``~`` sample plus the scope body, via
        the inherited
        [`explicit_latent_scope`][quivers.transpile.renderers._base.RendererBase.explicit_latent_scope]
        helper.

        JAGS samples discrete latents natively rather than enumerating
        like Stan, so the lowering is mechanical."""
        jctx = _as_jags_ctx(ctx)
        rewritten = self.explicit_latent_scope(node)
        latent = rewritten[0]
        if isinstance(latent, IRSample):
            renamed, rename_map = self._dedupe_plate(
                jctx, latent.plate, latent.name
            )
            # Stash the rename map so `_emit_sample` extends its
            # axis-to-loop-var map to cover the original axes too.
            jctx.axis_aliases.update(rename_map)
            latent_dedup = IRSample(
                name=latent.name,
                family=latent.family,
                args=latent.args,
                arg_names=latent.arg_names,
                constraint=latent.constraint,
                plate=renamed,
            )
            self._dispatch_jags_node(jctx, latent_dedup)
            for follow in rewritten[1:]:
                via_name = (
                    follow.via
                    if isinstance(follow, IRObserve) and follow.via is not None
                    else None
                )
                if via_name is not None:
                    jctx.latent_via[latent.name] = via_name
                self._dispatch_jags_node(jctx, follow)
        else:  # pragma: no cover -- explicit_latent_scope contract
            for child in rewritten:
                self._dispatch_jags_node(jctx, child)
        return ""

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Render a scalar-to-vector broadcast as a JAGS range index
        into a pre-bound data vector.

        JAGS has no scalar-to-vector broadcast primitive; the canonical
        convention is for the host to pre-bind a vector of the broadcast
        scalar's repeated value and index it as ``alpha[1:K]`` in the
        model. Literal-scalar broadcasts raise
        [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
        with ``arg:broadcast`` since JAGS cannot inline an arithmetic
        scalar at the vector slot.
        """
        jctx = _as_jags_ctx(ctx)
        if not isinstance(value, IRArgRef):
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                ["arg:broadcast"],
            )
        if not target_shape:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                ["arg:broadcast:empty-shape"],
            )
        return self._range_indexed(jctx, value.name, target_shape)

    def render_list(
        self, ctx: _JAGSCtx, arg: IRArgList
    ) -> SchemaFragment:
        """JAGS does not parse list literals in argument position;
        callers must pre-bind collections via let-decl."""
        del ctx, arg
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}",
            ["arg:list-literal"],
        )

    def render_matrix(
        self, ctx: _JAGSCtx, arg: IRArgMatrix
    ) -> SchemaFragment:
        """JAGS does not parse matrix literals in argument position;
        callers must pre-bind matrices via let-decl."""
        del ctx, arg
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}",
            ["arg:matrix-literal"],
        )

    # ------------------------------------------------------------------
    # IR node dispatch
    # ------------------------------------------------------------------

    def _dispatch_jags_node(
        self, ctx: _JAGSCtx, node: IRNode
    ) -> None:
        """Walk one IR node and emit its JAGS form into the active
        block."""
        if isinstance(node, IRDataInput):
            return
        if isinstance(node, IRSample):
            self._emit_sample(
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
            )
            return
        if isinstance(node, IRObserve):
            self._emit_sample(
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
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
            return
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}",
            [f"node:{type(node).__name__}"],
        )

    # ------------------------------------------------------------------
    # Sample / observe emission
    # ------------------------------------------------------------------

    def _emit_sample(
        self,
        ctx: _JAGSCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        via: str | None = None,
    ) -> SchemaFragment:
        """Build ``for (...) { name[m] ~ d<family>(args) }`` per batch
        dim and attach it to the surrounding block."""
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}", [f"family:unknown:{family}"]
            )
        target_name = meta.target_names.get(_BACKEND)
        if target_name is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}", [f"family:no-target-name:{family}"]
            )

        if family in _PREPEND_ZERO:
            args = (IRArgNumber(value=0.0), *args)
            arg_names = ("loc", *arg_names)

        aliases = meta.arg_aliases.get(_BACKEND, {})
        renamed_pairs: list[tuple[str, IRArg]] = []
        for arg_name, arg in zip(arg_names, args, strict=False):
            emitted_name = aliases.get(arg_name, arg_name)
            transform = _ALIAS_TRANSFORMS.get(emitted_name)
            if transform is not None and emitted_name != arg_name:
                arg = IRArgTransform(inner=arg, transform=transform)
            renamed_pairs.append((emitted_name, arg))

        # Compute loop var names. The observation-plate convention is to
        # use the canonical `n` when `via` is set and the plate has a
        # single dynamic dim.
        if via is not None and len(plate.batch_dims) == 1:
            loop_var_names: tuple[str, ...] = ("n",)
        else:
            loop_var_names = tuple(
                f"m_{_dim_name(dim)}" for dim in plate.batch_dims
            )

        # Build the axis-to-loop-var map for this sample's surrounding
        # plate. `_rewrite_arg` uses this to choose the right loop var
        # when a latent's recorded axis appears in the current plate.
        # Renamed axes (from `_dedupe_plate`) also resolve to their
        # original axis so a sample on the renamed plate can index
        # latents bound on the original.
        axis_to_lv: dict[str, str] = {}
        for dim, lv in zip(plate.batch_dims, loop_var_names, strict=False):
            renamed = _dim_name(dim)
            axis_to_lv[renamed] = lv
            original = ctx.axis_aliases.get(renamed)
            if original is not None:
                axis_to_lv[original] = lv

        # Pre-rewrite every arg ONCE: thread latent loop vars + event
        # ranges + via fibration through any ref to a previously-bound
        # latent. The rewrite is idempotent on its output by construction
        # (the appended event-range sentinels are not themselves latents).
        rewritten_args = tuple(
            (n, _rewrite_arg(ctx, a, loop_var_names, axis_to_lv, via))
            for n, a in renamed_pairs
        )

        sr = self._build_stochastic_relation(
            ctx,
            lhs_name=name,
            target_dist=target_name,
            renamed_pairs=rewritten_args,
            plate=plate,
            loop_vars=loop_var_names,
        )

        for dim in plate.batch_dims:
            ctx.emitted_plate_names.add(_dim_name(dim))

        # Record this sample's plate info so subsequent refs to `name`
        # thread the loop var + event ranges through. We record both the
        # axis name (so a reference can use the *current* surrounding
        # plate's loop var when the axes match) and a fallback loop var
        # for when no surrounding plate covers the axis.
        if plate.batch_dims:
            fallback_lv = loop_var_names[-1] if loop_var_names else f"m_{_dim_name(plate.batch_dims[-1])}"
            axes = tuple(_dim_name(d) for d in plate.batch_dims)
            ctx.latent_plate_info[name] = (
                fallback_lv, plate.event_dims, axes,
            )

        override_var = (
            "n"
            if via is not None and len(plate.batch_dims) == 1
            else None
        )
        wrapped = self._wrap_in_for_loops(
            ctx, sr, plate.batch_dims, override_var=override_var
        )
        if ctx.current_block is not None:
            ctx.sb.edge(ctx.current_block, wrapped, "child_of")
            ctx.block_children.setdefault(ctx.current_block, []).append(
                _block_child_kind(ctx, wrapped)
            )
        return wrapped

    def _build_stochastic_relation(
        self,
        ctx: _JAGSCtx,
        *,
        lhs_name: str,
        target_dist: str,
        renamed_pairs: tuple[tuple[str, IRArg], ...],
        plate: Plate,
        loop_vars: tuple[str, ...],
    ) -> str:
        """Build ``<lhs> ~ <dist>(...)`` as a ``stochastic_relation``."""
        sr = _fresh(ctx, "sr", "stochastic_relation")
        ctx.sb.constraint(sr, "chose-alt-fingerprint", "~")
        ctx.sb.constraint(
            sr,
            "chose-alt-child-kinds",
            f"{_lhs_kind(plate, loop_vars)} distribution_call",
        )
        ctx.sb.constraint(sr, "ptrace-0", f"C{_lhs_kind(plate, loop_vars)}")
        ctx.sb.constraint(sr, "ptrace-1", "T~")
        ctx.sb.constraint(sr, "ptrace-2", "Cdistribution_call")

        lhs = self._build_lhs(ctx, lhs_name, plate, loop_vars)
        ctx.sb.edge(sr, lhs, "variable")

        dc = self._build_distribution_call(
            ctx, target_dist, renamed_pairs
        )
        ctx.sb.edge(sr, dc, "distribution")
        return sr

    def _build_lhs(
        self,
        ctx: _JAGSCtx,
        name: str,
        plate: Plate,
        loop_vars: tuple[str, ...],
    ) -> str:
        """Build the LHS of a stochastic relation.

        With zero batch dims and no event dim, the LHS is a bare
        identifier. With batch dims, it is an
        ``indexed_variable``: ``name[m_0, m_1, ...]`` when the event
        is scalar, or ``name[m_0, ..., 1:E]`` when the event has a
        vector shape (e.g. Dirichlet draws on a Topic axis)."""
        if not plate.batch_dims and not plate.event_dims:
            return _identifier(ctx, name)
        return self._indexed_variable(
            ctx, name, loop_vars, plate.event_dims
        )

    def _indexed_variable(
        self,
        ctx: _JAGSCtx,
        name: str,
        loop_vars: tuple[str, ...],
        event_dims: tuple[object, ...],
    ) -> str:
        """``name[m_0, m_1, ..., 1:E0, 1:E1, ...]``."""
        iv = _fresh(ctx, "iv", "indexed_variable")
        ctx.sb.constraint(iv, "chose-alt-fingerprint", "[ ]")
        ctx.sb.constraint(
            iv, "chose-alt-child-kinds", "identifier index_list"
        )
        ctx.sb.constraint(iv, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(iv, "ptrace-1", "T[")
        ctx.sb.constraint(iv, "ptrace-2", "Cindex_list")
        ctx.sb.constraint(iv, "ptrace-3", "T]")

        ident = _identifier(ctx, name)
        ctx.sb.edge(iv, ident, "name")

        idx_list = _fresh(ctx, "il", "index_list")
        index_kinds: list[str] = []
        children: list[tuple[str, str]] = []
        for lv in loop_vars:
            child_id = _identifier(ctx, lv)
            children.append((child_id, "identifier"))
            index_kinds.append("identifier")
        for ed in event_dims:
            size_form, kind = self._event_range_form(ctx, ed)
            children.append((size_form, kind))
            index_kinds.append(kind)

        ptrace_idx = 0
        fingerprint_parts: list[str] = []
        for i, (_, kind) in enumerate(children):
            ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", f"C{kind}")
            ptrace_idx += 1
            if i < len(children) - 1:
                ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", "T,")
                ptrace_idx += 1
                fingerprint_parts.append(",")
        ctx.sb.constraint(
            idx_list,
            "chose-alt-fingerprint",
            " ".join(fingerprint_parts) if fingerprint_parts else "",
        )
        ctx.sb.constraint(
            idx_list, "chose-alt-child-kinds", " ".join(index_kinds)
        )

        for child_id, _kind in children:
            ctx.sb.edge(idx_list, child_id, "child_of")
        ctx.sb.edge(iv, idx_list, "indices")
        return iv

    def _event_range_form(
        self, ctx: _JAGSCtx, dim: object
    ) -> tuple[str, str]:
        """`1:E` range form for one event dim. Returns (vid, kind)."""
        if isinstance(dim, DimStatic):
            return self._range_static(ctx, dim.size), "range"
        if isinstance(dim, DimDynamic):
            return self._range_dynamic(ctx, dim.size_name), "range"
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"]
        )

    def _range_static(self, ctx: _JAGSCtx, upper: int) -> str:
        rng = _fresh(ctx, "rng", "range")
        ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
        ctx.sb.constraint(rng, "chose-alt-child-kinds", "number number")
        ctx.sb.constraint(rng, "ptrace-0", "Cnumber")
        ctx.sb.constraint(rng, "ptrace-1", "T:")
        ctx.sb.constraint(rng, "ptrace-2", "Cnumber")
        lo = _number(ctx, 1)
        hi = _number(ctx, upper)
        ctx.sb.edge(rng, lo, "lower")
        ctx.sb.edge(rng, hi, "upper")
        return rng

    def _range_dynamic(self, ctx: _JAGSCtx, upper_name: str) -> str:
        rng = _fresh(ctx, "rng", "range")
        ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
        ctx.sb.constraint(
            rng, "chose-alt-child-kinds", "number identifier"
        )
        ctx.sb.constraint(rng, "ptrace-0", "Cnumber")
        ctx.sb.constraint(rng, "ptrace-1", "T:")
        ctx.sb.constraint(rng, "ptrace-2", "Cidentifier")
        lo = _number(ctx, 1)
        hi = _identifier(ctx, upper_name)
        ctx.sb.edge(rng, lo, "lower")
        ctx.sb.edge(rng, hi, "upper")
        return rng

    def _range_indexed(
        self,
        ctx: _JAGSCtx,
        name: str,
        target_shape: tuple[int, ...],
    ) -> str:
        """Build ``<name>[1:K]`` (1D) or ``<name>[1:R, 1:C]`` (2D)."""
        iv = _fresh(ctx, "iv", "indexed_variable")
        ctx.sb.constraint(iv, "chose-alt-fingerprint", "[ ]")
        ctx.sb.constraint(
            iv, "chose-alt-child-kinds", "identifier index_list"
        )
        ctx.sb.constraint(iv, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(iv, "ptrace-1", "T[")
        ctx.sb.constraint(iv, "ptrace-2", "Cindex_list")
        ctx.sb.constraint(iv, "ptrace-3", "T]")

        ident = _identifier(ctx, name)
        ctx.sb.edge(iv, ident, "name")

        idx_list = _fresh(ctx, "il", "index_list")
        ranges: list[str] = []
        for size in target_shape:
            ranges.append(self._range_static(ctx, size))

        ptrace_idx = 0
        fingerprint_parts: list[str] = []
        kinds: list[str] = []
        for i, _ in enumerate(ranges):
            ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", "Crange")
            ptrace_idx += 1
            kinds.append("range")
            if i < len(ranges) - 1:
                ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", "T,")
                ptrace_idx += 1
                fingerprint_parts.append(",")
        ctx.sb.constraint(
            idx_list,
            "chose-alt-fingerprint",
            " ".join(fingerprint_parts) if fingerprint_parts else "",
        )
        ctx.sb.constraint(
            idx_list, "chose-alt-child-kinds", " ".join(kinds)
        )
        for vid in ranges:
            ctx.sb.edge(idx_list, vid, "child_of")
        ctx.sb.edge(iv, idx_list, "indices")
        return iv

    def _build_distribution_call(
        self,
        ctx: _JAGSCtx,
        target_dist: str,
        renamed_pairs: tuple[tuple[str, IRArg], ...],
    ) -> str:
        """Build ``<dist>(<arg0>, <arg1>, ...)``.

        JAGS positional-only calling convention: emit args in the
        order received.
        """
        dc = _fresh(ctx, "dc", "distribution_call")
        ctx.sb.constraint(dc, "chose-alt-fingerprint", "( )")
        ctx.sb.constraint(
            dc, "chose-alt-child-kinds", "identifier argument_list"
        )
        ctx.sb.constraint(dc, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(dc, "ptrace-1", "T(")
        ctx.sb.constraint(dc, "ptrace-2", "Cargument_list")
        ctx.sb.constraint(dc, "ptrace-3", "T)")

        name = _identifier(ctx, target_dist)
        ctx.sb.edge(dc, name, "name")

        al = self._build_argument_list(
            ctx,
            tuple(arg for _name, arg in renamed_pairs),
        )
        ctx.sb.edge(dc, al, "arguments")
        return dc

    def _build_argument_list(
        self,
        ctx: _JAGSCtx,
        args: tuple[IRArg, ...],
    ) -> str:
        """``arg0, arg1, ...`` as an ``argument_list``."""
        al = _fresh(ctx, "al", "argument_list")
        child_pairs: list[tuple[str, str]] = []
        for arg in args:
            vid, kind = self._render_arg_with_kind(ctx, arg)
            child_pairs.append((vid, kind))

        ptrace_idx = 0
        fingerprint_parts: list[str] = []
        kinds: list[str] = []
        for i, (_, kind) in enumerate(child_pairs):
            ctx.sb.constraint(al, f"ptrace-{ptrace_idx}", f"C{kind}")
            ptrace_idx += 1
            kinds.append(kind)
            if i < len(child_pairs) - 1:
                ctx.sb.constraint(al, f"ptrace-{ptrace_idx}", "T,")
                ptrace_idx += 1
                fingerprint_parts.append(",")
        ctx.sb.constraint(
            al,
            "chose-alt-fingerprint",
            " ".join(fingerprint_parts) if fingerprint_parts else "",
        )
        ctx.sb.constraint(
            al, "chose-alt-child-kinds", " ".join(kinds)
        )

        for vid, _kind in child_pairs:
            ctx.sb.edge(al, vid, "child_of")
        return al

    def _render_arg_with_kind(
        self,
        ctx: _JAGSCtx,
        arg: IRArg,
    ) -> tuple[str, str]:
        """Render one arg; return (vid, kind) so the parent can build
        ptrace constraints."""
        if isinstance(arg, IRArgNumber):
            return _number(ctx, arg.value), "number"
        if isinstance(arg, IRArgRef):
            return self._render_ref_with_kind(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return (
                self.broadcast(ctx, arg.value, arg.target_shape),
                "indexed_variable",
            )
        if isinstance(arg, IRArgList):
            return self.render_list(ctx, arg), "indexed_variable"
        if isinstance(arg, IRArgMatrix):
            return self.render_matrix(ctx, arg), "indexed_variable"
        if isinstance(arg, IRArgFamilyRef):
            return self._render_family_ref_with_kind(ctx, arg)
        if isinstance(arg, IRArgTransform):
            return self._render_transform_with_kind(ctx, arg)
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}", [f"arg:{type(arg).__name__}"]
        )

    def _render_ref_with_kind(
        self,
        ctx: _JAGSCtx,
        arg: IRArgRef,
    ) -> tuple[str, str]:
        """Render a bound-name reference. Bare names emit as
        ``identifier``; indexed references emit as ``indexed_variable``
        with the index expressions as children.

        The arg has already been rewritten by `_rewrite_arg` at sample
        time; no further latent-plate threading happens here.
        """
        # Sentinel: encode a JAGS `1:N` range.
        if arg.name.startswith(_RANGE_SENTINEL_PREFIX):
            upper = arg.name[len(_RANGE_SENTINEL_PREFIX) :]
            try:
                return self._range_static(ctx, int(upper)), "range"
            except ValueError:
                return self._range_dynamic(ctx, upper), "range"
        if not arg.indices:
            return _identifier(ctx, arg.name), "identifier"
        iv = _fresh(ctx, "iv", "indexed_variable")
        ctx.sb.constraint(iv, "chose-alt-fingerprint", "[ ]")
        ctx.sb.constraint(
            iv, "chose-alt-child-kinds", "identifier index_list"
        )
        ctx.sb.constraint(iv, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(iv, "ptrace-1", "T[")
        ctx.sb.constraint(iv, "ptrace-2", "Cindex_list")
        ctx.sb.constraint(iv, "ptrace-3", "T]")

        name = _identifier(ctx, arg.name)
        ctx.sb.edge(iv, name, "name")

        idx_list = _fresh(ctx, "il", "index_list")
        child_pairs: list[tuple[str, str]] = []
        for idx in arg.indices:
            vid, kind = self._render_arg_with_kind(ctx, idx)
            child_pairs.append((vid, kind))

        ptrace_idx = 0
        fingerprint_parts: list[str] = []
        kinds: list[str] = []
        for i, (_, kind) in enumerate(child_pairs):
            ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", f"C{kind}")
            ptrace_idx += 1
            kinds.append(kind)
            if i < len(child_pairs) - 1:
                ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", "T,")
                ptrace_idx += 1
                fingerprint_parts.append(",")
        ctx.sb.constraint(
            idx_list,
            "chose-alt-fingerprint",
            " ".join(fingerprint_parts) if fingerprint_parts else "",
        )
        ctx.sb.constraint(
            idx_list, "chose-alt-child-kinds", " ".join(kinds)
        )
        for vid, _ in child_pairs:
            ctx.sb.edge(idx_list, vid, "child_of")
        ctx.sb.edge(iv, idx_list, "indices")
        return iv, "indexed_variable"

    def _render_family_ref_with_kind(
        self, ctx: _JAGSCtx, arg: IRArgFamilyRef
    ) -> tuple[str, str]:
        """Emit ``d<family>(args) T(L, U)`` for a `Truncated` wrapper.

        For non-truncated wrappers and family-refs without a morphism
        declaration in `ctx.morphisms`, fall back to a bare identifier
        of the morphism's name (the host has already bound it via a
        let-decl)."""
        decl = ctx.morphisms.get(arg.name)
        if decl is None or decl.init_family is None:
            return _identifier(ctx, arg.name), "identifier"
        inner_family = decl.init_family.family
        inner_meta = FAMILY_META.get(inner_family)
        if inner_meta is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [f"family:wrapper-inner-unknown:{inner_family}"],
            )
        target_inner = inner_meta.target_names.get(_BACKEND)
        if target_inner is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [f"family:no-target-name:{inner_family}"],
            )
        inner_args: tuple[IRArg, ...] = tuple(
            _coerce_to_ir_arg(a) for a in (decl.init_family.args or ())
        )
        inner_arg_names = tuple(
            inner_meta.distribution_class.arg_constraints.keys()
            if isinstance(
                inner_meta.distribution_class.arg_constraints, dict
            )
            else ()
        )
        renamed_pairs = tuple(
            (n, a) for n, a in zip(inner_arg_names, inner_args, strict=False)
        )
        dc = self._build_distribution_call(
            ctx, target_inner, renamed_pairs
        )
        return dc, "distribution_call"

    def _render_transform_with_kind(
        self,
        ctx: _JAGSCtx,
        arg: IRArgTransform,
    ) -> tuple[str, str]:
        """Render an
        [`IRArgTransform`][quivers.transpile.renderers._base.IRArgTransform]:
        ``inv_square(x) -> 1/(x*x)`` etc.

        Emits a ``binary_expression`` tree using the JAGS grammar's
        arithmetic operators.
        """
        inner_vid, inner_kind = self._render_arg_with_kind(ctx, arg.inner)
        if arg.transform == "inv_square":
            sq = self._binary_expr(ctx, "*", inner_vid, inner_kind, inner_vid, inner_kind)
            paren = self._parenthesized(ctx, sq, "binary_expression")
            one = _number(ctx, 1)
            div = self._binary_expr(
                ctx, "/", one, "number", paren, "parenthesized_expression"
            )
            return div, "binary_expression"
        if arg.transform == "inv":
            one = _number(ctx, 1)
            div = self._binary_expr(
                ctx, "/", one, "number", inner_vid, inner_kind
            )
            return div, "binary_expression"
        if arg.transform == "neg":
            ue = _fresh(ctx, "ue", "unary_expression")
            ctx.sb.constraint(ue, "field:operator", "-")
            ctx.sb.constraint(ue, "chose-alt-fingerprint", "-")
            ctx.sb.edge(ue, inner_vid, "argument")
            return ue, "unary_expression"
        if arg.transform in ("log", "exp"):
            fc = _fresh(ctx, "fc", "function_call")
            ctx.sb.constraint(fc, "chose-alt-fingerprint", "( )")
            ctx.sb.constraint(
                fc, "chose-alt-child-kinds", f"identifier {inner_kind}"
            )
            ctx.sb.constraint(fc, "ptrace-0", "Cidentifier")
            ctx.sb.constraint(fc, "ptrace-1", "T(")
            ctx.sb.constraint(fc, "ptrace-2", f"C{inner_kind}")
            ctx.sb.constraint(fc, "ptrace-3", "T)")
            fn = _identifier(ctx, arg.transform)
            ctx.sb.edge(fc, fn, "name")
            ctx.sb.edge(fc, inner_vid, "child_of")
            return fc, "function_call"
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}", [f"transform:{arg.transform}"]
        )

    def _binary_expr(
        self,
        ctx: _JAGSCtx,
        op: str,
        left_vid: str,
        left_kind: str,
        right_vid: str,
        right_kind: str,
    ) -> str:
        """Build ``<left> <op> <right>`` as a ``binary_expression``."""
        be = _fresh(ctx, "be", "binary_expression")
        ctx.sb.constraint(be, "field:operator", op)
        ctx.sb.constraint(be, "chose-alt-fingerprint", op)
        ctx.sb.constraint(
            be, "chose-alt-child-kinds", f"{left_kind} {right_kind}"
        )
        ctx.sb.edge(be, left_vid, "left")
        ctx.sb.edge(be, right_vid, "right")
        return be

    def _parenthesized(
        self, ctx: _JAGSCtx, inner_vid: str, inner_kind: str
    ) -> str:
        """Build ``( <inner> )`` as a ``parenthesized_expression``."""
        pe = _fresh(ctx, "pe", "parenthesized_expression")
        ctx.sb.constraint(pe, "chose-alt-fingerprint", "( )")
        ctx.sb.constraint(pe, "chose-alt-child-kinds", inner_kind)
        ctx.sb.constraint(pe, "ptrace-0", "T(")
        ctx.sb.constraint(pe, "ptrace-1", f"C{inner_kind}")
        ctx.sb.constraint(pe, "ptrace-2", "T)")
        ctx.sb.edge(pe, inner_vid, "child_of")
        return pe

    # ------------------------------------------------------------------
    # For loop wrapping
    # ------------------------------------------------------------------

    def _wrap_in_for_loops(
        self,
        ctx: _JAGSCtx,
        inner_vid: str,
        batch_dims: tuple[object, ...],
        *,
        override_var: str | None = None,
    ) -> str:
        """Wrap `inner_vid` in nested ``for (m_<axis> in 1:N) { ... }``
        loops, outermost-first by source-axis order. With no batch
        dims, returns the inner vertex directly."""
        if not batch_dims:
            return inner_vid
        current = inner_vid
        current_kind = "stochastic_relation"
        for dim in reversed(batch_dims):
            blk = _fresh(ctx, "blk", "block")
            ctx.sb.constraint(blk, "chose-alt-fingerprint", "{ }")
            ctx.sb.constraint(blk, "chose-alt-child-kinds", current_kind)
            ctx.sb.constraint(blk, "ptrace-0", "T{")
            ctx.sb.constraint(blk, "ptrace-1", f"C{current_kind}")
            ctx.sb.constraint(blk, "ptrace-2", "T}")
            ctx.sb.edge(blk, current, "child_of")

            fl = _fresh(ctx, "fl", "for_loop")
            ctx.sb.constraint(fl, "chose-alt-fingerprint", "for ( in )")
            ctx.sb.constraint(
                fl, "chose-alt-child-kinds", "identifier range block"
            )
            ctx.sb.constraint(fl, "ptrace-0", "Tfor")
            ctx.sb.constraint(fl, "ptrace-1", "T(")
            ctx.sb.constraint(fl, "ptrace-2", "Cidentifier")
            ctx.sb.constraint(fl, "ptrace-3", "Tin")
            ctx.sb.constraint(fl, "ptrace-4", "Crange")
            ctx.sb.constraint(fl, "ptrace-5", "T)")
            ctx.sb.constraint(fl, "ptrace-6", "Cblock")

            loop_var_text = (
                override_var
                if override_var is not None and len(batch_dims) == 1
                else f"m_{_dim_name(dim)}"
            )
            lv = _identifier(ctx, loop_var_text)
            ctx.sb.edge(fl, lv, "variable")

            rng_vid, _ = self._dim_size_range(ctx, dim)
            ctx.sb.edge(fl, rng_vid, "range")
            ctx.sb.edge(fl, blk, "body")

            current = fl
            current_kind = "for_loop"
        return current

    def _dim_size_range(
        self, ctx: _JAGSCtx, dim: object
    ) -> tuple[str, str]:
        """Build a ``1:N`` range vertex for one batch dim. Returns
        (vid, kind)."""
        if isinstance(dim, DimStatic):
            return self._range_static(ctx, dim.size), "range"
        if isinstance(dim, DimDynamic):
            return self._range_dynamic(ctx, dim.size_name), "range"
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"]
        )

    # ------------------------------------------------------------------
    # Deterministic / score emission
    # ------------------------------------------------------------------

    def _emit_deterministic(
        self, ctx: _JAGSCtx, node: IRDeterministic
    ) -> None:
        """JAGS deterministic relation ``<name> <- <expr>``.

        The RHS goes through
        [`render_let_expr_bugs`][quivers.transpile.renderers._bugs_helpers.render_let_expr_bugs]
        (BUGS / JAGS share an expression grammar), with a thin
        ctx shim adapting `_JAGSCtx`'s `_fresh` /
        `panproto.SchemaBuilder` to the helper's protocol.

        When ``node.plate`` carries batch dims, the relation is
        wrapped in matching ``for (m_<axis> in 1:N) { ... }`` loops
        with the LHS indexed by the loop variables. Each let-expr
        reference to a plated binding is re-indexed against the
        same loop variables so the emitted RHS pulls per-iteration
        values out of the surrounding vectors.
        """
        loop_var_names = tuple(
            f"m_{_dim_name(dim)}" for dim in node.plate.batch_dims
        )
        # Open the deterministic relation. When plated, the LHS is
        # indexed by every loop variable; otherwise it is a bare name.
        dr = _fresh(ctx, "dr", "deterministic_relation")
        ctx.sb.constraint(dr, "chose-alt-fingerprint", "<-")
        ctx.sb.constraint(dr, "ptrace-1", "T<-")
        if loop_var_names:
            ctx.sb.constraint(dr, "ptrace-0", "Cindexed_variable")
            lhs = self._indexed_variable(
                ctx, node.name, loop_var_names, ()
            )
        else:
            ctx.sb.constraint(dr, "ptrace-0", "Cidentifier")
            lhs = _identifier(ctx, node.name)
        ctx.sb.edge(dr, lhs, "variable")
        rewritten = index_letexpr_refs(
            node.expr, ctx.decl_plates, node.plate, loop_var_names
        )
        let_ctx = _jags_let_ctx(ctx, self._cards)
        val = render_let_expr_bugs(let_ctx, rewritten)
        ctx.sb.edge(dr, val, "value")
        wrapped = self._wrap_in_for_loops(
            ctx, dr, node.plate.batch_dims, override_var=None
        )
        if ctx.current_block is not None:
            ctx.sb.edge(ctx.current_block, wrapped, "child_of")
            ctx.block_children.setdefault(ctx.current_block, []).append(
                _block_child_kind(ctx, wrapped)
            )
        # Record the deterministic's plate so downstream IRArgRef
        # references to `node.name` thread the right loop variable
        # through. Mirrors the IRSample registration in `_emit_sample`.
        if node.plate.batch_dims:
            fallback_lv = (
                loop_var_names[-1]
                if loop_var_names
                else f"m_{_dim_name(node.plate.batch_dims[-1])}"
            )
            axes = tuple(_dim_name(d) for d in node.plate.batch_dims)
            ctx.latent_plate_info[node.name] = (
                fallback_lv,
                node.plate.event_dims,
                axes,
            )
            for dim in node.plate.batch_dims:
                ctx.emitted_plate_names.add(_dim_name(dim))

    def _emit_score(self, ctx: _JAGSCtx, node: IRScore) -> None:  # type: ignore[override]
        """JAGS has no native target-statement; the zeros / ones
        trick demands a host-supplied phantom-observation carrier
        the IR does not currently express. Refuse rather than
        emit a placeholder."""
        del ctx, node
        raise UnsupportedConstruct(
            "qvr-jags",
            [
                "node:IRScore: jags has no native target-statement; "
                "the zeros / ones trick requires a host-supplied "
                "phantom-observation carrier the IR does not "
                "currently express"
            ],
        )

    # ------------------------------------------------------------------
    # Plate-name disambiguation
    # ------------------------------------------------------------------

    def _dedupe_plate(
        self,
        ctx: _JAGSCtx,
        plate: Plate,
        latent_name: str,
    ) -> tuple[Plate, dict[str, str]]:
        """Return a Plate whose batch_dim names get a ``_<latent>``
        suffix only when the name has already been emitted in this
        render call. Mirrors the NumPyro / Stan convention for the
        LDA-style marginalize-then-reuse pattern.

        Also returns a map from each renamed axis to its original axis
        name, so a sample inside the renamed plate can resolve refs to
        latents bound on the original axis (e.g. theta on Doc must
        thread through m_Doc_z when sampled inside the Doc_z plate).
        """
        seen = ctx.emitted_plate_names
        new_batch: list[Dim] = []
        rename_map: dict[str, str] = {}
        for dim in plate.batch_dims:
            original = _dim_name(dim)
            renamed = (
                f"{original}_{latent_name}"
                if original in seen
                else original
            )
            if renamed != original:
                rename_map[renamed] = original
            if isinstance(dim, DimStatic):
                new_batch.append(DimStatic(size=dim.size, name=renamed))
            elif isinstance(dim, DimDynamic):
                new_batch.append(
                    DimDynamic(size_name=dim.size_name, name=renamed)
                )
            else:
                new_batch.append(dim)
        return (
            Plate(
                event_dims=plate.event_dims,
                batch_dims=tuple(new_batch),
            ),
            rename_map,
        )

    # ------------------------------------------------------------------
    # model_block finalisation
    # ------------------------------------------------------------------

    def _finalise_model_block(self, ctx: _JAGSCtx) -> None:
        """Pin the ``model { ... }`` alternative on the model_block so
        the pretty-printer emits the keyword-prefixed form.

        The auto-derived theory supplies the inter-child layout from the
        grammar's whitespace conventions (newline + two-space indent
        between sibling statements). We only need to disambiguate the
        ``{ ... }`` vs. ``model { ... }`` alternative."""
        mb = ctx.model_block
        children = ctx.block_children.get(mb, [])
        ctx.sb.constraint(mb, "chose-alt-fingerprint", "model { }")
        ctx.sb.constraint(
            mb, "chose-alt-child-kinds", " ".join(children) if children else ""
        )
        ctx.sb.constraint(mb, "ptrace-0", "Tmodel")
        ctx.sb.constraint(mb, "ptrace-1", "T{")
        for i, kind in enumerate(children):
            ctx.sb.constraint(mb, f"ptrace-{2 + i}", f"C{kind}")
        ctx.sb.constraint(
            mb, f"ptrace-{2 + len(children)}", "T}"
        )


# ---------------------------------------------------------------------------
# Arg pre-rewrite (idempotent thread-once helper).
# ---------------------------------------------------------------------------


def _rewrite_arg(
    ctx: _JAGSCtx,
    arg: IRArg,
    loop_vars: tuple[str, ...],
    axis_to_lv: dict[str, str],
    via: str | None,
) -> IRArg:
    """Thread latent loop-vars + event ranges + via fibration through
    every ref nested inside `arg`.

    The rewrite is a SINGLE PASS: it inspects every IRArgRef in `arg`
    and rewrites it once. The output is idempotent on subsequent passes
    by construction because the appended indices for an already-rewritten
    latent reference are detected via the ``__jags_range__:`` sentinel
    prefix on the trailing event-range entries.

    `axis_to_lv` maps each axis name in the surrounding plate to the
    loop var that iterates it. A latent ref whose recorded axis appears
    in this map indexes through the *current* loop var (so two samples
    on the same axis share the same loop body), with via-wrapping when
    the latent's recorded fibration applies.
    """
    if isinstance(arg, IRArgRef):
        new_indices = tuple(
            _rewrite_arg(ctx, idx, loop_vars, axis_to_lv, via)
            for idx in arg.indices
        )
        info = ctx.latent_plate_info.get(arg.name)
        if info is None:
            if new_indices == arg.indices:
                return arg
            return IRArgRef(name=arg.name, indices=new_indices)
        fallback_lv, event_dims, axes = info
        via_for_latent = ctx.latent_via.get(arg.name)
        # Choose the loop-var index expression.
        if via_for_latent is not None and loop_vars:
            # Latent's recorded fibration: wrap the observe loop var
            # through it (`z[word_idx[n]]`).
            lv_idx: IRArg = IRArgRef(
                name=via_for_latent,
                indices=(IRArgRef(name=loop_vars[0]),),
            )
        elif axes and axes[-1] in axis_to_lv:
            # Surrounding plate iterates the latent's axis directly:
            # reuse the current loop var (`theta[m_Doc_z, ...]`).
            lv_idx = IRArgRef(name=axis_to_lv[axes[-1]])
        elif via is not None and loop_vars:
            # Caller-set via overrides (passed via fibration).
            lv_idx = IRArgRef(
                name=via,
                indices=(IRArgRef(name=loop_vars[0]),),
            )
        else:
            lv_idx = IRArgRef(name=fallback_lv)

        event_indices = tuple(_event_range_ir(ed) for ed in event_dims)
        # IDEMPOTENCE GUARD: if `arg.indices` already ends with the
        # event_indices we'd append, leave them alone.
        if arg.indices and _has_trailing(arg.indices, event_indices):
            return IRArgRef(name=arg.name, indices=new_indices)
        if arg.indices:
            return IRArgRef(
                name=arg.name,
                indices=new_indices + event_indices,
            )
        return IRArgRef(
            name=arg.name,
            indices=(lv_idx,) + event_indices,
        )
    if isinstance(arg, IRArgBroadcast):
        inner = _rewrite_arg(ctx, arg.value, loop_vars, axis_to_lv, via)
        if inner is arg.value:
            return arg
        return IRArgBroadcast(value=inner, target_shape=arg.target_shape)
    if isinstance(arg, IRArgList):
        elements = tuple(
            _rewrite_arg(ctx, e, loop_vars, axis_to_lv, via)
            for e in arg.elements
        )
        if elements == arg.elements:
            return arg
        return IRArgList(elements=elements)
    if isinstance(arg, IRArgMatrix):
        rows = tuple(
            IRArgList(elements=tuple(
                _rewrite_arg(ctx, e, loop_vars, axis_to_lv, via)
                for e in row.elements
            ))
            for row in arg.rows
        )
        return IRArgMatrix(rows=rows)
    if isinstance(arg, IRArgTransform):
        inner = _rewrite_arg(ctx, arg.inner, loop_vars, axis_to_lv, via)
        if inner is arg.inner:
            return arg
        return IRArgTransform(inner=inner, transform=arg.transform)
    return arg


def _has_trailing(
    indices: tuple[IRArg, ...], suffix: tuple[IRArg, ...]
) -> bool:
    """Return True iff `indices` ends with `suffix` (by name equality).

    Used by `_rewrite_arg` for idempotence: if a latent ref already has
    the event-range sentinel indices appended, do not re-append them on
    the next rewrite pass.
    """
    if not suffix:
        return False
    if len(indices) < len(suffix):
        return False
    tail = indices[-len(suffix):]
    for a, b in zip(tail, suffix, strict=False):
        if not (isinstance(a, IRArgRef) and isinstance(b, IRArgRef)):
            return False
        if a.name != b.name:
            return False
    return True


# ---------------------------------------------------------------------------
# Renderer-local context
# ---------------------------------------------------------------------------


class _JAGSCtx(_RenderCtx):
    """JAGS-specific extension of
    [`_RenderCtx`][quivers.transpile.renderers._base._RenderCtx]
    carrying the active block vertex, emitted plate names, and the
    per-block child-kind list used to assemble `chose-alt-child-kinds`
    constraints after the IR walk."""

    def __init__(
        self,
        *,
        sb: panproto.SchemaBuilder,
        morphisms: dict,
        lets: dict,
    ) -> None:
        super().__init__(sb=sb, morphisms=morphisms, lets=lets)
        self.current_block: str | None = None
        self.model_block: str = ""
        self.emitted_plate_names: set[str] = set()
        self.block_children: dict[str, list[str]] = {}
        #: Declared plate for every named binding (inputs, samples,
        #: observes, deterministics, marginalize-latents); read by
        #: `_emit_deterministic` to re-index plated var references
        #: inside lifted let-expressions.
        self.decl_plates: dict[str, Plate] = {}
        #: For every previously-bound latent name, record the
        #: (fallback_loop_var, event_dims, axis_names) used at its
        #: sample site. The fallback loop var is used when no
        #: surrounding plate covers any of the latent's axes; otherwise
        #: the rewriter looks up the axis in the current sample's
        #: axis-to-loop-var map.
        self.latent_plate_info: dict[
            str, tuple[str, tuple[object, ...], tuple[str, ...]]
        ] = {}
        #: For each latent on a per-observation plate, record the
        #: fibration that maps the observation row to its parent plate
        #: index.
        self.latent_via: dict[str, str] = {}
        #: For each renamed axis (from `_dedupe_plate`), record its
        #: original axis name. The arg rewriter consults this so a
        #: sample on the renamed plate can resolve refs to latents
        #: bound on the original axis.
        self.axis_aliases: dict[str, str] = {}


def _as_jags_ctx(ctx: _RenderCtx) -> _JAGSCtx:
    """Narrow a base `_RenderCtx` to the JAGS extension."""
    if not isinstance(ctx, _JAGSCtx):
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}", ["ctx:type-mismatch"]
        )
    return ctx


# ---------------------------------------------------------------------------
# Schema-builder helpers (vertex / identifier / number constructors).
# ---------------------------------------------------------------------------


def _fresh(ctx: _JAGSCtx, prefix: str, kind: str) -> str:
    """Allocate a fresh vertex with `prefix`-prefixed id of kind
    `kind`."""
    ctx.fresh_counter += 1
    vid = f"{prefix}_{ctx.fresh_counter}"
    ctx.sb.vertex(vid, kind)
    return vid


def _vertex(ctx: _JAGSCtx, vid: str, kind: str) -> str:
    """Register a vertex with an explicit id (used for the
    `source_file` root vertex)."""
    ctx.sb.vertex(vid, kind)
    return vid


class _JagsLetCtx:
    """Adapter exposing the protocol
    [`render_let_expr_bugs`][quivers.transpile.renderers._bugs_helpers.render_let_expr_bugs]
    expects (``v``, ``e``, ``lit``, ``fresh``, ``constraint``,
    ``target``, ``cards``) on top of a `_JAGSCtx`."""

    def __init__(self, ctx: _JAGSCtx, cards: dict[str, int]) -> None:
        self._ctx = ctx
        self.cards = cards
        self.target = "jags"

    def fresh(self, prefix: str) -> str:
        self._ctx.fresh_counter += 1
        return f"{prefix}_{self._ctx.fresh_counter}"

    def v(self, vid: str, kind: str) -> str:
        self._ctx.sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._ctx.sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._ctx.sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._ctx.sb.constraint(vid, sort, value)


def _jags_let_ctx(ctx: _JAGSCtx, cards: dict[str, int]) -> _JagsLetCtx:
    """Construct a `_JagsLetCtx` bound to `ctx` and `cards`."""
    return _JagsLetCtx(ctx, cards)


def _identifier(ctx: _JAGSCtx, text: str) -> str:
    """Build an `identifier` vertex with literal-value and
    fingerprint constraints."""
    vid = _fresh(ctx, "id", "identifier")
    ctx.sb.constraint(vid, "literal-value", text)
    ctx.sb.constraint(vid, "chose-alt-fingerprint", text)
    return vid


def _number(ctx: _JAGSCtx, value: float) -> str:
    """Build a `number` vertex carrying `value` as its
    literal-value."""
    vid = _fresh(ctx, "num", "number")
    text = str(int(value)) if value == int(value) else repr(value)
    ctx.sb.constraint(vid, "literal-value", text)
    ctx.sb.constraint(vid, "chose-alt-fingerprint", text)
    return vid


def _event_range_ir(dim: object) -> IRArg:
    """Encode a `1:E` range as a sentinel IRArgRef whose name carries
    the upper bound. The renderer recognises the prefix and emits a
    JAGS `range` vertex (`1:E`)."""
    if isinstance(dim, DimStatic):
        return IRArgRef(name=f"{_RANGE_SENTINEL_PREFIX}{dim.size}")
    if isinstance(dim, DimDynamic):
        return IRArgRef(name=f"{_RANGE_SENTINEL_PREFIX}{dim.size_name}")
    raise UnsupportedConstruct(
        f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"]
    )


def _dim_name(dim: object) -> str:
    """Return the source-axis name carried by a Dim."""
    if isinstance(dim, (DimStatic, DimDynamic)):
        return dim.name
    raise UnsupportedConstruct(
        f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"]
    )


def _lhs_kind(plate: Plate, loop_vars: tuple[str, ...]) -> str:
    """Return the grammar kind of the LHS of a stochastic_relation.

    Bare names (no batch / event dims) emit as ``identifier``;
    every other shape lands as ``indexed_variable``."""
    if not plate.batch_dims and not plate.event_dims:
        return "identifier"
    if not loop_vars and not plate.event_dims:
        return "identifier"
    return "indexed_variable"


def _block_child_kind(ctx: _JAGSCtx, vid: str) -> str:
    """Look up the grammar kind of `vid` via the fresh-id naming
    convention."""
    del ctx
    if vid.startswith("fl_"):
        return "for_loop"
    if vid.startswith("sr_"):
        return "stochastic_relation"
    if vid.startswith("dr_"):
        return "deterministic_relation"
    return "for_loop"


def _coerce_to_ir_arg(raw: object) -> IRArg:
    """Coerce a morphism's init_family arg (which may be a raw float
    or a name string) to an IRArg variant."""
    if isinstance(raw, IRArg):
        return raw
    if isinstance(raw, (int, float)):
        return IRArgNumber(value=float(raw))
    if isinstance(raw, str):
        return IRArgRef(name=raw)
    raise UnsupportedConstruct(
        f"qvr-{_BACKEND}", [f"arg:coerce:{type(raw).__name__}"]
    )


__all__ = ["JAGSRenderer"]
