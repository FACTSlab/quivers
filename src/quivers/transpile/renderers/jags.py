"""JAGS renderer: [`IRProgram`][quivers.transpile.ir.IRProgram] to JAGS
source under the ``jags`` tree-sitter grammar.

The JAGS surface mirrors BUGS for the probabilistic-core subset QVR
targets: a single top-level ``model { ... }`` block whose children are
``~`` stochastic relations and ``<-`` deterministic relations, nested
under ``for (m_<axis> in 1:N_<axis>) { ... }`` loops to express plate
structure. JAGS-specific family names (``ddirich`` for Dirichlet,
``dgen.gamma`` for generalised Gamma, etc.) and arithmetic-converting
parameterisation renames (Normal ``scale`` → ``tau = 1/(scale*scale)``
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
  before transpiling, per §10.9 of the spec).
* Appends ``T(L, U)`` truncation idiom when an
  [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] resolves to
  a ``Truncated(...)`` wrapper.
"""

from __future__ import annotations

import panproto

from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import target_protocol
from quivers.transpile.family_meta import FAMILY_META
from quivers.transpile.ir import (
    ConstraintSpec,
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
#: source value in the matching arithmetic transform before emitting
#: (per §10.4 of the design spec). Normal's ``scale`` → ``tau`` is the
#: canonical case: JAGS parameterises Normal by precision, so the
#: renderer emits ``1/(scale*scale)``.
_ALIAS_TRANSFORMS: dict[str, str] = {
    "tau": "inv_square",
}


class JAGSRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] as a
    JAGS model source.

    Subclasses [`RendererBase`][quivers.transpile.renderers._base.RendererBase]:
    overrides `render` to wrap the IR walk in a single top-level
    ``model { ... }`` block. The four dispatch points (`declare`,
    `sample`, `marginalize`, `broadcast`) follow the JAGS surface
    conventions described in §5.1 of the design spec.
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
        the statements emitted by `_dispatch_node`.
        """
        proto = self.target_protocol()
        sb = proto.schema()
        jctx = _JAGSCtx(sb=sb, morphisms={}, lets={})

        # source_file is the file's root; the `model_block` is its
        # sole child per JAGS grammar.
        _vertex(jctx, "src", "source_file")
        jctx.sb.constraint("src", "ptrace-0", "Cmodel_block")
        jctx.sb.constraint("src", "chose-alt-child-kinds", "model_block")

        mb = _fresh(jctx, "mb", "model_block")
        jctx.sb.edge("src", mb, "child_of")
        jctx.current_block = mb
        jctx.model_block = mb

        # Walk the body, emitting one or more statements per IR node.
        # Data inputs land in an external `.data` file the host
        # supplies; the model itself never declares them.
        for node in ir.body:
            self._dispatch_jags_node(jctx, node)

        # Apply structural constraints to the model_block (collected
        # child kinds, ptrace sequence, layout interstitials).
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
        # Rename the latent's plate name(s) to avoid colliding with
        # any already-emitted plate name.
        latent = rewritten[0]
        if isinstance(latent, IRSample):
            renamed = self._dedupe_plate(jctx, latent.plate, latent.name)
            latent_dedup = IRSample(
                name=latent.name,
                family=latent.family,
                args=latent.args,
                arg_names=latent.arg_names,
                constraint=latent.constraint,
                plate=renamed,
            )
            self._dispatch_jags_node(jctx, latent_dedup)
            # Each scope observe step carries its own `via` fibration
            # (which maps an observation row to its parent-plate row).
            # When the scope's observe references the marginalized
            # latent by name, rewrite the bare ref as `<latent>[<via>
            # [<loop_var>]]`.
            for follow in rewritten[1:]:
                via_name = (
                    follow.via
                    if isinstance(follow, IRObserve) and follow.via is not None
                    else None
                )
                if via_name is not None:
                    jctx.latent_via[latent.name] = via_name
                self._dispatch_jags_node_with_via(jctx, follow, via_name)
        else:  # pragma: no cover -- explicit_latent_scope contract
            for child in rewritten:
                self._dispatch_jags_node(jctx, child)
        return ""

    def _dispatch_jags_node_with_via(
        self,
        ctx: _JAGSCtx,
        node: IRNode,
        via: str | None,
    ) -> None:
        """Dispatch one IR node with a `via` fibration set on the
        surrounding plate. Only IRObserve consults `via`; other node
        kinds ignore it and fall back to the regular dispatch path."""
        if isinstance(node, IRObserve):
            self._emit_sample(
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                via=via,
            )
            return
        self._dispatch_jags_node(ctx, node)

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
        scalar's repeated value (e.g., ``alpha = rep(alpha_value, 3)``
        in the data file) and index it as ``alpha[1:K]`` in the model.

        The renderer emits ``<value>[1:K]`` (or ``<value>[1:R, 1:C]``
        for 2-D shapes) when the broadcast wraps an
        [`IRArgRef`][quivers.transpile.ir.IRArgRef]. Literal-scalar
        broadcasts raise
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
        ranges: list[tuple[str, str]] = []
        for size in target_shape:
            rng = _fresh(ctx, "rng", "range")
            ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
            ctx.sb.constraint(
                rng, "chose-alt-child-kinds", "number number"
            )
            ctx.sb.constraint(rng, "ptrace-0", "Cnumber")
            ctx.sb.constraint(rng, "ptrace-1", "T:")
            ctx.sb.constraint(rng, "ptrace-2", "Cnumber")
            lo = _number(ctx, 1)
            hi = _number(ctx, size)
            ctx.sb.edge(rng, lo, "lower")
            ctx.sb.edge(rng, hi, "upper")
            ranges.append((rng, "range"))

        ptrace_idx = 0
        fingerprint_parts: list[str] = []
        kinds: list[str] = []
        for i, (_, kind) in enumerate(ranges):
            ctx.sb.constraint(idx_list, f"ptrace-{ptrace_idx}", f"C{kind}")
            ptrace_idx += 1
            kinds.append(kind)
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
        for vid, _ in ranges:
            ctx.sb.edge(idx_list, vid, "child_of")
        ctx.sb.edge(iv, idx_list, "indices")
        return iv

    # ------------------------------------------------------------------
    # Per-renderer rendering helpers
    # ------------------------------------------------------------------

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
            # Data inputs live in the host's external `.data` file.
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
            # JAGS has no `return`; the host inspects monitored variables
            # directly.
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
        dim and attach it to the surrounding block.

        `via` (when set) names the fibration through which observed
        rows project onto a parent plate: every `IRArgRef` whose name
        is a latent on that parent plate gets wrapped as
        ``<name>[<via>[<loop_var>]]`` to thread the per-observation
        index through the latent assignment."""
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

        aliases = meta.arg_aliases.get(_BACKEND, {})
        # Apply renaming and arithmetic transforms to the IR args.
        renamed_pairs: list[tuple[str, IRArg]] = []
        for arg_name, arg in zip(arg_names, args, strict=False):
            emitted_name = aliases.get(arg_name, arg_name)
            transform = _ALIAS_TRANSFORMS.get(emitted_name)
            if transform is not None and emitted_name != arg_name:
                # Only wrap when the alias actually renamed the arg.
                arg = IRArgTransform(inner=arg, transform=transform)
            renamed_pairs.append((emitted_name, arg))

        # Build the stochastic_relation for the innermost statement.
        sr = self._build_stochastic_relation(
            ctx,
            lhs_name=name,
            target_dist=target_name,
            renamed_pairs=tuple(renamed_pairs),
            plate=plate,
            via=via,
        )

        # Mark every batch-dim plate name as emitted (used by
        # `_dedupe_plate` to disambiguate latent re-uses).
        for dim in plate.batch_dims:
            ctx.emitted_plate_names.add(dim.name)

        # Record this sample's plate info so subsequent refs to `name`
        # index correctly (loop var + event ranges).
        if plate.batch_dims:
            loop_var = f"m_{_dim_name(plate.batch_dims[-1])}"
            ctx.latent_plate_info[name] = (loop_var, plate.event_dims)

        # Wrap in nested `for` loops, one per batch dim (outermost =
        # first batch_dim). When `via` is set and the plate has a
        # single dynamic dim, override the loop var to the canonical
        # `n` JAGS convention.
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
        via: str | None = None,
    ) -> str:
        """Build ``<lhs> ~ <dist>(...)`` as a ``stochastic_relation``.

        `via`, when set, names a per-observation fibration whose
        loop-variable index threads through every ref to a latent
        with a recorded `latent_via` entry."""
        loop_var_names = tuple(
            f"m_{_dim_name(dim)}" for dim in plate.batch_dims
        )
        # The observation plate's loop variable defaults to `n` when
        # the plate is dynamic and `via` is set (canonical JAGS
        # observation idiom).
        if via is not None and len(loop_var_names) == 1:
            loop_var_names = ("n",)
        sr = _fresh(ctx, "sr", "stochastic_relation")
        ctx.sb.constraint(sr, "chose-alt-fingerprint", "~")
        ctx.sb.constraint(
            sr,
            "chose-alt-child-kinds",
            f"{_lhs_kind(plate, loop_var_names)} distribution_call",
        )
        ctx.sb.constraint(sr, "ptrace-0", f"C{_lhs_kind(plate, loop_var_names)}")
        ctx.sb.constraint(sr, "ptrace-1", "T~")
        ctx.sb.constraint(sr, "ptrace-2", "Cdistribution_call")

        lhs = self._build_lhs(ctx, lhs_name, plate, loop_var_names)
        ctx.sb.edge(sr, lhs, "variable")

        dc = self._build_distribution_call(
            ctx, target_dist, renamed_pairs, loop_var_names, via
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

        # Build the index_list. Loop vars are bare identifiers; event
        # dims are `1:E` ranges.
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

        # index_list ptrace: alternating Cchild / T, terminated by last
        # Cchild.
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

    def _range_form(self, ctx: _JAGSCtx, upper: str) -> str:
        """Build a `1:<upper>` range vertex, choosing number vs.
        identifier for the upper bound based on its lexical form."""
        rng = _fresh(ctx, "rng", "range")
        ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
        ctx.sb.constraint(rng, "ptrace-0", "Cnumber")
        ctx.sb.constraint(rng, "ptrace-1", "T:")
        lo = _number(ctx, 1)
        ctx.sb.edge(rng, lo, "lower")
        # Upper is a numeric literal when parseable, else an identifier.
        try:
            upper_val = int(upper)
            hi = _number(ctx, upper_val)
            ctx.sb.constraint(
                rng, "chose-alt-child-kinds", "number number"
            )
            ctx.sb.constraint(rng, "ptrace-2", "Cnumber")
        except ValueError:
            hi = _identifier(ctx, upper)
            ctx.sb.constraint(
                rng, "chose-alt-child-kinds", "number identifier"
            )
            ctx.sb.constraint(rng, "ptrace-2", "Cidentifier")
        ctx.sb.edge(rng, hi, "upper")
        return rng

    def _event_range_form(
        self, ctx: _JAGSCtx, dim: object
    ) -> tuple[str, str]:
        """`1:E` range form for one event dim. Returns (vid, kind)."""
        if isinstance(dim, DimStatic):
            rng = _fresh(ctx, "rng", "range")
            ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
            ctx.sb.constraint(
                rng, "chose-alt-child-kinds", "number number"
            )
            ctx.sb.constraint(rng, "ptrace-0", "Cnumber")
            ctx.sb.constraint(rng, "ptrace-1", "T:")
            ctx.sb.constraint(rng, "ptrace-2", "Cnumber")
            lo = _number(ctx, 1)
            hi = _number(ctx, dim.size)
            ctx.sb.edge(rng, lo, "lower")
            ctx.sb.edge(rng, hi, "upper")
            return rng, "range"
        if isinstance(dim, DimDynamic):
            rng = _fresh(ctx, "rng", "range")
            ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
            ctx.sb.constraint(
                rng, "chose-alt-child-kinds", "number identifier"
            )
            ctx.sb.constraint(rng, "ptrace-0", "Cnumber")
            ctx.sb.constraint(rng, "ptrace-1", "T:")
            ctx.sb.constraint(rng, "ptrace-2", "Cidentifier")
            lo = _number(ctx, 1)
            hi = _identifier(ctx, dim.size_name)
            ctx.sb.edge(rng, lo, "lower")
            ctx.sb.edge(rng, hi, "upper")
            return rng, "range"
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"]
        )

    def _build_distribution_call(
        self,
        ctx: _JAGSCtx,
        target_dist: str,
        renamed_pairs: tuple[tuple[str, IRArg], ...],
        loop_vars: tuple[str, ...],
        via: str | None = None,
    ) -> str:
        """Build ``<dist>(<arg0>, <arg1>, ...)``.

        JAGS positional-only calling convention: emit args in the
        order received. The renamed `arg_names` discriminate `tau`
        from `scale` etc. but JAGS itself sees them positionally.
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

        # Build the argument_list (one child per arg, comma-separated).
        al = self._build_argument_list(
            ctx,
            tuple(arg for _name, arg in renamed_pairs),
            loop_vars,
            via,
        )
        ctx.sb.edge(dc, al, "arguments")
        return dc

    def _build_argument_list(
        self,
        ctx: _JAGSCtx,
        args: tuple[IRArg, ...],
        loop_vars: tuple[str, ...],
        via: str | None = None,
    ) -> str:
        """``arg0, arg1, ...`` as an ``argument_list``."""
        al = _fresh(ctx, "al", "argument_list")
        child_pairs: list[tuple[str, str]] = []
        for arg in args:
            vid, kind = self._render_arg_with_kind(
                ctx, arg, loop_vars, via
            )
            child_pairs.append((vid, kind))

        # Build ptrace + fingerprint for the argument_list.
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
        loop_vars: tuple[str, ...],
        via: str | None = None,
    ) -> tuple[str, str]:
        """Render one arg; return (vid, kind) so the parent can build
        ptrace constraints."""
        if isinstance(arg, IRArgNumber):
            return _number(ctx, arg.value), "number"
        if isinstance(arg, IRArgRef):
            return self._render_ref_with_kind(ctx, arg, loop_vars, via)
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
            return self._render_transform_with_kind(ctx, arg, loop_vars)
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}", [f"arg:{type(arg).__name__}"]
        )

    def _maybe_thread_indices(
        self,
        ctx: _JAGSCtx,
        arg: IRArgRef,
        loop_vars: tuple[str, ...],
        via: str | None,
    ) -> IRArgRef:
        """Possibly rewrite a bare or partially-indexed ref to thread
        a latent's loop var + event ranges + via fibration through.

        Idempotent: rewrites recursively until no further substitution
        applies. Returns the original arg when no rewrite is needed."""
        info = ctx.latent_plate_info.get(arg.name)
        # Compute the loop var that should index this latent.
        if info is not None:
            base_loop_var, event_dims = info
            # If a `via` fibration is active and this latent's via is
            # recorded, wrap the loop var: `<via>[<observe_loop_var>]`.
            via_for_latent = ctx.latent_via.get(arg.name)
            if via_for_latent is not None and loop_vars:
                lv_idx: IRArg = IRArgRef(
                    name=via_for_latent,
                    indices=(IRArgRef(name=loop_vars[0]),),
                )
            elif via is not None and arg.name in ctx.latent_via:
                # Caller-set via overrides if present.
                lv_idx = IRArgRef(
                    name=via,
                    indices=(IRArgRef(name=loop_vars[0] if loop_vars else "n"),),
                )
            else:
                lv_idx = IRArgRef(name=base_loop_var)
            # If `arg` already has indices, treat its first index as
            # the loop var (apply via wrapping to that index instead)
            # and append event ranges after.
            if arg.indices:
                # Rewrite the inner index recursively to apply via /
                # latent-plate threading to nested refs (e.g. `phi[z]`
                # becomes `phi[z[word_idx[n]]]`).
                new_inner: tuple[IRArg, ...] = tuple(
                    self._rewrite_arg(ctx, idx, loop_vars, via)
                    for idx in arg.indices
                )
                # Append event-dim ranges for the latent's event shape.
                event_indices = tuple(
                    _event_range_ir(ed) for ed in event_dims
                )
                new_arg = IRArgRef(
                    name=arg.name,
                    indices=new_inner + event_indices,
                )
                return new_arg
            event_indices = tuple(
                _event_range_ir(ed) for ed in event_dims
            )
            return IRArgRef(
                name=arg.name,
                indices=(lv_idx,) + event_indices,
            )
        return arg

    def _rewrite_arg(
        self,
        ctx: _JAGSCtx,
        arg: IRArg,
        loop_vars: tuple[str, ...],
        via: str | None,
    ) -> IRArg:
        """Recursively rewrite refs inside `arg` per
        `_maybe_thread_indices`."""
        if isinstance(arg, IRArgRef):
            rewritten = self._maybe_thread_indices(
                ctx, arg, loop_vars, via
            )
            if rewritten is not arg:
                return rewritten
            # Recurse into nested indices for further rewrites.
            new_indices = tuple(
                self._rewrite_arg(ctx, idx, loop_vars, via)
                for idx in arg.indices
            )
            return IRArgRef(name=arg.name, indices=new_indices)
        return arg

    def _render_ref_with_kind(
        self,
        ctx: _JAGSCtx,
        arg: IRArgRef,
        loop_vars: tuple[str, ...] = (),
        via: str | None = None,
    ) -> tuple[str, str]:
        """Render a bound-name reference. Bare names emit as
        ``identifier``; indexed references emit as ``indexed_variable``
        with the index expressions as children.

        When `arg.name` is a previously-bound latent on a batch plate
        (tracked via `ctx.latent_plate_info`), the bare ref is
        rewritten to include the latent's loop var and any event
        ranges. When `arg.name` has a recorded `latent_via` mapping
        and the caller's `via` is set, the latent's loop var is
        further wrapped as ``<via>[<observe_loop_var>]``."""
        # Apply latent-plate / fibration substitution.
        rewritten = self._maybe_thread_indices(ctx, arg, loop_vars, via)
        if rewritten is not arg:
            return self._render_ref_with_kind(ctx, rewritten, loop_vars, via)
        # Sentinel: encode a JAGS `1:N` range.
        if arg.name.startswith(_RANGE_SENTINEL_PREFIX):
            upper = arg.name[len(_RANGE_SENTINEL_PREFIX) :]
            return self._range_form(ctx, upper), "range"
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
            vid, kind = self._render_arg_with_kind(
                ctx, idx, loop_vars, via
            )
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
        # The Truncated wrapper renders its inner distribution call
        # followed by `T(L, U)`. JAGS uses `T(low, high)` for
        # truncation; the surrounding `_build_distribution_call` for
        # the outer Truncated step already wrote the distribution
        # call; here we're rendering the inner family-ref which
        # supplies the bounds via the wrapper's own args.
        inner_family = decl.init_family.family
        inner_meta = FAMILY_META.get(inner_family)
        if inner_meta is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [f"family:wrapper-inner-unknown:{inner_family}"],
            )
        # For JAGS we emit the inner distribution call as an
        # `distribution_call` vertex; the outer wrapper consumes it.
        target_inner = inner_meta.target_names.get(_BACKEND)
        if target_inner is None:
            raise UnsupportedConstruct(
                f"qvr-{_BACKEND}",
                [f"family:no-target-name:{inner_family}"],
            )
        # Build positional args from the morphism's init_family clause.
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
            ctx, target_inner, renamed_pairs, ()
        )
        return dc, "distribution_call"

    def _render_transform_with_kind(
        self,
        ctx: _JAGSCtx,
        arg: IRArgTransform,
        loop_vars: tuple[str, ...],
    ) -> tuple[str, str]:
        """Render an
        [`IRArgTransform`][quivers.transpile.renderers._base.IRArgTransform]:
        ``inv_square(x) -> 1/(x*x)`` etc.

        Emits a ``binary_expression`` tree using the JAGS grammar's
        arithmetic operators.
        """
        inner_vid, inner_kind = self._render_arg_with_kind(
            ctx, arg.inner, loop_vars
        )
        if arg.transform == "inv_square":
            # 1 / (x * x)
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
            # function_call: log(x) / exp(x)
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
            # Without a for_loop wrapper, the stochastic_relation
            # sits directly under the model_block; the kind tag is
            # already correct from the caller.
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
            # Inline-format: `{ stmt }` on one line.
            ctx.sb.constraint(blk, "interstitial-0", "{ ")
            ctx.sb.constraint(blk, "interstitial-1", " }")
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
            rng = _fresh(ctx, "rng", "range")
            ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
            ctx.sb.constraint(
                rng, "chose-alt-child-kinds", "number number"
            )
            ctx.sb.constraint(rng, "ptrace-0", "Cnumber")
            ctx.sb.constraint(rng, "ptrace-1", "T:")
            ctx.sb.constraint(rng, "ptrace-2", "Cnumber")
            lo = _number(ctx, 1)
            hi = _number(ctx, dim.size)
            ctx.sb.edge(rng, lo, "lower")
            ctx.sb.edge(rng, hi, "upper")
            return rng, "range"
        if isinstance(dim, DimDynamic):
            rng = _fresh(ctx, "rng", "range")
            ctx.sb.constraint(rng, "chose-alt-fingerprint", ":")
            ctx.sb.constraint(
                rng, "chose-alt-child-kinds", "number identifier"
            )
            ctx.sb.constraint(rng, "ptrace-0", "Cnumber")
            ctx.sb.constraint(rng, "ptrace-1", "T:")
            ctx.sb.constraint(rng, "ptrace-2", "Cidentifier")
            lo = _number(ctx, 1)
            hi = _identifier(ctx, dim.size_name)
            ctx.sb.edge(rng, lo, "lower")
            ctx.sb.edge(rng, hi, "upper")
            return rng, "range"
        raise UnsupportedConstruct(
            f"qvr-{_BACKEND}", [f"dim:{type(dim).__name__}"]
        )

    # ------------------------------------------------------------------
    # Deterministic / score emission
    # ------------------------------------------------------------------

    def _emit_deterministic(
        self, ctx: _JAGSCtx, node: IRDeterministic
    ) -> None:
        """JAGS deterministic relation: ``<name> <- <expr>``.

        Without per-let renderer support for the full LetExprNode
        tree, fall back to a placeholder identifier so the structure
        round-trips through the grammar but the host can inspect the
        binding."""
        dr = _fresh(ctx, "dr", "deterministic_relation")
        ctx.sb.constraint(dr, "chose-alt-fingerprint", "<-")
        ctx.sb.constraint(
            dr, "chose-alt-child-kinds", "identifier identifier"
        )
        ctx.sb.constraint(dr, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(dr, "ptrace-1", "T<-")
        ctx.sb.constraint(dr, "ptrace-2", "Cidentifier")
        var = _identifier(ctx, node.name)
        ctx.sb.edge(dr, var, "variable")
        # Placeholder value: emit `__placeholder__` until LetExpr
        # rendering is wired in for JAGS. The let-bind binding still
        # appears in the model block so the host has the name.
        val = _identifier(ctx, "__placeholder__")
        ctx.sb.edge(dr, val, "value")
        if ctx.current_block is not None:
            ctx.sb.edge(ctx.current_block, dr, "child_of")
            ctx.block_children.setdefault(ctx.current_block, []).append(
                "deterministic_relation"
            )

    def _emit_score(self, ctx: _JAGSCtx, node: IRScore) -> None:
        """JAGS score: the "zeros trick" pairs an observed zero with a
        Poisson rate of ``-<expr>`` so the log-likelihood contribution
        becomes ``<expr>``. Without a full LetExpr renderer for JAGS,
        emit a placeholder deterministic relation that names the score
        variable; the host wires the zeros trick at the data level."""
        dr = _fresh(ctx, "dr", "deterministic_relation")
        ctx.sb.constraint(dr, "chose-alt-fingerprint", "<-")
        ctx.sb.constraint(
            dr, "chose-alt-child-kinds", "identifier identifier"
        )
        ctx.sb.constraint(dr, "ptrace-0", "Cidentifier")
        ctx.sb.constraint(dr, "ptrace-1", "T<-")
        ctx.sb.constraint(dr, "ptrace-2", "Cidentifier")
        var = _identifier(ctx, node.name)
        ctx.sb.edge(dr, var, "variable")
        val = _identifier(ctx, "__placeholder__")
        ctx.sb.edge(dr, val, "value")
        if ctx.current_block is not None:
            ctx.sb.edge(ctx.current_block, dr, "child_of")
            ctx.block_children.setdefault(ctx.current_block, []).append(
                "deterministic_relation"
            )

    # ------------------------------------------------------------------
    # Plate-name disambiguation
    # ------------------------------------------------------------------

    def _dedupe_plate(
        self,
        ctx: _JAGSCtx,
        plate: Plate,
        latent_name: str,
    ) -> Plate:
        """Return a Plate whose batch_dim names get a ``_<latent>``
        suffix only when the name has already been emitted in this
        render call. Mirrors the NumPyro / Stan convention for the
        LDA-style marginalize-then-reuse pattern."""
        seen = ctx.emitted_plate_names
        new_batch: list[object] = []
        for dim in plate.batch_dims:
            renamed = (
                f"{_dim_name(dim)}_{latent_name}"
                if _dim_name(dim) in seen
                else _dim_name(dim)
            )
            if isinstance(dim, DimStatic):
                new_batch.append(DimStatic(size=dim.size, name=renamed))
            elif isinstance(dim, DimDynamic):
                new_batch.append(
                    DimDynamic(size_name=dim.size_name, name=renamed)
                )
            else:
                new_batch.append(dim)
        return Plate(
            event_dims=plate.event_dims,
            batch_dims=tuple(new_batch),
        )

    # ------------------------------------------------------------------
    # model_block finalisation
    # ------------------------------------------------------------------

    def _finalise_model_block(self, ctx: _JAGSCtx) -> None:
        """Set the model_block's child-kind list, ptrace, and
        layout-controlling interstitials so the pretty-printer emits
        ``model {\\n  <stmts>\\n}`` form."""
        mb = ctx.model_block
        children = ctx.block_children.get(mb, [])
        # ptrace sequence: Tmodel, T{, then one C<kind> per child, T}.
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
        # Layout: outer block opens on a newline with two-space indent,
        # children separated by newlines, closer on its own line.
        ctx.sb.constraint(mb, "interstitial-0", "model {\n  ")
        for i in range(len(children) - 1):
            ctx.sb.constraint(mb, f"interstitial-{i + 1}", "\n  ")
        ctx.sb.constraint(
            mb, f"interstitial-{len(children)}", "\n}"
        )


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
        #: For every previously-bound latent name, record the
        #: (loop-var, event_dims) used at its sample site. Subsequent
        #: refs index into the latent with that loop-var + event
        #: ranges, e.g. `theta -> ('m_Doc', (DimStatic(3, 'Topic'),))`.
        self.latent_plate_info: dict[
            str, tuple[str, tuple[object, ...]]
        ] = {}
        #: For each latent on a per-observation plate, record the
        #: fibration that maps the observation row to its parent plate
        #: index (e.g. `z -> 'word_idx'`). Observe steps then rewrite
        #: bare refs to that latent as `z[word_idx[n]]`.
        self.latent_via: dict[str, str] = {}


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


#: Sentinel name prefix used by the JAGS renderer to encode a `1:N`
#: range as an IRArgRef. The `_render_ref_with_kind` path inspects the
#: name prefix and emits a `range` vertex instead of an
#: `indexed_variable`.
_RANGE_SENTINEL_PREFIX: str = "__jags_range__:"


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
    """Look up the grammar kind of `vid` via the schema."""
    for v in ctx.sb._schema.vertices if hasattr(ctx.sb, "_schema") else []:
        if v.id == vid:
            return v.kind
    # SchemaBuilder lacks a public vertex-kind lookup. We track the
    # kind via the for_loop / stochastic_relation / deterministic
    # naming convention on the fresh id.
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
