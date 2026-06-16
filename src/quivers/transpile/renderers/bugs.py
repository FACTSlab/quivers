"""`BUGSRenderer`: lower the transpile IR to BUGS source.

BUGS programs have a single `model { ... }` block containing
stochastic (`~`), deterministic (`<-`), and `for (i in 1:N) { ... }`
relations. BUGS has no separate data / parameters blocks: every
variable that appears on the LHS of `~` is implicitly declared by
that relation; every variable that appears only on the RHS or only
in an LHS index list is an exogenous data input the caller supplies
through the BUGS data list.

The renderer follows the structural protocol of
[`RendererBase`][quivers.transpile.renderers._base.RendererBase]:

* [`declare`][quivers.transpile.renderers.bugs.BUGSRenderer.declare]
  is a no-op outside the `data` block; BUGS declarations are
  implicit. Data-block declarations are also no-op: BUGS reads
  data from the calling environment's data list.
* [`sample`][quivers.transpile.renderers.bugs.BUGSRenderer.sample]
  emits one nested `for (m_<axis> in 1:N_<axis>)` per batch dim of
  the plate, with a `<lhs>[m_0, m_1, ..., 1:E_0, 1:E_1, ...]` LHS
  and the family's distribution call on the RHS, indexing every
  arg whose plate overlaps the surrounding batch dims.
* [`marginalize`][quivers.transpile.renderers.bugs.BUGSRenderer.marginalize]
  lowers [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] to
  an explicit `IRSample(latent)` followed by the scope body. BUGS
  supports discrete latents natively, so explicit sampling is the
  right idiom.
* [`broadcast`][quivers.transpile.renderers.bugs.BUGSRenderer.broadcast]
  raises [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
  with kind `"arg:broadcast"`: BUGS has no native scalar-to-vector
  broadcast op; the caller must pre-bind a vector data input.

`FAMILY_META[family].target_names["bugs"]` supplies the BUGS
distribution name (`"dnorm"`, `"ddirch"`, ...). Per-family argument
renames live in `FAMILY_META[family].arg_aliases["bugs"]`; when a
rename targets a parameterisation that needs arithmetic conversion
(BUGS `tau = 1/(scale * scale)` for Normal's `scale -> tau`), the
[`_ALIAS_TRANSFORMS`][quivers.transpile.renderers.bugs._ALIAS_TRANSFORMS]
table on the renderer keys the transform on the renamed target name
and the renderer wraps the arg in an
[`IRArgTransform`][quivers.transpile.renderers._base.IRArgTransform]
before emitting `1 / (scale * scale)`.

[`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] arguments
(wrapper families like `Truncated`) render via the BUGS
`d<family>(args) T(lower, upper)` truncation idiom: the renderer
inlines the referenced morphism's `~ Family(args)` clause as the
distribution call and appends a `truncation` child to the
`stochastic_relation` carrying `(lower, upper)`.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import panproto

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgName,
    DrawArgScalar,
    MorphismDecl,
    MorphismInitFamily,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import target_protocol
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
from quivers.transpile.ir import (
    ConstraintSpec,
    Dim,
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
    assert_no_dangling_refs,
)


#: Per-renderer alias-transform table. Keys are the *renamed* arg
#: names from `FAMILY_META[family].arg_aliases["bugs"]`; values are
#: the arithmetic transform the renderer applies before emission.
#: BUGS uses precision (`tau = 1/(scale * scale)`) where torch's
#: `Normal` uses scale, so `arg_aliases["bugs"]["scale"] = "tau"`
#: pairs with `_ALIAS_TRANSFORMS["tau"] = "inv_square"`.
_ALIAS_TRANSFORMS: dict[str, str] = {
    "tau": "inv_square",
}


@dataclasses.dataclass
class _BugsCtx(_RenderCtx):
    """BUGS-renderer-internal carrier extending `_RenderCtx`.

    Adds the declaration-plate table (axis-name lookup for index
    emission), the current `via` fibration in scope (the active
    scope's [`IRObserve.via`][quivers.transpile.ir.IRObserve.via]
    when rendering arg indices), and the enclosing plate / loop-
    name pair so an arg's index expansion can resolve loop
    variables by axis name.
    """

    decl_plates: dict[str, Plate] = dataclasses.field(default_factory=dict)
    via: str | None = None
    enclosing_plate: Plate | None = None
    enclosing_loop_names: tuple[str, ...] = ()
    block_id: str = ""


class BUGSRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to BUGS.

    Implements the four
    [`RendererBase`][quivers.transpile.renderers._base.RendererBase]
    dispatch points (`declare`, `sample`, `marginalize`, `broadcast`)
    plus `target_protocol` and overrides `render` to install the
    `model { ... }` block prologue and to wire the BUGS-specific
    declaration-plate table that drives slice-emission for vector
    arg positions.
    """

    target: str = "bugs"

    def target_protocol(self) -> panproto.Protocol:
        """Return the panproto protocol for the BUGS tree-sitter grammar."""
        return target_protocol("bugs")

    # ------------------------------------------------------------------
    # Top-level render.
    # ------------------------------------------------------------------

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Emit the BUGS source schema for `ir`.

        Wraps the IR walk in a `source_file -> model_block` shell so
        the emitted bytes are a complete BUGS file. The model_block's
        `ptrace-0 = Tmodel` constraint forces the emitter to render
        the optional `model` keyword.
        """
        assert_no_dangling_refs(ir)
        self._reject_list_args(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        ctx = _BugsCtx(sb=sb, morphisms={}, lets={})
        # Populate decl_plates for every IRDataInput and IRSample /
        # IRObserve / IRDeterministic / IRMarginalize-latent.
        self._populate_decl_plates(ir, ctx)
        src_id = self._fresh(ctx, "src")
        sb.vertex(src_id, "source_file")
        mb_id = self._fresh(ctx, "mb")
        sb.vertex(mb_id, "model_block")
        sb.constraint(mb_id, "ptrace-0", "Tmodel")
        sb.edge(src_id, mb_id, "model_block")
        ctx.block_id = mb_id
        for node in ir.body:
            self._dispatch_bugs_node(ctx, node)
        return sb.build()

    def _populate_decl_plates(self, ir: IRProgram, ctx: _BugsCtx) -> None:
        for inp in ir.inputs:
            ctx.decl_plates[inp.name] = inp.plate
        for node in self._all_nodes(ir):
            if isinstance(node, IRSample):
                ctx.decl_plates[node.name] = node.plate
            elif isinstance(node, IRObserve):
                ctx.decl_plates[node.name] = node.plate
            elif isinstance(node, IRDeterministic):
                ctx.decl_plates[node.name] = node.plate
            elif isinstance(node, IRMarginalize):
                ctx.decl_plates[node.latent] = node.plate

    def _all_nodes(self, ir: IRProgram) -> list[IRNode]:
        out: list[IRNode] = []
        for node in ir.body:
            self._collect_nodes(node, out)
        return out

    def _collect_nodes(self, node: IRNode, out: list[IRNode]) -> None:
        out.append(node)
        if isinstance(node, IRMarginalize):
            for inner in node.scope:
                self._collect_nodes(inner, out)

    # ------------------------------------------------------------------
    # Dispatch.
    # ------------------------------------------------------------------

    def _dispatch_bugs_node(self, ctx: _BugsCtx, node: IRNode) -> None:
        if isinstance(node, IRDataInput):
            # BUGS has no data block syntax; data inputs flow through
            # the caller's data list. No emission.
            return
        if isinstance(node, IRSample):
            self._emit_sample_node(ctx, node, loop_suffix="")
            return
        if isinstance(node, IRObserve):
            self._emit_observe_node(ctx, node)
            return
        if isinstance(node, IRDeterministic):
            # BUGS deterministic `<-` is rendered via the let-expr
            # tree; not exercised by the current acceptance gallery.
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["node:IRDeterministic: not yet wired"],
            )
        if isinstance(node, IRScore):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["node:IRScore: not yet wired"],
            )
        if isinstance(node, IRMarginalize):
            self._emit_marginalize_node(ctx, node)
            return
        if isinstance(node, IRReturn):
            # BUGS has no `return`; the model defines the joint and
            # the caller pulls posterior values out via the
            # inference engine's monitor list.
            return
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"node:{type(node).__name__}"],
        )

    # ------------------------------------------------------------------
    # Required RendererBase abstract methods (the dispatch points the
    # `Renderer` Protocol mandates). The BUGS renderer routes through
    # its `_emit_*_node` helpers internally; these forward.
    # ------------------------------------------------------------------

    def declare(
        self,
        ctx: _RenderCtx,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """No-op: BUGS has no declaration syntax distinct from `~` / data."""
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
        """Emit a `~` statement with surrounding `for` loops.

        BUGS treats observed and latent draws identically at the
        syntactic level: the caller's data list pins the value of
        observed variables and the same `~` line carries the
        likelihood factor.
        """
        del observed
        if not isinstance(ctx, _BugsCtx):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["ctx:wrong-type"],
            )
        node = IRSample(
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            constraint=constraint,
            plate=plate,
        )
        self._emit_sample_node(ctx, node, loop_suffix="")
        return ""

    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Lower [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        to an explicit latent draw followed by the scope body.

        BUGS supports discrete latents natively (it samples them
        in the chain), so the explicit-latent rewrite is the right
        idiom.
        """
        if not isinstance(ctx, _BugsCtx):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["ctx:wrong-type"],
            )
        self._emit_marginalize_node(ctx, node)
        return ""

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Raise: BUGS cannot broadcast a literal scalar to a vector
        position; the caller must pre-bind a vector data input."""
        del ctx, value, target_shape
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            ["arg:broadcast"],
        )

    # ------------------------------------------------------------------
    # List / matrix literal args: BUGS has no inline literal form.
    # ------------------------------------------------------------------

    def render_list(self, arg: IRArgList) -> SchemaFragment:
        """Raise: BUGS does not parse `[a, b, c]` in arg position."""
        del arg
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            ["arg:list-literal"],
        )

    def render_matrix(self, arg: IRArgMatrix) -> SchemaFragment:
        """Raise: BUGS does not parse `[[a, b], ...]` in arg position."""
        del arg
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            ["arg:matrix-literal"],
        )

    def _reject_list_args(self, ir: IRProgram) -> None:
        """Raise on the first
        [`IRArgList`][quivers.transpile.ir.IRArgList] /
        [`IRArgMatrix`][quivers.transpile.ir.IRArgMatrix] anywhere in
        the program's distribution-call args."""
        for node in self._all_nodes(ir):
            args = getattr(node, "args", ())
            for arg in args:
                self._check_no_literal(arg)

    def _check_no_literal(self, arg: IRArg) -> None:
        if isinstance(arg, IRArgList):
            self.render_list(arg)
        if isinstance(arg, IRArgMatrix):
            self.render_matrix(arg)
        if isinstance(arg, IRArgBroadcast):
            self._check_no_literal(arg.value)
        if isinstance(arg, IRArgRef):
            for idx in arg.indices:
                self._check_no_literal(idx)

    # ------------------------------------------------------------------
    # Family lookup.
    # ------------------------------------------------------------------

    def _lookup_family(self, family: str) -> FamilyMeta:
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"family:{family}: not in FAMILY_META"],
            )
        if "bugs" not in meta.target_names:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"family:{family}: no BUGS target name"],
            )
        return meta

    # ------------------------------------------------------------------
    # Sample / observe / marginalize emission.
    # ------------------------------------------------------------------

    def _emit_sample_node(
        self,
        ctx: _BugsCtx,
        node: IRSample,
        *,
        loop_suffix: str,
    ) -> None:
        """Render an [`IRSample`][quivers.transpile.ir.IRSample] node.

        A first-arg [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        signals a wrapper-family encoding (`Truncated`, `Mixture`,
        ...) whose runtime distribution comes from the referenced
        morphism's `init_family`. The BUGS renderer handles the
        Truncated case as the `T(lower, upper)` idiom; other
        wrappers route through the generic path (which will fail
        downstream if the wrapper has no BUGS encoding).
        """
        if _is_wrapper_family_call(node.args):
            self._emit_truncated(ctx, node)
            return
        self._emit_relation(
            ctx,
            name=node.name,
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            plate=node.plate,
            via=None,
            loop_suffix=loop_suffix,
        )

    def _emit_observe_node(self, ctx: _BugsCtx, node: IRObserve) -> None:
        """Render an [`IRObserve`][quivers.transpile.ir.IRObserve] node.

        Observes carry an optional `via` fibration: when present,
        index-references in the args whose target sits on a
        different plate are rewritten as `<name>[via[loop]]`. This
        is the BUGS idiom for "observation `n` maps to latent group
        `via[n]`".
        """
        if _is_wrapper_family_call(node.args):
            self._emit_truncated(ctx, IRSample(
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                constraint=node.constraint,
                plate=node.plate,
            ))
            return
        self._emit_relation(
            ctx,
            name=node.name,
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            plate=node.plate,
            via=node.via,
            loop_suffix="",
        )

    def _emit_marginalize_node(
        self, ctx: _BugsCtx, node: IRMarginalize
    ) -> None:
        """Lower marginalize to explicit-latent + scope body.

        The latent's loop variable is suffixed with the latent
        name so it stays distinct from any prior loop over the same
        plate axis."""
        latent_sample = IRSample(
            name=node.latent,
            family=node.family,
            args=node.args,
            arg_names=node.arg_names,
            constraint=node.constraint,
            plate=node.plate,
        )
        self._emit_sample_node(
            ctx, latent_sample, loop_suffix=f"_{node.latent}"
        )
        for inner in node.scope:
            self._dispatch_bugs_node(ctx, inner)

    def _emit_truncated(self, ctx: _BugsCtx, node: IRSample) -> None:
        """Emit `<lhs> ~ d<base>(base_args) T(lo, hi)`."""
        family_ref = node.args[0]
        if not isinstance(family_ref, IRArgFamilyRef):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["truncated:expected IRArgFamilyRef at arg 0"],
            )
        decl = ctx.morphisms.get(family_ref.name)
        if not isinstance(decl, MorphismDecl):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"truncated:base:{family_ref.name}: morphism not bound"],
            )
        init = decl.init
        if not isinstance(init, MorphismInitFamily):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    f"truncated:base:{family_ref.name}: init is not "
                    f"`~ Family(args)`"
                ],
            )
        base_family = init.family
        base_args = tuple(
            _draw_arg_to_ir(a) for a in (init.args or ())
        )
        base_meta = self._lookup_family(base_family)
        base_arg_names = self._infer_arg_names(base_meta, base_args)
        if len(node.args) < 3:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["truncated:missing-bounds"],
            )
        lo, hi = node.args[1], node.args[2]
        self._emit_relation(
            ctx,
            name=node.name,
            family=base_family,
            args=base_args,
            arg_names=base_arg_names,
            plate=node.plate,
            via=None,
            loop_suffix="",
            truncation=(lo, hi),
        )

    # ------------------------------------------------------------------
    # Core relation emitter.
    # ------------------------------------------------------------------

    def _emit_relation(
        self,
        ctx: _BugsCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        via: str | None,
        loop_suffix: str,
        truncation: tuple[IRArg, IRArg] | None = None,
    ) -> None:
        """Open the for-loops over `plate.batch_dims`, then emit the
        `<lhs> ~ <dist>(args)` line."""
        meta = self._lookup_family(family)
        loop_names = self._loop_names(plate, loop_suffix)
        body_id = self._open_loops(ctx, ctx.block_id, plate, loop_names)
        # Stash enclosing-plate + via for arg emission.
        prev_via = ctx.via
        prev_plate = ctx.enclosing_plate
        prev_loops = ctx.enclosing_loop_names
        ctx.via = via
        ctx.enclosing_plate = plate
        ctx.enclosing_loop_names = loop_names
        try:
            sr_id = self._fresh(ctx, "sr")
            ctx.sb.vertex(sr_id, "stochastic_relation")
            ctx.sb.edge(body_id, sr_id, "stochastic_relation")
            lhs_id = self._emit_lhs(
                ctx, name, plate, loop_names
            )
            ctx.sb.edge(sr_id, lhs_id, "variable")
            dc_id = self._emit_distribution_call(
                ctx, meta, args, arg_names
            )
            ctx.sb.edge(sr_id, dc_id, "distribution")
            if truncation is not None:
                trunc_id = self._emit_truncation(
                    ctx, truncation[0], truncation[1]
                )
                ctx.sb.edge(sr_id, trunc_id, "truncation")
        finally:
            ctx.via = prev_via
            ctx.enclosing_plate = prev_plate
            ctx.enclosing_loop_names = prev_loops

    # ------------------------------------------------------------------
    # Loop emission.
    # ------------------------------------------------------------------

    def _loop_names(
        self, plate: Plate, suffix: str
    ) -> tuple[str, ...]:
        """Return the loop-variable name for each `plate.batch_dim`."""
        return tuple(f"m_{dim.name}{suffix}" for dim in plate.batch_dims)

    def _open_loops(
        self,
        ctx: _BugsCtx,
        parent: str,
        plate: Plate,
        loop_names: tuple[str, ...],
    ) -> str:
        """Open one `for_loop` per batch dim. Return the innermost
        block id that the caller hangs the `~` line on."""
        current = parent
        for dim, name in zip(plate.batch_dims, loop_names, strict=True):
            for_id = self._fresh(ctx, "for")
            ctx.sb.vertex(for_id, "for_loop")
            ctx.sb.edge(current, for_id, "for_loop")
            var_id = self._fresh(ctx, "id")
            ctx.sb.vertex(var_id, "identifier")
            ctx.sb.constraint(var_id, "literal-value", name)
            ctx.sb.edge(for_id, var_id, "variable")
            rng_id = self._fresh(ctx, "rng")
            ctx.sb.vertex(rng_id, "range")
            ctx.sb.edge(for_id, rng_id, "range")
            lo_id = self._fresh(ctx, "num")
            ctx.sb.vertex(lo_id, "number")
            ctx.sb.constraint(lo_id, "literal-value", "1")
            ctx.sb.edge(rng_id, lo_id, "lower")
            hi_id = self._fresh(ctx, "num")
            upper_text = self._dim_upper_text(dim)
            self._emit_upper_literal(ctx, hi_id, upper_text)
            ctx.sb.edge(rng_id, hi_id, "upper")
            body_id = self._fresh(ctx, "blk")
            ctx.sb.vertex(body_id, "block")
            ctx.sb.edge(for_id, body_id, "body")
            current = body_id
        return current

    def _emit_upper_literal(
        self, ctx: _BugsCtx, vid: str, text: str
    ) -> None:
        """Emit either a number or an identifier vertex for the loop
        upper bound, depending on whether `text` is a digit string."""
        if text.lstrip("-").isdigit():
            ctx.sb.vertex(vid, "number")
        else:
            ctx.sb.vertex(vid, "identifier")
        ctx.sb.constraint(vid, "literal-value", text)

    def _dim_upper_text(self, dim: Dim) -> str:
        """Return the upper bound text for the for-loop range:
        either the static cardinality or the dynamic size name."""
        size = getattr(dim, "size", None)
        if size is not None:
            return str(int(size))
        size_name = getattr(dim, "size_name", None)
        if size_name is not None:
            return str(size_name)
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"dim:{type(dim).__name__}: unknown shape"],
        )

    # ------------------------------------------------------------------
    # LHS emission.
    # ------------------------------------------------------------------

    def _emit_lhs(
        self,
        ctx: _BugsCtx,
        name: str,
        plate: Plate,
        loop_names: tuple[str, ...],
    ) -> str:
        """Emit the LHS variable, indexed by batch loop names plus
        event-axis slices."""
        event_slices = tuple(
            self._dim_upper_text(d) for d in plate.event_dims
        )
        if not loop_names and not event_slices:
            return self._emit_bare_identifier(ctx, name)
        return self._emit_indexed_name(
            ctx,
            name,
            tuple(_LoopRef(n) for n in loop_names),
            event_slices,
        )

    # ------------------------------------------------------------------
    # Distribution-call emission.
    # ------------------------------------------------------------------

    def _emit_distribution_call(
        self,
        ctx: _BugsCtx,
        meta: FamilyMeta,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str:
        """Emit the `<dist_name>(<args>)` distribution_call."""
        dc_id = self._fresh(ctx, "dc")
        ctx.sb.vertex(dc_id, "distribution_call")
        dist_name = meta.target_names["bugs"]
        name_id = self._fresh(ctx, "id")
        ctx.sb.vertex(name_id, "identifier")
        ctx.sb.constraint(name_id, "literal-value", dist_name)
        ctx.sb.edge(dc_id, name_id, "name")
        if not args:
            return dc_id
        al_id = self._fresh(ctx, "al")
        ctx.sb.vertex(al_id, "argument_list")
        ctx.sb.edge(dc_id, al_id, "arguments")
        renames = meta.arg_aliases.get("bugs", {})
        for arg, aname in zip(args, arg_names, strict=True):
            wrapped = self._apply_alias_transform(arg, aname, renames)
            receiver_event_rank = self._receiver_event_rank(meta, aname)
            child_id = self._emit_arg(
                ctx, wrapped, receiver_event_rank=receiver_event_rank
            )
            ctx.sb.edge(al_id, child_id, self._arg_edge_kind(wrapped))
        return dc_id

    def _receiver_event_rank(
        self, meta: FamilyMeta, arg_name: str
    ) -> int:
        """Return the receiver arg-constraint's event rank.

        Reads from `meta.distribution_class.arg_constraints[arg_name]`
        when the table is class-level. Property-form
        `arg_constraints` returns rank 0 (the renderer has no
        sentinel; lack of slice in the emit is conservative)."""
        cls_attr = meta.distribution_class.arg_constraints
        if not isinstance(cls_attr, dict):
            return 0
        expected = cls_attr.get(arg_name)
        if expected is None:
            return 0
        return int(getattr(expected, "event_dim", 0))

    def _apply_alias_transform(
        self, arg: IRArg, arg_name: str, renames: dict[str, str]
    ) -> IRArg:
        """If `renames[arg_name]` targets a name with an arithmetic
        transform in [`_ALIAS_TRANSFORMS`][quivers.transpile.renderers.bugs._ALIAS_TRANSFORMS],
        wrap `arg` in
        [`IRArgTransform`][quivers.transpile.renderers._base.IRArgTransform];
        otherwise return `arg` unchanged."""
        target_name = renames.get(arg_name)
        if target_name is None:
            return arg
        transform = _ALIAS_TRANSFORMS.get(target_name)
        if transform is None:
            return arg
        return IRArgTransform(inner=arg, transform=_as_transform(transform))

    def _arg_edge_kind(self, arg: IRArg) -> str:
        if isinstance(arg, IRArgNumber):
            return "number"
        if isinstance(arg, IRArgTransform):
            return "binary_expression"
        if isinstance(arg, IRArgRef):
            decl_plate = self._lookup_decl_plate_if_known(arg)
            if decl_plate is None:
                return "indexed_variable" if arg.indices else "identifier"
            has_loop = bool(decl_plate.batch_dims)
            has_event = bool(decl_plate.event_dims)
            return (
                "indexed_variable"
                if arg.indices or has_loop or has_event
                else "identifier"
            )
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"arg:{type(arg).__name__}"],
        )

    def _lookup_decl_plate_if_known(self, arg: IRArgRef) -> Plate | None:
        # Class-method only used for edge kind selection; full lookup
        # happens during _emit_arg via context. The arg's name might
        # not be in decl_plates if it's a function-call etc.; return
        # None to fall back to "no slicing".
        return None

    # ------------------------------------------------------------------
    # Arg emission.
    # ------------------------------------------------------------------

    def _emit_arg(
        self,
        ctx: _BugsCtx,
        arg: IRArg,
        *,
        receiver_event_rank: int = 0,
    ) -> str:
        if isinstance(arg, IRArgNumber):
            return self._emit_number(ctx, arg.value)
        if isinstance(arg, IRArgRef):
            return self._emit_ref(
                ctx, arg, receiver_event_rank=receiver_event_rank
            )
        if isinstance(arg, IRArgTransform):
            return self._emit_transform(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self.broadcast(ctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self.render_list(arg)
        if isinstance(arg, IRArgMatrix):
            return self.render_matrix(arg)
        if isinstance(arg, IRArgFamilyRef):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    "arg:family-ref: BUGS wrappers inline via "
                    "truncation idiom; not legal as a free arg"
                ],
            )
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"arg:{type(arg).__name__}"],
        )

    def _emit_number(self, ctx: _BugsCtx, value: float) -> str:
        nid = self._fresh(ctx, "num")
        ctx.sb.vertex(nid, "number")
        text = str(int(value)) if float(value).is_integer() else repr(value)
        ctx.sb.constraint(nid, "literal-value", text)
        return nid

    def _emit_bare_identifier(self, ctx: _BugsCtx, name: str) -> str:
        nid = self._fresh(ctx, "id")
        ctx.sb.vertex(nid, "identifier")
        ctx.sb.constraint(nid, "literal-value", name)
        return nid

    def _emit_ref(
        self,
        ctx: _BugsCtx,
        ref: IRArgRef,
        *,
        receiver_event_rank: int = 0,
    ) -> str:
        """Emit a name reference, with auto-derived indices.

        The full BUGS index list for a referenced variable is:

        1. The user-supplied bracket indices (`ref.indices`),
           recursively rendered. They fill the leftmost batch_dim
           positions of the referenced variable.
        2. Auto-leading: one entry per UNFILLED batch_dim of the
           referenced variable's declaration plate, drawn from the
           surrounding loop variables by axis-name lookup. When the
           axis does not match any enclosing loop and the enclosing
           observe has a `via` fibration in scope, the axis renders
           as `via[enclosing_loop]`.
        3. One `1:E` slice per event dim of the referenced
           variable's declaration plate. When the referenced
           variable has no event dims but the *receiver* arg
           position expects a vector (`receiver_event_rank > 0`),
           the renderer emits one slice per receiver event rank
           using the receiver's enclosing-plate event-dim sizes
           (e.g. `alpha` referenced in a Dirichlet position emits
           `alpha[1:K]` where K is the surrounding Dirichlet's
           event dim).
        """
        decl_plate = ctx.decl_plates.get(ref.name)
        # Decide whether the reference is "scalar-into-vector" (the
        # referenced var has no declared event_dims AND no
        # batch_dims; the receiver expects a vector). Emit
        # receiver-driven slice in that case.
        if (
            decl_plate is not None
            and not decl_plate.event_dims
            and not decl_plate.batch_dims
            and receiver_event_rank > 0
            and not ref.indices
        ):
            slice_uppers = self._receiver_slice_uppers(
                ctx, receiver_event_rank
            )
            if slice_uppers:
                return self._emit_indexed_from_pieces(
                    ctx, ref.name, (), slice_uppers
                )
        if decl_plate is None:
            if not ref.indices:
                return self._emit_bare_identifier(ctx, ref.name)
            indices = tuple(self._emit_index_child(ctx, ix) for ix in ref.indices)
            kinds = tuple(self._index_child_kind(ix) for ix in ref.indices)
            return self._emit_indexed_with_pre_emitted(
                ctx, ref.name, indices, kinds, ()
            )
        # User indices fill the leftmost batch_dim positions.
        user_pieces = tuple(
            self._user_index_piece(ctx, ix) for ix in ref.indices
        )
        # Auto-leading covers the remaining batch dims.
        auto_leading: list[_IndexPiece] = []
        for dim in decl_plate.batch_dims[len(user_pieces):]:
            axis = str(dim.name)
            loop_var = self._loop_for_axis(ctx, axis)
            if loop_var is not None:
                auto_leading.append(_IndexPiece.ident(loop_var))
                continue
            # No surrounding loop on this axis. If the enclosing
            # observe has a via fibration, apply it.
            if ctx.via is not None and ctx.enclosing_loop_names:
                via_inner = ctx.enclosing_loop_names[-1]
                auto_leading.append(
                    _IndexPiece.indexed(ctx.via, (_LoopRef(via_inner),))
                )
                continue
            # Otherwise the reference is degenerate; emit the axis
            # name as a free identifier (will probably fail in BUGS
            # but is a signal of an under-specified IR).
            auto_leading.append(_IndexPiece.ident(axis))
        # Event-dim slices: prefer the referenced var's own event
        # dims; fall back to the receiver's expected event rank
        # when the referenced var has none but the receiver does.
        slice_uppers = tuple(
            self._dim_upper_text(d) for d in decl_plate.event_dims
        )
        if not slice_uppers and receiver_event_rank > len(user_pieces):
            slice_uppers = self._receiver_slice_uppers(
                ctx, receiver_event_rank
            )
        if not (auto_leading or user_pieces or slice_uppers):
            return self._emit_bare_identifier(ctx, ref.name)
        all_pieces = tuple(list(user_pieces) + auto_leading)
        return self._emit_indexed_from_pieces(
            ctx, ref.name, all_pieces, slice_uppers
        )

    def _receiver_slice_uppers(
        self, ctx: _BugsCtx, receiver_event_rank: int
    ) -> tuple[str, ...]:
        """Return slice upper-bound texts derived from the receiver
        family's enclosing-plate event dims.

        Used when a scalar-input (e.g. `alpha`) is referenced in a
        receiver position that expects a vector (e.g. Dirichlet's
        `concentration`). The receiver's enclosing plate's
        event_dims supply the per-axis size.
        """
        if ctx.enclosing_plate is None:
            return ()
        ev = ctx.enclosing_plate.event_dims
        if len(ev) < receiver_event_rank:
            return ()
        return tuple(
            self._dim_upper_text(d) for d in ev[:receiver_event_rank]
        )

    def _loop_for_axis(self, ctx: _BugsCtx, axis: str) -> str | None:
        """Return the enclosing loop variable for `axis`, or None.

        Searches the enclosing plate's batch dims for a dim whose
        name matches `axis`; the parallel entry in
        `enclosing_loop_names` is the loop variable.
        """
        if ctx.enclosing_plate is None:
            return None
        for dim, ln in zip(
            ctx.enclosing_plate.batch_dims,
            ctx.enclosing_loop_names,
            strict=True,
        ):
            if dim.name == axis:
                return ln
        return None

    def _user_index_piece(
        self, ctx: _BugsCtx, arg: IRArg
    ) -> _IndexPiece:
        """Build an `_IndexPiece` for a user-supplied bracket index.

        Numeric literals become number-pieces. Bare identifier refs
        whose name happens to be a declared variable (e.g. `z` in
        `phi[z]`) get the full per-batch-dim index expansion so
        `phi[z]` over a Doc-batched z under a Word-batched observe
        with via=word_idx renders as `z[word_idx[m_Word]]`.
        """
        if isinstance(arg, IRArgNumber):
            return _IndexPiece.number(
                str(int(arg.value)) if arg.value.is_integer() else repr(arg.value)
            )
        if isinstance(arg, IRArgRef):
            inner_plate = ctx.decl_plates.get(arg.name)
            if inner_plate is None or not inner_plate.batch_dims:
                if not arg.indices:
                    return _IndexPiece.ident(arg.name)
                children = tuple(
                    self._user_index_piece(ctx, ix) for ix in arg.indices
                )
                return _IndexPiece.indexed_pieces(arg.name, children)
            # arg has declared batch dims: build nested indexing.
            sub: list[_IndexPiece] = []
            for dim in inner_plate.batch_dims:
                axis = str(dim.name)
                loop_var = self._loop_for_axis(ctx, axis)
                if loop_var is not None:
                    sub.append(_IndexPiece.ident(loop_var))
                    continue
                if ctx.via is not None and ctx.enclosing_loop_names:
                    via_inner = ctx.enclosing_loop_names[-1]
                    sub.append(
                        _IndexPiece.indexed(
                            ctx.via, (_LoopRef(via_inner),)
                        )
                    )
                    continue
                sub.append(_IndexPiece.ident(axis))
            for ix in arg.indices:
                sub.append(self._user_index_piece(ctx, ix))
            return _IndexPiece.indexed_pieces(arg.name, tuple(sub))
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"user-index:{type(arg).__name__}"],
        )

    # ------------------------------------------------------------------
    # Indexed-variable emission helpers.
    # ------------------------------------------------------------------

    def _emit_indexed_name(
        self,
        ctx: _BugsCtx,
        name: str,
        index_args: tuple[_LoopRef | IRArg, ...],
        slice_uppers: tuple[str, ...],
    ) -> str:
        """Emit `name[<index_args>, <1:slice_uppers>...]`.

        `index_args` carries already-prepared pieces (loop-ref or
        IRArg); each prepared piece becomes one index_list child.
        """
        iv_id = self._fresh(ctx, "iv")
        ctx.sb.vertex(iv_id, "indexed_variable")
        nid = self._emit_bare_identifier(ctx, name)
        ctx.sb.edge(iv_id, nid, "name")
        il_id = self._fresh(ctx, "il")
        ctx.sb.vertex(il_id, "index_list")
        ctx.sb.edge(iv_id, il_id, "indices")
        for piece in index_args:
            if isinstance(piece, _LoopRef):
                pid = self._emit_bare_identifier(ctx, piece.name)
                ctx.sb.edge(il_id, pid, "identifier")
                continue
            child_id = self._emit_index_child(ctx, piece)
            ctx.sb.edge(il_id, child_id, self._index_child_kind(piece))
        for upper in slice_uppers:
            self._emit_one_to(ctx, il_id, upper)
        return iv_id

    def _emit_indexed_from_pieces(
        self,
        ctx: _BugsCtx,
        name: str,
        pieces: tuple[_IndexPiece, ...],
        slice_uppers: tuple[str, ...],
    ) -> str:
        """Emit `name[<piece_1>, <piece_2>, ..., <1:slice_uppers>]`."""
        iv_id = self._fresh(ctx, "iv")
        ctx.sb.vertex(iv_id, "indexed_variable")
        nid = self._emit_bare_identifier(ctx, name)
        ctx.sb.edge(iv_id, nid, "name")
        il_id = self._fresh(ctx, "il")
        ctx.sb.vertex(il_id, "index_list")
        ctx.sb.edge(iv_id, il_id, "indices")
        for piece in pieces:
            child_id, kind = piece.emit(ctx, self)
            ctx.sb.edge(il_id, child_id, kind)
        for upper in slice_uppers:
            self._emit_one_to(ctx, il_id, upper)
        return iv_id

    def _emit_indexed_with_pre_emitted(
        self,
        ctx: _BugsCtx,
        name: str,
        index_child_ids: tuple[str, ...],
        index_child_kinds: tuple[str, ...],
        slice_uppers: tuple[str, ...],
    ) -> str:
        """Emit `name[<pre_emitted_indices>..., <slices>...]` from
        pre-rendered index_list children."""
        iv_id = self._fresh(ctx, "iv")
        ctx.sb.vertex(iv_id, "indexed_variable")
        nid = self._emit_bare_identifier(ctx, name)
        ctx.sb.edge(iv_id, nid, "name")
        il_id = self._fresh(ctx, "il")
        ctx.sb.vertex(il_id, "index_list")
        ctx.sb.edge(iv_id, il_id, "indices")
        for cid, kind in zip(index_child_ids, index_child_kinds, strict=True):
            ctx.sb.edge(il_id, cid, kind)
        for upper in slice_uppers:
            self._emit_one_to(ctx, il_id, upper)
        return iv_id

    def _emit_index_child(self, ctx: _BugsCtx, arg: IRArg) -> str:
        """Emit one index_list child for a user-supplied index."""
        if isinstance(arg, IRArgNumber):
            return self._emit_number(ctx, arg.value)
        if isinstance(arg, IRArgRef):
            if not arg.indices:
                return self._emit_bare_identifier(ctx, arg.name)
            children = tuple(
                self._emit_index_child(ctx, ix) for ix in arg.indices
            )
            kinds = tuple(self._index_child_kind(ix) for ix in arg.indices)
            return self._emit_indexed_with_pre_emitted(
                ctx, arg.name, children, kinds, ()
            )
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"index:{type(arg).__name__}"],
        )

    def _index_child_kind(self, arg: IRArg) -> str:
        if isinstance(arg, IRArgNumber):
            return "number"
        if isinstance(arg, IRArgRef):
            return "indexed_variable" if arg.indices else "identifier"
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"index:{type(arg).__name__}"],
        )

    def _emit_one_to(
        self, ctx: _BugsCtx, parent_index_list: str, upper: str
    ) -> None:
        """Emit a `1:upper` range as a child of `parent_index_list`.

        `upper` is the raw upper-bound text (digit string for a
        static dim, identifier text for a dynamic dim)."""
        rng_id = self._fresh(ctx, "rng")
        ctx.sb.vertex(rng_id, "range")
        ctx.sb.edge(parent_index_list, rng_id, "range")
        lo_id = self._fresh(ctx, "num")
        ctx.sb.vertex(lo_id, "number")
        ctx.sb.constraint(lo_id, "literal-value", "1")
        ctx.sb.edge(rng_id, lo_id, "lower")
        hi_id = self._fresh(ctx, "num")
        self._emit_upper_literal(ctx, hi_id, upper)
        ctx.sb.edge(rng_id, hi_id, "upper")

    # ------------------------------------------------------------------
    # Transform emission (1 / (x*x), 1/x, -x, log(x), exp(x)).
    # ------------------------------------------------------------------

    def _emit_transform(
        self, ctx: _BugsCtx, wrapped: IRArgTransform
    ) -> str:
        """Emit the arithmetic transform's expression schema.

        BUGS infix arithmetic. The root of the emitted fragment is
        a `binary_expression` (or `function_call` for log / exp).
        """
        first_inner = self._emit_arg(ctx, wrapped.inner)
        if wrapped.transform == "inv_square":
            second_inner = self._emit_arg(ctx, wrapped.inner)
            return self._emit_inv_square(
                ctx, first_inner, second_inner
            )
        if wrapped.transform == "inv":
            return self._emit_inv(ctx, first_inner)
        if wrapped.transform == "neg":
            return self._emit_neg(ctx, first_inner)
        if wrapped.transform == "log":
            return self._emit_unary_call(ctx, "log", first_inner)
        if wrapped.transform == "exp":
            return self._emit_unary_call(ctx, "exp", first_inner)
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"transform:{wrapped.transform}"],
        )

    def _emit_inv_square(
        self,
        ctx: _BugsCtx,
        first_inner: str,
        second_inner: str,
    ) -> str:
        """Emit `1 / (<inner> * <inner>)`."""
        mul = self._fresh(ctx, "be")
        ctx.sb.vertex(mul, "binary_expression")
        ctx.sb.constraint(mul, "field:operator", "*")
        ctx.sb.constraint(mul, "chose-alt-fingerprint", "*")
        ctx.sb.edge(mul, first_inner, "left")
        ctx.sb.edge(mul, second_inner, "right")
        paren = self._fresh(ctx, "par")
        ctx.sb.vertex(paren, "parenthesized_expression")
        ctx.sb.edge(paren, mul, "parenthesized_expression")
        one = self._emit_number(ctx, 1.0)
        div = self._fresh(ctx, "be")
        ctx.sb.vertex(div, "binary_expression")
        ctx.sb.constraint(div, "field:operator", "/")
        ctx.sb.constraint(div, "chose-alt-fingerprint", "/")
        ctx.sb.edge(div, one, "left")
        ctx.sb.edge(div, paren, "right")
        return div

    def _emit_inv(self, ctx: _BugsCtx, inner_id: str) -> str:
        """Emit `1 / <inner>`."""
        one = self._emit_number(ctx, 1.0)
        div = self._fresh(ctx, "be")
        ctx.sb.vertex(div, "binary_expression")
        ctx.sb.constraint(div, "field:operator", "/")
        ctx.sb.constraint(div, "chose-alt-fingerprint", "/")
        ctx.sb.edge(div, one, "left")
        ctx.sb.edge(div, inner_id, "right")
        return div

    def _emit_neg(self, ctx: _BugsCtx, inner_id: str) -> str:
        """Emit `-<inner>` via unary_expression."""
        u = self._fresh(ctx, "ue")
        ctx.sb.vertex(u, "unary_expression")
        ctx.sb.constraint(u, "field:operator", "-")
        ctx.sb.constraint(u, "chose-alt-fingerprint", "-")
        ctx.sb.edge(u, inner_id, "operand")
        return u

    def _emit_unary_call(
        self, ctx: _BugsCtx, fn_name: str, inner_id: str
    ) -> str:
        """Emit `<fn_name>(<inner>)` as a function_call."""
        call = self._fresh(ctx, "call")
        ctx.sb.vertex(call, "function_call")
        fn = self._emit_bare_identifier(ctx, fn_name)
        ctx.sb.edge(call, fn, "name")
        al = self._fresh(ctx, "al")
        ctx.sb.vertex(al, "argument_list")
        ctx.sb.edge(call, al, "arguments")
        ctx.sb.edge(al, inner_id, "identifier")
        return call

    # ------------------------------------------------------------------
    # Truncation: T(lower, upper).
    # ------------------------------------------------------------------

    def _emit_truncation(
        self,
        ctx: _BugsCtx,
        lower: IRArg,
        upper: IRArg,
    ) -> str:
        """Emit a `truncation` node carrying `T(lower, upper)`.

        The fingerprint `T( , )` picks the T-alternative of the
        BUGS truncation rule (T / I / C).
        """
        tr = self._fresh(ctx, "tr")
        ctx.sb.vertex(tr, "truncation")
        ctx.sb.constraint(tr, "chose-alt-fingerprint", "T( , )")
        lo_id = self._emit_arg(ctx, lower)
        hi_id = self._emit_arg(ctx, upper)
        ctx.sb.edge(tr, lo_id, self._arg_edge_kind(lower))
        ctx.sb.edge(tr, hi_id, self._arg_edge_kind(upper))
        return tr

    # ------------------------------------------------------------------
    # Misc helpers.
    # ------------------------------------------------------------------

    def _infer_arg_names(
        self, meta: FamilyMeta, args: tuple[IRArg, ...]
    ) -> tuple[str, ...]:
        """Return the arg-name tuple for `args` against `meta`."""
        cls_attr = meta.distribution_class.arg_constraints
        if isinstance(cls_attr, dict):
            names = tuple(cls_attr.keys())
        else:
            names = ()
        return names[: len(args)]

    def _fresh(self, ctx: _BugsCtx, prefix: str) -> str:
        ctx.fresh_counter += 1
        return f"{prefix}_{ctx.fresh_counter}"


# ----------------------------------------------------------------------
# Helpers outside the class.
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _LoopRef:
    """A bare-identifier reference to a for-loop variable."""

    name: str


@dataclasses.dataclass(frozen=True)
class _IndexPiece:
    """One piece of an index list, post-substitution.

    Either a bare identifier (`ident`), a number literal, or an
    indexed-variable form (`indexed`) with nested pieces.
    """

    kind: Literal["ident", "number", "indexed"]
    text: str
    children: tuple[_IndexPiece | _LoopRef, ...] = ()

    @classmethod
    def ident(cls, name: str) -> _IndexPiece:
        return cls(kind="ident", text=name)

    @classmethod
    def number(cls, text: str) -> _IndexPiece:
        return cls(kind="number", text=text)

    @classmethod
    def indexed(
        cls, name: str, children: tuple[_IndexPiece | _LoopRef, ...]
    ) -> _IndexPiece:
        return cls(kind="indexed", text=name, children=children)

    @classmethod
    def indexed_pieces(
        cls, name: str, children: tuple[_IndexPiece, ...]
    ) -> _IndexPiece:
        return cls(
            kind="indexed", text=name, children=tuple(children)
        )

    def emit(
        self, ctx: _BugsCtx, renderer: BUGSRenderer
    ) -> tuple[str, str]:
        """Emit this piece as a schema fragment; return `(id, kind)`."""
        if self.kind == "ident":
            return renderer._emit_bare_identifier(ctx, self.text), "identifier"
        if self.kind == "number":
            nid = renderer._fresh(ctx, "num")
            ctx.sb.vertex(nid, "number")
            ctx.sb.constraint(nid, "literal-value", self.text)
            return nid, "number"
        # indexed:
        iv_id = renderer._fresh(ctx, "iv")
        ctx.sb.vertex(iv_id, "indexed_variable")
        name_id = renderer._emit_bare_identifier(ctx, self.text)
        ctx.sb.edge(iv_id, name_id, "name")
        il_id = renderer._fresh(ctx, "il")
        ctx.sb.vertex(il_id, "index_list")
        ctx.sb.edge(iv_id, il_id, "indices")
        for child in self.children:
            if isinstance(child, _LoopRef):
                cid = renderer._emit_bare_identifier(ctx, child.name)
                ctx.sb.edge(il_id, cid, "identifier")
                continue
            cid, kind = child.emit(ctx, renderer)
            ctx.sb.edge(il_id, cid, kind)
        return iv_id, "indexed_variable"


def _is_wrapper_family_call(args: tuple[IRArg, ...]) -> bool:
    """Return True iff the first arg is an
    [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef].

    The first-arg [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
    is the structural signal Lower emits for wrapper-family calls
    (`Truncated(base, lo, hi)`, `Mixture(weights, component)`, ...);
    the renderer dispatches on this shape rather than on the family
    name string."""
    return bool(args) and isinstance(args[0], IRArgFamilyRef)


def _draw_arg_to_ir(a: DrawArg) -> IRArg:
    """Convert a DSL-level
    [`DrawArg`][quivers.dsl.ast_nodes.DrawArg] tagged-union variant
    into the corresponding [`IRArg`][quivers.transpile.ir.IRArg].

    Used by the wrapper-family (Truncated) handler to lift the
    referenced morphism's `~ Family(args)` clause into IR form for
    re-emission as the truncated call's args.
    """
    if isinstance(a, DrawArgScalar):
        return IRArgNumber(value=a.value)
    if isinstance(a, DrawArgName):
        return IRArgRef(name=a.text)
    raise UnsupportedConstruct(
        "qvr-bugs",
        [f"draw-arg:{type(a).__name__}"],
    )


def _as_transform(
    t: str,
) -> Literal["inv_square", "inv", "neg", "log", "exp"]:
    """Coerce a string to the `IRArgTransform.transform` literal type."""
    if t in ("inv_square", "inv", "neg", "log", "exp"):
        return t  # type: ignore[return-value]
    raise UnsupportedConstruct(
        "qvr-bugs",
        [f"transform:{t}"],
    )


__all__ = ["BUGSRenderer"]
