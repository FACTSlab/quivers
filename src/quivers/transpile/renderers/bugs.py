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
(wrapper families like `Truncated`) render via the
`d<family>(args) T(lower, upper)` truncation idiom: the renderer
inlines the referenced morphism's `~ Family(args)` clause as the
distribution call and appends a `truncation` child to the
`stochastic_relation` carrying `(lower, upper)`. The `bugs` backend
executes through the JAGS engine (the probe image installs the
`jags` binary and pyjags), so it emits JAGS's renormalized
`T(lower, upper)` suffix rather than the `I(lower, upper)` censoring
form, which JAGS rejects on any latent-parent node.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Literal

import panproto

from quivers.dsl.ast_nodes import (
    MorphismDecl,
    MorphismInitFamily,
)
from quivers.dsl.ast_nodes.let_expressions import (
    LetExprBinOp,
    LetExprCall,
    LetExprFactor,
    LetExprIndex,
    LetExprLiteral,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import target_protocol
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
from quivers.transpile.ir import (
    CSReal,
    ConstraintSpec,
    Dim,
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
from quivers.transpile.renderers._base import (
    BlockKind,
    IRArgTransform,
    RendererBase,
    SchemaFragment,
    _RenderCtx,
    assert_no_dangling_refs,
    reorder_negbin_args,
    reorder_weibull_args,
)
from quivers.transpile.renderers._bugs_helpers import (
    TRUNCATION_FINGERPRINT,
    factor_axis_sizes,
    factor_cells,
    half_support_truncation,
    index_letexpr_refs,
    push_scalar_dets_into_loops,
    render_let_expr_bugs,
    split_event_dims,
)


class _BugsLetCtx:
    """Bridge ``_BugsCtx.sb`` to the
    [`render_let_expr_bugs`][quivers.transpile.renderers._bugs_helpers.render_let_expr_bugs]
    helper's ctx protocol (``v``, ``e``, ``lit``, ``fresh``,
    ``constraint``) and carry the per-render ``cards`` map (for
    factor unrolling) plus the ``target`` discriminator the helper
    stamps onto its error tags."""

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        fresh: Callable[[str], str],
        cards: dict[str, int],
        target: str,
    ) -> None:
        self._sb = sb
        self._fresh_fn = fresh
        self.cards = cards
        self.target = target

    def fresh(self, prefix: str) -> str:
        return self._fresh_fn(prefix)

    def v(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)

    def range_1_to(self, upper: str) -> str:
        """Build the `1:<upper>` range vertex the BUGS grammar wants."""
        rng_id = self._fresh_fn("rng")
        self._sb.vertex(rng_id, "range")
        lo_id = self._fresh_fn("num")
        self._sb.vertex(lo_id, "number")
        self._sb.constraint(lo_id, "literal-value", "1")
        self._sb.edge(rng_id, lo_id, "lower")
        hi_id = self._fresh_fn("num")
        self._sb.vertex(
            hi_id,
            "number" if upper.lstrip("-").isdigit() else "identifier",
        )
        self._sb.constraint(hi_id, "literal-value", upper)
        self._sb.edge(rng_id, hi_id, "upper")
        return rng_id


#: Per-renderer alias-transform table. Keys are the *renamed* arg
#: names from `FAMILY_META[family].arg_aliases["bugs"]`; values are
#: the arithmetic transform the renderer applies before emission.
#: BUGS uses precision (`tau = 1/(scale * scale)`) where torch's
#: `Normal` uses scale, so `arg_aliases["bugs"]["scale"] = "tau"`
#: pairs with `_ALIAS_TRANSFORMS["tau"] = "inv_square"`.
_ALIAS_TRANSFORMS: dict[str, str] = {
    "tau": "inv_square",
}

#: Per-family override of the arg-alias arithmetic transform. The
#: shared ``tau`` alias assumes a precision (``1/scale^2``)
#: parameterisation, right for ``dnorm`` / ``dt`` but wrong for
#: ``ddexp``. BUGS' ``ddexp(mu, tau)`` is rate-parameterised (density
#: ``(tau/2) * exp(-tau*|x-mu|)``), so ``Laplace``'s scale maps to the
#: rate ``tau = 1/scale`` rather than ``1/scale^2``.
_FAMILY_ALIAS_TRANSFORM_OVERRIDE: dict[str, dict[str, str]] = {
    "Laplace": {"tau": "inv"},
}


#: BUGS-side argument injection for QVR families whose underlying
#: torch distribution carries fewer parameters than the BUGS
#: distribution it maps to. ``HalfNormal(scale)`` maps to BUGS'
#: ``dnorm(0, tau)``; the renderer prepends an ``IRArgNumber(0)``
#: under the loc-position arg name so the alias-transform pipeline
#: still rewrites the scale into ``tau = 1/(scale*scale)``.
#: The constant ``log(2)`` offset that distinguishes HalfNormal from
#: the full Normal is absorbed by the constant-spread tolerance in
#: [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match].
_PREPEND_ZERO: frozenset[str] = frozenset({"HalfNormal", "HalfCauchy"})

#: BUGS-side argument injection for QVR families that map to BUGS'
#: ``dt(mu, tau, k)`` distribution. BUGS Student-t requires three
#: parameters (location, precision, degrees of freedom); Cauchy is
#: the special case ``k = 1``. The renderer appends ``IRArgNumber(1)``
#: as a trailing ``df`` argument after the alias-renaming pipeline so
#: the emitted call is ``dt(mu, tau, 1)``.
_APPEND_DF_ONE: frozenset[str] = frozenset({"Cauchy", "HalfCauchy"})


#: BUGS / JAGS zeros-trick constant offset. The Poisson PMF satisfies
#: ``log P(X = 0; lambda) = -lambda``, so emitting
#: ``zero_<name> ~ dpois(C - <expr>)`` with a host-bound
#: ``zero_<name> = 0`` adds ``<expr>`` to the joint log-likelihood up
#: to the additive constant ``-C`` (which absorbs into the
#: normalising constant and does not affect inference). ``C`` must
#: stay strictly larger than ``<expr>`` over the entire parameter
#: support; ``1.0e6`` is the conventional safe default for typical
#: BUGS / JAGS fixtures and matches the offset shipped in WinBUGS /
#: OpenBUGS examples.
_ZEROS_TRICK_OFFSET: float = 1.0e6


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
        # BUGS / JAGS have no scalar-to-vector broadcast: an
        # IRDeterministic with empty plate whose expression references
        # a plate-less free data input that ends up consumed inside a
        # non-empty-plate observe must be lifted into the consumer's
        # loop, with the references re-indexed by the loop variable.
        ir = push_scalar_dets_into_loops(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        ctx = _BugsCtx(sb=sb, morphisms={}, defines={})
        self._cards = dict(ir.cards)
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
            if node.family == "GP":
                self._emit_gp_block(ctx, node)
                return
            self._emit_sample_node(ctx, node, loop_suffix="")
            return
        if isinstance(node, IRObserve):
            self._emit_observe_node(ctx, node)
            return
        if isinstance(node, IRDeterministic):
            if isinstance(node.expr, LetExprFactor):
                self._emit_factor_deterministic_node(ctx, node)
                return
            self._emit_deterministic_node(ctx, node)
            return
        if isinstance(node, IRScore):
            self._emit_score_node(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            self._emit_marginalize_node(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._emit_export(ctx, node.names)
            return
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"node:{type(node).__name__}"],
        )

    def _emit_export(
        self, ctx: _BugsCtx, names: tuple[str, ...]
    ) -> None:
        """Expose each returned name as a deterministic relation.

        The BUGS language has no `return`: a model block declares
        relations and the inference engine reports whatever the caller
        monitors. The construct that carries "this quantity is part of
        what the model reports" is therefore a deterministic relation
        under a name of its own, `<name>_value <- <name>`, which is
        the same idiom the Stan renderer uses for its
        `generated quantities` alias and the PyMC renderer for its
        `pymc.Deterministic`. A relation adds no term to the joint
        (a deterministic node contributes nothing to the deviance),
        so the export rides alongside the density rather than into it.

        The alias reuses the deterministic emitter, which supplies the
        `for (m_<axis> in 1:N)` nest and the per-iteration indexing on
        both sides when the exported name is plated.
        """
        for name in names:
            plate = ctx.decl_plates.get(name)
            if plate is None:
                raise UnsupportedConstruct(
                    f"qvr-{self.target}",
                    [
                        f"return:unbound:{name}: the program returns a "
                        f"name no sample, observe, let, or data input "
                        f"binds, so the emit has no relation to alias"
                    ],
                )
            self._emit_deterministic_node(
                ctx,
                IRDeterministic(
                    name=f"{name}_value",
                    expr=LetExprVar(name=name),
                    constraint=CSReal(),
                    plate=plate,
                ),
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
        self._emit_sample_node(ctx, node, loop_suffix="", observed=observed)
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
        observed: bool = False,
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
        if node.family == "TruncatedNormal":
            self._emit_truncated_normal_native(ctx, node, loop_suffix)
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
            observed=observed,
        )

    def _emit_truncated_normal_native(
        self,
        ctx: _BugsCtx,
        node: IRSample,
        loop_suffix: str,
    ) -> None:
        """Emit ``<lhs> ~ dnorm(loc, tau) T(low, high)`` for
        ``TruncatedNormal(loc, scale, low, high)``.

        Splits the 4-arg call into (loc, scale) for the base ``dnorm``
        plus (low, high) for the truncation suffix; the family's
        ``arg_aliases[bugs]`` renames ``scale -> tau`` and the alias
        pipeline applies the ``inv_square`` transform so the emitted
        precision is ``1/(scale*scale)``.
        """
        if len(node.args) != 4:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    f"family:TruncatedNormal: expected 4 args "
                    f"(loc, scale, low, high), got {len(node.args)}"
                ],
            )
        family_args = node.args[:2]
        family_arg_names = node.arg_names[:2] if node.arg_names else ("loc", "scale")
        lo, hi = node.args[2], node.args[3]
        self._emit_relation(
            ctx,
            name=node.name,
            family="TruncatedNormal",
            args=family_args,
            arg_names=family_arg_names,
            plate=node.plate,
            via=None,
            loop_suffix=loop_suffix,
            truncation=(lo, hi),
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
            self._emit_truncated(
                ctx,
                IRSample(
                    name=node.name,
                    family=node.family,
                    args=node.args,
                    arg_names=node.arg_names,
                    constraint=node.constraint,
                    plate=node.plate,
                ),
            )
            return
        if node.family == "TruncatedNormal":
            self._emit_truncated_normal_native(
                ctx,
                IRSample(
                    name=node.name,
                    family=node.family,
                    args=node.args,
                    arg_names=node.arg_names,
                    constraint=node.constraint,
                    plate=node.plate,
                ),
                "",
            )
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
            observed=True,
        )

    def _emit_marginalize_node(self, ctx: _BugsCtx, node: IRMarginalize) -> None:
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
        self._emit_sample_node(ctx, latent_sample, loop_suffix=f"_{node.latent}")
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
                [f"truncated:base:{family_ref.name}: init is not `~ Family(args)`"],
            )
        base_family = init.family
        base_args = tuple(_draw_arg_to_ir(a) for a in (init.args or ()))
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

    def _emit_gp_block(
        self,
        ctx: _BugsCtx,
        node: IRSample,
    ) -> None:
        """Emit a Gaussian-process sample as BUGS deterministic loops
        plus a multivariate-normal stochastic relation.

        BUGS `dmnorm` parameterises by precision (the inverse of the
        covariance), so the RBF kernel matrix is constructed in a
        double loop and then inverted before being passed as the
        precision argument:

            for (i in 1:N) {
              __gp_zeros_<name>[i] <- 0
              for (j in 1:N) {
                __gp_K_<name>[i, j] <-
                    exp(-0.5 * pow(x[i] - x[j], 2)
                              / pow(length_scale, 2))
                    + equals(i, j) * jitter
              }
            }
            __gp_tau_<name> <- inverse(__gp_K_<name>)
            <name> ~ dmnorm(__gp_zeros_<name>, __gp_tau_<name>)

        BUGS lacks `ifelse`, so the jitter is added via the
        ``equals(i, j) * jitter`` idiom (equals returns 1 if true,
        0 otherwise).
        """
        if len(node.args) != 2 or not isinstance(node.args[1], IRArgKernel):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf is implemented"],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        # BUGS rejects identifiers starting with underscore; use a
        # `gp_` prefix instead of `__gp_` for the synthesized names.
        kmat_name = f"gp_K_{node.name}"
        zeros_name = f"gp_zeros_{node.name}"
        tau_name = f"gp_tau_{node.name}"
        # Synthesize an IRDeterministic for the zeros: K_zeros[i] <- 0
        # with batch_dim "i" (so the existing emit produces the outer
        # for(i) loop and the LHS index).
        i_dim = DimStatic(size=n, name="i")
        j_dim = DimStatic(size=n, name="j")
        zeros_det = IRDeterministic(
            name=zeros_name,
            expr=LetExprLiteral(value=0.0),
            constraint=node.constraint,
            plate=Plate(event_dims=(), batch_dims=(i_dim,)),
        )
        self._emit_deterministic_node(ctx, zeros_det)
        # Synthesize an IRDeterministic for K[i, j] <- exp(...) + ...
        # with batch_dims (i, j). The let-expr references `m_i`,
        # `m_j` as loop variables; index_letexpr_refs would rewrite
        # IRArgRef-ish refs to plate-bound names, but our expression
        # uses raw LetExprVar("m_i") / LetExprVar("m_j") and
        # LetExprIndex on `x` directly so the rewrite step is a no-op.
        i_var = LetExprVar(name="m_i")
        j_var = LetExprVar(name="m_j")
        x_i = LetExprIndex(
            array=LetExprVar(name=x),
            indices=(i_var,),
        )
        x_j = LetExprIndex(
            array=LetExprVar(name=x),
            indices=(j_var,),
        )
        diff = LetExprBinOp(op="-", left=x_i, right=x_j)
        diff_sq = LetExprCall(
            func="pow",
            args=(diff, LetExprLiteral(value=2.0)),
        )
        ls_sq = LetExprCall(
            func="pow",
            args=(
                LetExprLiteral(value=ls),
                LetExprLiteral(value=2.0),
            ),
        )
        neg_half = LetExprUnaryOp(operand=LetExprLiteral(value=0.5))
        scaled = LetExprBinOp(
            op="/",
            left=LetExprBinOp(op="*", left=neg_half, right=diff_sq),
            right=ls_sq,
        )
        exp_call = LetExprCall(func="exp", args=(scaled,))
        # BUGS lacks `ifelse`; use `equals(i, j) * jitter` (equals
        # returns 1 if its args are equal, 0 otherwise).
        eq_call = LetExprCall(func="equals", args=(i_var, j_var))
        jitter_term = LetExprBinOp(
            op="*",
            left=eq_call,
            right=LetExprLiteral(value=jitter),
        )
        kij_expr = LetExprBinOp(
            op="+",
            left=exp_call,
            right=jitter_term,
        )
        kij_det = IRDeterministic(
            name=kmat_name,
            expr=kij_expr,
            constraint=node.constraint,
            plate=Plate(event_dims=(), batch_dims=(i_dim, j_dim)),
        )
        self._emit_deterministic_node(ctx, kij_det)
        # __gp_tau_<name> <- inverse(__gp_K_<name>)
        tau_det = IRDeterministic(
            name=tau_name,
            expr=LetExprCall(
                func="inverse",
                args=(LetExprVar(name=kmat_name),),
            ),
            constraint=node.constraint,
            plate=Plate(event_dims=(), batch_dims=()),
        )
        self._emit_deterministic_node(ctx, tau_det)
        # <name> ~ dmnorm(__gp_zeros_<name>, __gp_tau_<name>)
        # Build a sample IR node for MultivariateNormal whose args
        # reference the bound names; the existing _emit_sample_node
        # handles emission via the family's BUGS target_name (dmnorm).
        mvn_sample = IRSample(
            name=node.name,
            family="MultivariateNormal",
            args=(
                IRArgRef(name=zeros_name),
                IRArgRef(name=tau_name),
            ),
            arg_names=("loc", "covariance_matrix"),
            constraint=node.constraint,
            plate=Plate(event_dims=(), batch_dims=()),
        )
        self._emit_sample_node(ctx, mvn_sample, loop_suffix="")

    def _emit_factor_deterministic_node(
        self, ctx: _BugsCtx, node: IRDeterministic
    ) -> None:
        """Emit a `factor` binding as one relation per cell.

        A rank-`n` factor denotes a rank-`n` tensor of scalar cells.
        BUGS has no array literal beyond the flat `c(...)` combine and
        no reshape to give one a rank, so the tensor is written out
        cell by cell: `<name>[i_1, ..., i_n] <- <body>` with every
        binder substituted by its coordinate. The cells are already
        enumerated, so no surrounding loop is emitted.
        """
        expr = node.expr
        if not isinstance(expr, LetExprFactor):
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"let-expr:factor:expected-factor:{node.name}"],
            )
        let_ctx = _BugsLetCtx(
            ctx.sb,
            lambda p: self._fresh(ctx, p),
            self._cards,
            self.target,
        )
        self._check_factor_plate(node, factor_axis_sizes(let_ctx, expr))
        for indices, body in factor_cells(let_ctx, expr):
            dr_id = self._fresh(ctx, "dr")
            ctx.sb.vertex(dr_id, "deterministic_relation")
            ctx.sb.edge(ctx.block_id, dr_id, "deterministic_relation")
            lhs_id = self._emit_indexed_from_pieces(
                ctx,
                node.name,
                tuple(
                    _IndexPiece.number(str(value + 1)) for value in indices
                ),
                (),
            )
            ctx.sb.edge(dr_id, lhs_id, "variable")
            ctx.sb.edge(
                dr_id,
                render_let_expr_bugs(
                    let_ctx, body, decl_plates=ctx.decl_plates,
                ),
                "value",
            )

    def _check_factor_plate(
        self, node: IRDeterministic, sizes: tuple[int, ...]
    ) -> None:
        """Assert the factor's binder axes are exactly the binding's
        plate, so the per-cell subscripts address the whole node."""
        if node.plate.event_dims:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    f"let-expr:LetExprFactor:{node.name}: a factor "
                    f"binds a tensor of scalar cells, so its plate "
                    f"carries no event axis, but this one declares "
                    f"{len(node.plate.event_dims)}"
                ],
            )
        declared = tuple(
            dim.size if isinstance(dim, DimStatic) else None
            for dim in node.plate.batch_dims
        )
        if declared != sizes:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [
                    f"let-expr:LetExprFactor:{node.name}: binder axes "
                    f"{sizes} do not match the binding's declared "
                    f"plate {declared}"
                ],
            )

    def _emit_deterministic_node(self, ctx: _BugsCtx, node: IRDeterministic) -> None:
        """Emit a BUGS deterministic relation ``<name> <- <expr>``.

        Wraps the relation in `for (m_<axis> in 1:N_<axis>)` loops
        when the node's plate carries batch dims (one loop per
        dim, BUGS-style nesting); the LHS is then indexed by each
        loop variable so the relation populates one element per
        iteration. The RHS expression goes through
        [`render_let_expr_bugs`][quivers.transpile.renderers._bugs_helpers.render_let_expr_bugs],
        which lowers `LetExpr*` nodes to BUGS expression vertices
        (`binary_expression`, `function_call`, `indexed_variable`,
        ...).
        """
        loop_names = self._loop_names(node.plate, "")
        body_id = self._open_loops(ctx, ctx.block_id, node.plate, loop_names)
        prev_plate = ctx.enclosing_plate
        prev_loops = ctx.enclosing_loop_names
        ctx.enclosing_plate = node.plate
        ctx.enclosing_loop_names = loop_names
        try:
            dr_id = self._fresh(ctx, "dr")
            ctx.sb.vertex(dr_id, "deterministic_relation")
            ctx.sb.edge(body_id, dr_id, "deterministic_relation")
            lhs_id = self._emit_lhs(ctx, node.name, node.plate, loop_names)
            ctx.sb.edge(dr_id, lhs_id, "variable")
            # Rewrite the expression so any var reference whose
            # declared plate axes match the surrounding loop axes
            # emits as `name[loop_var, ...]`. Required when an empty-
            # plate det was lifted into a non-empty plate by the
            # `_push_scalar_dets_into_loops` pre-pass: the lifted
            # det's expression still names a free data input
            # (`x_design`) that the lift retagged with the new plate.
            expr = index_letexpr_refs(
                node.expr, ctx.decl_plates, node.plate, loop_names
            )
            let_ctx = _BugsLetCtx(
                ctx.sb,
                lambda p: self._fresh(ctx, p),
                self._cards,
                self.target,
            )
            rhs_id = render_let_expr_bugs(
                let_ctx, expr, decl_plates=ctx.decl_plates,
            )
            ctx.sb.edge(dr_id, rhs_id, "value")
        finally:
            ctx.enclosing_plate = prev_plate
            ctx.enclosing_loop_names = prev_loops

    def _emit_score_node(self, ctx: _BugsCtx, node: IRScore) -> None:
        """Emit ``score <name> = <expr>`` via the BUGS zeros trick.

        BUGS has no native ``target +=`` statement; the canonical idiom
        for adding ``<expr>`` to the joint log-likelihood is the
        zeros trick, which exploits ``log P(0; lambda) = -lambda`` for
        the Poisson distribution. Concretely the renderer emits:

        ```
        C_<name> <- 1.0e6 - (<expr>)
        zero_<name> ~ dpois(C_<name>)
        ```

        The host supplies ``zero_<name> = 0`` in the data list so the
        stochastic relation contributes ``-(1.0e6 - <expr>) = <expr> -
        1.0e6`` to the log-density; the additive constant ``-1.0e6``
        absorbs into the normalising constant and does not affect
        posterior inference.

        The offset ``1.0e6`` must stay strictly larger than ``<expr>``
        over the entire parameter support so the Poisson rate argument
        remains positive; ``1.0e6`` is the canonical safe default for
        typical BUGS / JAGS fixtures.

        The score expression is wrapped in a ``parenthesized_expression``
        so the emitted source associates the subtraction correctly:
        without the parens an inner additive expression
        (``x*x + y*y``) would re-parse as
        ``1e6 - x*x + y*y = 1e6 - x^2 + y^2`` instead of the intended
        ``1e6 - (x^2 + y^2)``.
        """
        c_name = f"C_{node.name}"
        zero_name = f"zero_{node.name}"
        empty_plate = Plate(event_dims=(), batch_dims=())
        # Open the deterministic relation `C_<name> <- 1.0e6 - (<expr>)`.
        # IRScore carries no plate, so the relation lives at the model
        # block's top level with a bare-identifier LHS.
        dr_id = self._fresh(ctx, "dr")
        ctx.sb.vertex(dr_id, "deterministic_relation")
        ctx.sb.edge(ctx.block_id, dr_id, "deterministic_relation")
        lhs_id = self._emit_bare_identifier(ctx, c_name)
        ctx.sb.edge(dr_id, lhs_id, "variable")
        # Build the RHS as `1.0e6 - (<expr>)`: an outer
        # `binary_expression` with `field:operator = -`, left child a
        # number for the offset, right child a `parenthesized_expression`
        # wrapping the score expression. The parens force right-side
        # grouping when the score expression itself contains a `+` / `-`
        # operator.
        offset_id = self._emit_number(ctx, _ZEROS_TRICK_OFFSET)
        # Render the inner score expression through the standard
        # let-expression pipeline.
        let_ctx = _BugsLetCtx(
            ctx.sb,
            lambda p: self._fresh(ctx, p),
            self._cards,
            self.target,
        )
        inner_expr_id = render_let_expr_bugs(
            let_ctx, node.expr, decl_plates=ctx.decl_plates,
        )
        paren_id = self._fresh(ctx, "par")
        ctx.sb.vertex(paren_id, "parenthesized_expression")
        ctx.sb.edge(paren_id, inner_expr_id, "parenthesized_expression")
        sub_id = self._fresh(ctx, "be")
        ctx.sb.vertex(sub_id, "binary_expression")
        ctx.sb.constraint(sub_id, "field:operator", "-")
        ctx.sb.constraint(sub_id, "chose-alt-fingerprint", "-")
        ctx.sb.edge(sub_id, offset_id, "left")
        ctx.sb.edge(sub_id, paren_id, "right")
        ctx.sb.edge(dr_id, sub_id, "value")
        # Record the carrier's plate so the subsequent `dpois(C_<name>)`
        # call renders `C_<name>` as a bare identifier (no auto-leading
        # indices). `_emit_relation`'s ref-emission path consults
        # `decl_plates` for the rate-arg IRArgRef.
        ctx.decl_plates[c_name] = empty_plate
        ctx.decl_plates[zero_name] = empty_plate
        # Emit the stochastic relation `zero_<name> ~ dpois(C_<name>)`.
        # The host supplies `zero_<name> = 0` in the data list; BUGS
        # has no in-model data declaration so the model source itself
        # carries no `zero_<name> <- 0` line.
        self._emit_relation(
            ctx,
            name=zero_name,
            family="Poisson",
            args=(IRArgRef(name=c_name),),
            arg_names=("rate",),
            plate=empty_plate,
            via=None,
            loop_suffix="",
        )

    # ------------------------------------------------------------------
    # Core relation emitter.
    # ------------------------------------------------------------------

    def _inject_bugs_specific_args(
        self,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
        """Prepend the canonical zero-location argument for QVR
        families whose torch distribution carries fewer parameters
        than the BUGS distribution they map to.

        ``HalfNormal(scale)`` maps to BUGS' ``dnorm(0, tau)``; this
        helper prepends an ``IRArgNumber(0)`` plus the parallel
        ``"loc"`` arg-name entry so the alias-transform pipeline
        still rewrites the scale into ``tau = 1/(scale*scale)``.

        ``Cauchy(loc, scale)`` and ``HalfCauchy(scale)`` map to BUGS'
        ``dt(mu, tau, k)`` (Student-t parameterised by precision and
        degrees of freedom); this helper appends ``IRArgNumber(1)``
        under the ``"df"`` arg-name entry so the emitted call is
        ``dt(mu, tau, 1)``.
        """
        if family in _PREPEND_ZERO:
            args = (IRArgNumber(value=0.0), *args)
            arg_names = ("loc", *arg_names)
        if family in _APPEND_DF_ONE:
            args = (*args, IRArgNumber(value=1.0))
            arg_names = (*arg_names, "df")
        return args, arg_names

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
        truncation: tuple[IRArg, ...] | None = None,
        observed: bool = False,
    ) -> None:
        """Open the for-loops over `plate.batch_dims`, then emit the
        `<lhs> ~ <dist>(args)` line."""
        if truncation is None:
            truncation = half_support_truncation(family, observed=observed)
        meta = self._lookup_family(family)
        args, arg_names = self._inject_bugs_specific_args(family, args, arg_names)
        # Split the site's event dims into the family's own event
        # shape and the residual axes that merely replicate it. BUGS
        # has no vector form for a scalar family, so each residual
        # axis becomes an extra innermost loop rather than a slice on
        # the left-hand side.
        native_event, residual_event = split_event_dims(
            plate.event_dims, meta.event_rank
        )
        loop_plate = Plate(
            event_dims=native_event,
            batch_dims=(*plate.batch_dims, *residual_event),
        )
        loop_names = self._loop_names(loop_plate, loop_suffix)
        body_id = self._open_loops(ctx, ctx.block_id, loop_plate, loop_names)
        plate = loop_plate
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
            # When a truncation suffix is present, the
            # stochastic_relation's child-kind alternative needs to
            # advertise it so the panproto pretty-printer picks the
            # `~ ... T( , )` alternative rather than the no-suffix
            # alternative. Both the `jags` and the `bugs` backend
            # emit the JAGS `T( , )` spelling.
            if truncation is not None:
                lhs_kind = (
                    "identifier"
                    if not loop_names and not plate.event_dims
                    else "indexed_variable"
                )
                ctx.sb.constraint(sr_id, "chose-alt-fingerprint", "~")
                ctx.sb.constraint(
                    sr_id,
                    "chose-alt-child-kinds",
                    f"{lhs_kind} distribution_call truncation",
                )
            lhs_id = self._emit_lhs(ctx, name, plate, loop_names)
            ctx.sb.edge(sr_id, lhs_id, "variable")
            dc_id = self._emit_distribution_call(ctx, meta, args, arg_names)
            ctx.sb.edge(sr_id, dc_id, "distribution")
            if truncation is not None:
                trunc_id = self._emit_truncation(ctx, truncation)
                ctx.sb.edge(sr_id, trunc_id, "child_of")
        finally:
            ctx.via = prev_via
            ctx.enclosing_plate = prev_plate
            ctx.enclosing_loop_names = prev_loops

    # ------------------------------------------------------------------
    # Loop emission.
    # ------------------------------------------------------------------

    def _loop_names(self, plate: Plate, suffix: str) -> tuple[str, ...]:
        """Return the loop-variable name for each `plate.batch_dim`.

        A residual event axis lifted into the batch list can repeat an
        axis the site already iterates (a square row-stochastic matrix
        names the same object on both sides), and two `for` loops over
        the same variable name would silently alias, so a repeat gets
        a numeric suffix.
        """
        used: set[str] = set()
        out: list[str] = []
        for dim in plate.batch_dims:
            base = f"m_{dim.name}{suffix}"
            candidate = base
            index = 1
            while candidate in used:
                index += 1
                candidate = f"{base}_{index}"
            used.add(candidate)
            out.append(candidate)
        return tuple(out)

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

    def _emit_upper_literal(self, ctx: _BugsCtx, vid: str, text: str) -> None:
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
        event_slices = tuple(self._dim_upper_text(d) for d in plate.event_dims)
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
        if meta.qvr_name == "StudentT":
            args, arg_names = _reorder_studentt_dt(args, arg_names)
        elif meta.qvr_name == "NegativeBinomial":
            args, arg_names = reorder_negbin_args(args, arg_names)
        elif meta.qvr_name == "Weibull":
            args, arg_names = reorder_weibull_args(args, arg_names)
        renames = meta.arg_aliases.get("bugs", {})
        for arg, aname in zip(args, arg_names, strict=True):
            wrapped = self._apply_alias_transform(arg, aname, renames, meta.qvr_name)
            receiver_event_rank = self._receiver_event_rank(meta, aname)
            child_id = self._emit_arg(
                ctx, wrapped, receiver_event_rank=receiver_event_rank
            )
            ctx.sb.edge(al_id, child_id, self._arg_edge_kind(wrapped))
        return dc_id

    def _receiver_event_rank(self, meta: FamilyMeta, arg_name: str) -> int:
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
        self,
        arg: IRArg,
        arg_name: str,
        renames: dict[str, str],
        family: str,
    ) -> IRArg:
        """If `renames[arg_name]` targets a name with an arithmetic
        transform in [`_ALIAS_TRANSFORMS`][quivers.transpile.renderers.bugs._ALIAS_TRANSFORMS]
        (or the per-family override in
        [`_FAMILY_ALIAS_TRANSFORM_OVERRIDE`][quivers.transpile.renderers.bugs._FAMILY_ALIAS_TRANSFORM_OVERRIDE]),
        wrap `arg` in
        [`IRArgTransform`][quivers.transpile.renderers._base.IRArgTransform];
        otherwise return `arg` unchanged."""
        target_name = renames.get(arg_name)
        if target_name is None:
            return arg
        override = _FAMILY_ALIAS_TRANSFORM_OVERRIDE.get(family)
        if override is not None and target_name in override:
            transform: str | None = override[target_name]
        else:
            transform = _ALIAS_TRANSFORMS.get(target_name)
        if transform is None:
            return arg
        return IRArgTransform(inner=arg, transform=_as_transform(transform))

    def _arg_edge_kind(self, arg: IRArg) -> str:
        if isinstance(arg, IRArgNumber):
            return "number"
        if isinstance(arg, IRArgTransform):
            if arg.transform in ("log", "exp", "pow_neg"):
                return "function_call"
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
        """Edge-kind hook reserved for callers that need to discriminate
        on the arg's declaration plate. The class-method form returns
        `None` so the renderer falls through to no-slicing; the real
        lookup runs inside `_emit_arg` against the per-render context.
        """
        del arg
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
            return self._emit_ref(ctx, arg, receiver_event_rank=receiver_event_rank)
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
            slice_uppers = self._receiver_slice_uppers(ctx, receiver_event_rank)
            if slice_uppers:
                return self._emit_indexed_from_pieces(ctx, ref.name, (), slice_uppers)
        if decl_plate is None:
            if not ref.indices:
                return self._emit_bare_identifier(ctx, ref.name)
            indices = tuple(self._emit_index_child(ctx, ix) for ix in ref.indices)
            kinds = tuple(self._index_child_kind(ix) for ix in ref.indices)
            return self._emit_indexed_with_pre_emitted(
                ctx, ref.name, indices, kinds, ()
            )
        # User indices fill the leftmost batch_dim positions.
        user_pieces = tuple(self._user_index_piece(ctx, ix) for ix in ref.indices)
        # Auto-leading covers the remaining batch dims.
        auto_leading: list[_IndexPiece] = []
        for dim in decl_plate.batch_dims[len(user_pieces) :]:
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
        slice_uppers = tuple(self._dim_upper_text(d) for d in decl_plate.event_dims)
        if not slice_uppers and receiver_event_rank > len(user_pieces):
            slice_uppers = self._receiver_slice_uppers(ctx, receiver_event_rank)
        if not (auto_leading or user_pieces or slice_uppers):
            return self._emit_bare_identifier(ctx, ref.name)
        all_pieces = tuple(list(user_pieces) + auto_leading)
        return self._emit_indexed_from_pieces(ctx, ref.name, all_pieces, slice_uppers)

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
        return tuple(self._dim_upper_text(d) for d in ev[:receiver_event_rank])

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

    def _user_index_piece(self, ctx: _BugsCtx, arg: IRArg) -> _IndexPiece:
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
                children = tuple(self._user_index_piece(ctx, ix) for ix in arg.indices)
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
                    sub.append(_IndexPiece.indexed(ctx.via, (_LoopRef(via_inner),)))
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
            children = tuple(self._emit_index_child(ctx, ix) for ix in arg.indices)
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

    def _emit_one_to(self, ctx: _BugsCtx, parent_index_list: str, upper: str) -> None:
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

    def _emit_transform(self, ctx: _BugsCtx, wrapped: IRArgTransform) -> str:
        """Emit the arithmetic transform's expression schema.

        BUGS infix arithmetic. The root of the emitted fragment is
        a `binary_expression` (or `function_call` for log / exp).
        """
        first_inner = self._emit_arg(ctx, wrapped.inner)
        if wrapped.transform == "inv_square":
            second_inner = self._emit_arg(ctx, wrapped.inner)
            return self._emit_inv_square(ctx, first_inner, second_inner)
        if wrapped.transform == "inv":
            return self._emit_inv(ctx, first_inner)
        if wrapped.transform == "neg":
            return self._emit_neg(ctx, first_inner)
        if wrapped.transform == "log":
            return self._emit_unary_call(ctx, "log", first_inner)
        if wrapped.transform == "exp":
            return self._emit_unary_call(ctx, "exp", first_inner)
        if wrapped.transform == "one_minus":
            return self._emit_one_minus(ctx, first_inner)
        if wrapped.transform == "pow_neg":
            if wrapped.operand is None:
                raise UnsupportedConstruct(
                    f"qvr-{self.target}",
                    ["transform:pow_neg: missing exponent operand"],
                )
            operand_id = self._emit_arg(ctx, wrapped.operand)
            return self._emit_pow_neg(
                ctx,
                first_inner,
                self._arg_edge_kind(wrapped.inner),
                operand_id,
                self._arg_edge_kind(wrapped.operand),
            )
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"transform:{wrapped.transform}"],
        )

    def _emit_one_minus(self, ctx: _BugsCtx, inner_id: str) -> str:
        """Emit `1 - <inner>`."""
        one = self._emit_number(ctx, 1.0)
        diff = self._fresh(ctx, "be")
        ctx.sb.vertex(diff, "binary_expression")
        ctx.sb.constraint(diff, "field:operator", "-")
        ctx.sb.constraint(diff, "chose-alt-fingerprint", "-")
        ctx.sb.edge(diff, one, "left")
        ctx.sb.edge(diff, inner_id, "right")
        return diff

    def _emit_pow_neg(
        self,
        ctx: _BugsCtx,
        inner_id: str,
        inner_kind: str,
        operand_id: str,
        operand_kind: str,
    ) -> str:
        """Emit `pow(<inner>, -<operand>)` as a two-argument
        `function_call`."""
        neg = self._fresh(ctx, "ue")
        ctx.sb.vertex(neg, "unary_expression")
        ctx.sb.constraint(neg, "field:operator", "-")
        ctx.sb.constraint(neg, "chose-alt-fingerprint", "-")
        ctx.sb.constraint(neg, "chose-alt-child-kinds", operand_kind)
        ctx.sb.edge(neg, operand_id, "operand")
        call = self._fresh(ctx, "call")
        ctx.sb.vertex(call, "function_call")
        fn = self._emit_bare_identifier(ctx, "pow")
        ctx.sb.edge(call, fn, "name")
        al = self._fresh(ctx, "al")
        ctx.sb.vertex(al, "argument_list")
        ctx.sb.edge(call, al, "arguments")
        ctx.sb.edge(al, inner_id, inner_kind)
        ctx.sb.edge(al, neg, "unary_expression")
        return call

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

    def _emit_unary_call(self, ctx: _BugsCtx, fn_name: str, inner_id: str) -> str:
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
        bounds: tuple[IRArg, ...],
    ) -> str:
        """Emit a `truncation` node carrying `T(lower, upper)`, or
        `T(lower,)` when `bounds` holds a single lower bound.

        The `bugs` backend executes through the JAGS engine, so it
        emits JAGS's renormalized `T( , )` suffix, the same spelling
        the `jags` backend uses. The grammar's truncation rule offers
        `T( , )` and `I( , )` as alternatives keyed by the
        `chose-alt-fingerprint` constraint; the renderer selects
        `T( , )` for both backends because `I( , )` is JAGS interval
        censoring, which JAGS rejects on any latent-parent node.
        """
        if not bounds or len(bounds) > 2:
            raise UnsupportedConstruct(
                f"qvr-{self.target}",
                [f"truncation:expected 1 or 2 bounds, got {len(bounds)}"],
            )
        tr = self._fresh(ctx, "tr")
        ctx.sb.vertex(tr, "truncation")
        ctx.sb.constraint(
            tr, "chose-alt-fingerprint", TRUNCATION_FINGERPRINT[self.target]
        )
        for bound in bounds:
            ctx.sb.edge(tr, self._emit_arg(ctx, bound), "child_of")
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
        return cls(kind="indexed", text=name, children=tuple(children))

    def emit(self, ctx: _BugsCtx, renderer: BUGSRenderer) -> tuple[str, str]:
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


def _draw_arg_to_ir(a: str | float) -> IRArg:
    """Convert a morphism ``~ Family(args)`` init arg into the
    corresponding [`IRArg`][quivers.transpile.ir.IRArg].

    Init-family args arrive in wire form: a ``float`` literal or a
    ``str`` identifier. Used by the wrapper-family (Truncated)
    handler to lift the referenced morphism's init clause into IR
    form for re-emission as the truncated call's args.
    """
    if isinstance(a, (int, float)):
        return IRArgNumber(value=float(a))
    stripped = a.strip()
    try:
        return IRArgNumber(value=float(stripped))
    except ValueError:
        return IRArgRef(name=stripped)


def _reorder_studentt_dt(
    args: tuple[IRArg, ...], arg_names: tuple[str, ...]
) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
    """Reshape a location-scale ``StudentT(df, loc, scale)`` into BUGS'
    ``dt(mu, tau, k)`` argument order.

    BUGS' Student-t is parameterised by location, precision, and
    degrees of freedom, in that order; torch's ``StudentT`` carries
    ``(df, loc, scale)``. This reorders to ``(loc, scale, df)`` and
    pre-wraps the scale in the precision transform
    ``tau = 1/(scale*scale)`` so the emitted call is
    ``dt(loc, 1/(scale*scale), df)``.
    """
    by_name = dict(zip(arg_names, args, strict=True))
    return (
        (
            by_name["loc"],
            IRArgTransform(inner=by_name["scale"], transform="inv_square"),
            by_name["df"],
        ),
        ("loc", "tau", "df"),
    )


def _as_transform(
    t: str,
) -> Literal["inv_square", "inv", "neg", "log", "exp"]:
    """Coerce a string to the `IRArgTransform.transform` literal type."""
    if t == "inv_square":
        return "inv_square"
    if t == "inv":
        return "inv"
    if t == "neg":
        return "neg"
    if t == "log":
        return "log"
    if t == "exp":
        return "exp"
    raise UnsupportedConstruct(
        "qvr-bugs",
        [f"transform:{t}"],
    )


__all__ = ["BUGSRenderer"]
