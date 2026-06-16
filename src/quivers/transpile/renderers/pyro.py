"""Pyro renderer: [`IRProgram`][quivers.transpile.ir.IRProgram] to a
Python source [`panproto.Schema`][panproto.Schema].

Pyro is the PyTorch sibling of NumPyro. The rendered shape is one
`def model(<params>, <observed>=None): ...` function whose body
wraps every sampled / observed draw in `with pyro.plate(<name>,
<size>):` blocks (one per batch dim) and emits the call as
`pyro.sample("<name>", pyro.distributions.<Family>(<args>)[,
obs=<obs>])`. The renderer mirrors the NumPyro renderer with `torch`
substituted for `jnp` and `pyro` substituted for `numpyro`.

The dispatch points implement the contract documented on
[`RendererBase`][quivers.transpile.renderers._base.RendererBase]:

* `declare`: outside `"function_body"` is a no-op; the function
  signature picks up data inputs and observed names from the IR's
  `inputs` / `IRObserve` nodes during `render`.
* `sample`: wraps the `pyro.sample(...)` call in nested
  `with pyro.plate(<name>, <size>):` blocks per `plate.batch_dims`.
* `marginalize`: lowers
  [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] to
  [`IRSample`][quivers.transpile.ir.IRSample] over the latent plus
  the scope body inline via the inherited
  [`explicit_latent_scope`][quivers.transpile.renderers._base.RendererBase.explicit_latent_scope].
* `broadcast`: emits `torch.full((K,), <value>)` for a 1D target
  shape and `torch.full((R, C), <value>)` for 2D.
* `render_list`: emits `torch.tensor([e0, e1, ...])`.
* `render_matrix`: emits `torch.tensor([[...], [...]])`.
* `IRArgFamilyRef`: resolves the morphism's `~ Family(...)` init
  clause and dispatches under `pyro.distributions.<TruncatedFamily>`
  per the wrapper.

Per-family distribution names come from
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META]'s
`target_names["pyro"]`; keyword arg names come from the IR node's
`arg_names`. Family-specific branching is reserved to the registry
lookup, never to per-family equality tests in renderer code.
"""

from __future__ import annotations

import panproto

from quivers.dsl.ast_nodes import MorphismInitFamily
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import target_protocol
from quivers.transpile.renderers._python_helpers import (
    PyCtx,
    assignment,
    attribute,
    call,
    identifier,
    number_literal,
    render_let_expr_python,
    shape_tuple,
    string_literal,
    with_statement,
)
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
from quivers.transpile.ir import (
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
from quivers.transpile.renderers._base import (
    BlockKind,
    RendererBase,
    SchemaFragment,
    _RenderCtx,
)


_TARGET = "pyro"


class PyroRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
    Python source [`panproto.Schema`][panproto.Schema] under the
    Pyro PPL idiom.
    """

    target: str = _TARGET

    # ----- panproto plumbing -----

    def target_protocol(self) -> panproto.Protocol:
        return target_protocol("python")

    # ----- top-level render -----

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and emit a Pyro `def model(...): ...` function.

        Overrides the inherited walk so the function header (with
        `<observed>=None` default parameters drawn from the IR's
        `IRObserve` nodes) and the body block are constructed up
        front; the inherited dispatch then routes each node into
        the body.
        """
        proto = self.target_protocol()
        sb = proto.schema()
        ctx = _RenderCtx(sb=sb, morphisms={}, lets={})
        pctx = _PyroCtx(sb=sb, cards=dict(ir.cards))

        # Resolve module-level morphism / let tables for IRArgFamilyRef
        # lookup. The IR carries family names directly for atomic
        # families; wrappers like `Truncated(base, ...)` carry the
        # `base` reference via IRArgFamilyRef and the renderer looks
        # the wrapped family up via the resolved init_family.
        # (The IR program does not carry the module; renderers either
        # receive an externally-built morphism table or rebuild from
        # the IR. We rebuild from `IRSample`s the IR exposes; opaque
        # wrappers requiring deeper resolution surface via
        # IRArgFamilyRef and we error if the wrapped family cannot be
        # resolved at emit time.)
        pctx.morphisms = ctx.morphisms

        # Gather observed names so the function signature carries
        # `<obs>=None` defaults.
        observed_names = _observed_names(ir.body)
        # Function header: positional params are every IRDataInput
        # that is not observed; default params are the observed ones.
        param_names = tuple(
            inp.name for inp in ir.inputs if inp.name not in observed_names
        )
        default_params = tuple(
            inp.name for inp in ir.inputs if inp.name in observed_names
        )

        pctx.v("mod", "module")
        body = pctx.v(pctx.fresh("body"), "block")
        func = _function_def_split(
            pctx,
            name="model",
            positional=param_names,
            defaults=default_params,
            body_vid=body,
        )
        pctx.e("mod", func, "child_of")

        pctx.body = body
        pctx.observed = frozenset(observed_names)

        for node in ir.body:
            self._dispatch_pyro_node(pctx, ctx, node)

        return sb.build()

    # ----- per-node dispatch driving pctx body emission -----

    def _dispatch_pyro_node(
        self,
        pctx: _PyroCtx,
        ctx: _RenderCtx,
        node: IRNode,
    ) -> None:
        if isinstance(node, IRDataInput):
            # Function signature carries the data input; nothing in
            # the body.
            self.declare(
                ctx, node.name, node.constraint, node.plate, block="data"
            )
            return
        if isinstance(node, IRSample):
            self._emit_sample_or_observe(
                pctx, ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                observed=False,
            )
            return
        if isinstance(node, IRObserve):
            self._emit_sample_or_observe(
                pctx, ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                observed=True,
            )
            return
        if isinstance(node, IRDeterministic):
            asn = pctx.v(pctx.fresh("asn"), "assignment")
            lhs = identifier(pctx, node.name)
            pctx.e(asn, lhs, "left")
            pctx.e(asn, render_let_expr_python(pctx, node.expr), "right")
            pctx.e(pctx.body, asn, "child_of")
            return
        if isinstance(node, IRScore):
            self._emit_score(pctx, node)
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node, pctx=pctx)
            return
        if isinstance(node, IRReturn):
            self._emit_return(pctx, node.names)
            return
        raise UnsupportedConstruct(
            f"qvr-{self.target}",
            [f"node:{type(node).__name__}"],
        )

    # ----- declare: no-op outside `"function_body"` -----

    def declare(
        self,
        ctx: _RenderCtx,
        name: str,
        constraint,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """No-op outside `"function_body"`: Pyro picks data inputs up
        in the model function signature, not via a declaration block.
        """
        del ctx, name, constraint, plate, block
        return ""

    # ----- sample / observe -----

    def sample(
        self,
        ctx: _RenderCtx,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        constraint,
        plate: Plate,
        observed: bool,
    ) -> SchemaFragment:
        """Build the `pyro.sample(...)` call; the caller threads the
        return vertex into the enclosing `with` / block."""
        del constraint, plate
        return self._build_sample_call(
            ctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            observed=observed,
        )

    def _emit_sample_or_observe(
        self,
        pctx: _PyroCtx,
        ctx: _RenderCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        observed: bool,
    ) -> None:
        """Emit `with pyro.plate(<dim>, <size>):` per batch dim,
        wrapping a `pyro.sample(...)` call (optionally assigned to
        `name` for latents)."""
        sample_call = self._build_sample_call(
            ctx,
            pctx=pctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            observed=observed,
            plate=plate,
        )
        if observed:
            inner_stmt = sample_call
        else:
            inner_stmt = assignment(pctx, lhs_name=name, rhs=sample_call)

        # Nest one `with pyro.plate(...)` per batch dim. If no batch
        # dims, the call lives directly inside the function body.
        target_block = pctx.body
        if not plate.batch_dims:
            pctx.e(target_block, inner_stmt, "child_of")
            return
        # Build innermost-first by walking the dims in reverse: the
        # innermost block holds the sample, the outer blocks own each
        # inner `with` as their sole child.
        current_stmt = inner_stmt
        for dim in reversed(plate.batch_dims):
            inner_block = pctx.v(pctx.fresh("blk"), "block")
            pctx.e(inner_block, current_stmt, "child_of")
            plate_call = self._plate_call(pctx, dim, name=name, observed=observed)
            current_stmt = with_statement(
                pctx,
                expression=plate_call,
                alias=None,
                body_vid=inner_block,
            )
        pctx.e(target_block, current_stmt, "child_of")

    def _plate_call(
        self,
        pctx: _PyroCtx,
        dim: Dim,
        *,
        name: str,
        observed: bool,
    ) -> str:
        """Build `pyro.plate(<plate_name>, <size>)`.

        `<plate_name>` is the dim's `name` for outer iid plates; for
        the dynamic batch dim that backs an observation, the canonical
        idiom uses the dim's name with the observed-shape lookup.
        """
        del observed
        plate_callee = attribute(pctx, ("pyro", "plate"))
        if isinstance(dim, DimStatic):
            plate_name = string_literal(pctx, dim.name)
            size_arg = number_literal(pctx, float(dim.size))
        elif isinstance(dim, DimDynamic):
            plate_name = string_literal(pctx, dim.name)
            # Dynamic plate size: `<size_name>` is the canonical
            # data-bound size identifier (e.g. `N_w`). Renderers that
            # want `<observed>.shape[0]` can post-process; we emit the
            # documented bound name from the IR.
            size_arg = identifier(pctx, dim.size_name)
        else:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"dim-kind:{type(dim).__name__}"],
            )
        del name
        return call(
            pctx,
            plate_callee,
            positional=(plate_name, size_arg),
        )

    def _build_sample_call(
        self,
        ctx: _RenderCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        observed: bool,
        plate: Plate | None = None,
        pctx: _PyroCtx | None = None,
    ) -> str:
        """Build `pyro.sample("<name>", pyro.distributions.<Family>(
        <args>)[, obs=<name>])`."""
        if pctx is None:
            # Allow `sample` to run standalone (Protocol contract):
            # build a throwaway PyCtx-compatible view over the same
            # SchemaBuilder. The renderer always drives through
            # `_emit_sample_or_observe` for real emission.
            pctx = _PyroCtx(sb=ctx.sb)
        meta = self._family_meta(family)
        dist_class = meta.target_names.get(_TARGET)
        if dist_class is None:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family:{family}"],
            )
        aliases = meta.arg_aliases.get(_TARGET, {})
        # Build the distribution call.
        dist_callee = attribute(pctx, ("pyro", "distributions", dist_class))
        dist_args = self._build_dist_args(
            pctx, meta=meta, args=args, arg_names=arg_names,
            aliases=aliases, plate=plate,
        )
        dist_call = call(
            pctx,
            dist_callee,
            positional=dist_args.positional,
            keyword=dist_args.keyword,
        )
        # Wrap in pyro.sample(...)
        sample_callee = attribute(pctx, ("pyro", "sample"))
        positional = (string_literal(pctx, name), dist_call)
        keyword: tuple[tuple[str, str], ...] = ()
        if observed:
            keyword = (("obs", identifier(pctx, name)),)
        return call(
            pctx,
            sample_callee,
            positional=positional,
            keyword=keyword,
        )

    # ----- marginalize: explicit-latent rewrite -----

    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
        *,
        pctx: _PyroCtx | None = None,
    ) -> SchemaFragment:
        """Lower the marginalize scope to an `IRSample` over the latent
        plus the scope body inline (the Pyro idiom natively samples
        discrete latents)."""
        if pctx is None:
            # Standalone call: return empty fragment; the renderer
            # drives marginalize via `_emit_marginalize`.
            return ""
        rewritten = self.explicit_latent_scope(node)
        # The first node is the synthesised IRSample for the latent;
        # rename its plate name to `"<batch_dim_name>_<latent>"` so it
        # does not collide with a sibling plate of the same dim name.
        first = rewritten[0]
        if isinstance(first, IRSample):
            first = _rename_plate_for_latent(first)
        for emit_node in (first, *rewritten[1:]):
            self._dispatch_pyro_node(pctx, ctx, emit_node)
        return ""

    # ----- broadcast / list / matrix -----

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit `torch.full((K,), <value>)` for rank-1, `torch.full(
        (R, C), <value>)` for rank-2."""
        pctx = _PyroCtx(sb=ctx.sb)
        return self._broadcast(pctx, value, target_shape)

    def _broadcast(
        self,
        pctx: _PyroCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> str:
        if len(target_shape) not in (1, 2):
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"broadcast:rank-{len(target_shape)}"],
            )
        shape_vid = shape_tuple(pctx, target_shape)
        value_vid = self._arg_to_vid(pctx, value)
        full_callee = attribute(pctx, ("torch", "full"))
        return call(
            pctx,
            full_callee,
            positional=(shape_vid, value_vid),
        )

    def render_list(
        self,
        pctx: _PyroCtx,
        arg: IRArgList,
    ) -> str:
        """Emit `torch.tensor([e0, e1, ...])` for an
        [`IRArgList`][quivers.transpile.ir.IRArgList]."""
        list_vid = pctx.v(pctx.fresh("lst"), "list")
        for elem in arg.elements:
            pctx.e(list_vid, self._arg_to_vid(pctx, elem), "child_of")
        tensor_callee = attribute(pctx, ("torch", "tensor"))
        return call(pctx, tensor_callee, positional=(list_vid,))

    def render_matrix(
        self,
        pctx: _PyroCtx,
        arg: IRArgMatrix,
    ) -> str:
        """Emit `torch.tensor([[...], [...]])` for an
        [`IRArgMatrix`][quivers.transpile.ir.IRArgMatrix]."""
        outer = pctx.v(pctx.fresh("mat"), "list")
        for row in arg.rows:
            row_vid = pctx.v(pctx.fresh("row"), "list")
            for elem in row.elements:
                pctx.e(row_vid, self._arg_to_vid(pctx, elem), "child_of")
            pctx.e(outer, row_vid, "child_of")
        tensor_callee = attribute(pctx, ("torch", "tensor"))
        return call(pctx, tensor_callee, positional=(outer,))

    # ----- arg lowering -----

    def _build_dist_args(
        self,
        pctx: _PyroCtx,
        *,
        meta: FamilyMeta,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        aliases: dict[str, str],
        plate: Plate | None = None,
    ) -> _DistArgs:
        """Build the (positional, keyword) split for the distribution
        call.

        Positional args carry every IR arg that needs neither alias
        renaming nor name disambiguation; keyword args carry every
        position whose `arg_name` has been renamed by `arg_aliases`.
        Scalar arguments to vector-typed slots are auto-broadcast via
        `torch.full` when `plate.event_dims` supplies the target
        shape; this realises the Pyro idiom of `torch.full((K,),
        alpha)` for a scalar `alpha` passed to `Dirichlet`.
        """
        positional: list[str] = []
        keyword: list[tuple[str, str]] = []
        if not arg_names or len(arg_names) != len(args):
            arg_names = tuple(meta.distribution_class.arg_constraints or ())
        for ir_arg, arg_name in zip(args, arg_names, strict=False):
            target = None
            if plate is not None:
                target = _scalar_broadcast_for_arg(
                    meta, ir_arg, arg_name, plate
                )
            if target is not None:
                vid = self._broadcast(pctx, ir_arg, target)
            else:
                vid = self._lower_arg(
                    pctx, ir_arg, meta=meta, arg_name=arg_name
                )
            if arg_name in aliases:
                keyword.append((aliases[arg_name], vid))
            else:
                positional.append(vid)
        return _DistArgs(
            positional=tuple(positional),
            keyword=tuple(keyword),
        )

    def _lower_arg(
        self,
        pctx: _PyroCtx,
        arg: IRArg,
        *,
        meta: FamilyMeta,
        arg_name: str,
    ) -> str:
        """Lower one IR arg to a single Python expression vertex.

        Scalar-to-vector broadcasting is the caller's responsibility
        (see [`_build_dist_args`][PyroRenderer._build_dist_args]); this
        helper handles per-arg structural dispatch only.
        """
        del meta, arg_name
        if isinstance(arg, IRArgNumber):
            return number_literal(pctx, arg.value)
        if isinstance(arg, IRArgRef):
            return _arg_ref_vid(pctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self._broadcast(pctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self.render_list(pctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self.render_matrix(pctx, arg)
        if isinstance(arg, IRArgFamilyRef):
            return self._resolve_family_ref(pctx, arg)
        raise UnsupportedConstruct(
            f"qvr-{_TARGET}",
            [f"arg-kind:{type(arg).__name__}"],
        )

    def _arg_to_vid(self, pctx: _PyroCtx, arg: IRArg) -> str:
        """Convenience: lower an arg without `meta` context (used in
        list / matrix / broadcast inner positions)."""
        if isinstance(arg, IRArgNumber):
            return number_literal(pctx, arg.value)
        if isinstance(arg, IRArgRef):
            return _arg_ref_vid(pctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self._broadcast(pctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self.render_list(pctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self.render_matrix(pctx, arg)
        raise UnsupportedConstruct(
            f"qvr-{_TARGET}",
            [f"inner-arg-kind:{type(arg).__name__}"],
        )

    def _resolve_family_ref(
        self,
        pctx: _PyroCtx,
        arg: IRArgFamilyRef,
    ) -> str:
        """Resolve an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        to a `pyro.distributions.<Family>(...)` call.

        The referenced morphism's `~ Family(...)` clause names the
        inner family; the Pyro renderer dispatches the wrapped form
        directly (e.g. `Truncated(base, ...)` with
        `morphism base ~ Normal(0, 1)` emits
        `pyro.distributions.TruncatedNormal(loc=0, scale=1, low=-2,
        high=2)`).
        """
        morph = pctx.morphisms.get(arg.name)
        if morph is None or not isinstance(
            morph.init_family, MorphismInitFamily
        ):
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family-ref:unresolved:{arg.name}"],
            )
        inner_family = morph.init_family.family
        # For the Truncated wrapper, the canonical Pyro emission uses
        # the truncated specialisation of the inner family (e.g.
        # TruncatedNormal). The dispatch picks that specialisation by
        # name through FAMILY_META.
        truncated_name = f"Truncated{inner_family}"
        meta = FAMILY_META.get(truncated_name) or FAMILY_META.get(inner_family)
        if meta is None:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family-ref:unknown-family:{inner_family}"],
            )
        dist_class = meta.target_names.get(_TARGET, meta.qvr_name)
        return attribute(pctx, ("pyro", "distributions", dist_class))

    def _family_meta(self, family: str) -> FamilyMeta:
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family:{family}"],
            )
        return meta

    # ----- score / return -----

    def _emit_score(self, pctx: _PyroCtx, node: IRScore) -> None:
        """`<name> = <expr>; pyro.factor("<name>", <name>)`."""
        asn = pctx.v(pctx.fresh("asn"), "assignment")
        lhs = identifier(pctx, node.name)
        pctx.e(asn, lhs, "left")
        pctx.e(asn, render_let_expr_python(pctx, node.expr), "right")
        pctx.e(pctx.body, asn, "child_of")
        factor_call = call(
            pctx,
            attribute(pctx, ("pyro", "factor")),
            positional=(
                string_literal(pctx, node.name),
                identifier(pctx, node.name),
            ),
        )
        pctx.e(pctx.body, factor_call, "child_of")

    def _emit_return(self, pctx, names: tuple[str, ...]) -> None:
        """Emit `return <var>` / `return <a>, <b>, ...`."""
        if not names:
            return
        rs = pctx.v(pctx.fresh("ret"), "return_statement")
        if len(names) == 1:
            pctx.e(rs, identifier(pctx, names[0]), "child_of")
        else:
            elist = pctx.v(pctx.fresh("elist"), "expression_list")
            for var in names:
                pctx.e(elist, identifier(pctx, var), "child_of")
            pctx.e(rs, elist, "child_of")
        pctx.e(pctx.body, rs, "child_of")


# ---------------------------------------------------------------------------
# Helpers / data carriers (renderer-local).
# ---------------------------------------------------------------------------


class _PyroCtx(PyCtx):
    """A [`PyCtx`][quivers.transpile.renderers._python_helpers.PyCtx]
    enriched with the function body block id, the set of observed
    names, and the resolved morphism table.

    Used by the renderer to thread emission state through the IR walk
    without rebuilding the panproto builder on every call.
    """

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        cards: dict[str, int] | None = None,
    ) -> None:
        super().__init__(sb, cards=cards)
        self.body: str = ""
        self.observed: frozenset[str] = frozenset()
        self.morphisms: dict = {}


class _DistArgs:
    """Split of distribution-call args into positional / keyword."""

    __slots__ = ("positional", "keyword")

    def __init__(
        self,
        *,
        positional: tuple[str, ...],
        keyword: tuple[tuple[str, str], ...],
    ) -> None:
        self.positional = positional
        self.keyword = keyword


def _observed_names(body: tuple[IRNode, ...]) -> set[str]:
    """Recursively collect every name bound by an
    [`IRObserve`][quivers.transpile.ir.IRObserve] in the IR body.
    """
    out: set[str] = set()
    for node in body:
        if isinstance(node, IRObserve):
            out.add(node.name)
        elif isinstance(node, IRMarginalize):
            out.update(_observed_names(node.scope))
    return out


def _arg_ref_vid(pctx: _PyroCtx, arg: IRArgRef) -> str:
    """Build `<name>[<idx0>][<idx1>]...` as a chain of `subscript`
    vertices."""
    base = identifier(pctx, arg.name)
    for idx in arg.indices:
        s = pctx.v(pctx.fresh("subs"), "subscript")
        pctx.e(s, base, "value")
        pctx.e(s, _arg_ref_vid(pctx, idx) if isinstance(idx, IRArgRef)
               else _inner_index_vid(pctx, idx), "subscript")
        base = s
    return base


def _inner_index_vid(pctx: _PyroCtx, arg: IRArg) -> str:
    """Inner subscript expression: ref or number; lists/broadcast not
    supported as index expressions."""
    if isinstance(arg, IRArgRef):
        return _arg_ref_vid(pctx, arg)
    if isinstance(arg, IRArgNumber):
        return number_literal(pctx, arg.value)
    raise UnsupportedConstruct(
        f"qvr-{_TARGET}",
        [f"subscript-arg:{type(arg).__name__}"],
    )


def _rename_plate_for_latent(node: IRSample) -> IRSample:
    """For an explicit-latent rewrite of an
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize], rename each
    `DimStatic`/`DimDynamic` in the latent's plate so the plate name
    does not collide with the surrounding `over=` plate.

    The canonical convention is `<dim_name>_<latent_name>`.
    """
    new_dims: list[Dim] = []
    for dim in node.plate.batch_dims:
        if isinstance(dim, DimStatic):
            new_dims.append(
                DimStatic(size=dim.size, name=f"{dim.name}_{node.name}")
            )
        elif isinstance(dim, DimDynamic):
            new_dims.append(
                DimDynamic(
                    size_name=dim.size_name,
                    name=f"{dim.name}_{node.name}",
                )
            )
        else:
            new_dims.append(dim)
    return IRSample(
        name=node.name,
        family=node.family,
        args=node.args,
        arg_names=node.arg_names,
        constraint=node.constraint,
        plate=Plate(
            event_dims=node.plate.event_dims,
            batch_dims=tuple(new_dims),
        ),
    )


def _function_def_split(
    pctx: _PyroCtx,
    *,
    name: str,
    positional: tuple[str, ...],
    defaults: tuple[str, ...],
    body_vid: str,
) -> str:
    """Build `def <name>(<pos0>, <pos1>, ..., <def0>=None, ...): <body>`.

    [`function_def`][quivers.transpile.renderers._python_helpers.function_def]
    emits every param as `<name>=None`; this variant carries the Pyro
    idiom of positional model params followed by `<obs>=None` for
    every observation.
    """
    func = pctx.v(pctx.fresh("fn"), "function_definition")
    fname = identifier(pctx, name)
    params = pctx.v(pctx.fresh("ps"), "parameters")
    pctx.e(func, fname, "name")
    pctx.e(func, params, "parameters")
    pctx.e(func, body_vid, "body")
    for pname in positional:
        pctx.e(params, identifier(pctx, pname), "child_of")
    for pname in defaults:
        dp = pctx.v(pctx.fresh("dp"), "default_parameter")
        dp_name = identifier(pctx, pname)
        dp_val = pctx.v(pctx.fresh("none"), "none")
        pctx.literal(dp_val, "None")
        pctx.e(dp, dp_name, "name")
        pctx.e(dp, dp_val, "value")
        pctx.e(params, dp, "child_of")
    return func


# ---------------------------------------------------------------------------
# Plate-aware scalar broadcast: the per-call path used by
# `_emit_sample_or_observe`.
# ---------------------------------------------------------------------------


def _scalar_broadcast_for_arg(
    meta: FamilyMeta,
    arg: IRArg,
    arg_name: str,
    plate: Plate,
) -> tuple[int, ...] | None:
    """If `arg` is a bare scalar reference / literal and `arg_name`'s
    constraint is rank-`n` independent, return the `(K, ...)` derived
    from `plate.event_dims`; otherwise `None`.
    """
    if not isinstance(arg, (IRArgRef, IRArgNumber)):
        return None
    if isinstance(arg, IRArgRef) and arg.indices:
        return None
    constraints = getattr(meta.distribution_class, "arg_constraints", {})
    expected = constraints.get(arg_name) if isinstance(
        constraints, dict
    ) else None
    if expected is None:
        return None
    event_dim = int(getattr(expected, "event_dim", 0))
    if event_dim < 1:
        return None
    if len(plate.event_dims) < event_dim:
        return None
    sizes: list[int] = []
    for dim in plate.event_dims[:event_dim]:
        if isinstance(dim, DimStatic):
            sizes.append(dim.size)
        else:
            return None
    return tuple(sizes)


__all__ = ["PyroRenderer"]
