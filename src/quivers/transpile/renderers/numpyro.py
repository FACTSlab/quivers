"""NumPyro renderer: [`IRProgram`][quivers.transpile.ir.IRProgram] to
Python source under the `python` tree-sitter grammar.

The output is a single ``def model(<inputs>): ...`` function whose body
emits each [`IRSample`][quivers.transpile.ir.IRSample] /
[`IRObserve`][quivers.transpile.ir.IRObserve] inside a stack of
``with numpyro.plate(<axis>, <size>):`` context managers (one per
batch dim of the step's
[`Plate`][quivers.transpile.ir.Plate]). Distributions live under
``numpyro.distributions.<Family>`` and are emitted as keyword calls
keyed on each family's torch ``arg_constraints`` name (e.g.
``numpyro.distributions.Normal(loc=mu, scale=sigma)``); per-backend
renames declared in
[`FAMILY_META[family].arg_aliases["numpyro"]`][quivers.transpile.family_meta.FAMILY_META]
are applied before emission.

Marginalize lowering: [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
is rewritten via the
[`RendererBase.explicit_latent_scope`][quivers.transpile.renderers._base.RendererBase.explicit_latent_scope]
helper into an [`IRSample`][quivers.transpile.ir.IRSample] for the
latent followed by the scope body inline (NumPyro samples the discrete
latent natively rather than enumerating).

`declare` is a no-op outside ``"function_body"`` because NumPyro has
no declaration block: every variable is introduced by its
``numpyro.sample`` call. Function-body declarations (the model's
parameter list) are handled by the renderer's `render` override that
threads the [`IRDataInput`][quivers.transpile.ir.IRDataInput] list into
the ``def model(...)`` parameter list.

`broadcast(value, target_shape)` emits ``jnp.full((K,), value)`` for
1D shapes and ``jnp.full((R, C), value)`` for 2D shapes; the renderer
fully-qualifies ``jnp`` so the emitted source is import-aware (the
function header carries ``import jax.numpy as jnp`` plus
``import numpyro`` plus ``import numpyro.distributions``).
"""

from __future__ import annotations

import panproto

from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile.renderers._python_helpers import (
    PyCtx,
    arg_expr,
    attribute,
    call,
    identifier,
    number_literal,
    python_binary_op as _python_binary_op,
    python_method_call as _python_method_call,
    python_paren as _python_paren,
    python_unary_minus as _python_unary_minus,
    render_let_expr_python,
    string_literal,
    with_statement,
)
from quivers.transpile._pipeline import target_protocol
from quivers.transpile.family_meta import FAMILY_META
from quivers.transpile.ir import (
    ConstraintSpec,
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


#: The backend key used to look up `target_names` / `arg_aliases` in
#: [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META].
_BACKEND: str = "numpyro"


#: Inner-family-keyed lookup table for the specialised truncated
#: distribution classes NumPyro publishes (`TruncatedNormal`,
#: `TruncatedCauchy`, etc.). Inner families absent from this table use
#: the generic `TruncatedDistribution(<inner>, low, high)` wrapper.
_NUMPYRO_TRUNCATED_SPECIALISATION: dict[str, str] = {
    "Normal": "TruncatedNormal",
    "Cauchy": "TruncatedCauchy",
}


class NumPyroRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] as NumPyro
    Python source.

    Subclasses [`RendererBase`][quivers.transpile.renderers._base.RendererBase]
    and supplies the four required dispatch points
    (`declare`, `sample`, `marginalize`, `broadcast`) plus a `render`
    override that wraps the IR walk with the ``def model(...): ...``
    function header.
    """

    target: str = "numpyro"

    # ------------------------------------------------------------------
    # `RendererBase` dispatch points
    # ------------------------------------------------------------------

    def target_protocol(self) -> panproto.Protocol:
        """Use the auto-derived Python tree-sitter protocol."""
        return target_protocol("python")

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and emit a complete NumPyro module.

        Overrides the base IR walk because NumPyro programs need a
        function header that lists every
        [`IRDataInput`][quivers.transpile.ir.IRDataInput] in the
        signature (observed values get a ``=None`` default).
        """
        proto = self.target_protocol()
        sb = proto.schema()
        py = PyCtx(sb, cards=dict(ir.cards), target="numpyro")
        ctx = _NumPyroCtx(
            sb=sb,
            morphisms={},
            lets={},
            py=py,
            observed_names=self._collect_observed(ir),
        )

        py.v("mod", "module")
        body = py.v(py.fresh("body"), "block")
        params = self._function_params(ir)
        func = self._build_function(py, body, params)

        # Dispatch the body first so any ``LetExprCall`` records the
        # imports its symbol needs (jax.scipy.special / jax.nn), then emit
        # the import block and wire the function after it so the imports
        # lead the module.
        self._dispatch_body(ctx, body, ir.body)
        self._emit_imports(ctx)
        py.e("mod", func, "child_of")

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
        """No-op outside ``"function_body"``.

        NumPyro declarations ARE sample calls; the function-header
        parameter list is built up front by `render` from
        [`IRDataInput`][quivers.transpile.ir.IRDataInput] entries.
        Function-body declarations land at the
        [`sample`][quivers.transpile.renderers.numpyro.NumPyroRenderer.sample]
        dispatch point when the IR walker reaches the binding step.
        """
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
        """Emit a ``numpyro.sample`` call inside the surrounding plate
        stack."""
        del constraint
        npctx = _as_numpyro_ctx(ctx)
        return self._render_sample(
            npctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            plate=plate,
            observed=observed,
        )

    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Lower [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] to
        an explicit [`IRSample`][quivers.transpile.ir.IRSample] for the
        latent followed by the scope body inline.

        NumPyro samples the discrete latent natively rather than
        enumerating over it; the inherited
        [`explicit_latent_scope`][quivers.transpile.renderers._base.RendererBase.explicit_latent_scope]
        helper does the rewrite. Plate names that collide with a
        previously-emitted plate get a ``_<latent>`` suffix; otherwise
        the source axis name is used unchanged.
        """
        npctx = _as_numpyro_ctx(ctx)
        body_vid = npctx.current_body
        if body_vid is None:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                ["marginalize:no-enclosing-body"],
            )
        rewritten = self.explicit_latent_scope(node)
        latent = rewritten[0]
        if isinstance(latent, IRSample):
            renamed_plate = self._dedupe_plate(
                npctx, latent.plate, latent.name
            )
            latent_dedup = IRSample(
                name=latent.name,
                family=latent.family,
                args=latent.args,
                arg_names=latent.arg_names,
                constraint=latent.constraint,
                plate=renamed_plate,
            )
            self._dispatch_body(npctx, body_vid, (latent_dedup,))
        else:  # pragma: no cover -- explicit_latent_scope shape
            self._dispatch_body(npctx, body_vid, (latent,))
        self._dispatch_body(npctx, body_vid, rewritten[1:])
        return ""

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit ``jnp.full((K,), value)`` for 1D / ``jnp.full((R, C),
        value)`` for 2D."""
        npctx = _as_numpyro_ctx(ctx)
        py = npctx.py
        shape_tuple = self._render_shape_tuple(py, target_shape)
        value_vid = self._render_arg(npctx, value)
        return call(
            py,
            attribute(py, ("jnp", "full")),
            positional=(shape_tuple, value_vid),
        )

    # ------------------------------------------------------------------
    # Per-renderer rendering helpers
    # ------------------------------------------------------------------

    def render_list(
        self, ctx: _NumPyroCtx, arg: IRArgList
    ) -> SchemaFragment:
        """Emit ``jnp.array([e0, e1, ...])`` for an
        [`IRArgList`][quivers.transpile.ir.IRArgList]."""
        py = ctx.py
        lst = py.v(py.fresh("list"), "list")
        for elem in arg.elements:
            py.e(lst, self._render_arg(ctx, elem), "child_of")
        return call(
            py,
            attribute(py, ("jnp", "array")),
            positional=(lst,),
        )

    def render_matrix(
        self, ctx: _NumPyroCtx, arg: IRArgMatrix
    ) -> SchemaFragment:
        """Emit ``jnp.array([[...], [...]])`` for an
        [`IRArgMatrix`][quivers.transpile.ir.IRArgMatrix]."""
        py = ctx.py
        outer = py.v(py.fresh("list"), "list")
        for row in arg.rows:
            inner = py.v(py.fresh("list"), "list")
            for elem in row.elements:
                py.e(inner, self._render_arg(ctx, elem), "child_of")
            py.e(outer, inner, "child_of")
        return call(
            py,
            attribute(py, ("jnp", "array")),
            positional=(outer,),
        )

    # ------------------------------------------------------------------
    # Internal walk / emission
    # ------------------------------------------------------------------

    def _emit_imports(self, ctx: _NumPyroCtx) -> None:
        """Emit ``import jax.numpy as jnp``, ``import numpyro``,
        ``import numpyro.distributions``, plus any aliased imports a
        lowered ``LetExprCall`` symbol recorded (``jax.scipy.special`` /
        ``jax.nn``)."""
        py = ctx.py
        self._emit_aliased_import(py, ("jax", "numpy"), "jnp")
        self._emit_plain_import(py, ("numpyro",))
        self._emit_plain_import(py, ("numpyro", "distributions"))
        for chain, alias in sorted(py.required_imports):
            self._emit_aliased_import(py, chain, alias)

    def _emit_plain_import(
        self, py: PyCtx, chain: tuple[str, ...]
    ) -> None:
        """Emit ``import <a>.<b>.<c>``."""
        stmt = py.v(py.fresh("imp"), "import_statement")
        name = self._dotted_name(py, chain)
        py.e(stmt, name, "name")
        py.e("mod", stmt, "child_of")

    def _emit_aliased_import(
        self, py: PyCtx, chain: tuple[str, ...], alias: str
    ) -> None:
        """Emit ``import <a>.<b> as <alias>``."""
        stmt = py.v(py.fresh("imp"), "import_statement")
        aliased = py.v(py.fresh("alias"), "aliased_import")
        py.e(aliased, self._dotted_name(py, chain), "name")
        py.e(aliased, identifier(py, alias), "alias")
        py.e(stmt, aliased, "name")
        py.e("mod", stmt, "child_of")

    def _dotted_name(
        self, py: PyCtx, chain: tuple[str, ...]
    ) -> str:
        """Build a ``dotted_name`` vertex from the chain segments."""
        dn = py.v(py.fresh("dn"), "dotted_name")
        for seg in chain:
            py.e(dn, identifier(py, seg), "child_of")
        return dn

    def _function_params(self, ir: IRProgram) -> _Params:
        """Split function parameters into (positional, defaulted).

        Observed inputs (variables also bound as
        [`IRObserve`][quivers.transpile.ir.IRObserve]) get the
        ``=None`` default; all other inputs (scalar program params,
        via fibrations, free names) are positional.
        """
        observed = self._collect_observed(ir)
        positional: list[str] = []
        defaulted: list[str] = []
        for inp in ir.inputs:
            if inp.name in observed:
                defaulted.append(inp.name)
            else:
                positional.append(inp.name)
        return _Params(
            positional=tuple(positional),
            defaulted=tuple(defaulted),
        )

    def _build_function(
        self,
        py: PyCtx,
        body_vid: str,
        params: _Params,
    ) -> str:
        """Build ``def model(a, b, c=None, d=None): <body>`` with both
        positional and defaulted parameter slots.

        Mirrors `function_def` from the shared helpers but emits a
        mixed parameter list: positional identifiers first, then
        default-parameter entries with ``=None`` defaults. The grammar
        permits the mixed form natively.
        """
        func = py.v(py.fresh("fn"), "function_definition")
        fname = identifier(py, "model")
        ps = py.v(py.fresh("ps"), "parameters")
        py.e(func, fname, "name")
        py.e(func, ps, "parameters")
        py.e(func, body_vid, "body")
        for name in params.positional:
            py.e(ps, identifier(py, name), "child_of")
        for name in params.defaulted:
            dp = py.v(py.fresh("dp"), "default_parameter")
            dp_name = identifier(py, name)
            dp_val = py.v(py.fresh("none"), "none")
            py.literal(dp_val, "None")
            py.e(dp, dp_name, "name")
            py.e(dp, dp_val, "value")
            py.e(ps, dp, "child_of")
        return func

    def _collect_observed(self, ir: IRProgram) -> set[str]:
        """Return the set of names bound as
        [`IRObserve`][quivers.transpile.ir.IRObserve] anywhere in the
        body (recursive)."""
        observed: set[str] = set()
        self._collect_observed_in(ir.body, observed)
        return observed

    def _collect_observed_in(
        self,
        body: tuple,
        observed: set[str],
    ) -> None:
        for node in body:
            if isinstance(node, IRObserve):
                observed.add(node.name)
            elif isinstance(node, IRMarginalize):
                self._collect_observed_in(node.scope, observed)

    def _dispatch_body(
        self,
        ctx: _NumPyroCtx,
        body_vid: str,
        nodes: tuple,
    ) -> None:
        """Walk the IR nodes attached to ``body_vid`` (a Python
        ``block`` vertex)."""
        previous_body = ctx.current_body
        ctx.current_body = body_vid
        try:
            for node in nodes:
                self._dispatch_one(ctx, body_vid, node)
        finally:
            ctx.current_body = previous_body

    def _dispatch_one(
        self,
        ctx: _NumPyroCtx,
        body_vid: str,
        node,
    ) -> None:
        if isinstance(node, IRDataInput):
            # NumPyro: data inputs ride on the function signature; no
            # body emission. `render` already handled the header.
            return
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(ctx, body_vid, node, observed=False)
                return
            stmt = self._sample_statement(
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                observed=False,
            )
            ctx.py.e(body_vid, stmt, "child_of")
            return
        if isinstance(node, IRObserve):
            stmt = self._sample_statement(
                ctx,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                observed=True,
            )
            ctx.py.e(body_vid, stmt, "child_of")
            return
        if isinstance(node, IRDeterministic):
            stmt = self._deterministic_statement(ctx, node)
            ctx.py.e(body_vid, stmt, "child_of")
            return
        if isinstance(node, IRScore):
            self._score_statement(ctx, body_vid, node)
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node)
            return
        if isinstance(node, IRReturn):
            self._return_statement(ctx, body_vid, node.names)
            return
        raise UnsupportedConstruct(
            "qvr-numpyro",
            [f"node:{type(node).__name__}"],
        )

    def _emit_gp_block(
        self,
        ctx: _NumPyroCtx,
        body_vid: str,
        node: IRSample,
        *,
        observed: bool,
    ) -> None:
        """Emit a Gaussian-process sample as a triple:
        ``__gp_mean_<name> = jnp.zeros(N)``,
        ``__gp_cov_<name>  = jnp.exp(...) + jitter * jnp.eye(N)``,
        ``<name> = numpyro.sample("<name>",
        numpyro.distributions.MultivariateNormal(loc=__gp_mean_<name>,
        covariance_matrix=__gp_cov_<name>))``.

        The kernel-covariance expression is built by hand using the
        Python helper API plus explicit
        [`parenthesized_expression`][panproto.python.parenthesized_expression]
        nodes, since the let-expression renderer drops
        parenthesisation around nested
        [`LetExprBinOp`][quivers.transpile.ir.LetExprBinOp] children
        and would otherwise mis-group the exponent.
        """
        del observed  # GP samples are never observed in QVR.
        py = ctx.py
        _, kernel_arg = node.args
        if not isinstance(kernel_arg, IRArgKernel):
            raise UnsupportedConstruct(
                "qvr-numpyro",
                ["family:GP:expected IRArgKernel as second arg"],
            )
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"family:GP:kernel:{kernel_arg.kernel}: only rbf supported"],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{node.name}"
        cov_name = f"__gp_cov_{node.name}"
        # __gp_mean_<name> = jnp.zeros(N)
        mean_rhs = call(
            py,
            attribute(py, ("jnp", "zeros")),
            positional=(number_literal(py, n),),
        )
        mean_asn = self._assignment_statement(py, mean_name, mean_rhs)
        py.e(body_vid, mean_asn, "child_of")
        # __gp_cov_<name> = jnp.exp(-0.5 * (diff * diff) / (ls * ls)) + jitter * jnp.eye(N)
        # where diff = x.reshape(-1, 1) - x.reshape(1, -1)
        x_col = _python_method_call(
            py, identifier(py, x), "reshape",
            (_python_unary_minus(py, number_literal(py, 1)),
             number_literal(py, 1)),
        )
        x_row = _python_method_call(
            py, identifier(py, x), "reshape",
            (number_literal(py, 1),
             _python_unary_minus(py, number_literal(py, 1))),
        )
        diff = _python_paren(
            py,
            _python_binary_op(py, "-", x_col, x_row),
        )
        diff_sq = _python_paren(
            py,
            _python_binary_op(py, "*", diff, diff),
        )
        ls_sq = _python_paren(
            py,
            _python_binary_op(
                py, "*",
                number_literal(py, ls),
                number_literal(py, ls),
            ),
        )
        quotient = _python_binary_op(py, "/", diff_sq, ls_sq)
        neg_half = _python_unary_minus(py, number_literal(py, 0.5))
        exponent = _python_binary_op(py, "*", neg_half, quotient)
        kernel_call = call(
            py,
            attribute(py, ("jnp", "exp")),
            positional=(exponent,),
        )
        eye_call = call(
            py,
            attribute(py, ("jnp", "eye")),
            positional=(number_literal(py, n),),
        )
        jitter_term = _python_binary_op(
            py, "*", number_literal(py, jitter), eye_call,
        )
        cov_rhs = _python_binary_op(
            py, "+", kernel_call, jitter_term,
        )
        cov_asn = self._assignment_statement(py, cov_name, cov_rhs)
        py.e(body_vid, cov_asn, "child_of")
        # sample call
        mvn_call = call(
            py,
            attribute(
                py,
                ("numpyro", "distributions", "MultivariateNormal"),
            ),
            keyword=(
                ("loc", identifier(py, mean_name)),
                ("covariance_matrix", identifier(py, cov_name)),
            ),
        )
        sample_call = call(
            py,
            attribute(py, ("numpyro", "sample")),
            positional=(string_literal(py, node.name), mvn_call),
        )
        sample_stmt = self._assignment_statement(
            py, node.name, sample_call,
        )
        py.e(body_vid, sample_stmt, "child_of")

    def _render_sample(
        self,
        ctx: _NumPyroCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        observed: bool,
    ) -> SchemaFragment:
        """Build the sample / observe statement and attach it to the
        active body."""
        body_vid = ctx.current_body
        if body_vid is None:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                ["sample:no-enclosing-body"],
            )
        stmt = self._sample_statement(
            ctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            plate=plate,
            observed=observed,
        )
        ctx.py.e(body_vid, stmt, "child_of")
        return stmt

    def _sample_statement(
        self,
        ctx: _NumPyroCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        observed: bool,
    ) -> str:
        """Build ``with numpyro.plate(...): <name> = numpyro.sample(...)``
        (or a bare ``numpyro.sample(...)`` for observes)."""
        py = ctx.py
        # Innermost sample call:
        sample_call = self._numpyro_sample_call(
            ctx,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            obs_name=name if observed else None,
        )
        if observed:
            inner_stmt = self._expression_statement(py, sample_call)
        else:
            inner_stmt = self._assignment_statement(py, name, sample_call)

        # Wrap in nested `with numpyro.plate("<axis>", <size>):` per
        # batch_dim (outermost = first batch_dim, preserving source
        # `iid_over` order). Inner-most contains the sample statement.
        return self._wrap_in_plates(ctx, plate.batch_dims, inner_stmt)

    def _wrap_in_plates(
        self,
        ctx: _NumPyroCtx,
        batch_dims: tuple,
        inner_stmt: str,
    ) -> str:
        """Nest ``with numpyro.plate(name, size):`` blocks around
        ``inner_stmt``, one per batch dim.

        With no batch dims, returns ``inner_stmt`` directly. With one
        dim, returns a single ``with_statement``; with more dims,
        nests outermost-first so the source order of `iid_over`
        becomes the outermost plate.
        """
        if not batch_dims:
            return inner_stmt
        # Build innermost first, then wrap.
        current = inner_stmt
        for dim in reversed(batch_dims):
            plate_call = self._plate_call(ctx, dim)
            block = ctx.py.v(ctx.py.fresh("blk"), "block")
            ctx.py.e(block, current, "child_of")
            current = with_statement(
                ctx.py,
                expression=plate_call,
                alias=None,
                body_vid=block,
            )
        return current

    def _plate_call(self, ctx: _NumPyroCtx, dim) -> str:
        """Build ``numpyro.plate("<name>", <size>)`` for one
        [`Dim`][quivers.transpile.ir.Dim]."""
        py = ctx.py
        if isinstance(dim, DimStatic):
            size_vid = number_literal(py, dim.size)
        elif isinstance(dim, DimDynamic):
            size_vid = self._dynamic_size_expr(py, dim.size_name)
        else:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"plate:dim-kind:{type(dim).__name__}"],
            )
        ctx.emitted_plate_names.add(dim.name)
        return call(
            py,
            attribute(py, ("numpyro", "plate")),
            positional=(string_literal(py, dim.name), size_vid),
        )

    def _dynamic_size_expr(self, py: PyCtx, size_name: str) -> str:
        """Render the size expression for a
        [`DimDynamic`][quivers.transpile.ir.DimDynamic].

        Dynamic-axis sizes come from the data: the canonical pattern is
        ``<obs>.shape[0]``. Here we use the bare identifier
        (`size_name`) when it begins with ``N_``, since `Lower`
        synthesises ``N_<axis>`` for dynamic axes; otherwise we
        defer to ``<obs>.shape[0]`` via the first observed name.
        """
        # The canonical LDA emit reads `w.shape[0]` for the dynamic
        # word-observation axis. We match that idiom: prefer
        # `<observed>.shape[0]` when `size_name` is the synthesised
        # `N_<axis>` form.
        if size_name.startswith("N_"):
            axis = size_name[len("N_") :]
            observed_for_axis = self._observed_for_axis(axis)
            if observed_for_axis is not None:
                return self._shape_zero(py, observed_for_axis)
        return identifier(py, size_name)

    def _observed_for_axis(self, axis: str) -> str | None:
        """Heuristic for the canonical NumPyro idiom:
        ``Word_obs`` axis maps to the observed variable ``w``.

        Returns ``None`` when no mapping is known; the caller then
        falls back to the bare ``N_<axis>`` identifier."""
        # The canonical LDA emit pairs `Word_obs` with `w`; the broader
        # convention is to map an axis named `<X>_obs` to the lowercase
        # first character.
        axis_to_obs: dict[str, str] = {
            "Word_obs": "w",
        }
        return axis_to_obs.get(axis)

    def _shape_zero(self, py: PyCtx, var_name: str) -> str:
        """Emit ``<var>.shape[0]``."""
        attr = py.v(py.fresh("attr"), "attribute")
        py.e(attr, identifier(py, var_name), "object")
        py.e(attr, identifier(py, "shape"), "attribute")
        subs = py.v(py.fresh("subs"), "subscript")
        py.e(subs, attr, "value")
        py.e(subs, number_literal(py, 0), "subscript")
        return subs

    def _numpyro_sample_call(
        self,
        ctx: _NumPyroCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        obs_name: str | None,
    ) -> str:
        """Build ``numpyro.sample("<name>", <dist>, [obs=<obs>])``."""
        py = ctx.py
        dist_call = self._distribution_call(ctx, family, args, arg_names)
        sample_callee = attribute(py, ("numpyro", "sample"))
        positional = (string_literal(py, name), dist_call)
        keyword: tuple[tuple[str, str], ...] = ()
        if obs_name is not None:
            keyword = (("obs", identifier(py, obs_name)),)
        return call(
            py,
            sample_callee,
            positional=positional,
            keyword=keyword,
        )

    def _distribution_call(
        self,
        ctx: _NumPyroCtx,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str:
        """Build ``numpyro.distributions.<TargetName>(arg=value, ...)``.

        Looks up the target distribution name in `FAMILY_META`, applies
        any `arg_aliases` rename, and emits keyword arguments using
        each family's torch ``arg_constraints`` parameter names.

        Wrapper-family handling: when ``family == "Truncated"`` and the
        first arg is an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        pointing at a morphism with ``~ Normal(loc, scale)``, the
        emission collapses to
        ``numpyro.distributions.TruncatedNormal(loc=..., scale=...,
        low=lo, high=hi)`` per §10.10 of the spec.
        """
        py = ctx.py
        if family not in FAMILY_META:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"family:unknown:{family}"],
            )
        meta = FAMILY_META[family]
        if _BACKEND not in meta.target_names:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"family:no-target-name:{family}"],
            )

        # Wrapper-family inline emission for Truncated(base, lo, hi).
        wrapper_call = self._maybe_wrapper_call(
            ctx, family, args, arg_names
        )
        if wrapper_call is not None:
            return wrapper_call

        target_name = meta.target_names[_BACKEND]
        callee = attribute(
            py, ("numpyro", "distributions", target_name)
        )
        aliases = meta.arg_aliases.get(_BACKEND, {})
        keyword: list[tuple[str, str]] = []
        for arg, name in zip(args, arg_names, strict=False):
            emitted_name = aliases.get(name, name)
            value_vid = self._render_arg(ctx, arg)
            keyword.append((emitted_name, value_vid))
        return call(py, callee, keyword=tuple(keyword))

    def _maybe_wrapper_call(
        self,
        ctx: _NumPyroCtx,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str | None:
        """Inline-render a `Truncated(base, lo, hi)` call as
        ``TruncatedNormal(loc=..., scale=..., low=lo, high=hi)``
        when ``base`` resolves to a `Normal(...)` morphism.

        Falls through (returns ``None``) when the family is not a
        wrapper or the base resolution does not match a known
        truncation specialisation; the general path emits the wrapper
        via `TruncatedDistribution`.
        """
        if family != "Truncated":
            return None
        if not args:
            return None
        base = args[0]
        if not isinstance(base, IRArgFamilyRef):
            return None
        py = ctx.py
        decl = ctx.morphisms.get(base.name)
        if decl is None or decl.init_family is None:
            # Fall back to TruncatedDistribution(<base_ref>, low, high).
            return self._truncated_distribution_call(
                ctx, base_ref=identifier(py, base.name),
                args=args[1:], arg_names=arg_names[1:],
            )
        inner_family = decl.init_family.family
        inner_meta = FAMILY_META.get(inner_family)
        if inner_meta is None:
            raise UnsupportedConstruct(
                "qvr-numpyro",
                [f"family:wrapper-inner-unknown:{inner_family}"],
            )
        # Truncated-specialisation registry: when NumPyro publishes a
        # dedicated `Truncated<Inner>` class for the inner family, prefer
        # that emission over the generic `TruncatedDistribution(<inner>,
        # ...)` wrapper.
        specialised = _NUMPYRO_TRUNCATED_SPECIALISATION.get(inner_family)
        if specialised is not None:
            return self._truncated_specialised_call(
                ctx,
                target=specialised,
                base_decl=decl,
                inner_arg_names=tuple(
                    inner_meta.distribution_class.arg_constraints.keys()
                    if isinstance(
                        inner_meta.distribution_class.arg_constraints, dict
                    )
                    else ()
                ),
                rest_args=args[1:],
                rest_names=arg_names[1:],
            )
        # General wrapper: build the inner distribution call and pass
        # it as the first positional arg to TruncatedDistribution.
        inner_call = self._inner_morphism_call(ctx, decl)
        return self._truncated_distribution_call(
            ctx, base_ref=inner_call,
            args=args[1:], arg_names=arg_names[1:],
        )

    def _truncated_specialised_call(
        self,
        ctx: _NumPyroCtx,
        *,
        target: str,
        base_decl,
        inner_arg_names: tuple[str, ...],
        rest_args: tuple[IRArg, ...],
        rest_names: tuple[str, ...],
    ) -> str:
        """``numpyro.distributions.<TruncatedInner>(<inner_args>,
        low=lo, high=hi)`` for inner families that NumPyro publishes
        a specialised truncated wrapper for."""
        py = ctx.py
        callee = attribute(
            py, ("numpyro", "distributions", target)
        )
        base_args = base_decl.init_family.args or ()
        keyword: list[tuple[str, str]] = []
        for arg, name in zip(base_args, inner_arg_names, strict=False):
            keyword.append((name, arg_expr(py, arg)))
        for arg, name in zip(rest_args, rest_names, strict=False):
            keyword.append((name, self._render_arg(ctx, arg)))
        return call(py, callee, keyword=tuple(keyword))

    def _inner_morphism_call(self, ctx: _NumPyroCtx, decl) -> str:
        """Build a `numpyro.distributions.<Inner>(...)` call from the
        morphism's `~ Family(...)` init clause."""
        py = ctx.py
        inner_family = decl.init_family.family
        inner_meta = FAMILY_META[inner_family]
        target = inner_meta.target_names[_BACKEND]
        cls_attr = inner_meta.distribution_class.arg_constraints
        names: tuple[str, ...]
        if isinstance(cls_attr, dict):
            names = tuple(cls_attr.keys())
        else:
            names = ()
        callee = attribute(
            py, ("numpyro", "distributions", target)
        )
        keyword: list[tuple[str, str]] = []
        for arg, name in zip(
            decl.init_family.args or (), names, strict=False
        ):
            keyword.append((name, arg_expr(py, arg)))
        return call(py, callee, keyword=tuple(keyword))

    def _truncated_distribution_call(
        self,
        ctx: _NumPyroCtx,
        *,
        base_ref: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
    ) -> str:
        """``numpyro.distributions.TruncatedDistribution(<base>, low=lo,
        high=hi)``."""
        py = ctx.py
        callee = attribute(
            py, ("numpyro", "distributions", "TruncatedDistribution")
        )
        keyword: list[tuple[str, str]] = []
        for arg, name in zip(args, arg_names, strict=False):
            keyword.append((name, self._render_arg(ctx, arg)))
        return call(
            py,
            callee,
            positional=(base_ref,),
            keyword=tuple(keyword),
        )

    def _render_arg(
        self, ctx: _NumPyroCtx, arg: IRArg
    ) -> str:
        """Render one IR arg to a Python expression vertex."""
        py = ctx.py
        if isinstance(arg, IRArgNumber):
            return number_literal(py, arg.value)
        if isinstance(arg, IRArgRef):
            if not arg.indices:
                return identifier(py, arg.name)
            return self._render_indexed_ref(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self.broadcast(ctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self.render_list(ctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self.render_matrix(ctx, arg)
        if isinstance(arg, IRArgFamilyRef):
            decl = ctx.morphisms.get(arg.name)
            if decl is None or decl.init_family is None:
                return identifier(py, arg.name)
            return self._inner_morphism_call(ctx, decl)
        raise UnsupportedConstruct(
            "qvr-numpyro",
            [f"arg:unknown:{type(arg).__name__}"],
        )

    def _render_indexed_ref(
        self, ctx: _NumPyroCtx, arg: IRArgRef
    ) -> str:
        """Render ``name[i0][i1]...`` as nested ``subscript`` vertices."""
        py = ctx.py
        current = identifier(py, arg.name)
        for idx in arg.indices:
            subs = py.v(py.fresh("subs"), "subscript")
            py.e(subs, current, "value")
            py.e(subs, self._render_arg(ctx, idx), "subscript")
            current = subs
        return current

    def _render_shape_tuple(
        self, py: PyCtx, shape: tuple[int, ...]
    ) -> str:
        """Emit ``(K,)`` for a 1-tuple or ``(R, C)`` for a 2-tuple.

        Tree-sitter Python's `tuple` node carries the punctuation via
        `chose-alt-fingerprint` / `chose-alt-child-kinds` / `ptrace-*`
        constraints; without them the emitter drops the trailing
        comma on a 1-tuple and produces a `parenthesized_expression`
        rather than a `tuple`. The constraints below mirror what the
        Python tree-sitter parser produces when reading
        ``(<n>,)`` / ``(<r>, <c>)`` literally.
        """
        tup = py.v(py.fresh("tup"), "tuple")
        n = len(shape)
        if n == 0:
            py.constraint(tup, "chose-alt-fingerprint", "()")
            py.constraint(tup, "ptrace-0", "T(")
            py.constraint(tup, "ptrace-1", "T)")
            return tup
        kind_list = " ".join("integer" for _ in range(n))
        if n == 1:
            py.constraint(tup, "chose-alt-fingerprint", "( ,)")
            py.constraint(tup, "ptrace-0", "T(")
            py.constraint(tup, "ptrace-1", "Cinteger")
            py.constraint(tup, "ptrace-2", "T,")
            py.constraint(tup, "ptrace-3", "T)")
        else:
            fingerprint = "( " + " ".join("," for _ in range(n - 1)) + " )"
            py.constraint(tup, "chose-alt-fingerprint", fingerprint)
            py.constraint(tup, "ptrace-0", "T(")
            slot = 1
            for i in range(n):
                py.constraint(tup, f"ptrace-{slot}", "Cinteger")
                slot += 1
                if i < n - 1:
                    py.constraint(tup, f"ptrace-{slot}", "T,")
                    slot += 1
            py.constraint(tup, f"ptrace-{slot}", "T)")
        py.constraint(tup, "chose-alt-child-kinds", kind_list)
        for size in shape:
            py.e(tup, number_literal(py, size), "child_of")
        return tup

    def _expression_statement(self, py: PyCtx, expr: str) -> str:
        """Wrap an expression in an ``expression_statement`` vertex."""
        es = py.v(py.fresh("es"), "expression_statement")
        py.e(es, expr, "child_of")
        return es

    def _assignment_statement(
        self, py: PyCtx, lhs_name: str, rhs: str
    ) -> str:
        """``<lhs_name> = <rhs>``."""
        asn = py.v(py.fresh("asn"), "assignment")
        py.e(asn, identifier(py, lhs_name), "left")
        py.e(asn, rhs, "right")
        return asn

    def _deterministic_statement(
        self, ctx: _NumPyroCtx, node: IRDeterministic
    ) -> str:
        """``<name> = <expr>`` for an
        [`IRDeterministic`][quivers.transpile.ir.IRDeterministic]
        let-binding."""
        py = ctx.py
        rhs = render_let_expr_python(py, node.expr)
        return self._assignment_statement(py, node.name, rhs)

    def _score_statement(
        self, ctx: _NumPyroCtx, body_vid: str, node: IRScore
    ) -> None:
        """``<name> = <expr>``; ``numpyro.factor("<name>", <name>)``."""
        py = ctx.py
        rhs = render_let_expr_python(py, node.expr)
        py.e(
            body_vid,
            self._assignment_statement(py, node.name, rhs),
            "child_of",
        )
        factor_call = call(
            py,
            attribute(py, ("numpyro", "factor")),
            positional=(
                string_literal(py, node.name),
                identifier(py, node.name),
            ),
        )
        py.e(
            body_vid,
            self._expression_statement(py, factor_call),
            "child_of",
        )

    def _return_statement(
        self, ctx: _NumPyroCtx, body_vid: str, names: tuple[str, ...]
    ) -> None:
        """``return <var>`` (single) or ``return (<a>, <b>, ...)`` (tuple)."""
        if not names:
            return
        py = ctx.py
        rs = py.v(py.fresh("ret"), "return_statement")
        if len(names) == 1:
            py.e(rs, identifier(py, names[0]), "child_of")
        else:
            elist = py.v(py.fresh("elist"), "expression_list")
            for name in names:
                py.e(elist, identifier(py, name), "child_of")
            py.e(rs, elist, "child_of")
        py.e(body_vid, rs, "child_of")

    def _dedupe_plate(
        self,
        ctx: _NumPyroCtx,
        plate: Plate,
        latent_name: str,
    ) -> Plate:
        """Return a new Plate whose batch_dims carry a ``_<latent>``
        suffix only on those plate-names that have already been
        emitted in this `render` call. Names that have not been used
        before pass through unchanged.

        The canonical NumPyro LDA emit uses ``"Doc_z"`` for the
        marginalized-latent plate because ``"Doc"`` was already used
        by the prior plate; for mixture models without such reuse the
        latent's plate name is just the source axis name.
        """
        seen = ctx.emitted_plate_names
        new_batch: list = []
        for dim in plate.batch_dims:
            dim_name = str(dim.name)
            renamed: str = (
                f"{dim_name}_{latent_name}" if dim_name in seen else dim_name
            )
            if isinstance(dim, DimStatic):
                new_batch.append(DimStatic(size=int(dim.size), name=renamed))
            elif isinstance(dim, DimDynamic):
                new_batch.append(
                    DimDynamic(size_name=str(dim.size_name), name=renamed)
                )
            else:
                new_batch.append(dim)
        return Plate(
            event_dims=plate.event_dims,
            batch_dims=tuple(new_batch),
        )


# ---------------------------------------------------------------------------
# Renderer-local context
# ---------------------------------------------------------------------------


class _NumPyroCtx(_RenderCtx):
    """NumPyro-specific extension of
    [`_RenderCtx`][quivers.transpile.renderers._base._RenderCtx]
    carrying the Python-helpers context and the currently-active body
    vertex used by nested-plate emission."""

    def __init__(
        self,
        *,
        sb: panproto.SchemaBuilder,
        morphisms: dict,
        lets: dict,
        py: PyCtx,
        observed_names: set[str],
    ) -> None:
        super().__init__(sb=sb, morphisms=morphisms, defines=lets)
        self.py = py
        self.observed_names = observed_names
        self.current_body: str | None = None
        self.emitted_plate_names: set[str] = set()


def _as_numpyro_ctx(ctx: _RenderCtx) -> _NumPyroCtx:
    """Narrow a base `_RenderCtx` to the NumPyro extension. The
    renderer always constructs `_NumPyroCtx` instances, so this is a
    safe assertion at the boundary."""
    if not isinstance(ctx, _NumPyroCtx):
        raise UnsupportedConstruct(
            "qvr-numpyro",
            ["ctx:type-mismatch"],
        )
    return ctx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Params:
    """Split function parameters into positional vs. defaulted lists."""

    def __init__(
        self,
        *,
        positional: tuple[str, ...],
        defaulted: tuple[str, ...],
    ) -> None:
        self.positional = positional
        self.defaulted = defaulted


__all__ = ["NumPyroRenderer"]
