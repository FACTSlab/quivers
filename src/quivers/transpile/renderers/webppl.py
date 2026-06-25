"""[`WebPPLRenderer`][quivers.transpile.renderers.webppl.WebPPLRenderer]: IR to WebPPL source.

The renderer subclasses
[`RendererBase`][quivers.transpile.renderers._base.RendererBase] and
implements the four dispatch points (`declare`, `sample`,
`marginalize`, `broadcast`) plus the two arg helpers (`render_list`,
`render_matrix`). Distribution names live in
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META]'s
`target_names["webppl"]`; no per-renderer family table. Support
classification dispatches on the predicates exported from
[`ir.py`][quivers.transpile.ir]; the renderer dispatches purely on
support predicates and FAMILY_META lookups.

The WebPPL-specific layout decisions:

* Program shape: a single
  `var model = function(<inputs>) { ... };` declaration. The
  function parameters are the IR's
  [`IRDataInput`][quivers.transpile.ir.IRDataInput] names in
  declaration order; the body's statements come from the IR body.
* `declare` is a no-op: WebPPL's `var <name> = sample(...);`
  binding is constructed by `sample` directly.
* Sample-step plate loops: per batch dim, a `repeat(N, function() {
  return sample(<dist>); })` when no arg uses the per-element
  index, or `mapIndexed(function(m, _) { return sample(<dist
  using m>); }, repeat(N, 0))` when at least one arg refers to a
  name whose binding plate shares the surrounding batch dim.
* Observe: when a `via` fibration is present, the renderer threads
  the per-observe loop variable through every reference whose
  binding plate matches the fibration's group plate. The emit
  shape is `mapIndexed(function(n, <obs>_n) { observe(<dist>,
  <obs>_n); }, <obs>)`.
* Marginalize: WebPPL has no native enumeration; the inherited
  [`explicit_latent_scope`][quivers.transpile.renderers._base.RendererBase.explicit_latent_scope]
  helper lowers
  [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] to an
  explicit `IRSample(latent)` plus the scope body inline. The
  scope-body renderer then runs each scope node through the
  ordinary `sample` dispatch.
* Broadcast: `repeat(K, function() { return <value>; })` per
  target-shape entry, nested for higher rank.
* List / matrix args: `[<e0>, <e1>, ...]` for list; nested JS
  arrays for matrix.
* `IRArgFamilyRef`: WebPPL has no generic truncation idiom; the
  renderer resolves the referenced morphism's `init_family` clause
  and emits the inner distribution call inline.
* Every emitted statement carries the WebPPL terminating `;` per
  the JavaScript convention.
"""

from __future__ import annotations

import pathlib
from typing import Callable

import panproto

from quivers.dsl.ast_nodes import Expr, LetDecl, Module, MorphismDecl
from quivers.dsl.ast_nodes.let_expressions import (
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
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import parser_registry, target_protocol
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
from quivers.transpile.renderers._base import (
    BlockKind,
    RendererBase,
    SchemaFragment,
    _RenderCtx,
    assert_no_dangling_refs,
)
from quivers.transpile.renderers._javascript_helpers import (
    render_let_expr_javascript,
)


class _JsLetCtx:
    """Bridge `_RenderCtx.sb` to the
    [`render_let_expr_javascript`][quivers.transpile.renderers._javascript_helpers.render_let_expr_javascript]
    helper's protocol (`v`, `e`, `lit`, `constraint`, `fresh`).

    Carries the object-name -> static-cardinality map consulted when
    unrolling [`LetExprFactor`][quivers.dsl.ast_nodes.LetExprFactor]
    and the `target` tag used in error messages.

    The WebPPL IR-walk operates over
    [`_RenderCtx`][quivers.transpile.renderers._base._RenderCtx]; the
    let-expression helper expects a small carrier exposing
    [`panproto.SchemaBuilder`][panproto.SchemaBuilder] operations
    under terse method names. The shim keeps the helper independent
    of any specific renderer class.
    """

    target: str = "webppl"

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

    def v(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str) -> None:
        self._sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)


class WebPPLRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
    WebPPL [`panproto.Schema`][panproto.Schema].

    Subclasses
    [`RendererBase`][quivers.transpile.renderers._base.RendererBase]
    and overrides the four dispatch points
    (`declare`, `sample`, `marginalize`, `broadcast`) plus the two
    list / matrix arg helpers per the spec.
    """

    target: str = "webppl"

    def __init__(self, *, source_module: Module | None = None) -> None:
        """Initialise the renderer.

        Parameters
        ----------
        source_module
            Optional original [`Module`][quivers.dsl.ast_nodes.Module]
            the IR came from. Carries `MorphismDecl` / `LetDecl`
            entries the renderer reads when resolving
            [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
            args. When omitted, the renderer raises on
            `IRArgFamilyRef` rather than guess.
        """
        self._source_module = source_module
        # Per-render scratch state. Reset at the top of every
        # `render` call so a single renderer instance can serve
        # multiple programs back-to-back.
        self._fresh_n = 0
        # The body block id under the top-level function expression.
        self._body_vid: str = ""
        # Per-render map from a sample / observe / input name to its
        # binding plate. Powers the index-dependent / broadcast
        # decisions on later sample steps.
        self._binding_plates: dict[str, Plate] = {}
        # Per-render map from a sample / observe / input name to its
        # support constraint. Powers broadcast detection (a scalar
        # ref into a vector-arg slot needs `repeat` wrapping).
        self._binding_supports: dict[str, ConstraintSpec] = {}
        # `via` fibrations seen on the active observe scope; used by
        # ref substitution to thread the per-observe loop var through
        # references whose binding plate matches the fibration's
        # group axis.
        self._observe_via: str | None = None
        # The per-observe loop variable name, when an observe's
        # mapIndexed body is being emitted; None outside that scope.
        self._observe_loop_var: str | None = None
        # The active marginalize / group plate batch_dim names. When
        # an observe inside a marginalize scope renders, refs to
        # names whose binding plate aligns with this tuple get
        # `via[loop_var]` indexed through the fibration; refs to
        # other plates pass through unchanged.
        self._group_plate_axes: tuple[str, ...] = ()
        # Static-cardinality table snapshotted from `IRProgram.cards`
        # at the top of every `render` call. The let-expression shim
        # reads this when unrolling `LetExprFactor` binders whose
        # axis size needs to be known at render time.
        self._cards: dict[str, int] = {}
        # Counter for fresh loop-variable names generated by the
        # deterministic-binding lift (`__i_<n>`); kept independent of
        # the schema-vertex counter so the names stay short and
        # legible in the rendered source.
        self._lift_n = 0
        # The first observe step's plate seen during the walk, used
        # as the fallback shape for lifted deterministic bindings
        # whose data-input references the IR records as scalar but
        # which are array-shaped at runtime.
        self._observe_array_fallback: Plate | None = None
        # Names registered as IRDataInputs at render start; used by
        # ref-substitution to discriminate function parameters from
        # in-scope sample bindings.
        self._function_parameters_state: set[str] = set()

    # ------------------------------------------------------------------
    # Abstract overrides: target_protocol + four dispatch points.
    # ------------------------------------------------------------------

    def target_protocol(self) -> panproto.Protocol:
        return target_protocol("javascript")

    # ----- the full render override -----

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Walk the IR and emit a WebPPL schema.

        Override of the base `render` so the program-level shape
        (a single `var model = function(<inputs>) { ... };`
        declaration) is built once per call.
        """
        assert_no_dangling_refs(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        morphisms, lets = self._resolve_morphisms_and_lets()
        ctx = _RenderCtx(sb=sb, morphisms=morphisms, lets=lets)
        # Reset per-render state.
        self._fresh_n = 0
        self._binding_plates = {}
        self._binding_supports = {}
        self._observe_via = None
        self._observe_loop_var = None
        self._lift_n = 0
        # Snapshot the IR's static-cardinality table so the let-expr
        # shim can resolve factor binders.
        self._cards = dict(ir.cards)
        # Choose the first observe plate as the lift-fallback plate;
        # lifted deterministic bindings whose data inputs the IR
        # records as scalar still flow through this plate's `mapIndexed`
        # so the per-element computation runs over the right array.
        self._observe_array_fallback = self._first_observe_plate(ir)
        # Program root: a single var-decl wrapping a function expression.
        ctx.sb.vertex("prog", "program")
        # WebPPL's `dists` module ships `Gaussian`, `Beta`, `Categorical`,
        # etc. as built-in distributions but lacks `Logistic`,
        # `BetaBinomial`, `HalfStudentT`, `Kumaraswamy`, `LKJCholesky`,
        # and `ContinuousBernoulli`. When the IR samples or observes
        # from any of these, graft the helper at
        # [`runtime_webppl.js`][quivers.transpile.runtime_webppl] onto
        # the source above the `var model = function (...) {...};`
        # declaration so the body's `sample(<Family>({...}))` call
        # sites resolve through normal JS name lookup.
        if any(
            _ir_uses_family(ir.body, f)
            for f in _WEBPPL_RUNTIME_HELPER_FAMILIES
        ):
            _graft_runtime_webppl_helper(ctx.sb, self, "prog")
        var_decl = self._fresh(ctx, "vd")
        ctx.sb.vertex(var_decl, "variable_declaration")
        declarator = self._fresh(ctx, "dr")
        ctx.sb.vertex(declarator, "variable_declarator")
        model_name = self._ident(ctx, "model")
        ctx.sb.edge(declarator, model_name, "name")
        fn = self._fresh(ctx, "fn")
        ctx.sb.vertex(fn, "function_expression")
        params = self._fresh(ctx, "ps")
        ctx.sb.vertex(params, "formal_parameters")
        self._function_parameters.clear()
        for inp in ir.inputs:
            ctx.sb.edge(params, self._ident(ctx, inp.name), "child_of")
            self._binding_plates[inp.name] = inp.plate
            self._binding_supports[inp.name] = inp.constraint
            self._function_parameters.add(inp.name)
        body = self._fresh(ctx, "body")
        ctx.sb.vertex(body, "statement_block")
        ctx.sb.edge(fn, params, "parameters")
        ctx.sb.edge(fn, body, "body")
        ctx.sb.edge(declarator, fn, "value")
        ctx.sb.edge(var_decl, declarator, "child_of")
        ctx.sb.edge("prog", var_decl, "child_of")
        self._body_vid = body
        # Walk the body. Inputs were registered above; declare is a
        # no-op for WebPPL, so we dispatch only the body nodes.
        for node in ir.body:
            self._dispatch_node(ctx, node)
        return ctx.sb.build()

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
        """No-op: WebPPL's `var <name> = sample(...);` binding is
        emitted by `sample` directly. The declaration table threads
        the name's binding plate / support into the renderer's
        bookkeeping so later sample steps can index into the name
        correctly.
        """
        del ctx, block
        self._binding_plates[name] = plate
        self._binding_supports[name] = constraint
        return ""

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
        """Emit a `sample` / `observe` statement for a sample / observe
        step.

        Wraps the call in `repeat(B, function() { ... })` per batch
        dim when no arg uses the per-element index; uses
        `mapIndexed(function(m, _) { ... }, repeat(B, 0))` when an
        arg refers to a name whose binding plate aligns with the
        surrounding plate's batch dims.

        For an observed step the emit shape is
        `mapIndexed(function(n, <obs>_n) { observe(<dist>, <obs>_n);
        }, <obs>);` when batched; the unbatched case emits a bare
        `observe(<dist>, <obs>);` statement.
        """
        del constraint  # Constraint shaped the (no-op) declaration.
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [f"family:unknown:{family}"],
            )
        webppl_name = meta.target_names.get("webppl")
        if webppl_name is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [f"family:no-webppl-target:{family}"],
            )
        injected_args, injected_arg_names = _inject_webppl_specific_args(
            family, args, arg_names
        )
        if observed:
            return self._emit_observe(
                ctx, name, meta, webppl_name,
                injected_args, injected_arg_names, plate,
            )
        return self._emit_sample(
            ctx, name, meta, webppl_name,
            injected_args, injected_arg_names, plate,
        )

    def _emit_gp_block(
        self,
        ctx: _RenderCtx,
        node,
    ) -> None:
        """Emit a Gaussian-process sample as three WebPPL var-decls:

            var __gp_mean_<name> = _qvr_zeros(N);
            var __gp_cov_<name>  = _qvr_rbf_kernel(x, length_scale, jitter);
            var <name> = sample(MultivariateGaussian({mu: __gp_mean_<name>,
                                                       cov: __gp_cov_<name>}));

        The ``_qvr_rbf_kernel`` / ``_qvr_zeros`` helpers live in
        [`runtime_webppl.js`][quivers.transpile.runtime_webppl] and
        are grafted into the emit through the existing runtime-helper
        graft path; GP is added to the helper-family set so the
        helpers appear above the model body when the IR uses GP.
        """
        if len(node.args) != 2 or not isinstance(
            node.args[1], IRArgKernel
        ):
            raise UnsupportedConstruct(
                "qvr-webppl",
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"family:GP:kernel:{kernel_arg.kernel}: only rbf "
                    f"is implemented"
                ],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{node.name}"
        cov_name = f"__gp_cov_{node.name}"
        # var __gp_mean_<name> = _qvr_zeros(N);
        mean_rhs = self._call(
            ctx,
            self._ident(ctx, "_qvr_zeros"),
            (self._number_literal(ctx, n),),
        )
        self._emit_var_decl(ctx, self._body_vid, mean_name, mean_rhs)
        # var __gp_cov_<name> = _qvr_rbf_kernel(x, ls, jitter);
        cov_rhs = self._call(
            ctx,
            self._ident(ctx, "_qvr_rbf_kernel"),
            (
                self._ident(ctx, x),
                self._number_literal(ctx, ls),
                self._number_literal(ctx, jitter),
            ),
        )
        self._emit_var_decl(ctx, self._body_vid, cov_name, cov_rhs)
        # var <name> = sample(MultivariateGaussian({mu, cov}));
        params = self._object_literal(
            ctx,
            (
                ("mu", self._ident(ctx, mean_name)),
                ("cov", self._ident(ctx, cov_name)),
            ),
        )
        dist_call = self._call(
            ctx,
            self._ident(ctx, "MultivariateGaussian"),
            (params,),
        )
        sample_call = self._call(
            ctx, self._ident(ctx, "sample"), (dist_call,),
        )
        self._emit_var_decl(
            ctx, self._body_vid, node.name, sample_call,
        )

    def _emit_sample(
        self,
        ctx: _RenderCtx,
        name: str,
        meta: FamilyMeta,
        webppl_name: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> SchemaFragment:
        """Emit `var <name> = <maybe-batched-call>;` for a latent sample."""
        # Decide the batching strategy: `repeat` vs `mapIndexed`.
        loop_name = self._loop_name_for(name, plate)
        index_dependent = self._args_use_surrounding_index(args, plate)
        # Build the inner sample call: `sample(<Dist>({...}))`.
        # When index-dependent, arg refs whose binding plate aligns
        # with the surrounding plate get indexed by the loop var.
        rendered_args = self._render_arg_tuple(
            ctx, args, arg_names, meta, plate, loop_name, index_dependent
        )
        dist_obj = self._object_literal(ctx, rendered_args)
        dist_call = self._call(
            ctx, self._ident(ctx, webppl_name), (dist_obj,)
        )
        sample_call = self._call(
            ctx, self._ident(ctx, "sample"), (dist_call,)
        )
        # Wrap the call per the batching strategy.
        value_vid = self._wrap_for_batch(
            ctx, sample_call, plate, loop_name, index_dependent
        )
        # Bind the result to `var <name> = ...;`.
        self._emit_var_decl(ctx, self._body_vid, name, value_vid)
        # Record the binding plate so downstream refs see the right shape.
        return value_vid

    def _emit_observe(
        self,
        ctx: _RenderCtx,
        name: str,
        meta: FamilyMeta,
        webppl_name: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
    ) -> SchemaFragment:
        """Emit `mapIndexed(function(n, <obs>_n) { observe(<dist>,
        <obs>_n); }, <obs>);` for a batched observe; emit a bare
        `observe(<dist>, <obs>);` statement for the unbatched form.

        The per-observe loop variable is bound to `n` (mapIndexed's
        first arg); the per-element observed value is bound to
        `<name>_n`. The body's `<dist>` rendering threads `n` through
        the `via` fibration when present.
        """
        # When unbatched, emit a top-level observe(...) statement.
        if not plate.batch_dims:
            rendered_args = self._render_arg_tuple(
                ctx, args, arg_names, meta, plate,
                loop_name=None, index_dependent=False,
            )
            dist_obj = self._object_literal(ctx, rendered_args)
            dist_call = self._call(
                ctx, self._ident(ctx, webppl_name), (dist_obj,)
            )
            observe_call = self._call(
                ctx,
                self._ident(ctx, "observe"),
                (dist_call, self._ident(ctx, name)),
            )
            self._emit_expression_statement(
                ctx, self._body_vid, observe_call
            )
            return observe_call
        # Batched observe: mapIndexed over the observed data.
        # The loop var is `n`; the per-element bound var is `<name>_n`.
        loop_var = "n"
        per_elem_var = f"{name}_n"
        # Push the per-observe loop state so ref rendering can use it.
        prev_loop = self._observe_loop_var
        prev_via = self._observe_via
        self._observe_loop_var = loop_var
        # The observe's `via` is not surfaced on IRObserve directly
        # here; the caller passes plate. The dispatch site populates
        # `_observe_via` before calling _emit_observe via the
        # IRObserve node.
        try:
            rendered_args = self._render_arg_tuple(
                ctx, args, arg_names, meta, plate,
                loop_name=loop_var, index_dependent=True,
            )
            dist_obj = self._object_literal(ctx, rendered_args)
            dist_call = self._call(
                ctx, self._ident(ctx, webppl_name), (dist_obj,)
            )
            observe_call = self._call(
                ctx,
                self._ident(ctx, "observe"),
                (dist_call, self._ident(ctx, per_elem_var)),
            )
            lambda_body = self._fresh(ctx, "obody")
            ctx.sb.vertex(lambda_body, "statement_block")
            self._emit_expression_statement(
                ctx, lambda_body, observe_call
            )
            lambda_expr = self._function_expression(
                ctx, (loop_var, per_elem_var), lambda_body
            )
            mi_call = self._call(
                ctx,
                self._ident(ctx, "mapIndexed"),
                (lambda_expr, self._ident(ctx, name)),
            )
            self._emit_expression_statement(
                ctx, self._body_vid, mi_call
            )
            return mi_call
        finally:
            self._observe_loop_var = prev_loop
            self._observe_via = prev_via

    # ----- marginalize dispatch -----

    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Lower an
        [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] scope
        into an explicit
        [`IRSample`][quivers.transpile.ir.IRSample] for the latent
        plus the scope body inline.

        WebPPL has no native enumeration; the inherited
        `explicit_latent_scope` helper produces the rewritten node
        sequence, which is then dispatched through the ordinary
        node-walk so each scope node sees the same sample / observe
        emission path as a top-level node.

        The marginalize plate's batch axes are pushed onto the
        renderer's group-plate stack so a scope observe's ref
        substitution can apply the `via` fibration to refs whose
        binding plate matches the group plate.
        """
        rewritten = self.explicit_latent_scope(node)
        prev_group = self._group_plate_axes
        self._group_plate_axes = tuple(
            str(d.name) for d in node.plate.batch_dims
        )
        try:
            for inner in rewritten:
                self._dispatch_node(ctx, inner)
        finally:
            self._group_plate_axes = prev_group
        return ""

    # ----- broadcast dispatch -----

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit a WebPPL broadcast call:
        `repeat(K, function() { return <value>; })` for 1D, nested
        for higher rank.

        Rank-0 returns the value unchanged. The renderer raises with
        a precise kind tag for rank > 2 because the canonical
        WebPPL Dirichlet / MVN call shapes only exercise rank-1 and
        rank-2 broadcast positions.
        """
        if len(target_shape) == 0:
            return self._render_arg(ctx, value)
        if len(target_shape) > 2:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"broadcast:rank:{len(target_shape)}: WebPPL "
                    f"broadcast supports up to rank 2"
                ],
            )
        # Innermost: a function() returning the value.
        inner_value = self._render_arg(ctx, value)
        current = inner_value
        for size in reversed(target_shape):
            body = self._fresh(ctx, "rbody")
            ctx.sb.vertex(body, "statement_block")
            self._emit_return_statement(ctx, body, current)
            lam = self._function_expression(ctx, (), body)
            current = self._call(
                ctx,
                self._ident(ctx, "repeat"),
                (self._number_literal(ctx, size), lam),
            )
        return current

    # ----- arg rendering helpers (render_list, render_matrix) -----

    def render_list(
        self,
        ctx: _RenderCtx,
        arg: IRArgList,
    ) -> SchemaFragment:
        """Per §10.9 of the spec, WebPPL list args render as a JS
        array literal `[<e0>, <e1>, ...]`.
        """
        arr = self._fresh(ctx, "arr")
        ctx.sb.vertex(arr, "array")
        for element in arg.elements:
            child_vid = self._render_arg(ctx, element)
            ctx.sb.edge(arr, child_vid, "child_of")
        return arr

    def render_matrix(
        self,
        ctx: _RenderCtx,
        arg: IRArgMatrix,
    ) -> SchemaFragment:
        """Per §10.9 of the spec, WebPPL matrix args render as nested
        JS array literals `[[<r0_e0>, ...], [<r1_e0>, ...], ...]`.
        """
        outer = self._fresh(ctx, "arr")
        ctx.sb.vertex(outer, "array")
        for row in arg.rows:
            row_vid = self.render_list(ctx, row)
            ctx.sb.edge(outer, row_vid, "child_of")
        return outer

    # ------------------------------------------------------------------
    # IR-walk override: WebPPL-specific dispatch needs to thread the
    # observe's `via` fibration through to ref-rendering.
    # ------------------------------------------------------------------

    def _dispatch_node(self, ctx: _RenderCtx, node: IRNode) -> None:
        """WebPPL-specific dispatch.

        Tracks the active observe's `via` fibration so ref rendering
        can thread the per-element loop variable through references
        whose binding plate matches the fibration's group axis.
        """
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(ctx, node)
                return
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
                ctx, node.name, node.constraint, node.plate, block="data"
            )
            prev_via = self._observe_via
            self._observe_via = node.via
            try:
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
            finally:
                self._observe_via = prev_via
            return
        if isinstance(node, IRDataInput):
            self.declare(
                ctx, node.name, node.constraint, node.plate, block="data"
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
            "qvr-webppl",
            [f"node:{type(node).__name__}"],
        )

    def _emit_deterministic(
        self,
        ctx: _RenderCtx,
        node: IRDeterministic,
    ) -> None:
        """Emit `var <name> = <expr>;` for a deterministic let-binding.

        Lowers [`node.expr`][quivers.transpile.ir.IRDeterministic.expr]
        through
        [`render_let_expr_javascript`][quivers.transpile.renderers._javascript_helpers.render_let_expr_javascript]
        so binary ops, calls, indices, lambdas, list literals, and
        method calls reach the emitter as real JavaScript expression
        vertices rather than a self-referential identifier.

        When the expression references one or more data-input array
        names (free names registered as
        [`IRDataInput`][quivers.transpile.ir.IRDataInput]), the
        binding is lifted into a `mapIndexed(function(__i, _) {
        return <expr-with-array-refs-indexed>; }, <pivot>)` call so
        the per-position arithmetic happens elementwise rather than
        coercing arrays to NaN through JavaScript's scalar operators.
        The pivot is the first array data-input encountered in
        declaration order; every other array data-input reference in
        the expression is replaced with ``<name>[__i]`` so the
        per-position values align. The binding's plate is then
        promoted to carry the surrounding observe plate's batch dim
        so downstream
        [`IRArgRef`][quivers.transpile.ir.IRArgRef] references emit
        an index access.
        """
        array_inputs = self._array_input_refs_in_expr(node.expr)
        if not array_inputs:
            rhs = render_let_expr_javascript(
                _JsLetCtx(
                    ctx.sb, lambda p: self._fresh(ctx, p), self._cards
                ),
                node.expr,
            )
            self._emit_var_decl(ctx, self._body_vid, node.name, rhs)
            self._binding_plates[node.name] = node.plate
            self._binding_supports[node.name] = node.constraint
            return
        # Lift the binding through a `mapIndexed` over the pivot
        # array. The expression's body is rewritten so each array-
        # input reference becomes ``<name>[__i]``; the loop var
        # ``__i`` is injected as a bare identifier reference.
        loop_var = self._fresh_loop_var()
        rewritten = self._index_array_refs(
            node.expr, array_inputs, loop_var,
        )
        body_block = self._fresh(ctx, "lbody")
        ctx.sb.vertex(body_block, "statement_block")
        body_vid = render_let_expr_javascript(
            _JsLetCtx(
                ctx.sb, lambda p: self._fresh(ctx, p), self._cards
            ),
            rewritten,
        )
        self._emit_return_statement(ctx, body_block, body_vid)
        lam = self._function_expression(
            ctx, (loop_var, "_"), body_block,
        )
        pivot = array_inputs[0]
        mi_call = self._call(
            ctx,
            self._ident(ctx, "mapIndexed"),
            (lam, self._ident(ctx, pivot)),
        )
        self._emit_var_decl(ctx, self._body_vid, node.name, mi_call)
        # Promote the binding's plate so refs from inside an observe
        # mapIndexed body get the loop index threaded.
        pivot_plate = self._binding_plates.get(pivot)
        promoted_plate = (
            pivot_plate
            if pivot_plate is not None and pivot_plate.batch_dims
            else self._observed_array_plate(pivot)
        )
        self._binding_plates[node.name] = promoted_plate or node.plate
        self._binding_supports[node.name] = node.constraint

    def _fresh_loop_var(self) -> str:
        """Return a fresh `__i_<n>` loop-variable name for a lifted
        deterministic binding.

        The renderer counts these independently of the schema vertex
        counter so that two lifted bindings in the same program do
        not collide on the same loop name.
        """
        self._lift_n += 1
        return f"__i_{self._lift_n}"

    def _array_input_refs_in_expr(
        self, expr: LetExprNode
    ) -> tuple[str, ...]:
        """Return the array-shaped data-input names referenced in
        ``expr``, in left-to-right traversal order, deduplicated.

        A name counts as array-shaped when it appears in the
        renderer's function-parameters set and is not also bound as
        an IR sample / observe / let / score / marginalize name. The
        result drives the `mapIndexed` pivot choice in
        [`_emit_deterministic`][quivers.transpile.renderers.webppl.WebPPLRenderer._emit_deterministic].
        """
        ordered: list[str] = []
        seen: set[str] = set()
        for name in _free_names_in_let_expr(expr):
            if name in seen:
                continue
            if name not in self._function_parameters:
                continue
            # Function parameters that the renderer knows are
            # already plate-shaped (e.g. the observed-var input
            # itself) carry a non-empty `batch_dims`; bindings
            # referencing them flow through this same lifting path.
            # Names known to be scalar (no plate, not in
            # function parameters as an array) skip the lift via the
            # outer check above.
            seen.add(name)
            ordered.append(name)
        return tuple(ordered)

    def _observed_array_plate(self, input_name: str) -> Plate | None:
        """Return a `Plate` carrying ``input_name``'s array shape
        when the renderer has independent evidence of its size.

        Today the only such evidence is the binding plate already
        recorded for `input_name`; when the lower-time plate was
        empty (the common case for free scalar names that happen to
        be passed as arrays at runtime), the method falls back to
        the first observe step's plate in the IR walk via
        `_observe_array_fallback`.
        """
        plate = self._binding_plates.get(input_name)
        if plate is not None and plate.batch_dims:
            return plate
        # Fall back to the first observe plate seen during the walk;
        # the deterministic binding is consumed inside that observe
        # mapIndexed scope, so promoting to that plate aligns the
        # index threading.
        return self._observe_array_fallback

    def _index_array_refs(
        self,
        expr: LetExprNode,
        array_inputs: tuple[str, ...],
        loop_var: str,
    ) -> LetExprNode:
        """Rewrite every `LetExprVar` reference to a name in
        ``array_inputs`` into an indexed access ``<name>[loop_var]``.

        Used by the deterministic-binding lift to convert a scalar-
        style expression body into one that operates on per-position
        elements of each array.
        """
        return _substitute_array_refs(expr, set(array_inputs), loop_var)

    def _emit_score(self, ctx: _RenderCtx, node: IRScore) -> None:
        """Emit `var <name> = <expr>; factor(<name>);` for a score
        increment.

        WebPPL's `factor(value)` adds `value` to the log-density;
        the convention is to bind the expression to a local var
        first so the factor reads a name. The expression is rendered
        through
        [`render_let_expr_javascript`][quivers.transpile.renderers._javascript_helpers.render_let_expr_javascript]
        so the bound value is a real JavaScript expression.
        """
        rhs = render_let_expr_javascript(
            _JsLetCtx(ctx.sb, lambda p: self._fresh(ctx, p), self._cards),
            node.expr,
        )
        self._emit_var_decl(ctx, self._body_vid, node.name, rhs)
        factor_call = self._call(
            ctx,
            self._ident(ctx, "factor"),
            (self._ident(ctx, node.name),),
        )
        self._emit_expression_statement(
            ctx, self._body_vid, factor_call
        )

    def _emit_return(
        self,
        ctx: _RenderCtx,
        names: tuple[str, ...],
    ) -> None:
        """Emit `return <var>;` for a single return; `return [a, b,
        ...];` for multiple returns.
        """
        if not names:
            return
        rs = self._fresh(ctx, "rs")
        ctx.sb.vertex(rs, "return_statement")
        if len(names) == 1:
            ctx.sb.edge(rs, self._ident(ctx, names[0]), "child_of")
        else:
            arr = self._fresh(ctx, "rarr")
            ctx.sb.vertex(arr, "array")
            for var in names:
                ctx.sb.edge(arr, self._ident(ctx, var), "child_of")
            ctx.sb.edge(rs, arr, "child_of")
        ctx.sb.edge(self._body_vid, rs, "child_of")

    # ------------------------------------------------------------------
    # Argument tuple rendering: orchestrates broadcast / index
    # threading / arg-aliases per-call.
    # ------------------------------------------------------------------

    def _render_arg_tuple(
        self,
        ctx: _RenderCtx,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        meta: FamilyMeta,
        plate: Plate,
        loop_name: str | None,
        index_dependent: bool,
    ) -> tuple[tuple[str, str], ...]:
        """Render every arg in `args`, returning a tuple of
        `(keyword, vertex_id)` pairs suitable for the WebPPL
        distribution object literal.

        The keyword for each position comes from `arg_names`
        (carried on the IR per spec §2.4), filtered through the
        family's per-backend `arg_aliases["webppl"]` rename map.

        Vector / matrix-valued arg slots that receive a scalar ref
        get wrapped in a `repeat` broadcast call so the resulting
        WebPPL distribution sees a vector. Refs whose binding plate
        aligns with the surrounding sample plate get prepended
        loop-index access when the call site is index-dependent.
        """
        alias_map = meta.arg_aliases.get("webppl", {})
        # Pull the per-arg constraint table for broadcast detection.
        cls_attr = getattr(
            meta.distribution_class, "arg_constraints", None
        )
        per_arg_constraints: tuple[object, ...]
        if isinstance(cls_attr, dict):
            per_arg_constraints = tuple(cls_attr.values())
        else:
            per_arg_constraints = ()
        if len(args) != len(arg_names):
            # The IR contract is parallel arrays; a mismatch is a
            # genuine bug rather than a renderer concern.
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"arg-names-mismatch:{meta.qvr_name}: "
                    f"{len(args)} args vs {len(arg_names)} names"
                ],
            )
        out: list[tuple[str, str]] = []
        for idx, (arg, raw_name) in enumerate(
            zip(args, arg_names, strict=True)
        ):
            keyword = alias_map.get(raw_name, raw_name)
            expected = (
                per_arg_constraints[idx]
                if idx < len(per_arg_constraints)
                else None
            )
            broadcasted = self._maybe_broadcast(
                arg, expected, plate
            )
            substituted = self._substitute_for_indexing(
                broadcasted, plate, loop_name, index_dependent
            )
            vid = self._render_arg(ctx, substituted)
            out.append((keyword, vid))
        return tuple(out)

    def _maybe_broadcast(
        self,
        arg: IRArg,
        expected_constraint: object,
        plate: Plate,
    ) -> IRArg:
        """Wrap `arg` in
        [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast] when
        the call-site demands a vector / matrix and the user passed
        a scalar reference / literal.

        The expected event rank is read from the family's
        `arg_constraints` entry's `event_dim`; the target shape is
        taken from the surrounding sample's `plate.event_dims`.
        Refs whose binding plate is already vector-shaped pass
        through unchanged.
        """
        if expected_constraint is None:
            return arg
        expected_rank = int(getattr(expected_constraint, "event_dim", 0))
        if expected_rank < 1:
            return arg
        if len(plate.event_dims) < expected_rank:
            return arg
        # Skip wrapping when the IR already encodes the broadcast.
        if isinstance(arg, IRArgBroadcast):
            return arg
        # Lists / matrices are already vector-shaped.
        if isinstance(arg, (IRArgList, IRArgMatrix)):
            return arg
        # Family-refs resolve inline; broadcast wrapping is wrong.
        if isinstance(arg, IRArgFamilyRef):
            return arg
        # Refs to scalar bindings need wrapping; refs to a name
        # whose binding plate is non-scalar already carry the right
        # shape.
        if isinstance(arg, IRArgRef):
            decl_plate = self._binding_plates.get(arg.name)
            if decl_plate is not None and decl_plate.event_dims:
                return arg
        # Compute the target shape from the surrounding plate's
        # event dims (first `expected_rank` entries).
        sizes: list[int] = []
        for dim in plate.event_dims[:expected_rank]:
            if isinstance(dim, DimStatic):
                sizes.append(dim.size)
            else:
                # Dynamic-sized event dims would need a different
                # call form (e.g. `repeat(K, ...)` where K is a
                # data input); raise to keep the surface explicit.
                raise UnsupportedConstruct(
                    "qvr-webppl",
                    [
                        f"broadcast:dynamic-event-dim:{dim.name}: "
                        f"dynamic event-dim broadcast not "
                        f"implemented for WebPPL"
                    ],
                )
        return IRArgBroadcast(
            value=arg, target_shape=tuple(sizes)
        )

    def _substitute_for_indexing(
        self,
        arg: IRArg,
        plate: Plate,
        loop_name: str | None,
        index_dependent: bool,
    ) -> IRArg:
        """Thread the surrounding plate's loop variable through any
        [`IRArgRef`][quivers.transpile.ir.IRArgRef] whose binding
        plate's batch axes align with the surrounding plate's batch
        axes.

        For an unbatched call site or an index-independent call
        site, returns `arg` unchanged. For an observed step under a
        `via` fibration, the per-observe loop var is prepended to
        the ref's index list along the fibration's group axis.
        """
        if not index_dependent or loop_name is None:
            return arg
        return self._substitute_ref_indexing(arg, plate, loop_name)

    def _substitute_ref_indexing(
        self,
        arg: IRArg,
        plate: Plate,
        loop_name: str,
    ) -> IRArg:
        """Recursive worker for `_substitute_for_indexing`. Walks
        every IR-arg subtree threading the loop name into ref
        indices that align with the surrounding plate.
        """
        if isinstance(arg, IRArgRef):
            decl_plate = self._binding_plates.get(arg.name)
            new_indices = tuple(
                self._substitute_ref_indexing(idx, plate, loop_name)
                for idx in arg.indices
            )
            if decl_plate is not None and self._plates_align(
                decl_plate, plate
            ):
                # Prepend the loop var to the existing indices.
                loop_ref = IRArgRef(name=loop_name)
                return IRArgRef(
                    name=arg.name,
                    indices=(loop_ref, *new_indices),
                )
            # Apply `via` fibration to refs whose binding plate
            # equals the active marginalize / group plate. The
            # observe loop var goes through `<via>[<loop>]` so the
            # per-row observation indexes into the per-group latent
            # / parameter.
            if (
                self._observe_via is not None
                and decl_plate is not None
                and self._matches_group_plate(decl_plate)
                and not self._is_data_input(arg.name)
            ):
                via_ref = IRArgRef(
                    name=self._observe_via,
                    indices=(IRArgRef(name=loop_name),),
                )
                return IRArgRef(
                    name=arg.name,
                    indices=(via_ref, *new_indices),
                )
            return IRArgRef(name=arg.name, indices=new_indices)
        if isinstance(arg, IRArgBroadcast):
            return IRArgBroadcast(
                value=self._substitute_ref_indexing(
                    arg.value, plate, loop_name
                ),
                target_shape=arg.target_shape,
            )
        if isinstance(arg, IRArgList):
            return IRArgList(
                elements=tuple(
                    self._substitute_ref_indexing(e, plate, loop_name)
                    for e in arg.elements
                )
            )
        if isinstance(arg, IRArgMatrix):
            return IRArgMatrix(
                rows=tuple(
                    IRArgList(
                        elements=tuple(
                            self._substitute_ref_indexing(
                                e, plate, loop_name
                            )
                            for e in row.elements
                        )
                    )
                    for row in arg.rows
                )
            )
        return arg

    def _is_data_input(self, name: str) -> bool:
        """True iff `name` was registered as an IRDataInput (versus
        a sample / observe / let binding). Data inputs are the
        program's function parameters; their refs do not need
        per-element threading through a `via` fibration.
        """
        plate = self._binding_plates.get(name)
        if plate is None:
            return False
        # Data inputs are registered with no event_dims; samples
        # whose batch_dims encode a plate also live in
        # _binding_plates. The discriminator is whether the name
        # corresponds to a function parameter; we use the
        # function-parameter list as the authoritative source.
        return name in self._function_parameters

    @property
    def _function_parameters(self) -> set[str]:
        """The names registered as IRDataInputs at render start."""
        return self._function_parameters_state

    def _matches_group_plate(self, decl: Plate) -> bool:
        """True iff `decl`'s batch axes equal the active marginalize
        group plate axes.

        Drives the `via` fibration threading: a ref to a name bound
        on the marginalize plate (Doc) referenced from an observe
        on a different plate (Word) needs to be indexed by
        `<via>[<loop>]` when `via` maps the observe's plate to the
        marginalize's plate.
        """
        if not self._group_plate_axes:
            return False
        decl_names = tuple(d.name for d in decl.batch_dims)
        return decl_names == self._group_plate_axes

    def _plates_align(self, decl: Plate, surrounding: Plate) -> bool:
        """True iff `decl`'s batch axes are a left-aligned prefix of
        `surrounding`'s batch axes.

        The alignment rule is name-based: a ref to a name bound on
        a plate with batch axes `(A,)` referenced from a sample on
        a plate with batch axes `(A, B)` aligns on `A`. The
        check is purely name-based on `Dim.name`.
        """
        if not decl.batch_dims:
            return False
        decl_names = tuple(d.name for d in decl.batch_dims)
        surr_names = tuple(d.name for d in surrounding.batch_dims)
        if len(decl_names) > len(surr_names):
            return False
        return decl_names == surr_names[: len(decl_names)]

    def _args_use_surrounding_index(
        self,
        args: tuple[IRArg, ...],
        plate: Plate,
    ) -> bool:
        """True iff at least one arg in `args` references a name
        whose binding plate aligns with the surrounding plate's
        batch axes.

        Drives the `repeat` vs `mapIndexed` decision in
        `_emit_sample`: index-dependent call sites need the loop
        variable in scope to index into the referenced names.
        """
        if not plate.batch_dims:
            return False
        return any(self._arg_uses_surrounding_index(a, plate) for a in args)

    def _arg_uses_surrounding_index(
        self,
        arg: IRArg,
        plate: Plate,
    ) -> bool:
        """Recursive worker for `_args_use_surrounding_index`."""
        if isinstance(arg, IRArgRef):
            decl_plate = self._binding_plates.get(arg.name)
            if decl_plate is not None and self._plates_align(
                decl_plate, plate
            ):
                return True
            return any(
                self._arg_uses_surrounding_index(idx, plate)
                for idx in arg.indices
            )
        if isinstance(arg, IRArgBroadcast):
            return self._arg_uses_surrounding_index(arg.value, plate)
        if isinstance(arg, IRArgList):
            return any(
                self._arg_uses_surrounding_index(e, plate)
                for e in arg.elements
            )
        if isinstance(arg, IRArgMatrix):
            return any(
                self._arg_uses_surrounding_index(e, plate)
                for row in arg.rows
                for e in row.elements
            )
        return False

    def _loop_name_for(self, sample_name: str, plate: Plate) -> str | None:
        """Pick the loop-variable name for a batched sample step.

        Uses the spec's `m_<axis>_<name>` form so multi-batch
        nesting stays unambiguous and the name is stable across
        emits of the same program.
        """
        if not plate.batch_dims:
            return None
        first = plate.batch_dims[0]
        return f"m_{first.name}_{sample_name}"

    def _wrap_for_batch(
        self,
        ctx: _RenderCtx,
        inner_value: str,
        plate: Plate,
        loop_name: str | None,
        index_dependent: bool,
    ) -> str:
        """Wrap `inner_value` in `repeat(N, function() { return <v>;
        })` per batch dim or
        `mapIndexed(function(m, _) { return <v>; }, repeat(N, 0))`
        when an arg uses the loop index.

        The unbatched case returns `inner_value` unchanged.
        """
        batch_dims_list: list[Dim] = list(plate.batch_dims)
        if not batch_dims_list:
            return inner_value
        if len(batch_dims_list) > 1:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"batch-rank:{len(batch_dims_list)}: WebPPL "
                    f"emits one batch axis per sample; multi-axis "
                    f"plates not implemented"
                ],
            )
        dim = batch_dims_list[0]
        size_vid = self._dim_size_value(ctx, dim)
        body = self._fresh(ctx, "rbody")
        ctx.sb.vertex(body, "statement_block")
        self._emit_return_statement(ctx, body, inner_value)
        if index_dependent and loop_name is not None:
            # mapIndexed(function(m, _) { return <inner>; }, repeat(N, 0))
            lam = self._function_expression(
                ctx, (loop_name, "_"), body
            )
            zero = self._number_literal(ctx, 0)
            inner_repeat = self._call(
                ctx,
                self._ident(ctx, "repeat"),
                (size_vid, zero),
            )
            return self._call(
                ctx,
                self._ident(ctx, "mapIndexed"),
                (lam, inner_repeat),
            )
        # repeat(N, function() { return <inner>; })
        lam = self._function_expression(ctx, (), body)
        return self._call(
            ctx,
            self._ident(ctx, "repeat"),
            (size_vid, lam),
        )

    # ------------------------------------------------------------------
    # Render one IRArg to a JS expression vertex.
    # ------------------------------------------------------------------

    def _render_arg(
        self,
        ctx: _RenderCtx,
        arg: IRArg,
    ) -> SchemaFragment:
        """Render any [`IRArg`][quivers.transpile.ir.IRArg] to a JS
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
            "qvr-webppl",
            [f"arg:unknown:{type(arg).__name__}"],
        )

    def _render_number(self, ctx: _RenderCtx, value: float) -> str:
        return self._number_literal(ctx, value)

    def _render_ref(
        self, ctx: _RenderCtx, arg: IRArgRef
    ) -> SchemaFragment:
        """Render an IRArgRef. Bare-name refs emit an identifier;
        indexed refs build `subscript_expression` chains."""
        base = self._ident(ctx, arg.name)
        if not arg.indices:
            return base
        current = base
        for idx in arg.indices:
            idx_vid = self._render_arg(ctx, idx)
            current = self._subscript(ctx, current, idx_vid)
        return current

    def _render_family_ref(
        self,
        ctx: _RenderCtx,
        arg: IRArgFamilyRef,
    ) -> SchemaFragment:
        """Resolve an
        [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef] and
        emit the inner distribution call inline.

        WebPPL has no generic truncation idiom; the renderer reads
        the referenced morphism's `init_family` clause from
        `ctx.morphisms` and emits the inner distribution call. The
        outer wrapper family's renderer is responsible for shaping
        the wrapped call into the right WebPPL form.
        """
        decl = ctx.morphisms.get(arg.name)
        if decl is None or decl.init_family is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"arg:family-ref:{arg.name}: no morphism with "
                    f"`~ Family(...)` init clause in scope"
                ],
            )
        init = decl.init_family
        inner_meta = FAMILY_META.get(init.family)
        if inner_meta is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [f"family:unknown:{init.family}"],
            )
        inner_name = inner_meta.target_names.get("webppl")
        if inner_name is None:
            raise UnsupportedConstruct(
                "qvr-webppl",
                [f"family:no-webppl-target:{init.family}"],
            )
        # Best-effort: emit the inner call with positional args. The
        # init_family clause carries raw DrawArg-shaped values; we
        # translate to expression vertices and pack them into an
        # object literal using the inner family's `arg_names` from
        # its arg_constraints.
        alias_map = inner_meta.arg_aliases.get("webppl", {})
        cls_attr = getattr(
            inner_meta.distribution_class, "arg_constraints", None
        )
        if isinstance(cls_attr, dict):
            keys = tuple(cls_attr.keys())
        else:
            keys = ()
        raw_args = init.args or ()
        if len(raw_args) > len(keys):
            raise UnsupportedConstruct(
                "qvr-webppl",
                [
                    f"arg:family-ref:{arg.name}: too many args "
                    f"({len(raw_args)}) for {init.family} "
                    f"(expects {len(keys)})"
                ],
            )
        entries: list[tuple[str, str]] = []
        for raw, raw_key in zip(raw_args, keys, strict=False):
            keyword = str(alias_map.get(raw_key, raw_key))
            vid = self._render_init_family_arg(ctx, raw)
            entries.append((keyword, vid))
        dist_obj = self._object_literal(ctx, tuple(entries))
        return self._call(
            ctx,
            self._ident(ctx, inner_name),
            (dist_obj,),
        )

    def _render_init_family_arg(
        self,
        ctx: _RenderCtx,
        raw: object,
    ) -> SchemaFragment:
        """Render an `init_family` raw arg.

        The DSL surface gives us wire-form strings / floats /
        structured DrawArg variants; translate them to JS
        expression vertices.
        """
        if isinstance(raw, (int, float)):
            return self._number_literal(ctx, float(raw))
        if isinstance(raw, str):
            stripped = raw.strip()
            try:
                value = float(stripped)
            except ValueError:
                return self._ident(ctx, stripped)
            return self._number_literal(ctx, value)
        raise UnsupportedConstruct(
            "qvr-webppl",
            [
                f"arg:family-ref:init-arg:{type(raw).__name__}: "
                f"unsupported raw arg shape"
            ],
        )

    # ------------------------------------------------------------------
    # Low-level JS schema builders.
    # ------------------------------------------------------------------

    def _ident(self, ctx: _RenderCtx, text: str) -> str:
        vid = self._fresh(ctx, "id")
        ctx.sb.vertex(vid, "identifier")
        ctx.sb.constraint(vid, "literal-value", text)
        return vid

    def _prop_ident(self, ctx: _RenderCtx, text: str) -> str:
        vid = self._fresh(ctx, "pid")
        ctx.sb.vertex(vid, "property_identifier")
        ctx.sb.constraint(vid, "literal-value", text)
        return vid

    def _number_literal(
        self, ctx: _RenderCtx, value: int | float
    ) -> str:
        vid = self._fresh(ctx, "num")
        ctx.sb.vertex(vid, "number")
        text = (
            str(int(value)) if float(value).is_integer() else repr(float(value))
        )
        ctx.sb.constraint(vid, "literal-value", text)
        return vid

    def _object_literal(
        self,
        ctx: _RenderCtx,
        entries: tuple[tuple[str, str], ...],
    ) -> str:
        """Build a JS object literal `{k1: v1, k2: v2, ...}`."""
        obj = self._fresh(ctx, "obj")
        ctx.sb.vertex(obj, "object")
        for key, value_vid in entries:
            pair = self._fresh(ctx, "pair")
            ctx.sb.vertex(pair, "pair")
            ctx.sb.edge(pair, self._prop_ident(ctx, key), "key")
            ctx.sb.edge(pair, value_vid, "value")
            ctx.sb.edge(obj, pair, "child_of")
        return obj

    def _call(
        self,
        ctx: _RenderCtx,
        callee_vid: str,
        positional: tuple[str, ...],
    ) -> str:
        """Build a JS call expression `callee(arg1, arg2, ...)`."""
        call = self._fresh(ctx, "call")
        ctx.sb.vertex(call, "call_expression")
        args = self._fresh(ctx, "args")
        ctx.sb.vertex(args, "arguments")
        ctx.sb.edge(call, callee_vid, "function")
        ctx.sb.edge(call, args, "arguments")
        for pid in positional:
            ctx.sb.edge(args, pid, "child_of")
        return call

    def _function_expression(
        self,
        ctx: _RenderCtx,
        params: tuple[str, ...],
        body_vid: str,
    ) -> str:
        """Build a JS `function(<params>) { <body> }` expression."""
        fn = self._fresh(ctx, "fn")
        ctx.sb.vertex(fn, "function_expression")
        ps = self._fresh(ctx, "ps")
        ctx.sb.vertex(ps, "formal_parameters")
        for name in params:
            ctx.sb.edge(ps, self._ident(ctx, name), "child_of")
        ctx.sb.edge(fn, ps, "parameters")
        ctx.sb.edge(fn, body_vid, "body")
        return fn

    def _subscript(
        self,
        ctx: _RenderCtx,
        object_vid: str,
        index_vid: str,
    ) -> str:
        """Build a JS subscript expression `<object>[<index>]`."""
        sub = self._fresh(ctx, "sub")
        ctx.sb.vertex(sub, "subscript_expression")
        ctx.sb.edge(sub, object_vid, "object")
        ctx.sb.edge(sub, index_vid, "index")
        return sub

    def _emit_var_decl(
        self,
        ctx: _RenderCtx,
        parent_vid: str,
        name: str,
        value_vid: str,
    ) -> str:
        """Emit `var <name> = <value>;` into `parent_vid`."""
        vd = self._fresh(ctx, "vd")
        ctx.sb.vertex(vd, "variable_declaration")
        dr = self._fresh(ctx, "dr")
        ctx.sb.vertex(dr, "variable_declarator")
        ctx.sb.edge(dr, self._ident(ctx, name), "name")
        ctx.sb.edge(dr, value_vid, "value")
        ctx.sb.edge(vd, dr, "child_of")
        ctx.sb.edge(parent_vid, vd, "child_of")
        return vd

    def _emit_expression_statement(
        self,
        ctx: _RenderCtx,
        parent_vid: str,
        expr_vid: str,
    ) -> str:
        """Emit `<expr>;` into `parent_vid`."""
        es = self._fresh(ctx, "es")
        ctx.sb.vertex(es, "expression_statement")
        ctx.sb.edge(es, expr_vid, "child_of")
        ctx.sb.edge(parent_vid, es, "child_of")
        return es

    def _emit_return_statement(
        self,
        ctx: _RenderCtx,
        parent_vid: str,
        value_vid: str,
    ) -> str:
        """Emit `return <value>;` into `parent_vid`."""
        rs = self._fresh(ctx, "rs")
        ctx.sb.vertex(rs, "return_statement")
        ctx.sb.edge(rs, value_vid, "child_of")
        ctx.sb.edge(parent_vid, rs, "child_of")
        return rs

    def _dim_size_value(self, ctx: _RenderCtx, dim: Dim) -> str:
        """Render a plate dim's size as a JS expression.

        Static dims emit an integer literal; dynamic dims emit the
        `size_name` identifier so the surrounding scope (typically
        a function parameter) supplies the runtime value.
        """
        if isinstance(dim, DimStatic):
            return self._number_literal(ctx, dim.size)
        if isinstance(dim, DimDynamic):
            return self._ident(ctx, dim.size_name)
        raise UnsupportedConstruct(
            "qvr-webppl",
            [f"dim:unknown:{type(dim).__name__}"],
        )

    def _fresh(self, ctx: _RenderCtx, prefix: str) -> str:
        """Return a fresh vertex id with `prefix`.

        Renderer-internal counter; the base's
        `_RenderCtx.fresh_counter` is left untouched so per-walk
        node IDs stay stable across renderer instances.
        """
        del ctx
        self._fresh_n += 1
        return f"{prefix}_{self._fresh_n}"

    # ------------------------------------------------------------------
    # Morphism / let resolution for IRArgFamilyRef.
    # ------------------------------------------------------------------

    def _resolve_morphisms_and_lets(
        self,
    ) -> tuple[dict[str, MorphismDecl], dict[str, Expr]]:
        """Build the morphism / let tables from the source module.

        Used by `IRArgFamilyRef` resolution. Empty when the
        renderer was constructed without a source module; in that
        case `IRArgFamilyRef` rendering raises with a precise
        kind tag.
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


    def _first_observe_plate(self, ir: IRProgram) -> Plate | None:
        """Return the first
        [`IRObserve`][quivers.transpile.ir.IRObserve] step's plate
        encountered in declaration order, recursing into
        marginalize scopes.

        Used as the fallback `Plate` for a lifted deterministic
        binding whose data-input references the IR records as
        scalar; the binding's `mapIndexed` then runs over the same
        per-position iteration as the surrounding observe.
        """
        for node in ir.body:
            plate = _first_observe_plate_in_node(node)
            if plate is not None:
                return plate
        return None


def _first_observe_plate_in_node(node: IRNode) -> Plate | None:
    """Recursive worker for
    [`_first_observe_plate`][quivers.transpile.renderers.webppl.WebPPLRenderer._first_observe_plate].
    """
    if isinstance(node, IRObserve):
        if node.plate.batch_dims:
            return node.plate
        return None
    if isinstance(node, IRMarginalize):
        for inner in node.scope:
            plate = _first_observe_plate_in_node(inner)
            if plate is not None:
                return plate
    return None


def _free_names_in_let_expr(expr: LetExprNode) -> tuple[str, ...]:
    """Return the variable names referenced anywhere in ``expr``, in
    left-to-right pre-order.

    Reads bare-name `LetExprVar` nodes plus the function-name of
    `LetExprCall`. Lambda parameters bound inside the expression do
    not occur as free names by construction here (the renderer's
    upstream let-binding semantics rebind every name through
    `IRDeterministic`).
    """
    out: list[str] = []
    _collect_free_names(expr, out)
    return tuple(out)


def _collect_free_names(expr: LetExprNode, out: list[str]) -> None:
    """Walk ``expr`` adding each `LetExprVar` name to ``out``."""
    if isinstance(expr, LetExprVar):
        out.append(expr.name)
        return
    if isinstance(expr, LetExprLiteral):
        return
    if isinstance(expr, LetExprString):
        return
    if isinstance(expr, LetExprBinOp):
        _collect_free_names(expr.left, out)
        _collect_free_names(expr.right, out)
        return
    if isinstance(expr, LetExprUnaryOp):
        _collect_free_names(expr.operand, out)
        return
    if isinstance(expr, LetExprCall):
        for a in expr.args:
            _collect_free_names(a, out)
        return
    if isinstance(expr, LetExprIndex):
        _collect_free_names(expr.array, out)
        for idx in expr.indices:
            _collect_free_names(idx, out)
        return
    if isinstance(expr, LetExprList):
        for item in expr.items:
            _collect_free_names(item, out)
        return
    if isinstance(expr, LetExprLambda):
        _collect_free_names(expr.body, out)
        return
    if isinstance(expr, LetExprMethodCall):
        _collect_free_names(expr.receiver, out)
        for a in expr.args:
            _collect_free_names(a, out)
        return
    if isinstance(expr, LetExprFactor):
        if expr.body is not None:
            _collect_free_names(expr.body, out)
        for case in expr.cases:
            _collect_free_names(case.value, out)
        return


def _substitute_array_refs(
    expr: LetExprNode,
    array_names: set[str],
    loop_var: str,
) -> LetExprNode:
    """Rewrite every `LetExprVar(name)` where ``name in array_names``
    into a `LetExprIndex(LetExprVar(name), [LetExprVar(loop_var)])`.

    The substitution is structural; no scoping considerations
    because lambdas are not allowed to rebind data-input names in
    the surface language.
    """
    if isinstance(expr, LetExprVar):
        if expr.name in array_names:
            return LetExprIndex(
                array=LetExprVar(name=expr.name),
                indices=(LetExprVar(name=loop_var),),
            )
        return expr
    if isinstance(expr, (LetExprLiteral, LetExprString)):
        return expr
    if isinstance(expr, LetExprBinOp):
        return LetExprBinOp(
            op=expr.op,
            left=_substitute_array_refs(expr.left, array_names, loop_var),
            right=_substitute_array_refs(expr.right, array_names, loop_var),
        )
    if isinstance(expr, LetExprUnaryOp):
        return LetExprUnaryOp(
            op=expr.op,
            operand=_substitute_array_refs(
                expr.operand, array_names, loop_var
            ),
        )
    if isinstance(expr, LetExprCall):
        return LetExprCall(
            func=expr.func,
            args=tuple(
                _substitute_array_refs(a, array_names, loop_var)
                for a in expr.args
            ),
        )
    if isinstance(expr, LetExprIndex):
        return LetExprIndex(
            array=_substitute_array_refs(
                expr.array, array_names, loop_var
            ),
            indices=tuple(
                _substitute_array_refs(idx, array_names, loop_var)
                for idx in expr.indices
            ),
        )
    if isinstance(expr, LetExprList):
        return LetExprList(
            items=tuple(
                _substitute_array_refs(item, array_names, loop_var)
                for item in expr.items
            ),
        )
    if isinstance(expr, LetExprLambda):
        return LetExprLambda(
            param=expr.param,
            body=_substitute_array_refs(
                expr.body, array_names, loop_var
            ),
        )
    if isinstance(expr, LetExprMethodCall):
        return LetExprMethodCall(
            receiver=_substitute_array_refs(
                expr.receiver, array_names, loop_var
            ),
            method=expr.method,
            args=tuple(
                _substitute_array_refs(a, array_names, loop_var)
                for a in expr.args
            ),
        )
    if isinstance(expr, LetExprFactor):
        return expr
    raise UnsupportedConstruct(
        "qvr-webppl",
        [f"let-expr:substitute:{type(expr).__name__}"],
    )


# WebPPL-side argument injection for QVR families whose torch
# distribution carries fewer parameters than WebPPL's same-named
# distribution. `HalfNormal(scale)` maps to WebPPL's
# `Gaussian({mu: 0, sigma: scale})`; `HalfCauchy(scale)` maps to
# `Cauchy({location: 0, scale: scale})`. The renderer prepends an
# explicit zero-valued location argument so the WebPPL distribution
# constructor sees both parameters. The resulting log-density
# differs from QVR's half-distribution log-density by the constant
# ``+log(2) * N_observations``; the constant-spread equivalence
# check in
# [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
# tolerates this offset.
_PREPEND_MU_ZERO: frozenset[str] = frozenset({
    "HalfNormal", "HalfCauchy",
})


def _inject_webppl_specific_args(
    family: str,
    args: tuple[IRArg, ...],
    arg_names: tuple[str, ...],
) -> tuple[tuple[IRArg, ...], tuple[str, ...]]:
    """Inject the canonical extra arguments required by WebPPL for
    families whose torch shape is narrower than WebPPL's call shape.

    Returns the possibly-augmented (args, arg_names) tuple in
    parallel order. The injected keyword name uses the QVR-side
    name (e.g. ``loc``) so the family's ``arg_aliases["webppl"]`` map
    can rename it to the WebPPL-side keyword (``mu``) during
    rendering.
    """
    if family in _PREPEND_MU_ZERO:
        return (
            (IRArgNumber(value=0.0), *args),
            ("loc", *arg_names),
        )
    return args, arg_names


# ---------------------------------------------------------------------------
# Runtime-helper graft: `Logistic`, `BetaBinomial`, `HalfStudentT`,
# `Kumaraswamy`, `LKJCholesky`, and `ContinuousBernoulli` as plain
# JavaScript distribution constructors.
#
# WebPPL's `dists` module ships `Gaussian`, `Beta`, `Categorical`,
# `Dirichlet`, ... as built-in distributions but lacks these six.
# The transpile-time graft parses the hand-written helper at
# [`runtime_webppl.js`][quivers.transpile.runtime_webppl] once at
# module-load through panproto's JavaScript tree-sitter grammar;
# per-render, it copies every grafted vertex / constraint / edge into
# the per-render schema (with fresh vertex ids) and attaches the
# runtime's top-level statements as `child_of` of the emitted
# `program` vertex above the `var model = function (...) {...};`
# declaration.
#
# The emit is structurally a normal JavaScript module: a few
# numeric utility functions (`_lgamma`, `_lbeta`, `_gaussian_sample`,
# `_gamma_sample`, `_beta_sample`, `_binomial_sample`) and the six
# distribution constructors. Subsequent
# `sample(Logistic({loc, scale}))` etc. call sites in the model body
# then resolve to the grafted constructors via normal JS name lookup.
# ---------------------------------------------------------------------------


_RUNTIME_WEBPPL_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "runtime_webppl.js"
)


#: Families whose WebPPL emit relies on the
#: [`runtime_webppl.js`][quivers.transpile.runtime_webppl] helper subtree.
#: WebPPL's `dists` module ships `Gaussian`, `Beta`, `Categorical`, etc.
#: as built-in distributions but lacks these; the renderer grafts the
#: helper when the IR samples or observes from any of them.
_WEBPPL_RUNTIME_HELPER_FAMILIES: frozenset[str] = frozenset({
    "Logistic",
    "BetaBinomial",
    "HalfStudentT",
    "Kumaraswamy",
    "LKJCholesky",
    "ContinuousBernoulli",
    "GP",
    "MatrixNormal",
})


def _load_runtime_webppl_schema() -> tuple[
    panproto.Schema, str, tuple[str, ...]
]:
    """Parse [`runtime_webppl.js`][quivers.transpile.runtime_webppl]
    through panproto's JavaScript tree-sitter grammar at module-load
    time.

    Returns the parsed schema, the parsed `program` vertex id, and
    the tuple of top-level child ids in source order (sorted by
    `start-byte`). The graft replays these children in order beneath
    the per-render `program` so the emit's top-level statements
    appear in the original file's layout.
    """
    schema = parser_registry().parse_with_protocol(
        "javascript",
        _RUNTIME_WEBPPL_PATH.read_bytes(),
        str(_RUNTIME_WEBPPL_PATH),
    )
    src_id = next(
        (v.id for v in schema.vertices if v.kind == "program"),
        None,
    )
    if src_id is None:
        raise RuntimeError(
            f"`program` not found in parse of {_RUNTIME_WEBPPL_PATH}"
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


_RUNTIME_WEBPPL_SCHEMA, _RUNTIME_WEBPPL_PROGRAM_ID, _RUNTIME_WEBPPL_TOP_LEVEL = (
    _load_runtime_webppl_schema()
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


_RUNTIME_WEBPPL_SUBTREE = _subtree_vertex_ids(
    _RUNTIME_WEBPPL_SCHEMA, _RUNTIME_WEBPPL_TOP_LEVEL
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


def _graft_runtime_webppl_helper(
    sb: panproto.SchemaBuilder,
    renderer: WebPPLRenderer,
    program_vid: str,
) -> None:
    """Graft the runtime-helper subtree onto the per-render schema.

    Copies every vertex, every constraint, and every internal edge of
    the parsed `runtime_webppl.js` subtree into the per-render
    `SchemaBuilder` with fresh vertex ids, then attaches each
    top-level child as a `child_of` of `program_vid` in source order.
    The grafted top-level children appear above the `var model =
    function(...)` declaration in the emit.
    """
    src_schema = _RUNTIME_WEBPPL_SCHEMA
    subtree = _RUNTIME_WEBPPL_SUBTREE
    id_map: dict[str, str] = {}

    for old in subtree:
        renderer._fresh_n += 1
        new = f"rw_{renderer._fresh_n}"
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
    for child_old in _RUNTIME_WEBPPL_TOP_LEVEL:
        sb.edge(program_vid, id_map[child_old], "child_of")


__all__ = ["WebPPLRenderer"]
