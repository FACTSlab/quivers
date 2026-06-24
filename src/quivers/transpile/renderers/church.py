"""Church renderer: [`IRProgram`][quivers.transpile.ir.IRProgram] to a
Scheme [`panproto.Schema`][panproto.Schema] under the ``scheme``
tree-sitter grammar.

The Church idiom for a probabilistic program is a top-level
``(define (model ...) ...)``: every sample step is a
``(define <name> (map (lambda (m_<axis>) (sample <dist>)) (iota N)))``
form for each batch axis; observed steps are
``(for-each (lambda (n) (observe <dist> (list-ref <obs> n))) (iota
(length <obs>)))``. The IR's [`IRDataInput`][quivers.transpile.ir.IRDataInput]
entries become the model's formal parameter list (Scheme has no
separate declaration block; the function header carries the inputs).

The renderer reads
[`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] for each
family's Church distribution name (`target_names["church"]`), reuses
[`RendererBase`][quivers.transpile.renderers._base.RendererBase] for
the IR walk and the explicit-latent rewrite that lowers
[`IRMarginalize`][quivers.transpile.ir.IRMarginalize] inline (Church
has no native ``log_sum_exp`` enumeration construct), and dispatches
the four primitives (`declare`, `sample`, `marginalize`, `broadcast`)
per `Renderer`. Broadcast scalars emit ``(make-list K <value>)``; list
literals emit ``(list e0 e1 ...)``; matrix literals raise
[`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
with ``arg:matrix-literal`` since Scheme has no canonical matrix
form.
"""

from __future__ import annotations

import pathlib

import panproto

from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import parser_registry, target_protocol
from quivers.transpile.renderers._scheme_helpers import render_let_expr_scheme
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


_TARGET = "qvr-church"


class ChurchRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
    Scheme schema in the Church probabilistic-programming idiom.

    Subclass of [`RendererBase`][quivers.transpile.renderers._base.RendererBase]:
    overrides `render` to wrap the IR walk in a top-level
    ``(define (model <inputs...>) <body...>)`` form, then dispatches
    each [`IRNode`][quivers.transpile.ir.IRNode] through `declare` /
    `sample` / `marginalize` / `broadcast`.
    """

    target: str = _TARGET

    # ----- top-level wrapper -----

    def target_protocol(self) -> panproto.Protocol:
        return target_protocol("scheme")

    def render(self, ir: IRProgram) -> panproto.Schema:
        # Snapshot `IRProgram.cards` for the renderer-local cards map
        # threaded through every `_LetExprCtx`; the Scheme let-expr
        # helper consults `ctx.cards` when a factor binder's index
        # references a named QVR object whose cardinality must be
        # resolved at unroll time.
        self._cards: dict[str, int] = dict(ir.cards)
        proto = self.target_protocol()
        sb = proto.schema()
        ctx = _RenderCtx(sb=sb, morphisms={}, lets={}, cards=self._cards)
        prog_id = _v(ctx, "prog", "program")
        # Church ships `gaussian`, `beta`, `flip`, `multivariate-
        # gaussian`, ... as built-in distributions but lacks
        # `matrix-normal`. When the IR samples or observes from a
        # family whose Church emit relies on a user-defined helper,
        # graft the hand-written subtree at
        # [`runtime_church.scm`][quivers.transpile.runtime_church]
        # into the program above the model `(define ...)` form so the
        # sampled identifier resolves through Scheme's top-level
        # binding lookup.
        if any(
            _ir_uses_family(ir.body, f)
            for f in _CHURCH_RUNTIME_HELPER_FAMILIES
        ):
            _graft_runtime_church_helper(ctx, prog_id)

        # `(model <param1> <param2> ...)` -- function signature.
        signature_children: list[str] = [_sym(ctx, "model")]
        for inp in ir.inputs:
            signature_children.append(_sym(ctx, inp.name))
        signature = _list(ctx, tuple(signature_children))

        # Body forms: one or more forms per IR node, plus a trailing
        # return symbol. Most nodes emit one form; `IRMarginalize`
        # expands inline to the latent sample plus each scope step,
        # yielding several sibling forms at the model-body level.
        body_forms: list[str] = []
        for node in ir.body:
            body_forms.extend(self._render_body_forms(ctx, node))
        # Trailing return: Church returns the last expression of the
        # body. `_emit_return` appends one bare symbol per returned
        # name.
        return_form = self._return_form(ctx, ir.body)
        if return_form is not None:
            body_forms.append(return_form)

        top_define = _list(
            ctx, (_sym(ctx, "define"), signature, *body_forms)
        )
        _e(ctx, prog_id, top_define)
        return sb.build()

    # ----- IR-node dispatch -----

    def _render_body_forms(
        self, ctx: _RenderCtx, node: IRNode
    ) -> tuple[SchemaFragment, ...]:
        """Render one IR node into one or more sibling body forms.

        Most node kinds emit a single form;
        [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] expands
        to the latent sample plus the scope body inline, returning
        multiple forms so they sit at the same nesting level as the
        surrounding ``(define (model ...) ...)`` body.
        """
        if isinstance(node, IRDataInput):
            # Inputs land in the model signature; no body form.
            return ()
        if isinstance(node, IRSample):
            if node.family == "GP":
                return self._render_gp_forms(ctx, node)
            return (
                self.sample(
                    ctx, node.name, node.family, node.args, node.arg_names,
                    node.constraint, node.plate, observed=False,
                ),
            )
        if isinstance(node, IRObserve):
            return (
                self.sample(
                    ctx, node.name, node.family, node.args, node.arg_names,
                    node.constraint, node.plate, observed=True,
                ),
            )
        if isinstance(node, IRDeterministic):
            # `(define <name> <expr>)`. The expression tree comes from
            # the surface compiler; render it via the existing Scheme
            # let-expression helper.
            expr_id = render_let_expr_scheme(
                _LetExprCtx(ctx.sb, ctx, self._cards), node.expr
            )
            return (
                _list(
                    ctx,
                    (_sym(ctx, "define"), _sym(ctx, node.name), expr_id),
                ),
            )
        if isinstance(node, IRScore):
            # Church's score primitive: `(factor <expr>)`.
            expr_id = render_let_expr_scheme(
                _LetExprCtx(ctx.sb, ctx, self._cards), node.expr
            )
            return (_list(ctx, (_sym(ctx, "factor"), expr_id)),)
        if isinstance(node, IRMarginalize):
            return self._render_marginalize_forms(ctx, node)
        if isinstance(node, IRReturn):
            # Handled by `_return_form`; nothing to emit inline.
            return ()
        raise UnsupportedConstruct(
            _TARGET, [f"node:{type(node).__name__}"]
        )

    def _render_gp_forms(
        self, ctx: _RenderCtx, node: IRSample,
    ) -> tuple[SchemaFragment, ...]:
        """Emit three Scheme forms for a Gaussian-process sample:

        ``(define __gp_mean_<name> (repeat <N> (lambda () 0)))``
        ``(define __gp_cov_<name> <inline-rbf-matrix>)``
        ``(define <name> (multivariate-gaussian __gp_mean_<name>
                                                  __gp_cov_<name>))``

        The inline-rbf-matrix is a nested `map` over the grid index
        producing the N-by-N covariance matrix:
        ``K[i,j] = exp(-0.5 * (x[i]-x[j])^2 / length_scale^2)``
        plus diagonal jitter via `(if (= i j) <jitter> 0)`.
        """
        if len(node.args) != 2 or not isinstance(
            node.args[1], IRArgKernel
        ):
            raise UnsupportedConstruct(
                _TARGET,
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                _TARGET,
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
        # (define __gp_mean_<name> (repeat N (lambda () 0)))
        zero_lambda = _list(
            ctx,
            (
                _sym(ctx, "lambda"),
                _list(ctx, ()),
                _num(ctx, 0),
            ),
        )
        mean_form = _list(
            ctx,
            (
                _sym(ctx, "define"),
                _sym(ctx, mean_name),
                _list(
                    ctx,
                    (_sym(ctx, "repeat"), _num(ctx, n), zero_lambda),
                ),
            ),
        )
        # (define __gp_cov_<name>
        #   (map (lambda (i)
        #          (map (lambda (j)
        #                 (+ (exp (- (* 0.5 (* (- (list-ref x i) (list-ref x j))
        #                                       (- (list-ref x i) (list-ref x j))))
        #                            (* ls ls)))
        #                    (if (= i j) jitter 0)))
        #               (iota N)))
        #        (iota N)))
        def diff_form() -> str:
            return _list(
                ctx,
                (
                    _sym(ctx, "-"),
                    _list(ctx, (_sym(ctx, "list-ref"),
                                _sym(ctx, x), _sym(ctx, "i"))),
                    _list(ctx, (_sym(ctx, "list-ref"),
                                _sym(ctx, x), _sym(ctx, "j"))),
                ),
            )

        diff_sq = _list(
            ctx, (_sym(ctx, "*"), diff_form(), diff_form()),
        )
        # arg to exp: (- (/ (* 0.5 diff_sq) (* ls ls)))
        ls_sq = _list(
            ctx, (_sym(ctx, "*"), _num(ctx, ls), _num(ctx, ls)),
        )
        scaled = _list(
            ctx,
            (
                _sym(ctx, "/"),
                _list(
                    ctx,
                    (_sym(ctx, "*"), _num(ctx, 0.5), diff_sq),
                ),
                ls_sq,
            ),
        )
        neg_scaled = _list(
            ctx, (_sym(ctx, "-"), scaled),
        )
        exp_call = _list(
            ctx, (_sym(ctx, "exp"), neg_scaled),
        )
        jitter_if = _list(
            ctx,
            (
                _sym(ctx, "if"),
                _list(
                    ctx,
                    (_sym(ctx, "="),
                     _sym(ctx, "i"), _sym(ctx, "j")),
                ),
                _num(ctx, jitter),
                _num(ctx, 0),
            ),
        )
        entry = _list(
            ctx, (_sym(ctx, "+"), exp_call, jitter_if),
        )
        inner_lambda = _list(
            ctx,
            (
                _sym(ctx, "lambda"),
                _list(ctx, (_sym(ctx, "j"),)),
                entry,
            ),
        )
        iota_n_inner = _list(
            ctx, (_sym(ctx, "iota"), _num(ctx, n)),
        )
        inner_map = _list(
            ctx,
            (_sym(ctx, "map"), inner_lambda, iota_n_inner),
        )
        outer_lambda = _list(
            ctx,
            (
                _sym(ctx, "lambda"),
                _list(ctx, (_sym(ctx, "i"),)),
                inner_map,
            ),
        )
        iota_n_outer = _list(
            ctx, (_sym(ctx, "iota"), _num(ctx, n)),
        )
        outer_map = _list(
            ctx,
            (_sym(ctx, "map"), outer_lambda, iota_n_outer),
        )
        cov_form = _list(
            ctx,
            (
                _sym(ctx, "define"),
                _sym(ctx, cov_name),
                outer_map,
            ),
        )
        # (define <name> (multivariate-gaussian __gp_mean_<name>
        #                                         __gp_cov_<name>))
        sample_form = _list(
            ctx,
            (
                _sym(ctx, "define"),
                _sym(ctx, node.name),
                _list(
                    ctx,
                    (
                        _sym(ctx, "multivariate-gaussian"),
                        _sym(ctx, mean_name),
                        _sym(ctx, cov_name),
                    ),
                ),
            ),
        )
        return (mean_form, cov_form, sample_form)

    def _render_marginalize_forms(
        self, ctx: _RenderCtx, node: IRMarginalize
    ) -> tuple[SchemaFragment, ...]:
        """Lower [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        to its sibling forms: the latent sample define, then one form
        per scope step (typically an observe).
        """
        expanded = self.explicit_latent_scope(node)
        out: list[SchemaFragment] = []
        for child in expanded:
            out.extend(self._render_body_forms(ctx, child))
        return tuple(out)

    def _return_form(
        self, ctx: _RenderCtx, body: tuple[IRNode, ...]
    ) -> SchemaFragment | None:
        for node in body:
            if isinstance(node, IRReturn):
                names = node.names
                if not names:
                    return None
                if len(names) == 1:
                    return _sym(ctx, names[0])
                return _list(
                    ctx,
                    (_sym(ctx, "list"), *(_sym(ctx, n) for n in names)),
                )
        return None

    # ----- the four dispatch points -----

    def declare(
        self,
        ctx: _RenderCtx,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """Church has no separate declaration block.

        Sample / observe bind their result names directly via
        ``(define name (map ...))``; data inputs land in the model
        signature in `render`. This dispatch is a no-op for every
        block.
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
        """Emit the per-batch-axis sample / observe form.

        For a non-observed step with N batch dims:

        ``(define <name>
            (map (lambda (m_<a0>)
                   (map (lambda (m_<a1>) (sample <dist>))
                        (iota <B1>)))
                 (iota <B0>)))``

        For an observed step: a ``for-each`` over the observation
        plate, with ``(observe <dist> (list-ref <obs> n))`` inside
        each iteration.
        """
        del arg_names, constraint
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(_TARGET, [f"family:{family}"])
        target_symbol = meta.target_names.get("church")
        if target_symbol is None:
            raise UnsupportedConstruct(_TARGET, [f"family:{family}:church"])

        dist_form = self._build_dist_call(ctx, target_symbol, args)
        if observed:
            return self._wrap_observe(ctx, name, dist_form, plate)
        return self._wrap_sample_define(ctx, name, dist_form, plate)

    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Lower [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        to ``(define <latent> (map ...))`` plus the scope body inline.

        Church has no native ``log_sum_exp`` enumeration construct; the
        spec dictates inline lowering via
        [`explicit_latent_scope`][quivers.transpile.renderers._base.RendererBase.explicit_latent_scope]
        for every backend except Stan.

        The top-level
        [`render`][quivers.transpile.renderers.church.ChurchRenderer.render]
        consumes the multi-form expansion via
        ``_render_marginalize_forms`` so the latent sample and scope
        observes sit as siblings of the surrounding ``(define (model
        ...) ...)`` body. When `marginalize` is invoked as a single
        dispatch point (per the Renderer protocol contract), the
        forms collapse into a ``(begin ...)`` block.
        """
        forms = self._render_marginalize_forms(ctx, node)
        if not forms:
            return ""
        if len(forms) == 1:
            return forms[0]
        return _list(ctx, (_sym(ctx, "begin"), *forms))

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """`(make-list K <value>)` for 1-D broadcasts.

        Higher-rank broadcasts raise
        [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
        with ``arg:broadcast-rank-2+`` (Church has no canonical
        matrix form, so a rank-2 broadcast has no idiomatic
        rendering).
        """
        if len(target_shape) != 1:
            raise UnsupportedConstruct(
                _TARGET,
                [f"arg:broadcast-rank-{len(target_shape)}"],
            )
        (k,) = target_shape
        value_form = self._render_arg(ctx, value)
        return _list(
            ctx,
            (_sym(ctx, "make-list"), _num(ctx, float(k)), value_form),
        )

    # ----- helpers -----

    def _build_dist_call(
        self,
        ctx: _RenderCtx,
        symbol: str,
        args: tuple[IRArg, ...],
    ) -> SchemaFragment:
        """`(<symbol> <arg0> <arg1> ...)`."""
        children = [_sym(ctx, symbol)]
        for arg in args:
            children.append(self._render_arg(ctx, arg))
        return _list(ctx, tuple(children))

    def _render_arg(self, ctx: _RenderCtx, arg: IRArg) -> SchemaFragment:
        """Render one [`IRArg`][quivers.transpile.ir.IRArg] to a
        Scheme schema fragment.

        Dispatches on the IR variant; consults
        [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META] for
        `IRArgFamilyRef` lookups so the wrapper family's inner
        distribution renders inline.
        """
        if isinstance(arg, IRArgNumber):
            return _num(ctx, arg.value)
        if isinstance(arg, IRArgRef):
            return self._render_ref(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self.broadcast(ctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self._render_list(ctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(ctx, arg)
        if isinstance(arg, IRArgFamilyRef):
            return self._render_family_ref(ctx, arg)
        raise UnsupportedConstruct(
            _TARGET, [f"arg:{type(arg).__name__}"]
        )

    def _render_ref(
        self, ctx: _RenderCtx, arg: IRArgRef
    ) -> SchemaFragment:
        """`x` for a bare reference; `(list-ref x idx)` for one index;
        nested `list-ref` for higher-rank indexing."""
        if not arg.indices:
            return _sym(ctx, arg.name)
        current: SchemaFragment = _sym(ctx, arg.name)
        for idx in arg.indices:
            idx_form = self._render_arg(ctx, idx)
            current = _list(
                ctx, (_sym(ctx, "list-ref"), current, idx_form)
            )
        return current

    def _render_list(
        self, ctx: _RenderCtx, arg: IRArgList
    ) -> SchemaFragment:
        children = [_sym(ctx, "list")]
        for elem in arg.elements:
            children.append(self._render_arg(ctx, elem))
        return _list(ctx, tuple(children))

    def _render_matrix(
        self, ctx: _RenderCtx, arg: IRArgMatrix
    ) -> SchemaFragment:
        del ctx, arg
        raise UnsupportedConstruct(
            _TARGET, ["arg:matrix-literal"]
        )

    def _render_family_ref(
        self, ctx: _RenderCtx, arg: IRArgFamilyRef
    ) -> SchemaFragment:
        """Resolve an [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        through the morphism table and emit the inner distribution
        call inline.

        Wrapper families (`Truncated`, `Mixture`, `Independent`,
        `Transformed`, `LKJCorrelationFactor`) carry their wrapped
        distribution as a morphism name; the morphism table maps that
        name to its `~ Family(...)` clause. Church does not expose
        any of those wrappers natively, so the renderer raises
        [`UnsupportedConstruct`][quivers.transpile._api.UnsupportedConstruct]
        on absent morphism data; callers wanting truncated /
        composition / rejection-sampling semantics fold them into the
        outer distribution call before lowering.
        """
        del ctx
        raise UnsupportedConstruct(
            _TARGET, [f"arg:family_ref:{arg.name}"]
        )

    def _wrap_sample_define(
        self,
        ctx: _RenderCtx,
        name: str,
        dist_form: SchemaFragment,
        plate: Plate,
    ) -> SchemaFragment:
        """`(define <name> <body>)` where `<body>` is the (possibly
        nested) `(map (lambda (m_<axis>) (sample <dist>)) (iota N))`
        wrapping over the plate's batch dims."""
        sample_form = _list(ctx, (_sym(ctx, "sample"), dist_form))
        wrapped = self._wrap_in_maps(
            ctx, sample_form, plate.batch_dims, axis_suffix=""
        )
        return _list(
            ctx, (_sym(ctx, "define"), _sym(ctx, name), wrapped)
        )

    def _wrap_in_maps(
        self,
        ctx: _RenderCtx,
        inner: SchemaFragment,
        batch_dims: tuple[object, ...],
        *,
        axis_suffix: str,
    ) -> SchemaFragment:
        """Wrap `inner` in one `(map (lambda (m_<axis>) ...) (iota N))`
        per batch dim, outermost-first.

        ``axis_suffix`` distinguishes the loop variable across nested
        contexts (e.g. the marginalized latent's per-axis index uses
        ``"_z"`` so the scope's inner observation can rebind without
        shadowing).
        """
        current = inner
        for dim in reversed(batch_dims):
            current = self._one_map_layer(ctx, current, dim, axis_suffix)
        return current

    def _one_map_layer(
        self,
        ctx: _RenderCtx,
        inner: SchemaFragment,
        dim: object,
        axis_suffix: str,
    ) -> SchemaFragment:
        loop_var = f"m_{getattr(dim, 'name', 'i')}{axis_suffix}"
        size_form = self._dim_size_form(ctx, dim)
        lambda_form = _list(
            ctx,
            (
                _sym(ctx, "lambda"),
                _list(ctx, (_sym(ctx, loop_var),)),
                inner,
            ),
        )
        iota_form = _list(ctx, (_sym(ctx, "iota"), size_form))
        return _list(
            ctx,
            (_sym(ctx, "map"), lambda_form, iota_form),
        )

    def _dim_size_form(
        self, ctx: _RenderCtx, dim: object
    ) -> SchemaFragment:
        """Emit the size form for one plate dim.

        Static dims render as their integer cardinality; dynamic dims
        emit ``(length <size_name>)`` so the host can supply runtime
        data without recompilation.
        """
        if isinstance(dim, DimStatic):
            return _num(ctx, float(dim.size))
        if isinstance(dim, DimDynamic):
            return _list(
                ctx, (_sym(ctx, "length"), _sym(ctx, dim.size_name))
            )
        raise UnsupportedConstruct(
            _TARGET, [f"dim:{type(dim).__name__}"]
        )

    def _wrap_observe(
        self,
        ctx: _RenderCtx,
        obs_name: str,
        dist_form: SchemaFragment,
        plate: Plate,
    ) -> SchemaFragment:
        """`(for-each (lambda (n) (observe <dist> (list-ref <obs> n)))
        (iota (length <obs>)))` over the observation plate.

        For higher-rank batch dims the function nests `for-each` from
        outermost to innermost. For an empty plate the form is a
        single ``(observe <dist> <obs>)`` (no iteration).
        """
        if not plate.batch_dims:
            return _list(
                ctx,
                (
                    _sym(ctx, "observe"),
                    dist_form,
                    _sym(ctx, obs_name),
                ),
            )
        # Index variable per nested level: outermost = n0, n1, ...
        loop_vars = tuple(
            f"n_{i}" if len(plate.batch_dims) > 1 else "n"
            for i in range(len(plate.batch_dims))
        )
        # Build the indexed reference `(list-ref ... obs)` nested per
        # loop, innermost-first to match the lambda nesting.
        obs_ref: SchemaFragment = _sym(ctx, obs_name)
        for lv in loop_vars:
            obs_ref = _list(
                ctx, (_sym(ctx, "list-ref"), obs_ref, _sym(ctx, lv))
            )
        innermost = _list(
            ctx, (_sym(ctx, "observe"), dist_form, obs_ref)
        )
        # Wrap in nested `for-each` from inner to outer.
        current = innermost
        for lv, dim in zip(
            reversed(loop_vars),
            reversed(plate.batch_dims),
            strict=True,
        ):
            size_form = self._dim_size_form(ctx, dim)
            lambda_form = _list(
                ctx,
                (
                    _sym(ctx, "lambda"),
                    _list(ctx, (_sym(ctx, lv),)),
                    current,
                ),
            )
            iota_form = _list(ctx, (_sym(ctx, "iota"), size_form))
            current = _list(
                ctx,
                (_sym(ctx, "for-each"), lambda_form, iota_form),
            )
        return current


# ---------------------------------------------------------------------------
# Schema-builder helpers (vertex / edge / literal constructors).
# ---------------------------------------------------------------------------


def _fresh(ctx: _RenderCtx, prefix: str) -> str:
    ctx.fresh_counter += 1
    return f"{prefix}_{ctx.fresh_counter}"


def _v(ctx: _RenderCtx, vid: str, kind: str) -> str:
    ctx.sb.vertex(vid, kind)
    return vid


def _e(ctx: _RenderCtx, src: str, tgt: str, kind: str = "child_of") -> None:
    ctx.sb.edge(src, tgt, kind)


def _sym(ctx: _RenderCtx, text: str) -> SchemaFragment:
    vid = _v(ctx, _fresh(ctx, "sym"), "symbol")
    ctx.sb.constraint(vid, "literal-value", text)
    return vid


def _num(ctx: _RenderCtx, value: float) -> SchemaFragment:
    vid = _v(ctx, _fresh(ctx, "num"), "number")
    text = str(int(value)) if value == int(value) else repr(value)
    ctx.sb.constraint(vid, "literal-value", text)
    return vid


def _list(
    ctx: _RenderCtx, children: tuple[SchemaFragment, ...]
) -> SchemaFragment:
    """A parenthesised Scheme list with `children` in order."""
    lst = _v(ctx, _fresh(ctx, "lst"), "list")
    for child in children:
        if child:
            _e(ctx, lst, child)
    return lst


# ---------------------------------------------------------------------------
# Bridge adapter: the
# [`render_let_expr_scheme`][quivers.transpile.renderers._scheme_helpers.render_let_expr_scheme]
# helper consumes a duck-typed context with `.fresh`, `.v`, `.e`,
# `.lit`, `.constraint`, `.cards`, `.target` members. The IR renderer's
# [`_RenderCtx`][quivers.transpile.renderers._base._RenderCtx] exposes
# only the panproto `SchemaBuilder` and the morphism / let / cards
# tables; this adapter projects the helper's expected surface onto it
# without rebinding the fresh-id counter, so let / score expressions
# render through the same code path as the schema-extraction backend
# and benefit from cards-driven binder-size resolution when the helper
# adds it.
# ---------------------------------------------------------------------------


class _LetExprCtx:
    """Duck-typed adapter exposing the
    [`render_let_expr_scheme`][quivers.transpile.renderers._scheme_helpers.render_let_expr_scheme]
    context protocol (``fresh``, ``v``, ``e``, ``lit``, ``constraint``,
    ``cards``, ``target``) on top of a
    [`_RenderCtx`][quivers.transpile.renderers._base._RenderCtx]."""

    def __init__(
        self,
        sb: panproto.SchemaBuilder,
        owner: _RenderCtx,
        cards: dict[str, int],
    ) -> None:
        self._sb = sb
        self._owner = owner
        self.cards = cards
        self.target = "church"

    def fresh(self, prefix: str) -> str:
        return _fresh(self._owner, prefix)

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
# Runtime-helper graft: `matrix-normal` as a top-level `(define ...)`
# built on top of Church's `multivariate-gaussian` primitive plus
# small vec / Kronecker / reshape helpers.
#
# Church's built-in primitive set ships `gaussian`, `beta`, `flip`,
# `multivariate-gaussian`, ... but lacks a matrix-variate normal.
# The transpile-time graft parses
# [`runtime_church.scm`][quivers.transpile.runtime_church] once at
# module-load through panproto's Scheme tree-sitter grammar; per-render,
# it copies every grafted vertex / constraint / edge into the per-render
# schema (with fresh vertex ids) and attaches the runtime's top-level
# forms as `child_of` of the emitted `program` vertex above the
# `(define (model ...) ...)` form. The graft makes `matrix-normal` /
# `vec` / `reshape` / `mat-kron` resolve through the standard Scheme
# top-level binding lookup at evaluation time.
# ---------------------------------------------------------------------------


_RUNTIME_CHURCH_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "runtime_church.scm"
)


#: Families whose Church emit relies on the
#: [`runtime_church.scm`][quivers.transpile.runtime_church] helper subtree.
#: Church ships `gaussian`, `beta`, `flip`, `multivariate-gaussian`, ... as
#: built-in distributions but lacks `matrix-normal`; the renderer grafts
#: the helper when the IR samples or observes from any of them.
_CHURCH_RUNTIME_HELPER_FAMILIES: frozenset[str] = frozenset({
    "MatrixNormal",
})


def _load_runtime_church_schema() -> tuple[
    panproto.Schema, str, tuple[str, ...]
]:
    """Parse [`runtime_church.scm`][quivers.transpile.runtime_church]
    through panproto's Scheme tree-sitter grammar at module-load time.

    Returns the parsed schema, the parsed `program` vertex id, and the
    tuple of top-level child ids in source order (sorted by
    `start-byte`). The graft replays these children in order beneath
    the per-render `program` so the emit's top-level forms appear in
    the original file's layout.
    """
    schema = parser_registry().parse_with_protocol(
        "scheme",
        _RUNTIME_CHURCH_PATH.read_bytes(),
        str(_RUNTIME_CHURCH_PATH),
    )
    src_id = next(
        (v.id for v in schema.vertices if v.kind == "program"),
        None,
    )
    if src_id is None:
        raise RuntimeError(
            f"`program` not found in parse of {_RUNTIME_CHURCH_PATH}"
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


_RUNTIME_CHURCH_SCHEMA, _RUNTIME_CHURCH_PROGRAM_ID, _RUNTIME_CHURCH_TOP_LEVEL = (
    _load_runtime_church_schema()
)


def _church_subtree_vertex_ids(
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


_RUNTIME_CHURCH_SUBTREE = _church_subtree_vertex_ids(
    _RUNTIME_CHURCH_SCHEMA, _RUNTIME_CHURCH_TOP_LEVEL
)


def _ir_uses_family(body: tuple[IRNode, ...], family: str) -> bool:
    """True iff any [`IRSample`][quivers.transpile.ir.IRSample],
    [`IRObserve`][quivers.transpile.ir.IRObserve], or
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] in `body`
    (including nested marginalize scopes) draws from `family`."""
    for node in body:
        if (
            isinstance(node, (IRSample, IRObserve))
            and node.family == family
        ):
            return True
        if isinstance(node, IRMarginalize):
            if node.family == family:
                return True
            if _ir_uses_family(node.scope, family):
                return True
    return False


def _graft_runtime_church_helper(ctx: _RenderCtx, program_vid: str) -> None:
    """Graft the runtime-helper subtree onto the per-render schema.

    Copies every vertex, every constraint, and every internal edge of
    the parsed `runtime_church.scm` subtree into the per-render
    `SchemaBuilder` with fresh vertex ids, then attaches each top-level
    child as a `child_of` of `program_vid` in source order. The grafted
    forms appear above the model `(define ...)` in the emit so the
    helper bindings are in scope when the model is evaluated.
    """
    src_schema = _RUNTIME_CHURCH_SCHEMA
    subtree = _RUNTIME_CHURCH_SUBTREE
    id_map: dict[str, str] = {}

    for old in subtree:
        new = _fresh(ctx, "rc")
        id_map[old] = new
        kind = next(
            v.kind for v in src_schema.vertices if v.id == old
        )
        ctx.sb.vertex(new, kind)
        for cstr in src_schema.constraints_for(old):
            ctx.sb.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            ctx.sb.edge(id_map[edge.src], id_map[edge.tgt], edge.kind)
    for child_old in _RUNTIME_CHURCH_TOP_LEVEL:
        ctx.sb.edge(program_vid, id_map[child_old], "child_of")


__all__ = ["ChurchRenderer"]
