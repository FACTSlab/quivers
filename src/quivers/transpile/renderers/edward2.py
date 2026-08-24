"""Edward2 renderer: [`IRProgram`][quivers.transpile.ir.IRProgram] to
TFP-Edward2 Python source under the `python` tree-sitter grammar.

Edward2's idiom is a plain ``def model(<inputs>=None, ...): ...`` whose
body constructs random variables via
``edward2.<Family>(args, sample_shape=[B0, B1, ...], name="<name>")``.
There is no separate data block; every exogenous identifier (program
parameters, observed values, fibrations) becomes a function parameter
with a ``=None`` default. Discrete-latent integration scopes lower to
explicit ``IRSample(latent)`` plus the inlined scope body (Edward2
samples discrete latents natively; ``log_sum_exp`` enumeration is the
Stan-specific path).

Scalar-to-vector broadcast for a per-axis parameter (the ``alpha`` in
``Dirichlet(alpha)`` against the Topic plate) is emitted as
``tf.fill([K], <value>)``; for 2D targets as ``tf.fill([R, C], <value>)``.
``IRArgList`` literals render as plain Python lists ``[e0, e1, ...]``
(TF accepts those for vector-arg positions); ``IRArgMatrix`` as nested
lists ``[[...], [...]]``.

The renderer reads target distribution names from
``FAMILY_META[family].target_names["edward2"]`` and applies per-arg
renames from ``arg_aliases["edward2"]``; no per-family branching
appears here.
"""

from __future__ import annotations

import panproto

from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import EmitPretty, target_protocol
from quivers.transpile.renderers._python_helpers import (
    PyCtx,
    assignment,
    attribute,
    call,
    function_def,
    identifier,
    name_event_rank_map,
    number_literal,
    python_binary_op as _python_binary_op,
    python_method_call as _python_method_call,
    python_paren as _python_paren,
    python_unary_minus as _python_unary_minus,
    render_let_expr_python,
    string_literal,
)
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
from quivers.transpile.ir import (
    CSBoolean,
    CSIntegerInterval,
    CSInterval,
    CSNonnegativeInteger,
    CSPositive,
    CSPositiveInteger,
    CSReal,
    CSUnitInterval,
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
    is_real_cov_matrix,
    is_real_matrix,
    is_real_simplex,
    is_real_vector,
)
from quivers.transpile.renderers._base import (
    BlockKind,
    RendererBase,
    SchemaFragment,
    _RenderCtx,
    assert_no_dangling_refs,
)


_TARGET = "edward2"
_BACKEND_KEY = f"qvr-{_TARGET}"


#: Edward2-side argument injection for QVR families whose underlying
#: torch distribution carries fewer parameters than TFP's same-named
#: distribution. ``HalfCauchy(scale)`` maps to TFP's
#: ``HalfCauchy(loc, scale)`` (loc is required); the renderer prepends
#: ``IRArgNumber(0)`` under the ``"loc"`` arg name so TFP sees a
#: complete (loc, scale) pair. The injected ``loc=0`` is the standard
#: half-distribution origin and matches the QVR-side semantics for
#: HalfCauchy / HalfNormal.
_PREPEND_LOC_ZERO: frozenset[str] = frozenset({"HalfCauchy"})


#: Edward2-local argument renames layered on top of
#: ``FAMILY_META[family].arg_aliases["edward2"]``. TFP's ``Pareto``
#: names its shape parameter ``concentration`` where torch / QVR name
#: it ``alpha``; without the rename the emitted keyword ``alpha=`` is
#: not a valid TFP ``Pareto`` constructor argument and construction
#: raises ``TypeError``. The value is unchanged: torch's ``alpha`` and
#: TFP's ``concentration`` are the same Pareto shape parameter.
_EDWARD2_ARG_ALIASES: dict[str, dict[str, str]] = {
    "Pareto": {"alpha": "concentration"},
}


class Edward2Renderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to TFP
    Edward2 Python source.

    Output shape (single function, all inputs defaulted to ``None``):

    ```python
    def model(alpha=None, beta=None, word_idx=None, w=None):
        theta = edward2.Dirichlet(tf.fill([3], alpha),
                                  sample_shape=[20], name="theta")
        phi = edward2.Dirichlet(tf.fill([200], beta),
                                sample_shape=[3], name="phi")
        z = edward2.Categorical(probs=theta,
                                sample_shape=[20], name="z")
        w = edward2.Categorical(probs=phi[z[word_idx]], name="w")
        return theta
    ```

    The walk overrides
    [`RendererBase.render`][quivers.transpile.renderers._base.RendererBase.render]
    because Edward2's program shape (single function, no block
    structure) does not map onto the inherited block-by-block default
    walk.
    """

    target: str = _TARGET

    def target_protocol(self) -> panproto.Protocol:
        return target_protocol("python")

    # ------------------------------------------------------------------
    # Top-level render
    # ------------------------------------------------------------------

    def render(self, ir: IRProgram) -> panproto.Schema:
        assert_no_dangling_refs(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        py = PyCtx(
            sb,
            cards=dict(ir.cards),
            target="edward2",
            name_event_rank=name_event_rank_map(ir),
            gather_symbol=("tf", "gather"),
        )
        ctx = _RenderCtx(sb=sb, morphisms={}, defines={})

        sb.vertex("mod", "module")
        body_vid = py.v(py.fresh("body"), "block")
        param_names = tuple(inp.name for inp in ir.inputs)
        fn = function_def(
            py, name="model", default_params=param_names, body_vid=body_vid
        )
        sb.edge("mod", fn, "child_of")

        # Track input constraint info so the renderer can decide whether
        # to wrap a scalar `IRArgRef` into a `tf.fill(...)` broadcast.
        input_specs: dict[str, ConstraintSpec] = {
            inp.name: inp.constraint for inp in ir.inputs
        }
        bindings: dict[str, _Binding] = {
            inp.name: _Binding(constraint=inp.constraint, plate=inp.plate)
            for inp in ir.inputs
        }

        for node in ir.body:
            self._emit_node(py, ctx, body_vid, node, input_specs, bindings)

        return sb.build()

    def emit_bytes(self, ir: IRProgram) -> bytes:
        """Convenience: render `ir` and run `emit_pretty` to bytes."""
        schema = self.render(ir)
        return EmitPretty("python")(schema)

    # ------------------------------------------------------------------
    # IRNode dispatch
    # ------------------------------------------------------------------

    def _emit_node(
        self,
        py: PyCtx,
        ctx: _RenderCtx,
        body_vid: str,
        node: IRNode,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> None:
        if isinstance(node, IRDataInput):
            # Inputs are surfaced as function parameters; nothing
            # to emit in the body.
            return
        if isinstance(node, IRSample):
            if node.family == "GP":
                self._emit_gp_block(py, body_vid, node)
                bindings[node.name] = _Binding(
                    constraint=node.constraint, plate=node.plate
                )
                return
            rhs = self._dist_call(
                py,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                plate=node.plate,
                input_specs=input_specs,
                bindings=bindings,
                observed_name=None,
            )
            py.e(
                body_vid,
                assignment(py, lhs_name=node.name, rhs=rhs),
                "child_of",
            )
            bindings[node.name] = _Binding(
                constraint=node.constraint, plate=node.plate
            )
            return
        if isinstance(node, IRObserve):
            rhs = self._dist_call(
                py,
                name=node.name,
                family=node.family,
                args=node.args,
                arg_names=node.arg_names,
                # Observed leaves omit ``sample_shape``: the probe
                # detects the value-shape mismatch and scores the
                # observation directly via ``dist.log_prob(value)``,
                # which broadcasts ``loc`` / ``scale`` / ``probs``
                # against the user-supplied tensor's shape. Carrying
                # batch_dims into the RV constructor instead would
                # bake in a fixed ``value.shape`` that rejects any
                # observation whose runtime shape comes from a
                # deterministic input the IR does not know about.
                plate=Plate(event_dims=node.plate.event_dims, batch_dims=()),
                input_specs=input_specs,
                bindings=bindings,
                observed_name=node.name,
            )
            py.e(
                body_vid,
                assignment(py, lhs_name=node.name, rhs=rhs),
                "child_of",
            )
            bindings[node.name] = _Binding(
                constraint=node.constraint, plate=node.plate
            )
            return
        if isinstance(node, IRDeterministic):
            asn = py.v(py.fresh("asn"), "assignment")
            lhs = identifier(py, node.name)
            py.e(asn, lhs, "left")
            py.e(asn, render_let_expr_python(py, node.expr), "right")
            py.e(body_vid, asn, "child_of")
            bindings[node.name] = _Binding(
                constraint=node.constraint, plate=node.plate
            )
            return
        if isinstance(node, IRScore):
            self._emit_score_node(py, body_vid, node)
            return
        if isinstance(node, IRMarginalize):
            for sub in _thread_via_through_scope(
                self.explicit_latent_scope(node),
                latent_name=node.latent,
            ):
                self._emit_node(
                    py, ctx, body_vid, sub, input_specs, bindings
                )
            return
        if isinstance(node, IRReturn):
            self._emit_return(py, body_vid, node.names)
            return
        raise UnsupportedConstruct(
            _BACKEND_KEY, [f"node:{type(node).__name__}"]
        )

    def _emit_gp_block(
        self,
        py: PyCtx,
        body_vid: str,
        node: IRSample,
    ) -> None:
        """Emit a Gaussian-process sample as three Edward2 statements:

            __gp_mean_<name> = tf.zeros([N])
            __gp_cov_<name>  = tf.exp(-0.5 * (diff)*(diff) / (ls*ls))
                                + jitter * tf.eye(N)
            <name> = edward2.MultivariateNormalFullCovariance(
                        loc=__gp_mean_<name>,
                        covariance_matrix=__gp_cov_<name>, name="<name>")

        Uses tf.zeros / tf.exp / tf.eye math. The kernel matrix is
        the squared-exponential covariance plus diagonal jitter.
        Parens wrap the diff and squared-length-scale subexpressions
        so the Python pretty-printer keeps operator precedence.
        """
        if len(node.args) != 2 or not isinstance(
            node.args[1], IRArgKernel
        ):
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = node.args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                _BACKEND_KEY,
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
        # __gp_mean_<name> = tf.zeros([N])
        mean_rhs = call(
            py,
            attribute(py, ("tf", "zeros")),
            positional=(number_literal(py, n),),
        )
        py.e(
            body_vid,
            assignment(py, lhs_name=mean_name, rhs=mean_rhs),
            "child_of",
        )
        # __gp_cov_<name> = tf.exp(-0.5 * (diff)*(diff) / (ls*ls))
        #                    + jitter * tf.eye(N)
        x_col = _python_method_call(
            py, identifier(py, x), "reshape",
            (
                _python_unary_minus(py, number_literal(py, 1)),
                number_literal(py, 1),
            ),
        )
        x_row = _python_method_call(
            py, identifier(py, x), "reshape",
            (
                number_literal(py, 1),
                _python_unary_minus(py, number_literal(py, 1)),
            ),
        )
        diff = _python_paren(
            py, _python_binary_op(py, "-", x_col, x_row),
        )
        diff_sq = _python_paren(
            py, _python_binary_op(py, "*", diff, diff),
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
            attribute(py, ("tf", "exp")),
            positional=(exponent,),
        )
        eye_call = call(
            py,
            attribute(py, ("tf", "eye")),
            positional=(number_literal(py, n),),
        )
        jitter_term = _python_binary_op(
            py, "*", number_literal(py, jitter), eye_call,
        )
        cov_rhs = _python_binary_op(
            py, "+", kernel_call, jitter_term,
        )
        py.e(
            body_vid,
            assignment(py, lhs_name=cov_name, rhs=cov_rhs),
            "child_of",
        )
        # <name> = edward2.MultivariateNormalFullCovariance(
        #     loc=__gp_mean_<name>, covariance_matrix=__gp_cov_<name>,
        #     name="<name>")
        mvn_call = call(
            py,
            attribute(
                py, ("edward2", "MultivariateNormalFullCovariance"),
            ),
            keyword=(
                ("loc", identifier(py, mean_name)),
                ("covariance_matrix", identifier(py, cov_name)),
                ("name", string_literal(py, node.name)),
            ),
        )
        py.e(
            body_vid,
            assignment(py, lhs_name=node.name, rhs=mvn_call),
            "child_of",
        )

    def _emit_score_node(
        self, py: PyCtx, body_vid: str, node: IRScore
    ) -> None:
        """Bind ``<name> = <expr>``.

        Edward2 has no top-level factor primitive; the canonical idiom
        is to compute the log-density factor at trace time. For the
        static fragment we bind the expression and leave the factor
        accumulation to the consumer's tape / interceptor.
        """
        asn = py.v(py.fresh("asn"), "assignment")
        lhs = identifier(py, node.name)
        py.e(asn, lhs, "left")
        py.e(asn, render_let_expr_python(py, node.expr), "right")
        py.e(body_vid, asn, "child_of")

    def _emit_return(
        self, py: PyCtx, body_vid: str, names: tuple[str, ...]
    ) -> None:
        if not names:
            return
        rs = py.v(py.fresh("ret"), "return_statement")
        if len(names) == 1:
            py.e(rs, identifier(py, names[0]), "child_of")
        else:
            elist = py.v(py.fresh("elist"), "expression_list")
            for var in names:
                py.e(elist, identifier(py, var), "child_of")
            py.e(rs, elist, "child_of")
        py.e(body_vid, rs, "child_of")

    # ------------------------------------------------------------------
    # Dispatch points (Renderer protocol)
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
        """No-op: Edward2 has no separate declarations.

        Data inputs are emitted as function parameters by ``render``;
        sample / observe declarations are subsumed into the
        ``edward2.<Family>(...)`` call assigned to the bound name.
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
        """Build the ``edward2.<Family>(...)`` call vertex.

        The IR-walk in ``render`` calls this from a fresh `PyCtx`
        wrapping ``ctx.sb``. Edward2 has no native ``obs=`` keyword;
        observations are surfaced by carrying them as default-None
        function parameters and conditioning via the caller's
        interceptor mechanism, so ``observed`` does not change the
        emitted call.
        """
        del constraint, observed
        py = PyCtx(ctx.sb, target="edward2")
        return self._dist_call(
            py,
            name=name,
            family=family,
            args=args,
            arg_names=arg_names,
            plate=plate,
            input_specs={},
            bindings={},
            observed_name=None,
        )

    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Lower the integration scope to explicit sampling.

        Edward2 samples discrete latents natively; the explicit-latent
        rewrite from
        [`RendererBase.explicit_latent_scope`][quivers.transpile.renderers._base.RendererBase.explicit_latent_scope]
        produces ``IRSample(latent)`` followed by the scope body. The
        top-level ``render`` IR walk consumes the rewritten sequence.
        """
        del ctx, node
        return ""

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit a ``tf.fill([...], <value>)`` call.

        1D targets render as ``tf.fill([K], <value>)``; 2D as
        ``tf.fill([R, C], <value>)``. The shape literal is a plain
        Python list (Edward2 / TF accept it).
        """
        py = PyCtx(ctx.sb, target="edward2")
        return self._broadcast(py, value, target_shape, {}, {})

    # ------------------------------------------------------------------
    # Distribution-call construction
    # ------------------------------------------------------------------

    def _dist_call(
        self,
        py: PyCtx,
        *,
        name: str,
        family: str,
        args: tuple[IRArg, ...],
        arg_names: tuple[str, ...],
        plate: Plate,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
        observed_name: str | None,
    ) -> str:
        """Build ``edward2.<Family>(positional|kwargs,
        sample_shape=[..], name="<name>")``.

        Args are emitted positionally for single-arg distributions like
        ``Dirichlet(concentration)`` and as keyword args for
        multi-positional families to match Edward2's `tfp` conventions.
        ``arg_aliases["edward2"]`` rewrites argument names.
        """
        del observed_name
        meta = self._lookup_meta(family)
        dist_class = meta.target_names.get(_TARGET)
        if dist_class is None:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [f"family:{family}: no edward2 target name in FAMILY_META"],
            )

        # Families whose TFP constructor shape diverges from the
        # generic (rename-only) path get a dedicated builder.
        if family == "MatrixNormal":
            return self._matrix_normal_call(
                py,
                name=name,
                dist_class=dist_class,
                args=args,
                plate=plate,
                input_specs=input_specs,
                bindings=bindings,
            )
        if family == "LKJCholesky":
            return self._lkj_cholesky_call(
                py,
                name=name,
                dist_class=dist_class,
                args=args,
                plate=plate,
                input_specs=input_specs,
                bindings=bindings,
            )

        # ``HalfStudentT`` has no TFP class. A half-Student-t with
        # ``(df, scale)`` is the fold of ``StudentT(df, 0, scale)`` onto
        # the nonnegative half-line; its density there is
        # ``StudentT(df, 0, scale)`` scaled by 2, so the two agree up to
        # the additive constant ``log 2`` on the family's positive
        # support. The renderer emits the location-scale StudentT and
        # relies on the fold being an additive-constant shift.
        if family == "HalfStudentT":
            dist_class = "StudentT"
            args = (args[0], IRArgNumber(value=0.0), *args[1:])
            arg_names = (arg_names[0], "loc", *arg_names[1:])
            meta = self._lookup_meta("StudentT")

        callee = attribute(py, ("edward2", dist_class))
        aliases = {
            **meta.arg_aliases.get(_TARGET, {}),
            **_EDWARD2_ARG_ALIASES.get(family, {}),
        }

        # Inject the canonical zero-loc argument for families whose
        # torch distribution carries fewer parameters than TFP's
        # same-named distribution (``HalfCauchy(scale)`` ->
        # ``HalfCauchy(loc, scale)``). The injected name is the QVR
        # side ``"loc"``; ``arg_aliases["edward2"]`` may rename it
        # before the keyword form is emitted.
        if family in _PREPEND_LOC_ZERO:
            args = (IRArgNumber(value=0.0), *args)
            arg_names = ("loc", *arg_names)

        # Build per-arg expression vertices, lifting scalar refs to
        # broadcasts where the family's event_dims demand it.
        target_event_shape = _event_shape(plate)
        rendered_args = tuple(
            self._render_arg(
                py,
                a,
                expected_arg_event=target_event_shape,
                arg_position=i,
                meta=meta,
                input_specs=input_specs,
                bindings=bindings,
            )
            for i, a in enumerate(args)
        )

        # TFP / Edward2 distribution constructors accept every arg
        # by keyword; some (Categorical, Bernoulli) require the
        # keyword form because `logits` precedes `probs` in their
        # positional signature. The renderer emits the keyword form
        # for arguments whose target name is not the family's first
        # positional, and the positional form otherwise; this matches
        # the canonical Edward2 idiom (`Dirichlet(tf.fill([K], x))`
        # positional; `Categorical(probs=x)` keyword).
        keyword: list[tuple[str, str]] = []
        positional: list[str] = []
        for arg_name, rendered in zip(arg_names, rendered_args, strict=False):
            target_name = aliases.get(arg_name, arg_name)
            if _is_positional_arg(meta, target_name) and not positional:
                positional.append(rendered)
            else:
                keyword.append((target_name, rendered))
        for rendered in rendered_args[len(arg_names) :]:
            positional.append(rendered)

        sample_shape = self._sample_shape(
            py,
            plate,
            family,
            carried=self._carried_batch_keys(args, meta, bindings),
        )
        if sample_shape is not None:
            keyword.append(("sample_shape", sample_shape))
        keyword.append(("name", string_literal(py, name)))

        return call(
            py,
            callee,
            positional=tuple(positional),
            keyword=tuple(keyword),
        )

    def _lkj_cholesky_call(
        self,
        py: PyCtx,
        *,
        name: str,
        dist_class: str,
        args: tuple[IRArg, ...],
        plate: Plate,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Build ``edward2.LKJ(<d>, <concentration>,
        input_output_cholesky=True, name="<name>")``.

        TFP's ``LKJ`` takes the correlation-matrix dimension ``d`` as
        its first positional argument; the generic path emits only the
        concentration, so ``concentration`` binds to ``dimension`` and
        the true concentration is dropped. The dimension is read from
        the sample's event axis. ``input_output_cholesky=True`` makes
        TFP sample / score Cholesky factors, matching the QVR
        ``LKJCholesky`` support (a bare ``LKJ`` ranges over full
        correlation matrices, a different support and density).
        """
        if not plate.event_dims:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                ["family:LKJCholesky:missing matrix-dimension event axis"],
            )
        callee = attribute(py, ("edward2", dist_class))
        dimension = _dim_expr(py, plate.event_dims[-1])
        concentration = self._render_arg(
            py,
            args[0],
            expected_arg_event=(),
            arg_position=0,
            meta=self._lookup_meta("LKJCholesky"),
            input_specs=input_specs,
            bindings=bindings,
        )
        keyword: list[tuple[str, str]] = [
            ("input_output_cholesky", _true_literal(py)),
        ]
        sample_shape = self._sample_shape(
            py,
            plate,
            "LKJCholesky",
            carried=self._carried_batch_keys(
                args, self._lookup_meta("LKJCholesky"), bindings
            ),
        )
        if sample_shape is not None:
            keyword.append(("sample_shape", sample_shape))
        keyword.append(("name", string_literal(py, name)))
        return call(
            py,
            callee,
            positional=(dimension, concentration),
            keyword=tuple(keyword),
        )

    def _matrix_normal_call(
        self,
        py: PyCtx,
        *,
        name: str,
        dist_class: str,
        args: tuple[IRArg, ...],
        plate: Plate,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Build ``edward2.MatrixNormalLinearOperator(loc=...,
        scale_row=..., scale_column=..., name="<name>")``.

        TFP parameterizes the matrix-normal by the Cholesky factors of
        the row and column covariances, wrapped as ``LinearOperator``s,
        under the keywords ``scale_row`` / ``scale_column``. The generic
        path forwards the QVR ``row_covariance`` / ``col_covariance``
        keywords with the covariance matrices themselves, neither of
        which TFP accepts. Row / column covariance ``C`` becomes
        ``tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(C))``;
        since ``C = L Lᵀ`` this leaves the distribution (and its
        log-density) unchanged.
        """
        if len(args) != 3:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [
                    "family:MatrixNormal:expected "
                    "(loc, row_covariance, col_covariance)"
                ],
            )
        callee = attribute(py, ("edward2", dist_class))
        meta = self._lookup_meta("MatrixNormal")
        event = _event_shape(plate)
        rendered = tuple(
            self._render_arg(
                py,
                arg,
                expected_arg_event=event,
                arg_position=i,
                meta=meta,
                input_specs=input_specs,
                bindings=bindings,
            )
            for i, arg in enumerate(args)
        )
        keyword: list[tuple[str, str]] = [
            ("loc", rendered[0]),
            ("scale_row", _linop_cholesky(py, rendered[1])),
            ("scale_column", _linop_cholesky(py, rendered[2])),
        ]
        sample_shape = self._sample_shape(
            py,
            plate,
            "MatrixNormal",
            carried=self._carried_batch_keys(args, meta, bindings),
        )
        if sample_shape is not None:
            keyword.append(("sample_shape", sample_shape))
        keyword.append(("name", string_literal(py, name)))
        return call(py, callee, keyword=tuple(keyword))

    def _render_arg(
        self,
        py: PyCtx,
        arg: IRArg,
        *,
        expected_arg_event: tuple[int, ...],
        arg_position: int,
        meta: FamilyMeta,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Render one [`IRArg`][quivers.transpile.ir.IRArg] to a
        Python expression vertex.

        Scalar references against a vector / matrix arg position are
        lifted into a ``tf.fill(...)`` broadcast. The expected
        argument shape is read from the family's
        ``arg_constraints[arg_name]``; when that constraint is an
        ``IndependentConstraint`` of rank ``>=1`` the surrounding
        plate's ``event_dims`` give the broadcast target shape.
        """
        if isinstance(arg, IRArgNumber):
            return self._maybe_broadcast_scalar(
                py, arg, expected_arg_event, arg_position, meta,
                input_specs, bindings,
            )
        if isinstance(arg, IRArgRef):
            return self._maybe_broadcast_ref(
                py, arg, expected_arg_event, arg_position, meta,
                input_specs, bindings,
            )
        if isinstance(arg, IRArgBroadcast):
            return self._broadcast(
                py, arg.value, arg.target_shape, input_specs, bindings
            )
        if isinstance(arg, IRArgList):
            return self._render_list(py, arg, input_specs, bindings)
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(py, arg, input_specs, bindings)
        if isinstance(arg, IRArgFamilyRef):
            return self._render_family_ref(py, arg, input_specs, bindings)
        raise UnsupportedConstruct(
            _BACKEND_KEY, [f"arg:{type(arg).__name__}"]
        )

    def _maybe_broadcast_scalar(
        self,
        py: PyCtx,
        arg: IRArgNumber,
        expected_arg_event: tuple[int, ...],
        arg_position: int,
        meta: FamilyMeta,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        target_shape = self._broadcast_target_for(
            arg_position, meta, expected_arg_event
        )
        if target_shape is None:
            return number_literal(py, arg.value)
        return self._broadcast(py, arg, target_shape, input_specs, bindings)

    def _maybe_broadcast_ref(
        self,
        py: PyCtx,
        arg: IRArgRef,
        expected_arg_event: tuple[int, ...],
        arg_position: int,
        meta: FamilyMeta,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        target_shape = self._broadcast_target_for(
            arg_position, meta, expected_arg_event
        )
        if target_shape is None:
            return self._ref_expr(py, arg)
        # Broadcast only when the referenced value is scalar (an
        # exogenous Real / Nat input or a previously-bound scalar
        # binding). Tensor-shaped refs flow through unchanged.
        if not self._is_scalar_ref(arg, input_specs, bindings):
            return self._ref_expr(py, arg)
        return self._broadcast(py, arg, target_shape, input_specs, bindings)

    def _ref_expr(self, py: PyCtx, arg: IRArgRef) -> str:
        """Build a Python expression for an `IRArgRef`, threading
        bracket indices through nested ``subscript`` vertices."""
        base = identifier(py, arg.name)
        current = base
        for idx in arg.indices:
            sub = py.v(py.fresh("subs"), "subscript")
            py.e(sub, current, "value")
            py.e(sub, self._render_index(py, idx), "subscript")
            current = sub
        return current

    def _render_index(self, py: PyCtx, idx: IRArg) -> str:
        """Render an index expression: refs become identifiers,
        nested refs preserve their own bracket chain, numbers become
        integer literals."""
        if isinstance(idx, IRArgNumber):
            return number_literal(py, idx.value)
        if isinstance(idx, IRArgRef):
            return self._ref_expr(py, idx)
        raise UnsupportedConstruct(
            _BACKEND_KEY, [f"index:{type(idx).__name__}"]
        )

    def _render_list(
        self,
        py: PyCtx,
        arg: IRArgList,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Render ``[e0, e1, ...]`` as a Python list (TF accepts it)."""
        lst = py.v(py.fresh("list"), "list")
        for el in arg.elements:
            py.e(lst, self._render_arg_atom(py, el, input_specs, bindings),
                 "child_of")
        return lst

    def _render_matrix(
        self,
        py: PyCtx,
        arg: IRArgMatrix,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Render ``[[...], [...]]`` as nested Python lists."""
        outer = py.v(py.fresh("list"), "list")
        for row in arg.rows:
            py.e(outer, self._render_list(py, row, input_specs, bindings),
                 "child_of")
        return outer

    def _render_arg_atom(
        self,
        py: PyCtx,
        el: IRArg,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Render one element inside an `IRArgList` (no broadcast
        lifting; lists carry concrete element expressions)."""
        if isinstance(el, IRArgNumber):
            return number_literal(py, el.value)
        if isinstance(el, IRArgRef):
            return self._ref_expr(py, el)
        if isinstance(el, IRArgList):
            return self._render_list(py, el, input_specs, bindings)
        if isinstance(el, IRArgMatrix):
            return self._render_matrix(py, el, input_specs, bindings)
        if isinstance(el, IRArgBroadcast):
            return self._broadcast(
                py, el.value, el.target_shape, input_specs, bindings
            )
        raise UnsupportedConstruct(
            _BACKEND_KEY, [f"arg-atom:{type(el).__name__}"]
        )

    def _render_family_ref(
        self,
        py: PyCtx,
        arg: IRArgFamilyRef,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Render a wrapped-distribution reference as an identifier.

        Wrapper families (`Truncated`, `Mixture`, ...) reference a
        morphism by name; the morphism's declaration is consumed by the
        wrapper's call shape. Without the morphism table in this
        renderer's `ctx` (renderers consume the IR, not the source
        Module), the safe rendering is the wrapped morphism's name as
        an identifier; downstream wrapper handling is per-family
        future work tracked in the spec's §10.10 wrapper rendering
        clause.
        """
        del input_specs, bindings
        return identifier(py, arg.name)

    def _broadcast(
        self,
        py: PyCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> str:
        """Emit ``tf.fill([<shape>], <value>)``.

        1D and 2D targets are both supported; the shape literal is a
        Python list. Higher ranks raise
        [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
        because tree-sitter Python's list shape would render
        identically but TF semantics for rank-3 broadcasts vary by
        family.
        """
        if len(target_shape) not in (1, 2):
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [f"broadcast:rank-{len(target_shape)}"],
            )
        shape_list = py.v(py.fresh("list"), "list")
        for k in target_shape:
            py.e(shape_list, number_literal(py, float(k)), "child_of")
        if isinstance(value, IRArgNumber):
            value_vid = number_literal(py, value.value)
        elif isinstance(value, IRArgRef):
            value_vid = self._ref_expr(py, value)
        else:
            value_vid = self._render_arg_atom(
                py, value, input_specs, bindings
            )
        return call(
            py,
            attribute(py, ("tf", "fill")),
            positional=(shape_list, value_vid),
        )

    # ------------------------------------------------------------------
    # Sample-shape and scalar-detection helpers
    # ------------------------------------------------------------------

    def _sample_shape(
        self,
        py: PyCtx,
        plate: Plate,
        family: str,
        *,
        carried: tuple[str, ...] | None,
    ) -> str | None:
        """Build the ``sample_shape=[D0, D1, ...]`` keyword payload
        for an Edward2 RV constructor.

        Combines ``plate.batch_dims`` (iid axes) with the residual
        ``plate.event_dims`` that exceed the family's natural
        [`FamilyMeta.event_rank`][quivers.transpile.family_meta.FamilyMeta].
        A scalar family (`event_rank=0`) folds every plate event dim
        into the sample shape, so `Normal(0, 1) [over=LatentDim,
        iid_over=Item]` renders `sample_shape=[Item, LatentDim]`; a
        vector family (`event_rank=1`) folds only the dims beyond its
        natural axis; a matrix family (`event_rank=2`) only those
        beyond the last two.

        A TFP random variable's value shape is
        ``sample_shape + batch_shape + event_shape``, and the
        ``batch_shape`` is whatever the constructor's arguments
        already broadcast to. `carried` names the trailing plate axes
        the arguments supply on their own, as returned by
        [`_carried_batch_keys`][]; those axes must be dropped from the
        payload or the RV is replicated once per plate index and every
        index scores the whole plate. `Normal(loc=h_mean, scale=s)`
        with `h_mean` shaped by the Step axis therefore renders with
        no ``sample_shape`` at all, while `Normal(0, 1) over Step`
        keeps ``sample_shape=[Step]``.

        Returns ``None`` when nothing is left to declare (Edward2 omits
        the keyword in that case). Raises when the arguments carry axes
        the plate does not account for, or when their shape cannot be
        determined: emitting either the padded or the stripped payload
        would silently misscore the site.
        """
        meta = FAMILY_META.get(family)
        natural = meta.event_rank if meta is not None else 0
        residual_event = (
            plate.event_dims[: len(plate.event_dims) - natural]
            if natural else plate.event_dims
        )
        dims = (*plate.batch_dims, *residual_event)
        if not dims:
            return None
        if carried is None:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [
                    f"family:{family}: plate axes over an argument whose "
                    "broadcast shape is not statically determinable"
                ],
            )
        split = len(dims) - len(carried)
        if split < 0 or tuple(_dim_key(d) for d in dims[split:]) != carried:
            raise UnsupportedConstruct(
                _BACKEND_KEY,
                [
                    f"family:{family}: argument batch axes {carried} are "
                    "not a trailing run of the plate axes "
                    f"{tuple(_dim_key(d) for d in dims)}"
                ],
            )
        dims = dims[:split]
        if not dims:
            return None
        lst = py.v(py.fresh("list"), "list")
        for dim in dims:
            py.e(lst, _dim_expr(py, dim), "child_of")
        return lst

    def _carried_batch_keys(
        self,
        args: tuple[IRArg, ...],
        meta: FamilyMeta,
        bindings: dict[str, _Binding],
    ) -> tuple[str, ...] | None:
        """Return the batch axes the rendered arguments already
        broadcast to, or ``None`` when any argument's shape is not
        statically determinable.

        TFP broadcasts the constructor's arguments against one
        another, so the distribution's batch shape is the widest of
        the per-argument batch shapes; the widest tuple is returned.
        """
        carried: tuple[str, ...] = ()
        for position, arg in enumerate(args):
            keys = self._arg_batch_keys(
                arg,
                arg_event_rank=_arg_event_rank(meta, position),
                bindings=bindings,
            )
            if keys is None:
                return None
            if len(keys) > len(carried):
                carried = keys
        return carried

    def _arg_batch_keys(
        self,
        arg: IRArg,
        *,
        arg_event_rank: int,
        bindings: dict[str, _Binding],
    ) -> tuple[str, ...] | None:
        """Return the leading axes the rendered `arg` carries beyond
        the `arg_event_rank` axes its constructor slot consumes, keyed
        by [`_dim_key`][], or ``None`` when the shape is not
        statically determinable.

        Numeric literals are scalars. A `tf.fill` broadcast targets
        exactly the slot's event shape, so it contributes no batch
        axis. A list / matrix literal contributes its own extents. A
        bare reference contributes the axes of the plate its binding
        was declared under; a bracket-indexed reference lowers to
        ``tf.gather`` or a subscript whose result shape the IR does
        not pin down.
        """
        if isinstance(arg, IRArgNumber):
            return ()
        if isinstance(arg, IRArgBroadcast | IRArgFamilyRef | IRArgKernel):
            return ()
        if isinstance(arg, IRArgMatrix):
            columns = len(arg.rows[0].elements) if arg.rows else 0
            extents = (_static_key(len(arg.rows)), _static_key(columns))
            return _drop_event_axes(extents, arg_event_rank)
        if isinstance(arg, IRArgList):
            return _drop_event_axes(
                (_static_key(len(arg.elements)),), arg_event_rank
            )
        if isinstance(arg, IRArgRef):
            if arg.indices:
                return None
            binding = bindings.get(arg.name)
            if binding is None:
                return None
            extents = tuple(
                _dim_key(dim)
                for dim in (
                    *binding.plate.batch_dims,
                    *binding.plate.event_dims,
                )
            )
            return _drop_event_axes(extents, arg_event_rank)
        return None

    def _broadcast_target_for(
        self,
        arg_position: int,
        meta: FamilyMeta,
        plate_event: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        """Return the broadcast target shape for the arg at
        `arg_position` of `meta`, or ``None`` when the arg position
        does not require a broadcast.

        Reads `meta.distribution_class.arg_constraints` to detect
        rank-`n>=1` independent constraints, simplex constraints,
        positive-definite matrices, and the plain real-vector /
        real-matrix forms; combines them with the plate's
        `event_dims` to compute the target shape.
        """
        cls_attr = meta.distribution_class.arg_constraints
        if not isinstance(cls_attr, dict):
            return None
        items = list(cls_attr.items())
        if arg_position >= len(items):
            return None
        _, constraint = items[arg_position]
        if is_real_simplex(constraint):
            return plate_event[-1:] if plate_event else None
        if is_real_vector(constraint):
            return plate_event[-1:] if plate_event else None
        if is_real_cov_matrix(constraint) or is_real_matrix(constraint):
            if len(plate_event) >= 2:
                return plate_event[-2:]
            if len(plate_event) == 1:
                return (plate_event[0], plate_event[0])
            return None
        # `IndependentConstraint(base, n>=1)` not covered above.
        rank = int(getattr(constraint, "event_dim", 0))
        if rank == 0:
            return None
        if rank == 1 and plate_event:
            return plate_event[-1:]
        if rank == 2 and len(plate_event) >= 2:
            return plate_event[-2:]
        return None

    def _is_scalar_ref(
        self,
        arg: IRArgRef,
        input_specs: dict[str, ConstraintSpec],
        bindings: dict[str, _Binding],
    ) -> bool:
        """True iff the referenced name carries a scalar constraint
        with no bracket indexing peeling already applied.

        Exogenous inputs typed `Real` / `Nat` are scalar; sample /
        observe / deterministic bindings inherit their family's
        event_dims plus any batch_dims from the surrounding plate.
        """
        if arg.indices:
            return False
        spec = input_specs.get(arg.name)
        if spec is not None and _is_scalar_constraint(spec):
            binding = bindings.get(arg.name)
            if binding is None:
                return True
            return _is_scalar_plate(binding.plate)
        binding = bindings.get(arg.name)
        if binding is None:
            return False
        return _is_scalar_constraint(binding.constraint) and _is_scalar_plate(
            binding.plate
        )

    def _lookup_meta(self, family: str) -> FamilyMeta:
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                _BACKEND_KEY, [f"family:{family}: not in FAMILY_META"]
            )
        return meta


# ----------------------------------------------------------------------
# Small structural helpers (no per-family branching).
# ----------------------------------------------------------------------


class _Binding:
    """One previously-bound name's constraint + plate, used by the
    scalar-ref test for broadcast lifting decisions."""

    __slots__ = ("constraint", "plate")

    def __init__(self, constraint: ConstraintSpec, plate: Plate) -> None:
        self.constraint = constraint
        self.plate = plate


def _event_shape(plate: Plate) -> tuple[int, ...]:
    """Return the integer event shape implied by `plate.event_dims`.

    Dynamic dims contribute `0` (a placeholder; the renderer never
    embeds it into the source: it lives only as a marker that the dim
    is dynamic). The `_broadcast_target_for` caller treats `0`-valued
    entries by falling back to whatever the source plate already
    encodes.
    """
    out: list[int] = []
    for dim in plate.event_dims:
        if isinstance(dim, DimStatic):
            out.append(dim.size)
        else:
            out.append(0)
    return tuple(out)


def _true_literal(py: PyCtx) -> str:
    """Emit a Python ``True`` literal vertex."""
    vid = py.v(py.fresh("true"), "true")
    py.literal(vid, "True")
    return vid


def _linop_cholesky(py: PyCtx, cov_expr: str) -> str:
    """Wrap a covariance expression ``C`` as the ``LinearOperator``
    ``tf.linalg.LinearOperatorLowerTriangular(tf.linalg.cholesky(C))``,
    the Cholesky-factor scale TFP's ``MatrixNormalLinearOperator``
    expects for ``scale_row`` / ``scale_column``.
    """
    chol = call(
        py,
        attribute(py, ("tf", "linalg", "cholesky")),
        positional=(cov_expr,),
    )
    return call(
        py,
        attribute(py, ("tf", "linalg", "LinearOperatorLowerTriangular")),
        positional=(chol,),
    )


def _dim_key(dim: Dim) -> str:
    """Structural key for a plate dim's extent.

    Two dims share a key exactly when they denote the same axis
    length: a static cardinality matches on its size, a runtime length
    on the name of the input carrying it.
    """
    if isinstance(dim, DimStatic):
        return _static_key(dim.size)
    if isinstance(dim, DimDynamic):
        return f"dynamic:{dim.size_name}"
    raise UnsupportedConstruct(
        _BACKEND_KEY, [f"dim:{type(dim).__name__}"]
    )


def _static_key(size: int) -> str:
    """Key for a statically known extent, matching `_dim_key`."""
    return f"static:{size}"


def _drop_event_axes(
    extents: tuple[str, ...], event_rank: int
) -> tuple[str, ...]:
    """Strip the trailing `event_rank` axes from `extents`, leaving the
    batch axes the argument broadcasts over."""
    return extents[: max(0, len(extents) - event_rank)]


def _arg_event_rank(meta: FamilyMeta, arg_position: int) -> int:
    """Return the number of event axes the family's constructor slot at
    `arg_position` consumes, read off the torch distribution's
    ``arg_constraints``.

    A simplex or real-vector slot consumes one axis, a covariance or
    real-matrix slot two, an ``IndependentConstraint(base, n)`` slot
    ``n``; every other constraint is scalar-valued and consumes none.
    """
    cls_attr = meta.distribution_class.arg_constraints
    if not isinstance(cls_attr, dict):
        return 0
    items = list(cls_attr.items())
    if arg_position >= len(items):
        return 0
    _, constraint = items[arg_position]
    if is_real_simplex(constraint) or is_real_vector(constraint):
        return 1
    if is_real_cov_matrix(constraint) or is_real_matrix(constraint):
        return 2
    return int(getattr(constraint, "event_dim", 0))


def _dim_expr(py: PyCtx, dim: Dim) -> str:
    """Render a single plate dim into the expression that should sit
    inside ``sample_shape=[...]``.

    `DimStatic` becomes an integer literal; `DimDynamic` becomes the
    runtime size identifier (``N_Doc`` etc.).
    """
    if isinstance(dim, DimStatic):
        return number_literal(py, float(dim.size))
    if isinstance(dim, DimDynamic):
        return identifier(py, dim.size_name)
    raise UnsupportedConstruct(
        _BACKEND_KEY, [f"dim:{type(dim).__name__}"]
    )


_SCALAR_CONSTRAINT_CLASSES: tuple[type, ...] = (
    CSReal,
    CSPositive,
    CSUnitInterval,
    CSInterval,
    CSBoolean,
    CSIntegerInterval,
    CSNonnegativeInteger,
    CSPositiveInteger,
)


def _is_scalar_constraint(spec: ConstraintSpec) -> bool:
    """True iff `spec` denotes a 0-rank support.

    Scalar inputs (`CSReal`, the `Real`-typed program parameters) are
    eligible for broadcast lifting; vector / matrix constraints are
    already shaped and flow through unchanged.
    """
    return isinstance(spec, _SCALAR_CONSTRAINT_CLASSES)


def _is_scalar_plate(plate: Plate) -> bool:
    """True iff the plate has neither event nor batch dims."""
    return not plate.event_dims and not plate.batch_dims


def _is_positional_arg(meta: FamilyMeta, target_name: str) -> bool:
    """True iff the given arg should ride positionally in the
    ``edward2.<Family>(...)`` call.

    The rule: distributions with exactly one declared arg
    (`Dirichlet(concentration)`, `Exponential(rate)`, ...) render
    that arg positionally; distributions whose torch
    ``arg_constraints`` advertises multiple parameterisations
    (`Categorical(probs, logits)`, `Bernoulli(probs, logits)`, ...)
    render every arg by keyword so the caller doesn't accidentally
    bind to ``logits=`` when TFP's positional signature differs from
    torch's. ``target_name`` is unused today: the predicate is
    structural over the family's arg_constraints shape.
    """
    del target_name
    cls_attr = meta.distribution_class.arg_constraints
    if not isinstance(cls_attr, dict):
        return False
    return len(cls_attr) == 1


def _thread_via_through_scope(
    nodes: tuple[IRNode, ...], *, latent_name: str
) -> tuple[IRNode, ...]:
    """Rewrite references to `latent_name` inside scope-observe args
    so they are indexed by each observe's `via=` fibration.

    The lifted latent in an `IRMarginalize` rewrite ranges over the
    marginalize batch dim (e.g. ``Doc``); each scope-observe ranges
    over its own batch dim (e.g. ``Word_obs``). The `via=`
    fibration maps observation indices to the marginalize plate, so
    a reference to the latent inside the observe's args becomes
    ``z[word_idx]`` -- a single indexed lookup per observation.

    Nodes outside the observe surface (the lifted latent sample
    itself, deterministic / score nodes) are returned unchanged.
    """
    out: list[IRNode] = []
    for node in nodes:
        if isinstance(node, IRObserve) and node.via is not None:
            out.append(_rewrite_observe_via(node, latent_name))
        else:
            out.append(node)
    return tuple(out)


def _rewrite_observe_via(
    node: IRObserve, latent_name: str
) -> IRObserve:
    """Apply via indexing to every `latent_name` reference in the
    observe's args."""
    via = node.via
    if via is None:
        return node
    new_args = tuple(
        _index_latent_ref(a, latent_name, via) for a in node.args
    )
    return IRObserve(
        name=node.name,
        family=node.family,
        args=new_args,
        arg_names=node.arg_names,
        constraint=node.constraint,
        plate=node.plate,
        via=node.via,
    )


def _index_latent_ref(arg: IRArg, latent_name: str, via: str) -> IRArg:
    """Recursively rewrite `IRArgRef(latent_name)` into
    `IRArgRef(latent_name, indices=(IRArgRef(via),))`.
    """
    if isinstance(arg, IRArgRef):
        if arg.name == latent_name and not arg.indices:
            return IRArgRef(
                name=latent_name,
                indices=(IRArgRef(name=via),),
            )
        new_indices = tuple(
            _index_latent_ref(i, latent_name, via) for i in arg.indices
        )
        return IRArgRef(name=arg.name, indices=new_indices)
    if isinstance(arg, IRArgBroadcast):
        return IRArgBroadcast(
            value=_index_latent_ref(arg.value, latent_name, via),
            target_shape=arg.target_shape,
        )
    if isinstance(arg, IRArgList):
        return IRArgList(
            elements=tuple(
                _index_latent_ref(e, latent_name, via) for e in arg.elements
            )
        )
    if isinstance(arg, IRArgMatrix):
        return IRArgMatrix(
            rows=tuple(
                IRArgList(
                    elements=tuple(
                        _index_latent_ref(e, latent_name, via)
                        for e in row.elements
                    )
                )
                for row in arg.rows
            )
        )
    return arg


__all__ = ["Edward2Renderer"]
