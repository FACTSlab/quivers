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

import pathlib

import panproto

from quivers.dsl.ast_nodes import MorphismInitFamily
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import parser_registry, target_protocol
from quivers.transpile.renderers._python_helpers import (
    PyCtx,
    assignment,
    attribute,
    call,
    identifier,
    number_literal,
    python_binary_op as _python_binary_op,
    python_method_call as _python_method_call,
    python_paren as _python_paren,
    python_unary_minus as _python_unary_minus,
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
    StructuredDataArg,
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
        ctx = _RenderCtx(sb=sb, morphisms={}, defines={})
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
        # Pyro lacks several QVR families as built-ins (`TruncatedNormal`,
        # `LogitNormal`, `HalfStudentT`, `MatrixNormal`); their runtime
        # helpers live at
        # [`quivers.transpile.runtime_pyro`][quivers.transpile.runtime_pyro].
        # When the IR samples or observes from such a family, graft the
        # parsed helper `class` subtree onto the module above `model`
        # so a reader sees the helper classes first (the natural Python
        # idiom: define classes before consumers).
        for helper_name in sorted(_ir_helper_classes_used(ir.body)):
            _emit_runtime_helper(pctx, helper_name)
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
            self._emit_score_pyro(pctx, node)
            return
        if isinstance(node, IRMarginalize):
            self.marginalize(ctx, node, pctx=pctx)
            return
        if isinstance(node, IRReturn):
            self._emit_return_pyro(pctx, node.names)
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

    def _emit_gp_block(
        self,
        pctx: _PyroCtx,
        *,
        name: str,
        args: tuple[IRArg, ...],
        observed: bool,
    ) -> None:
        """Emit a Gaussian-process sample as a triple of Python
        statements: ``__gp_mean_<name> = torch.zeros(N)``,
        ``__gp_cov_<name> = torch.exp(...) + jitter * torch.eye(N)``,
        ``<name> = pyro.sample("<name>", pyro.distributions.MultivariateNormal(
            loc=__gp_mean_<name>, covariance_matrix=__gp_cov_<name>))``.

        Mirrors the NumPyro emission with `torch` substituted for
        `jnp`. The kernel-cov expression is built using the Python
        helper API plus explicit
        [`parenthesized_expression`][quivers.transpile.renderers._python_helpers.python_paren]
        wrappers so operator precedence around the squared diff and
        the squared length scale survives the printer's drop-paren
        default.
        """
        del observed  # GP samples are never observed in QVR.
        if len(args) != 2 or not isinstance(args[1], IRArgKernel):
            raise UnsupportedConstruct(
                "qvr-pyro",
                ["family:GP:expected IRArgKernel as second arg"],
            )
        kernel_arg = args[1]
        if kernel_arg.kernel != "rbf":
            raise UnsupportedConstruct(
                "qvr-pyro",
                [
                    f"family:GP:kernel:{kernel_arg.kernel}: only rbf "
                    f"is implemented"
                ],
            )
        n = kernel_arg.grid_size
        ls = kernel_arg.length_scale
        jitter = kernel_arg.jitter
        x = kernel_arg.x_name
        mean_name = f"__gp_mean_{name}"
        cov_name = f"__gp_cov_{name}"
        # __gp_mean_<name> = torch.zeros(N)
        mean_rhs = call(
            pctx,
            attribute(pctx, ("torch", "zeros")),
            positional=(number_literal(pctx, n),),
        )
        pctx.e(
            pctx.body,
            assignment(pctx, lhs_name=mean_name, rhs=mean_rhs),
            "child_of",
        )
        # __gp_cov_<name> = torch.exp(-0.5 * (diff * diff) / (ls * ls))
        #                    + jitter * torch.eye(N)
        x_col = _python_method_call(
            pctx, identifier(pctx, x), "reshape",
            (
                _python_unary_minus(pctx, number_literal(pctx, 1)),
                number_literal(pctx, 1),
            ),
        )
        x_row = _python_method_call(
            pctx, identifier(pctx, x), "reshape",
            (
                number_literal(pctx, 1),
                _python_unary_minus(pctx, number_literal(pctx, 1)),
            ),
        )
        diff = _python_paren(
            pctx,
            _python_binary_op(pctx, "-", x_col, x_row),
        )
        diff_sq = _python_paren(
            pctx,
            _python_binary_op(pctx, "*", diff, diff),
        )
        ls_sq = _python_paren(
            pctx,
            _python_binary_op(
                pctx, "*",
                number_literal(pctx, ls),
                number_literal(pctx, ls),
            ),
        )
        quotient = _python_binary_op(pctx, "/", diff_sq, ls_sq)
        neg_half = _python_unary_minus(pctx, number_literal(pctx, 0.5))
        exponent = _python_binary_op(pctx, "*", neg_half, quotient)
        kernel_call = call(
            pctx,
            attribute(pctx, ("torch", "exp")),
            positional=(exponent,),
        )
        eye_call = call(
            pctx,
            attribute(pctx, ("torch", "eye")),
            positional=(number_literal(pctx, n),),
        )
        jitter_term = _python_binary_op(
            pctx, "*", number_literal(pctx, jitter), eye_call,
        )
        cov_rhs = _python_binary_op(
            pctx, "+", kernel_call, jitter_term,
        )
        pctx.e(
            pctx.body,
            assignment(pctx, lhs_name=cov_name, rhs=cov_rhs),
            "child_of",
        )
        # <name> = pyro.sample("<name>", pyro.distributions.MultivariateNormal(
        #     loc=__gp_mean_<name>, covariance_matrix=__gp_cov_<name>))
        mvn_call = call(
            pctx,
            attribute(
                pctx,
                ("pyro", "distributions", "MultivariateNormal"),
            ),
            keyword=(
                ("loc", identifier(pctx, mean_name)),
                ("covariance_matrix", identifier(pctx, cov_name)),
            ),
        )
        sample_call = call(
            pctx,
            attribute(pctx, ("pyro", "sample")),
            positional=(string_literal(pctx, name), mvn_call),
        )
        sample_asn = assignment(
            pctx, lhs_name=name, rhs=sample_call,
        )
        pctx.e(pctx.body, sample_asn, "child_of")

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
        if family == "GP":
            self._emit_gp_block(
                pctx, name=name, args=args, observed=observed,
            )
            return
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
        # Build the distribution call. Pyro lacks several QVR families
        # as built-ins (`TruncatedNormal`, `LogitNormal`,
        # `HalfStudentT`, `MatrixNormal`); the renderer grafts a helper
        # class from `quivers.transpile.runtime_pyro` (see
        # [`_emit_runtime_helper`][quivers.transpile.renderers.pyro._emit_runtime_helper])
        # and dispatches the call to that bare identifier rather than
        # to `pyro.distributions.<name>`.
        if dist_class in _RUNTIME_PYRO_HELPER_ROOTS:
            dist_callee = identifier(pctx, dist_class)
        else:
            dist_callee = attribute(
                pctx, ("pyro", "distributions", dist_class)
            )
        dist_args = self._build_dist_args(
            pctx, meta=meta, args=args, arg_names=arg_names,
            aliases=aliases, plate=plate,
        )
        positional = dist_args.positional
        # The Cholesky-LKJ families take the correlation-matrix
        # dimension as a mandatory leading positional argument
        # (`LKJCorrCholesky(d, eta)`, `LKJ(d, eta)`); it is carried on
        # the sample's event axis, not in the QVR init clause.
        if family in _PYRO_LEADING_DIM_FAMILIES:
            positional = (
                self._leading_dim_arg(pctx, family, plate),
                *positional,
            )
        dist_call = call(
            pctx,
            dist_callee,
            positional=positional,
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
            torch_constraints = getattr(
                meta.distribution_class, "arg_constraints", None
            )
            if isinstance(torch_constraints, dict):
                arg_names = tuple(torch_constraints.keys())
            else:
                arg_names = ()
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

    def _leading_dim_arg(
        self,
        pctx: _PyroCtx,
        family: str,
        plate: Plate | None,
    ) -> str:
        """Emit the correlation-matrix dimension for a Cholesky-LKJ
        family, read off the sample's first event axis.

        Pyro's `LKJCorrCholesky` / `LKJ` require the matrix dimension
        `d` as the leading positional argument; the QVR init clause
        supplies only the concentration, so the dimension is recovered
        from the sample site's event axis (e.g. `sample chol : Dim`
        with `Dim : FinSet 4` yields `4`).
        """
        if plate is None or not plate.event_dims:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family:{family}:missing-dimension-axis"],
            )
        dim = plate.event_dims[0]
        if not isinstance(dim, DimStatic):
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family:{family}:dynamic-dimension-axis"],
            )
        return number_literal(pctx, float(dim.size))

    def _family_meta(self, family: str) -> FamilyMeta:
        meta = FAMILY_META.get(family)
        if meta is None:
            raise UnsupportedConstruct(
                f"qvr-{_TARGET}",
                [f"family:{family}"],
            )
        return meta

    # ----- score / return -----

    def _emit_score_pyro(
        self, pctx: _PyroCtx, node: IRScore
    ) -> None:
        """`<name> = <expr>; pyro.factor("<name>", <name>)`.

        Pyro-local emitter; never goes through the
        [`RendererBase`][quivers.transpile.renderers._base.RendererBase]
        dispatch (the Pyro renderer routes through
        [`_dispatch_pyro_node`][quivers.transpile.renderers.pyro.PyroRenderer._dispatch_pyro_node]
        instead, with its own
        [`_PyroCtx`][quivers.transpile.renderers.pyro._PyroCtx]
        rather than the base `_RenderCtx`).
        """
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

    def _emit_return_pyro(
        self, pctx: _PyroCtx, names: tuple[str, ...]
    ) -> None:
        """Emit `return <var>` / `return <a>, <b>, ...`.

        Pyro-local emitter; see
        [`_emit_score_pyro`][quivers.transpile.renderers.pyro.PyroRenderer._emit_score_pyro]
        for why this does not override the base method.
        """
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
        super().__init__(sb, cards=cards, target="pyro")
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
    slot is a rank-`n` array, return the target shape derived from
    `plate.event_dims`; otherwise `None`.

    When the family declares a
    [`StructuredSampleLowering`][quivers.transpile.ir.StructuredSampleLowering],
    the per-argument shape is read off the matching
    [`StructuredDataArg`][quivers.transpile.ir.StructuredDataArg]'s
    `axis_indices` so a Kronecker-factored family (e.g. `MatrixNormal`)
    fills its row covariance from event axis 0 (`(0, 0)`) and its
    column covariance from event axis 1 (`(1, 1)`), rather than
    filling every covariance to the full `loc` shape. Absent a
    structured lowering, the target shape falls back to the leading
    `event_dim` axes implied by the torch arg constraint (the
    `Dirichlet` idiom of `torch.full((K,), alpha)`).
    """
    if not isinstance(arg, (IRArgRef, IRArgNumber)):
        return None
    if isinstance(arg, IRArgRef) and arg.indices:
        return None
    axis_indices = _structured_axis_indices(meta, arg_name)
    if axis_indices is not None:
        if not axis_indices:
            return None
        return _shape_from_axis_indices(axis_indices, plate)
    constraints = getattr(meta.distribution_class, "arg_constraints", {})
    expected = constraints.get(arg_name) if isinstance(
        constraints, dict
    ) else None
    if expected is None:
        return None
    event_dim = int(getattr(expected, "event_dim", 0))
    if event_dim < 1:
        return None
    return _shape_from_axis_indices(tuple(range(event_dim)), plate)


def _structured_axis_indices(
    meta: FamilyMeta, arg_name: str
) -> tuple[int, ...] | None:
    """The `axis_indices` of the family's structured-lowering
    [`StructuredDataArg`][quivers.transpile.ir.StructuredDataArg] whose
    `arg_name` matches, or `None` when the family declares no
    structured lowering (or no data arg of that name)."""
    lowering = meta.structured_lowering
    if lowering is None:
        return None
    for spec in lowering.args:
        if isinstance(spec, StructuredDataArg) and spec.arg_name == arg_name:
            return spec.axis_indices
    return None


def _shape_from_axis_indices(
    axis_indices: tuple[int, ...], plate: Plate
) -> tuple[int, ...] | None:
    """Resolve `axis_indices` against `plate.event_dims`, returning the
    concrete static shape or `None` if any indexed axis is out of range
    or dynamic."""
    sizes: list[int] = []
    for i in axis_indices:
        if i >= len(plate.event_dims):
            return None
        dim = plate.event_dims[i]
        if not isinstance(dim, DimStatic):
            return None
        sizes.append(dim.size)
    return tuple(sizes)


#: The Cholesky-LKJ families whose Pyro distribution class takes the
#: correlation-matrix dimension `d` as a mandatory leading positional
#: argument (`LKJCorrCholesky(d, eta)`, `LKJ(d, eta)`). The QVR init
#: clause carries only the concentration, so the renderer prepends the
#: dimension read off the sample's first event axis.
_PYRO_LEADING_DIM_FAMILIES = frozenset({
    "LKJCholesky",
    "LKJCorrelationFactor",
})


_RUNTIME_PYRO_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "runtime_pyro.py"
)


def _class_definition_name(
    schema: panproto.Schema, class_vid: str
) -> str | None:
    """The literal name of a `class_definition` vertex, or `None`."""
    for edge in schema.edges:
        if edge.src == class_vid and edge.kind == "name":
            name_v = next(
                (vv for vv in schema.vertices if vv.id == edge.tgt),
                None,
            )
            if name_v is None:
                continue
            return next(
                (
                    c.value
                    for c in schema.constraints_for(name_v.id)
                    if c.sort == "literal-value"
                ),
                None,
            )
    return None


def _subtree_vertex_ids(
    schema: panproto.Schema, root: str
) -> set[str]:
    """Return every vertex id reachable from `root` via outgoing
    edges of `schema`."""
    seen: set[str] = {root}
    frontier: list[str] = [root]
    while frontier:
        src = frontier.pop()
        for edge in schema.edges:
            if edge.src == src and edge.tgt not in seen:
                seen.add(edge.tgt)
                frontier.append(edge.tgt)
    return seen


def _load_runtime_pyro_helpers() -> tuple[
    panproto.Schema, dict[str, str], dict[str, set[str]]
]:
    """Parse [`runtime_pyro.py`][quivers.transpile.runtime_pyro]
    through panproto's Python tree-sitter grammar at module-load time
    and index every helper `class_definition` subtree by class name.

    Returns the parsed schema, a map from class name to the
    class-definition vertex id, and a map from class name to the set
    of vertex ids in that class's subtree. The renderer's
    [`_emit_runtime_helper`][quivers.transpile.renderers.pyro._emit_runtime_helper]
    grafts a class subtree (vertex + all descendants + their
    constraints + edges) into the per-render schema as a `child_of`
    of the emitted module. The emit is structurally a real Python
    class definition, with no string literal, no `exec`, and no
    runtime self-injection.
    """
    schema = parser_registry().parse_with_protocol(
        "python",
        _RUNTIME_PYRO_PATH.read_bytes(),
        str(_RUNTIME_PYRO_PATH),
    )
    roots: dict[str, str] = {}
    for v in schema.vertices:
        if v.kind == "class_definition":
            name = _class_definition_name(schema, v.id)
            if name is not None:
                roots[name] = v.id
    if not roots:
        raise RuntimeError(
            f"no helper `class` definitions found in {_RUNTIME_PYRO_PATH}; "
            "the renderer expects it as the source of truth for the "
            "embedded Pyro runtime helpers."
        )
    subtrees = {
        name: _subtree_vertex_ids(schema, root)
        for name, root in roots.items()
    }
    return schema, roots, subtrees


(
    _RUNTIME_PYRO_HELPER_SCHEMA,
    _RUNTIME_PYRO_HELPER_ROOTS,
    _RUNTIME_PYRO_HELPER_SUBTREES,
) = _load_runtime_pyro_helpers()


def _ir_helper_classes_used(body: tuple[IRNode, ...]) -> set[str]:
    """The set of runtime-helper class names the IR body needs grafted.

    For each [`IRSample`][quivers.transpile.ir.IRSample] /
    [`IRObserve`][quivers.transpile.ir.IRObserve] (including nested
    [`IRMarginalize`][quivers.transpile.ir.IRMarginalize] scopes),
    a helper is needed when the family's Pyro target name matches a
    `class` defined in
    [`runtime_pyro.py`][quivers.transpile.runtime_pyro]."""
    used: set[str] = set()
    for node in body:
        if isinstance(node, (IRSample, IRObserve)):
            meta = FAMILY_META.get(node.family)
            if meta is not None:
                target = meta.target_names.get(_TARGET)
                if target in _RUNTIME_PYRO_HELPER_ROOTS:
                    used.add(target)
        elif isinstance(node, IRMarginalize):
            used |= _ir_helper_classes_used(node.scope)
    return used


def _emit_runtime_helper(pctx: _PyroCtx, class_name: str) -> None:
    """Graft the named helper `class` subtree from
    [`runtime_pyro.py`][quivers.transpile.runtime_pyro] into the
    per-render schema as a top-level child of `mod`.

    The class definition is a real
    [`class_definition`][panproto.schema.class_definition] panproto
    subtree (parsed once at module load via panproto's Python
    tree-sitter grammar). The renderer copies every vertex, every
    constraint, and every internal edge of the subtree into the
    per-render `SchemaBuilder`, then attaches the class root as a
    `child_of` of `mod`. Subsequent call sites (e.g.
    ``TruncatedNormal(loc, scale, low, high)`` or
    ``MatrixNormal(loc, row_covariance, col_covariance)``) in the
    rendered model body resolve to that class through normal Python
    name lookup. Vertex ids are rewritten through `pctx.fresh` so the
    graft does not collide with builder-allocated ids.
    """
    src_schema = _RUNTIME_PYRO_HELPER_SCHEMA
    subtree = _RUNTIME_PYRO_HELPER_SUBTREES[class_name]
    root = _RUNTIME_PYRO_HELPER_ROOTS[class_name]
    id_map: dict[str, str] = {}

    for old in subtree:
        new = pctx.fresh("rh")
        id_map[old] = new
        kind = next(
            v.kind for v in src_schema.vertices if v.id == old
        )
        pctx.v(new, kind)
        for cstr in src_schema.constraints_for(old):
            pctx.constraint(new, cstr.sort, cstr.value)
    for edge in src_schema.edges:
        if edge.src in id_map and edge.tgt in id_map:
            pctx.e(id_map[edge.src], id_map[edge.tgt], edge.kind)
    pctx.e("mod", id_map[root], "child_of")


__all__ = ["PyroRenderer"]
