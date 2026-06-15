"""PyMC renderer: lower the transpile IR to a PyMC `build_model` function.

PyMC declarations are constructor calls inside a
`with pymc.Model(coords={...}) as model:` block. There is no separate
data block; data inputs become parameters of a wrapping `build_model`
function. Each [`IRSample`][quivers.transpile.ir.IRSample] /
[`IRObserve`][quivers.transpile.ir.IRObserve] becomes one
`pymc.<Family>("name", **args, dims=(...), observed=<obs>)` call.

The dispatch points:

* `declare` is a no-op for PyMC: the declaration is the constructor
  call itself, emitted in `sample`. Data inputs land in the function
  signature; coords carry the per-axis cardinality.
* `sample` emits `name = pymc.<Family>("name", **args, dims=(...),
  observed=<obs>)`. Argument names come from `arg_names`; the per-
  backend `arg_aliases` map renames them
  ([`FamilyMeta.arg_aliases`][quivers.transpile.family_meta.FamilyMeta]).
* `marginalize` lowers to explicit `IRSample(latent)` plus the scope
  body inline (PyMC supports discrete latents under MCMC and via
  the marginalisation primitives).
* `broadcast` emits `np.full((K,), x)` for 1D, `np.full((R, C), x)`
  for 2D target shapes.

`render_list` / `render_matrix` emit `np.array([...])` /
`np.array([[...], [...]])` for list and matrix literal args.

`IRArgFamilyRef` rendering handles wrappers: `Truncated` emits
`pymc.<base>.dist(...)`; `Mixture` emits `pymc.<base>.dist(...)` inside
the `comp_dists` keyword. The renderer reads the referenced morphism's
`init_family` clause from `ctx.morphisms`.
"""

from __future__ import annotations

import torch.distributions.constraints as c
import panproto

from quivers.dsl.ast_nodes import (
    Expr,
    Module,
    MorphismDecl,
)
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._pipeline import (
    SchemaTransform,
    realize,
    target_protocol,
)
from quivers.transpile.backends._pyhelpers import (
    PyCtx,
    arg_expr,
    assignment,
    attribute,
    call,
    function_def,
    identifier,
    number_literal,
    string_literal,
    with_statement,
)
from quivers.transpile.backends._letexpr_python import (
    render_let_expr_python,
)
from quivers.transpile.backends._resolve import (
    build_let_table,
    build_morphism_table,
)
from quivers.transpile.family_meta import FAMILY_META, FamilyMeta
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
from quivers.transpile.lower import Lower
from quivers.transpile.renderers._base import (
    BlockKind,
    IRArgTransform,
    RendererBase,
    SchemaFragment,
    _RenderCtx,
    assert_no_dangling_refs,
)


# Per-backend supplement to FAMILY_META's `arg_aliases`. Families whose
# torch arg_constraints carry a name PyMC reparameterises (Categorical's
# `probs` is PyMC's `p`; Multinomial's `total_count` is PyMC's `n`).
# Merged with `meta.arg_aliases["pymc"]` at lookup time; keeps the
# renamings as data rather than per-family dispatch logic.
_PYMC_EXTRA_ALIASES: dict[str, dict[str, str]] = {
    "Categorical": {"probs": "p"},
    "OneHotCategorical": {"probs": "p"},
    "Multinomial": {"probs": "p", "total_count": "n"},
    "Bernoulli": {"probs": "p"},
    "Binomial": {"probs": "p", "total_count": "n"},
    "BetaBinomial": {
        "concentration1": "alpha",
        "concentration0": "beta",
        "total_count": "n",
    },
    "NegativeBinomial": {"probs": "p", "total_count": "n"},
    "Beta": {"concentration1": "alpha", "concentration0": "beta"},
    "Gamma": {"concentration": "alpha", "rate": "beta"},
    "InverseGamma": {"concentration": "alpha", "rate": "beta"},
    "StudentT": {"df": "nu", "loc": "mu", "scale": "sigma"},
    "HalfStudentT": {"df": "nu", "scale": "sigma"},
    "Normal": {"loc": "mu", "scale": "sigma"},
    "LogNormal": {"loc": "mu", "scale": "sigma"},
    "HalfNormal": {"scale": "sigma"},
    "Cauchy": {"loc": "alpha", "scale": "beta"},
    "HalfCauchy": {"scale": "beta"},
    "Laplace": {"loc": "mu", "scale": "b"},
    "LogitNormal": {"loc": "mu", "scale": "sigma"},
    "TruncatedNormal": {"loc": "mu", "scale": "sigma"},
    "Logistic": {"loc": "mu", "scale": "s"},
    "Gumbel": {"loc": "mu", "scale": "beta"},
    "Weibull": {"concentration": "alpha", "scale": "beta"},
    "Exponential": {"rate": "lam"},
    "Pareto": {"alpha": "alpha", "scale": "m"},
    "MultivariateNormal": {"loc": "mu", "covariance_matrix": "cov"},
    "LowRankMVN": {
        "loc": "mu",
        "cov_factor": "W",
        "cov_diag": "diag",
    },
    "Wishart": {"df": "nu", "covariance_matrix": "scale_matrix"},
    "InverseWishart": {"df": "nu", "covariance_matrix": "scale_matrix"},
    "MatrixNormal": {
        "loc": "mu",
        "row_covariance": "rowcov",
        "column_covariance": "colcov",
    },
    "Mixture": {"mixture_distribution": "w", "component_distribution": "comp_dists"},
    "Truncated": {"base_distribution": "dist"},
}


class PyMCRenderer(RendererBase):
    """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
    PyMC `build_model` function under the `python` tree-sitter grammar.

    The output has the shape:

    ```python
    def build_model(<inputs>=None):
        with pymc.Model(coords={...}) as model:
            theta = pymc.Dirichlet("theta", a=..., dims=(...))
            phi = pymc.Dirichlet("phi", a=..., dims=(...))
            pymc.Categorical("w", p=phi[...], observed=w, dims=(...))
        return model
    ```
    """

    target: str = "qvr-pymc"

    # ----- protocol -----

    def target_protocol(self) -> panproto.Protocol:
        return target_protocol("python")

    # ----- top-level render -----

    def render(self, ir: IRProgram) -> panproto.Schema:
        """Render an [`IRProgram`][quivers.transpile.ir.IRProgram] to a
        PyMC schema. Subclasses normally call
        [`render_with_tables`][quivers.transpile.renderers.pymc.PyMCRenderer.render_with_tables]
        when morphism / let tables are available."""
        return self.render_with_tables(ir, morphisms={}, lets={})

    def render_with_tables(
        self,
        ir: IRProgram,
        *,
        morphisms: dict[str, MorphismDecl],
        lets: dict[str, Expr],
    ) -> panproto.Schema:
        """Same as `render` but pre-populates `ctx.morphisms` / `ctx.lets`
        so [`IRArgFamilyRef`][quivers.transpile.ir.IRArgFamilyRef]
        rendering can read the referenced morphism's `init_family`
        clause."""
        assert_no_dangling_refs(ir)
        proto = self.target_protocol()
        sb = proto.schema()
        ctx = _RenderCtx(sb=sb, morphisms=dict(morphisms), lets=dict(lets))
        py = PyCtx(sb)
        bag = _PyMCCtx(ctx=ctx, py=py, ir=ir)

        # `module` root.
        py.v("mod", "module")

        # Function body block.
        fn_body = py.v(py.fresh("fbody"), "block")

        # `with pymc.Model(coords={...}) as model:` header.
        coords_dict = self._build_coords_dict(py, ir)
        model_call = call(
            py,
            attribute(py, ("pymc", "Model")),
            keyword=(("coords", coords_dict),),
        )
        with_body = py.v(py.fresh("wbody"), "block")
        ws = with_statement(
            py, expression=model_call, alias="model", body_vid=with_body
        )
        py.e(fn_body, ws, "child_of")

        # Walk the body, dispatching each node into the with-body block.
        bag.with_body = with_body
        for node in ir.body:
            self._dispatch_pymc(bag, node)

        # Trailing `return model` inside the function body (outside
        # the with-block). PyMC's idiom is for `build_model` to return
        # the constructed model object.
        ret = py.v(py.fresh("ret"), "return_statement")
        py.e(ret, identifier(py, "model"), "child_of")
        py.e(fn_body, ret, "child_of")

        # `def build_model(<data_inputs>=None): <fn_body>`.
        param_names = tuple(inp.name for inp in ir.inputs)
        fn = function_def(
            py, name="build_model",
            default_params=param_names, body_vid=fn_body,
        )
        py.e("mod", fn, "child_of")

        return sb.build()

    # ----- coord dict construction -----

    def _build_coords_dict(self, py: PyCtx, ir: IRProgram) -> str:
        """Build the `coords={"axis": np.arange(N), ...}` dict.

        Walks every plate in the IR (inputs + body), collecting each
        unique [`Dim`][quivers.transpile.ir.Dim] axis name. Static
        dims use `np.arange(<size>)`; dynamic dims use
        `np.arange(<size_name>)` (the size variable arrives via the
        function signature).
        """
        seen: dict[str, Dim] = {}
        for inp in ir.inputs:
            self._collect_dims(inp.plate, seen)
        self._collect_dims_in_body(ir.body, seen)

        d = py.v(py.fresh("dict"), "dictionary")
        for name, dim in seen.items():
            key = string_literal(py, name)
            value = self._np_arange(py, dim)
            pair = py.v(py.fresh("pair"), "pair")
            py.e(pair, key, "key")
            py.e(pair, value, "value")
            py.e(d, pair, "child_of")
        return d

    def _collect_dims(self, plate: Plate, seen: dict[str, Dim]) -> None:
        for dim in (*plate.batch_dims, *plate.event_dims):
            name = _dim_name(dim)
            seen.setdefault(name, dim)

    def _collect_dims_in_body(
        self, body: tuple[IRNode, ...], seen: dict[str, Dim]
    ) -> None:
        for node in body:
            if isinstance(node, (IRSample, IRObserve, IRDeterministic, IRDataInput)):
                self._collect_dims(node.plate, seen)
            elif isinstance(node, IRMarginalize):
                self._collect_dims(node.plate, seen)
                self._collect_dims_in_body(node.scope, seen)

    def _np_arange(self, py: PyCtx, dim: Dim) -> str:
        """Emit `np.arange(<size>)` for a static dim, or
        `np.arange(<size_name>)` for a dynamic dim."""
        callee = attribute(py, ("np", "arange"))
        if isinstance(dim, DimStatic):
            return call(
                py, callee,
                positional=(number_literal(py, float(dim.size)),),
            )
        if isinstance(dim, DimDynamic):
            return call(
                py, callee, positional=(identifier(py, dim.size_name),),
            )
        msg = f"unknown Dim kind: {type(dim).__name__}"
        raise UnsupportedConstruct(self.target, [msg])

    # ----- per-node dispatch (PyMC walks over with-body) ----

    def _dispatch_pymc(
        self, ctx: _PyMCCtx, node: IRNode
    ) -> None:
        if isinstance(node, IRDataInput):
            # Data inputs are function parameters, declared in
            # `function_def`. No additional emit inside the with-body.
            return
        if isinstance(node, IRSample):
            self._emit_sample(ctx, node, observed_name=None)
            return
        if isinstance(node, IRObserve):
            self._emit_sample(ctx, node, observed_name=node.name)
            return
        if isinstance(node, IRDeterministic):
            self._emit_deterministic(ctx, node)
            return
        if isinstance(node, IRScore):
            self._emit_score_step(ctx, node)
            return
        if isinstance(node, IRMarginalize):
            previous = ctx.current_latent
            ctx.current_latent = node.latent
            try:
                for child in self.explicit_latent_scope(node):
                    self._dispatch_pymc(ctx, child)
            finally:
                ctx.current_latent = previous
            return
        if isinstance(node, IRReturn):
            # The function-level `return model` is emitted in `render`
            # after the with-block; per-program return names are not
            # surfaced separately in PyMC's idiom.
            return
        raise UnsupportedConstruct(
            self.target, [f"node:{type(node).__name__}"]
        )

    # ----- declare: no-op for PyMC (declarations are constructor calls) ----

    def declare(
        self,
        ctx: _RenderCtx,
        name: str,
        constraint: ConstraintSpec,
        plate: Plate,
        *,
        block: BlockKind,
    ) -> SchemaFragment:
        """PyMC declarations ARE the constructor calls (emitted in
        `sample`). `declare` is therefore a no-op; the caller's emit
        handles both declaration and assignment in one node."""
        del ctx, name, constraint, plate, block
        return ""

    # ----- sample / observe -----

    def _emit_sample(
        self,
        ctx: _PyMCCtx,
        node: IRSample | IRObserve,
        observed_name: str | None,
    ) -> None:
        """Build `<name> = pymc.<Family>("<name>", **args, dims=(...),
        observed=<observed>)` and append to the with-body block."""
        meta = _resolve_meta(node.family, self.target)
        dist_class = meta.target_names.get("pymc")
        if dist_class is None:
            raise UnsupportedConstruct(
                self.target, [f"family:{node.family}"]
            )

        callee = attribute(ctx.py, ("pymc", dist_class))
        positional = (string_literal(ctx.py, node.name),)

        # Apply via=<idx> rewrite: when an observation declares
        # `via=<idx>`, wrap every reference to the surrounding-
        # marginalize latent with `[<idx>]` so `phi[z]` becomes
        # `phi[z[<idx>]]` at the call site.
        via = getattr(node, "via", None)
        latent_name = ctx.current_latent

        # arg_constraints for this family (read class-level only;
        # property-form parameterisations land in a future pass).
        cls_constraints = getattr(
            meta.distribution_class, "arg_constraints", {},
        )
        if not isinstance(cls_constraints, dict):
            cls_constraints = {}

        # Build the keyword list, applying alias renaming and
        # broadcasting for scalar refs in vector-expected slots.
        aliases = _merged_aliases(meta)
        keyword: list[tuple[str, str]] = []
        for arg, arg_name in zip(node.args, node.arg_names, strict=True):
            renamed = aliases.get(arg_name, arg_name)
            arg_to_emit = arg
            if via is not None and latent_name is not None:
                arg_to_emit = _wrap_latent_with_via(
                    arg, latent_name=latent_name, via=via,
                )
            arg_to_emit = self._maybe_broadcast_ref(
                arg_to_emit,
                arg_name=arg_name,
                cls_constraints=cls_constraints,
                plate=node.plate,
                ir=ctx.ir,
            )
            keyword.append((renamed, self._render_arg(ctx, arg_to_emit)))

        # `dims=(<batch+event names>,)`.
        dims_tuple = self._dims_tuple(ctx.py, node.plate)
        if dims_tuple is not None:
            keyword.append(("dims", dims_tuple))

        # `observed=<observed>` for IRObserve.
        if observed_name is not None:
            keyword.append(("observed", identifier(ctx.py, observed_name)))

        rhs = call(
            ctx.py, callee,
            positional=positional, keyword=tuple(keyword),
        )

        if observed_name is None:
            stmt = assignment(ctx.py, lhs_name=node.name, rhs=rhs)
        else:
            stmt = ctx.py.v(ctx.py.fresh("es"), "expression_statement")
            ctx.py.e(stmt, rhs, "child_of")
        ctx.py.e(ctx.with_body, stmt, "child_of")

    def _maybe_broadcast_ref(
        self,
        arg: IRArg,
        *,
        arg_name: str,
        cls_constraints: dict[str, c.Constraint],
        plate: Plate,
        ir: IRProgram,
    ) -> IRArg:
        """Wrap a scalar [`IRArgRef`][quivers.transpile.ir.IRArgRef] in
        [`IRArgBroadcast`][quivers.transpile.ir.IRArgBroadcast] when
        the family's `arg_constraints[arg_name]` expects a vector /
        matrix.

        `Lower` skips broadcasting for [`IRArgRef`][quivers.transpile.ir.IRArgRef]
        on the assumption the referenced binding is itself the right
        shape. PyMC's `Dirichlet(a=...)` wants the concentration as a
        vector; when the user passes a scalar program parameter
        (`alpha : Real`), this method injects an `np.full(...)` call so
        the emitted source remains shape-correct."""
        if not isinstance(arg, IRArgRef):
            return arg
        expected = cls_constraints.get(arg_name)
        if expected is None:
            return arg
        if not isinstance(expected, c._IndependentConstraint):
            return arg
        # Only broadcast when the referenced name is a scalar program
        # parameter (an input with no batch or event axes).
        bound = self._input_for(arg.name, ir)
        if bound is None:
            # Not a known input; assume the binding carries the right
            # shape (e.g. a let-bound vector).
            return arg
        if bound.plate.batch_dims or bound.plate.event_dims:
            return arg
        # Build the target shape from the surrounding plate's
        # event_dims (the broadcast target for a vector/matrix arg).
        event = tuple(
            dim.size for dim in plate.event_dims
            if isinstance(dim, DimStatic)
        )
        if not event or len(event) < expected.event_dim:
            return arg
        target_shape = event[: expected.event_dim]
        return IRArgBroadcast(value=arg, target_shape=target_shape)

    def _input_for(
        self, name: str, ir: IRProgram
    ) -> IRDataInput | None:
        for inp in ir.inputs:
            if inp.name == name:
                return inp
        return None

    def _dims_tuple(self, py: PyCtx, plate: Plate) -> str | None:
        """Build the `dims=(<batch>..., <event>...)` tuple.

        PyMC's `dims` carries every axis the RV broadcasts over:
        first the batch (replication) axes, then the event axes.
        Both are passed as string axis names matching the coord
        dict's keys."""
        all_dims = (*plate.batch_dims, *plate.event_dims)
        if not all_dims:
            return None
        t = py.v(py.fresh("tup"), "tuple")
        names = [_dim_name(dim) for dim in all_dims]
        for name in names:
            py.e(t, string_literal(py, name), "child_of")
        # Force trailing comma on single-element tuples so pymc parses
        # `("Doc",)` as a tuple rather than a parenthesised string.
        if len(names) == 1:
            py.constraint(t, "ptrace-0", "T(")
            py.constraint(t, "ptrace-1", "Cstring")
            py.constraint(t, "ptrace-2", "T,")
            py.constraint(t, "ptrace-3", "T)")
        return t

    # ----- marginalize: lower to explicit IRSample + scope inline -----

    def marginalize(
        self,
        ctx: _RenderCtx,
        node: IRMarginalize,
    ) -> SchemaFragment:
        """Lower [`IRMarginalize`][quivers.transpile.ir.IRMarginalize]
        to `IRSample(latent)` plus the scope body inline.

        PyMC supports discrete latents under MCMC and via the
        marginalisation primitives, so the explicit-latent rewrite is
        the canonical lowering. The walk through the rewritten scope
        happens in `_dispatch_pymc`."""
        del ctx, node
        return ""

    # ----- broadcast: np.full((K,), x) for 1D / np.full((R, C), x) for 2D --

    def broadcast(
        self,
        ctx: _RenderCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> SchemaFragment:
        """Emit `np.full((K,), <value>)` for 1D, `np.full((R, C),
        <value>)` for 2D target shapes."""
        del ctx
        msg = (
            "PyMCRenderer.broadcast: standalone broadcast emission "
            "is internal; use `_render_arg` on an `IRArgBroadcast` "
            "instead."
        )
        raise RuntimeError(msg)

    def _render_broadcast(
        self,
        bag: _PyMCCtx,
        value: IRArg,
        target_shape: tuple[int, ...],
    ) -> str:
        if len(target_shape) not in (1, 2):
            raise UnsupportedConstruct(
                self.target, [f"broadcast:rank-{len(target_shape)}"],
            )
        shape_tuple = bag.py.v(bag.py.fresh("tup"), "tuple")
        for n in target_shape:
            bag.py.e(
                shape_tuple,
                number_literal(bag.py, float(n)),
                "child_of",
            )
        # Trailing-comma constraint for 1-tuples.
        if len(target_shape) == 1:
            bag.py.constraint(shape_tuple, "ptrace-0", "T(")
            bag.py.constraint(shape_tuple, "ptrace-1", "Cinteger")
            bag.py.constraint(shape_tuple, "ptrace-2", "T,")
            bag.py.constraint(shape_tuple, "ptrace-3", "T)")
        callee = attribute(bag.py, ("np", "full"))
        return call(
            bag.py, callee,
            positional=(shape_tuple, self._render_arg(bag, value)),
        )

    # ----- arg rendering: dispatch on IRArg variant -----

    def _render_arg(self, ctx: _PyMCCtx, arg: IRArg) -> str:
        if isinstance(arg, IRArgNumber):
            return number_literal(ctx.py, arg.value)
        if isinstance(arg, IRArgRef):
            return self._render_ref(ctx, arg)
        if isinstance(arg, IRArgBroadcast):
            return self._render_broadcast(ctx, arg.value, arg.target_shape)
        if isinstance(arg, IRArgList):
            return self._render_list(ctx, arg)
        if isinstance(arg, IRArgMatrix):
            return self._render_matrix(ctx, arg)
        if isinstance(arg, IRArgFamilyRef):
            return self._render_family_ref(ctx, arg)
        if isinstance(arg, IRArgTransform):
            return self._render_transform(ctx, arg)
        raise UnsupportedConstruct(
            self.target, [f"arg:{type(arg).__name__}"]
        )

    def _render_ref(self, ctx: _PyMCCtx, ref: IRArgRef) -> str:
        """Emit `name` or `name[i0][i1]...` as a subscript chain."""
        current = identifier(ctx.py, ref.name)
        for idx in ref.indices:
            s = ctx.py.v(ctx.py.fresh("subs"), "subscript")
            ctx.py.e(s, current, "value")
            ctx.py.e(s, self._render_arg(ctx, idx), "subscript")
            current = s
        return current

    def _render_list(self, ctx: _PyMCCtx, lst: IRArgList) -> str:
        """Render a list literal arg as `np.array([e0, e1, ...])`."""
        list_v = ctx.py.v(ctx.py.fresh("list"), "list")
        for e in lst.elements:
            ctx.py.e(list_v, self._render_arg(ctx, e), "child_of")
        return call(
            ctx.py, attribute(ctx.py, ("np", "array")),
            positional=(list_v,),
        )

    def _render_matrix(self, ctx: _PyMCCtx, mat: IRArgMatrix) -> str:
        """Render a matrix literal arg as `np.array([[...], [...]])`."""
        outer = ctx.py.v(ctx.py.fresh("list"), "list")
        for row in mat.rows:
            inner = ctx.py.v(ctx.py.fresh("list"), "list")
            for e in row.elements:
                ctx.py.e(inner, self._render_arg(ctx, e), "child_of")
            ctx.py.e(outer, inner, "child_of")
        return call(
            ctx.py, attribute(ctx.py, ("np", "array")),
            positional=(outer,),
        )

    def _render_family_ref(
        self, ctx: _PyMCCtx, ref: IRArgFamilyRef
    ) -> str:
        """Emit `pymc.<base>.dist(<base_args>)` for a referenced
        morphism whose `init_family` clause names a base distribution.

        Used by wrappers (`Truncated`, `Mixture`, etc.): the wrapper
        renderer emits the wrapped distribution as a frozen `.dist()`
        call so PyMC's wrapper can consume it."""
        morph = ctx.ctx.morphisms.get(ref.name)
        if morph is None:
            raise UnsupportedConstruct(
                self.target,
                [f"family_ref:{ref.name}: morphism not in context"],
            )
        init = morph.init_family
        if init is None:
            raise UnsupportedConstruct(
                self.target,
                [
                    f"family_ref:{ref.name}: morphism has no "
                    f"`~ Family(...)` init clause"
                ],
            )
        base_meta = _resolve_meta(init.family, self.target)
        base_dist_class = base_meta.target_names.get("pymc")
        if base_dist_class is None:
            raise UnsupportedConstruct(
                self.target,
                [f"family_ref:{init.family}: pymc target name missing"],
            )
        dist_attr = attribute(ctx.py, ("pymc", base_dist_class, "dist"))
        positional = tuple(
            arg_expr(ctx.py, _draw_arg_to_wire(a))
            for a in (init.args or ())
        )
        return call(ctx.py, dist_attr, positional=positional)

    def _render_transform(
        self, ctx: _PyMCCtx, t: IRArgTransform
    ) -> str:
        """Renderer-applied arithmetic transform on an arg.

        PyMC's [`Potential`][pymc.Potential] does not currently use
        `_ALIAS_TRANSFORMS`, but the renderer ships the mechanism so
        future parameterisation renames plug in without a separate
        dispatch surface."""
        inner = self._render_arg(ctx, t.inner)
        if t.transform == "inv":
            one = number_literal(ctx.py, 1.0)
            return _bin_op(ctx.py, one, "/", inner)
        if t.transform == "neg":
            return _unary_op(ctx.py, "-", inner)
        if t.transform == "inv_square":
            square = _bin_op(ctx.py, inner, "*", inner)
            one = number_literal(ctx.py, 1.0)
            return _bin_op(ctx.py, one, "/", square)
        if t.transform == "log":
            return call(
                ctx.py, attribute(ctx.py, ("np", "log")),
                positional=(inner,),
            )
        if t.transform == "exp":
            return call(
                ctx.py, attribute(ctx.py, ("np", "exp")),
                positional=(inner,),
            )
        raise UnsupportedConstruct(
            self.target, [f"transform:{t.transform}"]
        )

    # ----- deterministic / score -----

    def _emit_deterministic(
        self, ctx: _PyMCCtx, node: IRDeterministic
    ) -> None:
        """Emit `<name> = <expr>` for a let-bound deterministic
        computation."""
        rhs = render_let_expr_python(ctx.py, node.expr)
        stmt = assignment(ctx.py, lhs_name=node.name, rhs=rhs)
        ctx.py.e(ctx.with_body, stmt, "child_of")

    def _emit_score_step(
        self, ctx: _PyMCCtx, node: IRScore
    ) -> None:
        """Emit `<name> = <expr>` then `pymc.Potential("<name>",
        <name>)` for a score step.

        PyMC's [`Potential`][pymc.Potential] adds an arbitrary scalar
        log-density factor to the model's joint, mirroring NumPyro's
        `factor` primitive."""
        rhs = render_let_expr_python(ctx.py, node.expr)
        stmt = assignment(ctx.py, lhs_name=node.name, rhs=rhs)
        ctx.py.e(ctx.with_body, stmt, "child_of")
        factor_call = call(
            ctx.py, attribute(ctx.py, ("pymc", "Potential")),
            positional=(
                string_literal(ctx.py, node.name),
                identifier(ctx.py, node.name),
            ),
        )
        fstmt = ctx.py.v(ctx.py.fresh("es"), "expression_statement")
        ctx.py.e(fstmt, factor_call, "child_of")
        ctx.py.e(ctx.with_body, fstmt, "child_of")

    # ----- Renderer protocol's `sample` (proxies into _emit_sample) -----

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
        """`Renderer` protocol entry: PyMC's emit is structurally
        joined with declare (the constructor call IS the declaration).
        Not called by the PyMC walker (which calls `_emit_sample`
        directly); kept here so external callers using the abstract
        protocol can drive the renderer."""
        del ctx, name, family, args, arg_names, constraint, plate, observed
        msg = (
            "PyMCRenderer.sample is not called directly; PyMC dispatch "
            "goes through `_emit_sample` to thread the per-walk bag."
        )
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _PyMCCtx:
    """Renderer-internal mutable bag threaded through the dispatch
    points: shared [`_RenderCtx`][quivers.transpile.renderers._base._RenderCtx],
    the [`PyCtx`][quivers.transpile.backends._pyhelpers.PyCtx] adapter,
    and per-walk state (`with`-body block, current marginalize latent,
    parent IR for input lookups)."""

    def __init__(
        self, *, ctx: _RenderCtx, py: PyCtx, ir: IRProgram,
    ) -> None:
        self.ctx = ctx
        self.py = py
        self.ir = ir
        self.with_body: str = ""
        self.current_latent: str | None = None


def _dim_name(dim: Dim) -> str:
    """Return a [`Dim`][quivers.transpile.ir.Dim]'s axis name as a
    plain `str`.

    `Dim` is a [`dx.TaggedUnion`][didactic.api.TaggedUnion] base whose
    `.name` slot types as `FieldValue`; narrowing the variant lets the
    type checker prove the slot is a `str`."""
    if isinstance(dim, (DimStatic, DimDynamic)):
        return dim.name
    msg = f"unknown Dim kind: {type(dim).__name__}"
    raise UnsupportedConstruct("qvr-pymc", [msg])


def _resolve_meta(family: str, target: str) -> FamilyMeta:
    meta = FAMILY_META.get(family)
    if meta is None:
        raise UnsupportedConstruct(target, [f"family:{family}"])
    return meta


def _merged_aliases(meta: FamilyMeta) -> dict[str, str]:
    """Merge `meta.arg_aliases['pymc']` with `_PYMC_EXTRA_ALIASES` so
    renderer-side renames (PyMC's `probs → p`, `loc → mu`, ...) layer
    on top of the per-family rename table in
    [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META]."""
    out: dict[str, str] = dict(_PYMC_EXTRA_ALIASES.get(meta.qvr_name, {}))
    out.update(meta.arg_aliases.get("pymc", {}))
    return out


def _wrap_latent_with_via(
    arg: IRArg, *, latent_name: str, via: str
) -> IRArg:
    """Rewrite an IR arg by appending a `[via]` index to every
    [`IRArgRef`][quivers.transpile.ir.IRArgRef] that names the
    marginalize-scoped latent.

    `phi[z]` with `latent='z'` and `via='word_idx'` becomes
    `phi[z[word_idx]]`. The renamed `z` reference inside `phi`'s
    indices gains a trailing `[word_idx]` index so PyMC indexes the
    latent positionally at the observation's fibration site."""
    if isinstance(arg, IRArgRef):
        if arg.name == latent_name:
            return IRArgRef(
                name=arg.name,
                indices=(
                    *(
                        _wrap_latent_with_via(
                            i, latent_name=latent_name, via=via
                        )
                        for i in arg.indices
                    ),
                    IRArgRef(name=via),
                ),
            )
        return IRArgRef(
            name=arg.name,
            indices=tuple(
                _wrap_latent_with_via(
                    i, latent_name=latent_name, via=via
                )
                for i in arg.indices
            ),
        )
    if isinstance(arg, IRArgBroadcast):
        return IRArgBroadcast(
            value=_wrap_latent_with_via(
                arg.value, latent_name=latent_name, via=via
            ),
            target_shape=arg.target_shape,
        )
    if isinstance(arg, IRArgList):
        return IRArgList(
            elements=tuple(
                _wrap_latent_with_via(
                    e, latent_name=latent_name, via=via
                )
                for e in arg.elements
            ),
        )
    if isinstance(arg, IRArgMatrix):
        return IRArgMatrix(
            rows=tuple(
                IRArgList(
                    elements=tuple(
                        _wrap_latent_with_via(
                            e, latent_name=latent_name, via=via
                        )
                        for e in row.elements
                    ),
                )
                for row in arg.rows
            ),
        )
    return arg


def _draw_arg_to_wire(a: object) -> str | float:
    """Coerce a referenced morphism's init-family argument to the
    `arg_expr`-acceptable wire form."""
    if isinstance(a, (int, float)):
        return float(a)
    return str(a)


def _bin_op(py: PyCtx, lhs: str, op: str, rhs: str) -> str:
    """Build a `binary_operator` vertex with `left`/`operator`/`right`."""
    b = py.v(py.fresh("bop"), "binary_operator")
    py.e(b, lhs, "left")
    op_vid = py.v(py.fresh("op"), "operator")
    py.literal(op_vid, op)
    py.e(b, op_vid, "operator")
    py.e(b, rhs, "right")
    return b


def _unary_op(py: PyCtx, op: str, operand: str) -> str:
    """Build a `unary_operator` vertex with `operator`/`argument`."""
    u = py.v(py.fresh("uop"), "unary_operator")
    op_vid = py.v(py.fresh("op"), "operator")
    py.literal(op_vid, op)
    py.e(u, op_vid, "operator")
    py.e(u, operand, "argument")
    return u


# ---------------------------------------------------------------------------
# `Mapping[Module, bytes]` adapter for the legacy `realize(...)` pipeline.
# ---------------------------------------------------------------------------


class _PyMCWalker(SchemaTransform):
    """SchemaTransform shim: lower the `Module` to IR, then run the
    `PyMCRenderer` to a panproto schema."""

    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        ir = Lower().forward(module)
        renderer = PyMCRenderer()
        morphisms = build_morphism_table(module)
        lets = build_let_table(module)
        return renderer.render_with_tables(
            ir, morphisms=morphisms, lets=lets,
        )


def render_module_bytes(module: Module) -> bytes:
    """Convenience: parse `module` through Lower + PyMCRenderer +
    `emit_pretty` and return the emitted Python source bytes."""
    return realize(module, grammar="python", transform=_PyMCWalker())


__all__ = ["PyMCRenderer", "render_module_bytes"]
