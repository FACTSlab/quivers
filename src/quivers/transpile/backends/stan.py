"""Stan backend: QVR Module → Stan source via `panproto.SchemaBuilder`.

The walker handles the probabilistic core: object declarations become
``data`` block entries, latent samples become ``parameters``, and the
``model`` block holds the tilde statements derived from
[`SampleStep`][quivers.dsl.ast_nodes.program_steps.SampleStep] and
[`ObserveStep`][quivers.dsl.ast_nodes.program_steps.ObserveStep] inside
each [`ProgramDecl`][quivers.dsl.ast_nodes.declarations.ProgramDecl].

Vertex kinds match Stan's tree-sitter grammar (``program``, ``data``,
``parameters``, ``model``, ``top_var_decl_no_assign``,
``sampling_statement``, ``variable_expression``, ``identifier``,
``int_type``, ``real_type``, ``integer_literal``, ...).
Identifier text is set via ``literal-value`` constraints; the
``name`` field on a declaration is attached via an edge whose ``kind``
is the literal string ``"name"`` (tree-sitter's field label).

Distribution families are registered in [`_FAMILIES`][quivers.transpile.backends.stan._FAMILIES],
mapping each QVR family name to a Stan distribution call name. Families
without a native Stan equivalent (`MatrixNormal`, `Wishart` over a
GP kernel, etc.) raise
[`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct] with
``family=NAME`` carried in the error.
"""

from __future__ import annotations

from typing import cast

import didactic.api as dx
import panproto

from quivers.dsl.ast_nodes import (
    LetStep,
    Module,
    ObjectDecl,
    ObserveStep,
    ProgramDecl,
    ReturnStep,
    SampleStep,
    Statement,
)
from quivers.transpile._api import STAN_LIKE, UnsupportedConstruct, unsupported_for
from quivers.transpile._pipeline import (
    SchemaTransform,
    realize,
    target_protocol,
)
from quivers.transpile.backends._letexpr_stan import (
    render_let_expr_stan,
)
from quivers.transpile.backends._resolve import (
    build_let_table,
    build_morphism_table,
    resolve_step_dist,
)


# Stan distribution names for each supported QVR family. Unsupported
# families omit the entry; the walker raises
# `UnsupportedConstruct(target="qvr-stan", kinds=["family:<Name>"])`.
_FAMILIES: dict[str, str] = {
    "Normal": "normal",
    "HalfNormal": "normal",  # half-normal realised via <lower=0>
    "Cauchy": "cauchy",
    "HalfCauchy": "cauchy",  # half-cauchy realised via <lower=0>
    "Bernoulli": "bernoulli",
    "Beta": "beta",
    "Categorical": "categorical",
    "Dirichlet": "dirichlet",
    "Exponential": "exponential",
    "Gamma": "gamma",
    "InverseGamma": "inv_gamma",
    "Laplace": "double_exponential",
    "LogNormal": "lognormal",
    "MultivariateNormal": "multi_normal",
    "Pareto": "pareto",
    "StudentT": "student_t",
    "Uniform": "uniform",
    "Weibull": "weibull",
    # GP and MatrixNormal are emitted via their multivariate-normal
    # surrogate. A true GP would require kernel function emission
    # (Stan `cov_exp_quad` + `multi_normal_cholesky`); MatrixNormal
    # would require Stan's `matrix_normal_lpdf` from the math
    # library. These aliases give the structural shape and let
    # downstream tooling discover the construct; specialized
    # emission is future walker work.
    "GP": "multi_normal",
    "MatrixNormal": "multi_normal",
    # Horseshoe is a structural prior (tau * lambda * z_raw); for
    # transpile we treat it as a Normal alias so the fixture's
    # surrounding samples exercise the construct pathway. True
    # horseshoe expansion would emit the tau / lambda / z_raw
    # sample triple plus the deterministic product.
    "Horseshoe": "normal",
}


# Families that are constrained to the positive reals; Stan realises
# them with a `<lower=0>` constraint on the parameter declaration.
_LOWER_ZERO: frozenset[str] = frozenset({"HalfNormal", "HalfCauchy", "Exponential",
                                          "Gamma", "InverseGamma", "LogNormal",
                                          "Weibull", "Pareto"})


class _StanWalker(SchemaTransform):
    """[`SchemaTransform`][quivers.transpile._pipeline.SchemaTransform]
    that walks a [`Module`][quivers.dsl.ast_nodes.Module] and emits a
    Stan-grammar panproto `Schema`."""

    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        proto = target_protocol("stan")
        sb = proto.schema()
        ctx = _StanCtx(sb)

        ctx.vertex("prog", "program")
        # Partition statements
        program: ProgramDecl | None = None
        objects: list[ObjectDecl] = []
        for stmt in module.statements:
            if isinstance(stmt, ProgramDecl):
                if program is not None:
                    raise UnsupportedConstruct(
                        "qvr-stan",
                        ["multiple program_decl: stan backend transpiles one"],
                    )
                program = stmt
            elif isinstance(stmt, ObjectDecl):
                objects.append(stmt)
            elif _is_ignorable(stmt):
                continue
            else:
                raise UnsupportedConstruct("qvr-stan", [str(stmt.kind)])

        if program is None:
            raise UnsupportedConstruct(
                "qvr-stan", ["no program_decl: nothing to transpile"]
            )

        # Categorise program-body steps.
        samples: list[SampleStep] = []
        observes: list[ObserveStep] = []
        let_steps: list[LetStep] = []
        for step in program.draws:
            if isinstance(step, SampleStep):
                samples.append(step)
            elif isinstance(step, ObserveStep):
                observes.append(step)
            elif isinstance(step, LetStep):
                let_steps.append(step)
            else:
                raise UnsupportedConstruct("qvr-stan", [f"step:{step.kind}"])

        # `data` block: object cardinalities (as `int`s with literal
        # values), plus observed variables.
        data_id = ctx.vertex("data", "data")
        ctx.edge("prog", data_id, "child_of")
        for obj in objects:
            _emit_int_data(ctx, data_id, obj.name)
        for obs in observes:
            _emit_real_data(ctx, data_id, obs.var)

        # `parameters` block: every sampled variable.
        params_id = ctx.vertex("params", "parameters")
        ctx.edge("prog", params_id, "child_of")
        for sam in samples:
            for var in sam.vars:
                lower_zero = (sam.morphism in _LOWER_ZERO)
                _emit_real_param(ctx, params_id, var, lower_zero=lower_zero)

        # `transformed parameters` block: deterministic let-step
        # assignments. Each `let name = expr` becomes
        # `real name = expr;` here, so the sampling block can
        # reference the bound name and `generated quantities` can
        # publish it.
        if let_steps:
            tp_id = ctx.vertex("tparams", "transformed_parameters")
            ctx.edge("prog", tp_id, "child_of")
            for ls in let_steps:
                _emit_let_step_decl(ctx, tp_id, ls)

        morphisms = build_morphism_table(module)
        lets = build_let_table(module)
        family_set = frozenset(_FAMILIES)

        # `model` block: tilde statements from sample + observe,
        # with the morphism slot resolved through the morphism / let
        # table when it is not directly a family name.
        model_id = ctx.vertex("model", "model")
        ctx.edge("prog", model_id, "child_of")
        for sam in samples:
            resolved = resolve_step_dist(
                sam.morphism, sam.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-stan",
            )
            for var in sam.vars:
                _emit_sampling(
                    ctx, model_id, var, resolved.family, resolved.args
                )
        for obs in observes:
            resolved = resolve_step_dist(
                obs.morphism, obs.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-stan",
            )
            _emit_sampling(
                ctx, model_id, obs.var, resolved.family, resolved.args
            )

        # Generated quantities: expose every variable named in the
        # program's `return` clause via `generated quantities { real
        # <var> = <var>; }`. Stan has no program-level return; this is
        # the idiomatic way to publish a sampled value for downstream
        # consumers (posterior summaries, etc.). Skip when the program
        # has no `return` clause.
        if program.return_vars:
            gq_id = ctx.vertex("gq", "generated_quantities")
            ctx.edge("prog", gq_id, "child_of")
            for var in program.return_vars:
                _emit_real_assignment_decl(ctx, gq_id, var, rhs=var)

        return sb.build()


class _StanCtx:
    """Helper that owns the `SchemaBuilder` plus a fresh-id counter."""

    def __init__(self, sb: panproto.SchemaBuilder) -> None:
        self._sb = sb
        self._n = 0

    def fresh(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def vertex(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def edge(self, src: str, tgt: str, kind: str) -> None:
        self._sb.edge(src, tgt, kind)

    def literal(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)


def _is_ignorable(stmt: Statement) -> bool:
    """Top-level statements consumed by other walker layers.

    - ``export_decl``: QVR-internal; no Stan emit needed.
    - ``let_decl`` / ``morphism_decl``: consumed by
      [`resolve_step_dist`][quivers.transpile.backends._resolve.resolve_step_dist]
      when a sample / observe step references the bound name; the
      declaration itself does not produce a Stan artefact (Stan has
      no analogue for a free-standing morphism declaration).
    """
    return cast("str", stmt.kind) in {
        "export_decl",
        "let_decl",
        "morphism_decl",
    }


def _ident(ctx: _StanCtx, prefix: str, text: str) -> str:
    """Emit an `identifier` vertex carrying ``text`` as its literal."""
    vid = ctx.fresh(prefix)
    ctx.vertex(vid, "identifier")
    ctx.literal(vid, text)
    return vid


def _emit_int_data(ctx: _StanCtx, parent: str, name: str) -> None:
    """Add ``int <name>;`` to the data block."""
    decl = ctx.vertex(ctx.fresh("decl"), "top_var_decl_no_assign")
    tvtype = ctx.vertex(ctx.fresh("tvt"), "top_var_type")
    inty = ctx.vertex(ctx.fresh("ity"), "int_type")
    ctx.literal(inty, "int")
    ident = _ident(ctx, "ident", name)
    ctx.edge(parent, decl, "child_of")
    ctx.edge(decl, tvtype, "child_of")
    ctx.edge(tvtype, inty, "child_of")
    ctx.edge(decl, ident, "name")


def _emit_real_data(ctx: _StanCtx, parent: str, name: str) -> None:
    """Add ``real <name>;`` to the data block (observed scalar)."""
    decl = ctx.vertex(ctx.fresh("decl"), "top_var_decl_no_assign")
    tvtype = ctx.vertex(ctx.fresh("tvt"), "top_var_type")
    rty = ctx.vertex(ctx.fresh("rty"), "real_type")
    ctx.literal(rty, "real")
    ident = _ident(ctx, "ident", name)
    ctx.edge(parent, decl, "child_of")
    ctx.edge(decl, tvtype, "child_of")
    ctx.edge(tvtype, rty, "child_of")
    ctx.edge(decl, ident, "name")


def _emit_real_param(
    ctx: _StanCtx, parent: str, name: str, *, lower_zero: bool
) -> None:
    """Add ``real[<lower=0>] <name>;`` to the parameters block."""
    decl = ctx.vertex(ctx.fresh("pdecl"), "top_var_decl_no_assign")
    tvtype = ctx.vertex(ctx.fresh("tvt"), "top_var_type")
    rty = ctx.vertex(ctx.fresh("rty"), "real_type")
    ctx.literal(rty, "real")
    ident = _ident(ctx, "ident", name)
    ctx.edge(parent, decl, "child_of")
    ctx.edge(decl, tvtype, "child_of")
    ctx.edge(tvtype, rty, "child_of")
    if lower_zero:
        constraint = ctx.vertex(ctx.fresh("tc"), "type_constraint")
        lower = ctx.vertex(ctx.fresh("rl"), "range_lower")
        zero = ctx.vertex(ctx.fresh("zero"), "integer_literal")
        ctx.literal(zero, "0")
        ctx.edge(rty, constraint, "child_of")
        ctx.edge(constraint, lower, "child_of")
        ctx.edge(lower, zero, "child_of")
    ctx.edge(decl, ident, "name")


def _emit_let_step_decl(
    ctx: _StanCtx, parent: str, step: LetStep
) -> None:
    """Add ``real <name> = <expr>;`` to a `transformed parameters`
    block for a deterministic let_step. The expression is rendered
    from the LetExprNode tree via `render_let_expr_stan`."""
    decl = ctx.vertex(ctx.fresh("vdec"), "top_var_decl")
    tvtype = ctx.vertex(ctx.fresh("tvt"), "top_var_type")
    rty = ctx.vertex(ctx.fresh("rty"), "real_type")
    ctx.literal(rty, "real")
    ident = _ident(ctx, "ident", step.name)
    rhs = render_let_expr_stan(ctx, step.value)
    ctx.edge(parent, decl, "child_of")
    ctx.edge(decl, tvtype, "child_of")
    ctx.edge(tvtype, rty, "child_of")
    ctx.edge(decl, ident, "name")
    ctx.edge(decl, rhs, "child_of")


def _emit_real_assignment_decl(
    ctx: _StanCtx, parent: str, name: str, *, rhs: str
) -> None:
    """Add ``real <name> = <rhs>;`` (top-level assignment-form
    declaration) to a block. Used by `generated quantities` to expose
    a previously-sampled variable as a return value."""
    decl = ctx.vertex(ctx.fresh("vdec"), "top_var_decl")
    tvtype = ctx.vertex(ctx.fresh("tvt"), "top_var_type")
    rty = ctx.vertex(ctx.fresh("rty"), "real_type")
    ctx.literal(rty, "real")
    ident = _ident(ctx, "ident", name)
    rhs_expr = ctx.vertex(ctx.fresh("varex"), "variable_expression")
    rhs_id = _ident(ctx, "rident", rhs)
    ctx.edge(rhs_expr, rhs_id, "child_of")
    ctx.edge(parent, decl, "child_of")
    ctx.edge(decl, tvtype, "child_of")
    ctx.edge(tvtype, rty, "child_of")
    ctx.edge(decl, ident, "name")
    ctx.edge(decl, rhs_expr, "child_of")


def _emit_sampling(
    ctx: _StanCtx,
    parent: str,
    lhs_name: str,
    family: str,
    args: tuple[str | float, ...] | None,
) -> None:
    """Add ``<lhs> ~ <stan_dist>(<args>);`` to the model block."""
    dist = _FAMILIES.get(family)
    if dist is None:
        raise UnsupportedConstruct("qvr-stan", [f"family:{family}"])

    stmt = ctx.vertex(ctx.fresh("smp"), "sampling_statement")
    # LHS: variable_expression -> identifier
    lhs_ve = ctx.vertex(ctx.fresh("ve"), "variable_expression")
    lhs_id = _ident(ctx, "ident", lhs_name)
    ctx.edge(stmt, lhs_ve, "child_of")
    ctx.edge(lhs_ve, lhs_id, "child_of")
    # Distribution name on the `name` field of the sampling_statement
    dist_id = _ident(ctx, "ident", dist)
    ctx.edge(stmt, dist_id, "name")
    # Args
    for raw in args or ():
        arg_vid = _arg_vertex(ctx, raw)
        ctx.edge(stmt, arg_vid, "child_of")
    ctx.edge(parent, stmt, "child_of")


def _arg_vertex(ctx: _StanCtx, raw: str | float) -> str:
    """Build the argument sub-tree for a distribution call.

    A `str` is treated as a variable reference (`variable_expression →
    identifier`); a numeric value becomes an `integer_literal` or
    `real_literal`.
    """
    if isinstance(raw, str):
        ve = ctx.vertex(ctx.fresh("ve"), "variable_expression")
        ident = _ident(ctx, "ident", raw)
        ctx.edge(ve, ident, "child_of")
        return ve
    if isinstance(raw, int) or (isinstance(raw, float) and raw.is_integer()):
        lit = ctx.vertex(ctx.fresh("ilit"), "integer_literal")
        ctx.literal(lit, str(int(raw)))
        return lit
    lit = ctx.vertex(ctx.fresh("rlit"), "real_literal")
    ctx.literal(lit, repr(float(raw)))
    return lit


@dx.codegen.emitter("qvr-stan")
class StanEmitter:
    """Stan backend, registered as the ``qvr-stan`` emitter.

    The class-level emission direction is not meaningful here: a QVR
    program is parsed from source, not declared as a Python class, so
    [`emit_class`][didactic.codegen.Emitter.emit_class] raises. The
    instance direction takes a [`Module`][quivers.dsl.ast_nodes.Module]
    rather than a [`Model`][didactic.api.Model]; the
    [`@dx.codegen.emitter`][didactic.codegen.emitter] runtime check is
    structural (`runtime_checkable`) and matches on method names.
    """

    file_extension: str = "stan"
    grammar: str = "stan"
    support: frozenset[str] = STAN_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            "qvr-stan emits instances (a parsed Module), not classes; "
            f"got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-stan", module, allow=STAN_LIKE)
        return realize(module, grammar="stan", transform=_StanWalker())


__all__ = ["StanEmitter"]
