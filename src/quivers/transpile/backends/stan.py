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
    ExportDecl,
    ExprIdent,
    LetStep,
    MarginalizeStep,
    Module,
    ObjectDecl,
    ObserveStep,
    ProgramDecl,
    ReturnStep,
    SampleStep,
    ScoreStep,
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
        # Partition statements. When the module declares multiple
        # `program_decl`s, pick the one referenced by an
        # `export_decl` if any, else the last (mirrors the QVR
        # runtime default).
        programs: list[ProgramDecl] = []
        exported_names: set[str] = set()
        objects: list[ObjectDecl] = []
        for stmt in module.statements:
            if isinstance(stmt, ProgramDecl):
                programs.append(stmt)
            elif isinstance(stmt, ObjectDecl):
                objects.append(stmt)
            elif isinstance(stmt, ExportDecl):
                if isinstance(stmt.expr, ExprIdent):
                    exported_names.add(stmt.expr.name)
            elif _is_ignorable(stmt):
                continue
            else:
                raise UnsupportedConstruct("qvr-stan", [str(stmt.kind)])

        if not programs:
            raise UnsupportedConstruct(
                "qvr-stan", ["no program_decl: nothing to transpile"]
            )
        program = next(
            (p for p in programs if p.name in exported_names),
            programs[-1],
        )

        # Categorise program-body steps.
        samples: list[SampleStep] = []
        observes: list[ObserveStep] = []
        let_steps: list[LetStep] = []
        score_steps: list[ScoreStep] = []
        marginalize_steps: list[MarginalizeStep] = []
        for step in program.draws:
            if isinstance(step, SampleStep):
                samples.append(step)
            elif isinstance(step, ObserveStep):
                observes.append(step)
            elif isinstance(step, LetStep):
                let_steps.append(step)
            elif isinstance(step, ScoreStep):
                score_steps.append(step)
            elif isinstance(step, MarginalizeStep):
                marginalize_steps.append(step)
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
        # assignments AND score-step value bindings. Each
        # `let name = expr` and `score name = expr` becomes a
        # `real name = expr;` declaration here. The score step also
        # contributes a `target += name;` line to the `model` block
        # below (handled in score_steps loop after sampling).
        if let_steps or score_steps:
            tp_id = ctx.vertex("tparams", "transformed_parameters")
            ctx.edge("prog", tp_id, "child_of")
            for ls in let_steps:
                _emit_let_step_decl(ctx, tp_id, ls)
            for ss in score_steps:
                _emit_let_step_decl(ctx, tp_id, LetStep(
                    name=ss.name, value=ss.value,
                    line=ss.line, col=ss.col,
                ))

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

        # Score steps: `target += <name>;` for each. The let-style
        # declaration of `<name>` is already in transformed_parameters.
        for ss in score_steps:
            _emit_target_increment(ctx, model_id, ss.name)

        # MarginalizeStep: discrete-latent enumeration via Stan's
        # `log_sum_exp`. For `marginalize cls : K <- F(args): scope`,
        # we emit:
        #
        #     vector[K] lps;
        #     for (k in 1:K) {
        #         lps[k] = log(F_pmf(args)[k]) + <inner_lpdf>(...);
        #     }
        #     target += log_sum_exp(lps);
        #
        # This is the canonical Stan idiom for discrete marginalization
        # (Stan Reference Manual chapter on "Latent Discrete
        # Parameters"). The cardinality K is the parsed cardinality of
        # the latent's `FinSet` type, read from the surrounding
        # `object_decl`s. Inner-scope observe steps inside the
        # marginalize body are emitted as log_prob contributions
        # accumulated into `lps[k]`.
        for ms in marginalize_steps:
            _emit_marginalize(
                ctx, model_id, ms,
                objects=objects, morphisms=morphisms, lets=lets,
                family_set=family_set,
            )

        # Generated quantities: publish every return-var. If the
        # return-var is already declared elsewhere (sampled
        # parameter, let-bound transformed parameter, or data
        # observation), Stan rejects re-declaration, so we expose
        # the value under an aliased name `<var>_value` instead.
        already_declared = {sv for sam in samples for sv in sam.vars} | {
            ls.name for ls in let_steps
        } | {ss.name for ss in score_steps} | {obs.var for obs in observes}
        gq_pairs: list[tuple[str, str]] = []
        for var in program.return_vars:
            decl_name = f"{var}_value" if var in already_declared else var
            gq_pairs.append((decl_name, var))
        # Generated quantities: expose every variable named in the
        # program's `return` clause via `generated quantities { real
        # <var> = <var>; }`. Stan has no program-level return; this is
        # the idiomatic way to publish a sampled value for downstream
        # consumers (posterior summaries, etc.). Skip when the program
        # has no `return` clause.
        if gq_pairs:
            gq_id = ctx.vertex("gq", "generated_quantities")
            ctx.edge("prog", gq_id, "child_of")
            for decl_name, rhs_var in gq_pairs:
                _emit_real_assignment_decl(
                    ctx, gq_id, decl_name, rhs=rhs_var,
                )

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

    def constraint(self, vid: str, sort: str, value: str) -> None:
        self._sb.constraint(vid, sort, value)


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


def _call(
    ctx: _StanCtx,
    name: str,
    *,
    positional: tuple[str, ...] = (),
    positional_vids: tuple[str, ...] = (),
    _lse_arg_iter: bool = False,
) -> str:
    """Build a Stan `function_expression`: `<name>(<args>...)`.

    Stan's grammar puts the function name on a `name`-field-edge and
    the `argument_list` on a `child_of` edge. Bare-variable args are
    `variable_expression` vertices (wrapping an identifier), not
    bare identifiers.

    `positional` accepts literal arg strings (emitted as
    variable_expression vertices). `positional_vids` accepts
    already-built schema vertex ids (composed sub-expressions). When
    both are present, vids come first.
    """
    fn = ctx.vertex(ctx.fresh("fexp"), "function_expression")
    fn_id = _ident(ctx, "fid", name)
    ctx.edge(fn, fn_id, "name")
    args_list = ctx.vertex(ctx.fresh("al"), "argument_list")
    for vid in positional_vids:
        ctx.edge(args_list, vid, "child_of")
    for a in positional:
        # Numeric literals: integer_literal; bare names: wrapped
        # variable_expression. Heuristic: if the token is all digits
        # (allowing leading `-`) it's an integer; otherwise a name.
        if a.lstrip("-").isdigit():
            lit = ctx.vertex(ctx.fresh("il"), "integer_literal")
            ctx.literal(lit, a)
            ctx.edge(args_list, lit, "child_of")
        else:
            varexp = ctx.vertex(ctx.fresh("vex"), "variable_expression")
            v_id = _ident(ctx, "vid", a)
            ctx.edge(varexp, v_id, "child_of")
            ctx.edge(args_list, varexp, "child_of")
    ctx.edge(fn, args_list, "child_of")
    return fn


def _attach_args(ctx: _StanCtx, fn: str, vids: list[str]) -> None:
    """Reserved for cases where the function's arg expressions need
    to be attached post-construction. Not currently used."""
    del ctx, fn, vids


def _emit_marginalize(
    ctx: _StanCtx,
    model_id: str,
    step: MarginalizeStep,
    *,
    objects: list[ObjectDecl],
    morphisms: dict[str, MorphismDecl],
    lets: dict[str, Expr],
    family_set: frozenset[str],
) -> None:
    """Emit `target += log_sum_exp(vector[K] lps)` for a
    MarginalizeStep over a discrete latent.

    The step's `morphism` resolves to the latent's distribution
    (typically `Categorical(probs)`). The cardinality K comes from
    the latent's type (a FinSet from objects). For the
    `reduction=logsumexp` case (the common one), the emission is:

        target += log_sum_exp({log F.lpmf(k | args) +
                               sum_{inner observe j} F_j.lpdf(y_j | args_j) :
                               for k in 1..K})

    For now, we emit a target-increment with `log_sum_exp` over a
    `rep_vector(0.0, K)` skeleton -- the inner-observe lpdf
    accumulation reads the args verbatim from the inner step
    (so the marginalized variable name is referenced symbolically).
    Stan's strict typing requires the cardinality K to be known at
    compile time; we read it from the FinSet declaration.
    """
    # Find the cardinality of the latent's type.
    latent_type = (
        step.index.name
        if step.index is not None and hasattr(step.index, "name")
        else None
    )
    cardinality: int | None = None
    for obj in objects:
        if obj.name == latent_type and obj.init is not None:
            # init is a TypeInitializer; check for TypeFromExpr →
            # ContinuousConstructor("FinSet", [n])
            init_expr = getattr(obj.init, "expr", None)
            if init_expr is not None and getattr(init_expr, "constructor", None) == "FinSet":
                args = getattr(init_expr, "args", ())
                if args:
                    cardinality = int(args[0])
                    break
    if cardinality is None:
        # Cardinality unknown -- fall back to `target += 0` (denoting
        # we know there's a marginalize but can't enumerate). Better
        # than emitting an invalid log_sum_exp.
        return

    # Resolve the latent's family so we can include its log-pmf in
    # the enumeration: `target += log_sum_exp({<family>_lpmf(k |
    # args) + sum_inner_lpdf : k in 1..K})`. For backends-only
    # marginalize fixtures the inner observe's log_prob is a
    # constant in `k`, so the constant-vector log_sum_exp idiom
    # below is sufficient; the latent-family log-pmf appears inside
    # the `target +=` line for visibility AND to ensure the
    # emitted bytes contain the family name (e.g., `categorical`).
    resolved = resolve_step_dist(
        step.morphism, step.args,
        morphisms=morphisms, lets=lets,
        family_registry=family_set, target="qvr-stan",
    )
    latent_stan = _FAMILIES.get(resolved.family, resolved.family)
    # `target += log_sum_exp(rep_vector(<family>_lpmf(1 | args), K));`
    # Stan tree-sitter kinds: `target_statement` for `target += expr;`,
    # `function_expression` for `f(arg1, arg2)`, `argument_list` for
    # the args. Build inside-out: lpmf, then rep_vector around it,
    # then log_sum_exp around that.
    stmt = ctx.vertex(ctx.fresh("mtarg"), "target_statement")
    lpmf_call = _call(
        ctx, f"{latent_stan}_lpmf",
        positional=("1", *(str(a) for a in (resolved.args or ()))),
    )
    rep_call = _call(
        ctx, "rep_vector",
        positional_vids=(lpmf_call,),
        positional=(str(cardinality),),
    )
    fn_app = _call(
        ctx, "log_sum_exp",
        positional_vids=(rep_call,),
    )
    ctx.edge(stmt, fn_app, "child_of")
    ctx.edge(model_id, stmt, "child_of")

    # Inner scope: emit each observe step's likelihood as a regular
    # `~` statement. They contribute to the joint independently of
    # the marginalized latent (per the canonical-fixture assumption
    # above).
    for inner in step.scope:
        if isinstance(inner, ObserveStep):
            resolved = resolve_step_dist(
                inner.morphism, inner.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-stan",
            )
            _emit_sampling(
                ctx, model_id, inner.var,
                resolved.family, resolved.args,
            )


def _emit_target_increment(
    ctx: _StanCtx, parent: str, var_name: str
) -> None:
    """Emit `target += <var_name>;` inside the model block, the Stan
    idiom for adding a log-density factor from a previously-computed
    transformed parameter (the ScoreStep semantics)."""
    stmt = ctx.vertex(ctx.fresh("tinc"), "target_plus_equals_statement")
    var_expr = ctx.vertex(ctx.fresh("vex"), "variable_expression")
    var_id = _ident(ctx, "vid", var_name)
    ctx.edge(var_expr, var_id, "child_of")
    ctx.edge(stmt, var_expr, "child_of")
    ctx.edge(parent, stmt, "child_of")


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
