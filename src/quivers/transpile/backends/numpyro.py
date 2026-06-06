"""NumPyro backend: QVR Module → Python source under the `python`
tree-sitter grammar.

Output is a single ``def model(<observed>=None): ...`` function whose
body assigns each latent via ``numpyro.sample("<var>", <dist>)`` and
each observation via ``numpyro.sample("<var>", <dist>, obs=<var>)``.
Distributions live in ``numpyro.distributions.<Family>`` (fully
qualified throughout to avoid emitting imports).
"""

from __future__ import annotations

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
    ScoreStep,
    Statement,
)
from quivers.transpile._api import STAN_LIKE, UnsupportedConstruct, unsupported_for
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
    string_literal,  # noqa: F401  -- exposed for score-step factor calls
)
from quivers.transpile.backends._letexpr_python import (
    render_let_expr_python,
)
from quivers.transpile.backends._resolve import (
    build_let_table,
    build_morphism_table,
    resolve_step_dist,
)


def _emit_python_let_step(ctx: PyCtx, body_vid: str, step: LetStep) -> None:
    """Emit a deterministic `<name> = <expr>` assignment inside the
    model body for a `LetStep`."""
    asn = ctx.v(ctx.fresh("asn"), "assignment")
    lhs = ctx.v(ctx.fresh("id"), "identifier")
    ctx.literal(lhs, step.name)
    ctx.e(asn, lhs, "left")
    ctx.e(asn, render_let_expr_python(ctx, step.value), "right")
    ctx.e(body_vid, asn, "child_of")


def _emit_python_score_step(
    ctx: PyCtx,
    body_vid: str,
    step: ScoreStep,
    *,
    factor_namespace: tuple[str, ...] = ("numpyro",),
    factor_fn: str = "factor",
) -> None:
    """Emit `<name> = <expr>; <namespace>.<factor_fn>("<name>", <name>)`
    for a `ScoreStep`.

    The semantics of `score name = expr` is two-fold (cf.
    [Programs §2.7a](../docs/semantics/programs.md#27a-score-factor)):
    bind `name` to the value of `expr` AND add `expr` as a log-density
    factor to the program's joint. Python PPLs realize the factor via
    the backend's `factor` primitive (NumPyro: `numpyro.factor`; Pyro:
    `pyro.factor`; PyMC: `pymc.Potential`; Edward2: explicit
    `tape().factor`).
    """
    asn = ctx.v(ctx.fresh("asn"), "assignment")
    lhs = ctx.v(ctx.fresh("id"), "identifier")
    ctx.literal(lhs, step.name)
    ctx.e(asn, lhs, "left")
    ctx.e(asn, render_let_expr_python(ctx, step.value), "right")
    ctx.e(body_vid, asn, "child_of")

    factor_call = call(
        ctx,
        attribute(ctx, (*factor_namespace, factor_fn)),
        positional=(
            string_literal(ctx, step.name),
            identifier(ctx, step.name),
        ),
    )
    ctx.e(body_vid, factor_call, "child_of")


def _emit_python_return(
    ctx: PyCtx, body_vid: str, return_vars: tuple[str, ...]
) -> None:
    """Emit ``return <var>`` or ``return (<a>, <b>, ...)`` as a Python
    `return_statement` inside ``body_vid``.

    Tree-sitter Python represents a single-variable return as
    ``return_statement → identifier``; a tuple return wraps in an
    ``expression_list`` of children.
    """
    if not return_vars:
        return
    rs = ctx.v(ctx.fresh("ret"), "return_statement")
    if len(return_vars) == 1:
        ctx.e(rs, identifier(ctx, return_vars[0]), "child_of")
    else:
        elist = ctx.v(ctx.fresh("elist"), "expression_list")
        for var in return_vars:
            ctx.e(elist, identifier(ctx, var), "child_of")
        ctx.e(rs, elist, "child_of")
    ctx.e(body_vid, rs, "child_of")


# QVR family name → NumPyro distribution class name.
_FAMILIES: dict[str, str] = {
    "Normal": "Normal", "HalfNormal": "HalfNormal", "Cauchy": "Cauchy",
    "HalfCauchy": "HalfCauchy", "Bernoulli": "Bernoulli", "Beta": "Beta",
    "Categorical": "Categorical", "Dirichlet": "Dirichlet",
    "Exponential": "Exponential", "Gamma": "Gamma",
    "InverseGamma": "InverseGamma", "Laplace": "Laplace",
    "LogNormal": "LogNormal", "MultivariateNormal": "MultivariateNormal",
    "GP": "MultivariateNormal", "MatrixNormal": "MultivariateNormal",
    "Horseshoe": "Normal",
    "Poisson": "Poisson", "NegativeBinomial": "NegativeBinomial2",
    "Binomial": "Binomial", "Geometric": "Geometric",
    "Chi2": "Chi2", "ChiSquared": "Chi2",
    "ContinuousBernoulli": "ContinuousBernoulli",
    "FisherSnedecor": "FisherSnedecor",
    "Gumbel": "Gumbel", "Kumaraswamy": "Kumaraswamy",
    "LogitNormal": "LogitNormal",
    "RelaxedBernoulli": "RelaxedBernoulli",
    "RelaxedOneHotCategorical": "RelaxedOneHotCategorical",
    "TruncatedNormal": "TruncatedNormal",
    "LowRankMVN": "LowRankMultivariateNormal",
    "GeneralizedPareto": "GeneralizedPareto",
    "Wishart": "Wishart", "InverseWishart": "InverseWishart",
    "LKJ": "LKJ", "LKJCholesky": "LKJCholesky",
    "Pareto": "Pareto", "StudentT": "StudentT", "Uniform": "Uniform",
    "Weibull": "Weibull", "Gumbel": "Gumbel", "Chi2": "Chi2",
    "ContinuousBernoulli": "ContinuousBernoulli", "Wishart": "Wishart",
    "InverseWishart": "InverseWishart", "MatrixNormal": "MatrixNormal",
}


class _NumPyroWalker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        proto = target_protocol("python")
        sb = proto.schema()
        ctx = PyCtx(sb)

        ctx.v("mod", "module")
        program, _objects = _partition(module, "qvr-numpyro")

        samples, observes = _program_steps(program, "qvr-numpyro")

        body = ctx.v(ctx.fresh("body"), "block")
        func = function_def(
            ctx,
            name="model",
            default_params=tuple(o.var for o in observes),
            body_vid=body,
        )
        ctx.e("mod", func, "child_of")

        morphisms = build_morphism_table(module)
        lets = build_let_table(module)
        family_set = frozenset(_FAMILIES)
        for sam in samples:
            resolved = resolve_step_dist(
                sam.morphism, sam.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-numpyro",
            )
            for var in sam.vars:
                rhs = _numpyro_sample(
                    ctx, name=var, family=resolved.family,
                    args=resolved.args, obs_name=None,
                )
                ctx.e(body, assignment(ctx, lhs_name=var, rhs=rhs), "child_of")
        for body_step in program.draws:
            if isinstance(body_step, LetStep):
                _emit_python_let_step(ctx, body, body_step)
            elif isinstance(body_step, ScoreStep):
                _emit_python_score_step(ctx, body, body_step)
        for obs in observes:
            resolved = resolve_step_dist(
                obs.morphism, obs.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-numpyro",
            )
            call_expr = _numpyro_sample(
                ctx, name=obs.var, family=resolved.family,
                args=resolved.args, obs_name=obs.var,
            )
            ctx.e(body, call_expr, "child_of")

        # Emit `return <vars>` from the ProgramDecl's return_vars
        # (the parser stores the return on the declaration, not as
        # a separate ReturnStep in `draws`).
        _emit_python_return(ctx, body, tuple(program.return_vars))

        return sb.build()


def _partition(
    module: Module, target: str
) -> tuple[ProgramDecl, list[ObjectDecl]]:
    program: ProgramDecl | None = None
    objects: list[ObjectDecl] = []
    ignored_kinds: list[str] = []
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            if program is not None:
                raise UnsupportedConstruct(
                    target, ["multiple program_decl: backend transpiles one"]
                )
            program = stmt
        elif isinstance(stmt, ObjectDecl):
            objects.append(stmt)
        elif _ignorable(stmt):
            ignored_kinds.append(str(stmt.kind))
            continue
        else:
            raise UnsupportedConstruct(target, [str(stmt.kind)])
    if program is None:
        # No probabilistic program to transpile. When the module only
        # carries categorical-metadata declarations (composition_decl,
        # category_decl, schema_decl, ...), report those statement
        # kinds in the error so the construct-matrix test confirms
        # the construct cell is correctly rejected at this layer.
        kinds = ignored_kinds or ["no program_decl: nothing to transpile"]
        raise UnsupportedConstruct(target, kinds)
    return program, objects


def _program_steps(
    program: ProgramDecl, target: str
) -> tuple[list[SampleStep], list[ObserveStep]]:
    samples: list[SampleStep] = []
    observes: list[ObserveStep] = []
    for step in program.draws:
        if isinstance(step, SampleStep):
            samples.append(step)
        elif isinstance(step, ObserveStep):
            observes.append(step)
        elif isinstance(step, (LetStep, ScoreStep)):
            # LetStep and ScoreStep are emitted separately in the
            # walker body after the sample loop; skip here.
            continue
        else:
            raise UnsupportedConstruct(target, [f"step:{step.kind}"])
    return samples, observes


def _ignorable(stmt: Statement) -> bool:
    """Top-level statements consumed by other walker layers.

    - ``export_decl``: QVR-internal; no target emit required.
    - ``let_decl`` / ``morphism_decl``: consumed by the
      [`resolve_step_dist`][quivers.transpile.backends._resolve.resolve_step_dist]
      layer when a sample / observe step references the bound name.
      A declared morphism's ``~ Family(args)`` init clause becomes
      the resolved family + args for that step. A let-binding to a
      bare identifier resolves to whatever that identifier resolves
      to.
    """
    return str(stmt.kind) in {
        "export_decl",
        "let_decl",
        "morphism_decl",
        # Categorical metadata declarations: do not emit target-
        # language code; QVR-internal structural / algebraic info.
        "category_decl",
        "schema_decl",
        "composition_decl",
        "bundle_decl",
        "rule_decl",
        "contraction_decl",
        "signature_decl",
        "deduction_decl",
        "encoder_decl",
        "decoder_decl",
        "loss_decl",
    }


def _numpyro_sample(
    ctx: PyCtx,
    *,
    name: str,
    family: str,
    args: tuple[str | float, ...] | None,
    obs_name: str | None,
) -> str:
    dist_class = _FAMILIES.get(family)
    if dist_class is None:
        raise UnsupportedConstruct("qvr-numpyro", [f"family:{family}"])
    dist_callee = attribute(ctx, ("numpyro", "distributions", dist_class))
    dist_args = tuple(arg_expr(ctx, a) for a in (args or ()))
    dist_call = call(ctx, dist_callee, positional=dist_args)
    sample_callee = attribute(ctx, ("numpyro", "sample"))
    positional = (string_literal(ctx, name), dist_call)
    keyword: tuple[tuple[str, str], ...] = ()
    if obs_name is not None:
        keyword = (("obs", identifier(ctx, obs_name)),)
    return call(ctx, sample_callee, positional=positional, keyword=keyword)


@dx.codegen.emitter("qvr-numpyro")
class NumPyroEmitter:
    file_extension: str = "py"
    grammar: str = "python"
    support: frozenset[str] = STAN_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            f"qvr-numpyro emits instances, not classes; got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-numpyro", module, allow=STAN_LIKE)
        return realize(module, grammar="python", transform=_NumPyroWalker())


__all__ = ["NumPyroEmitter"]
