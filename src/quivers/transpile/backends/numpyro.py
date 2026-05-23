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
from quivers.transpile.backends._pyhelpers import (
    PyCtx,
    arg_expr,
    assignment,
    attribute,
    call,
    function_def,
    identifier,
    string_literal,
)


# QVR family name → NumPyro distribution class name.
_FAMILIES: dict[str, str] = {
    "Normal": "Normal", "HalfNormal": "HalfNormal", "Cauchy": "Cauchy",
    "HalfCauchy": "HalfCauchy", "Bernoulli": "Bernoulli", "Beta": "Beta",
    "Categorical": "Categorical", "Dirichlet": "Dirichlet",
    "Exponential": "Exponential", "Gamma": "Gamma",
    "InverseGamma": "InverseGamma", "Laplace": "Laplace",
    "LogNormal": "LogNormal", "MultivariateNormal": "MultivariateNormal",
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

        for sam in samples:
            for var in sam.vars:
                rhs = _numpyro_sample(
                    ctx, name=var, family=sam.morphism, args=sam.args,
                    obs_name=None,
                )
                ctx.e(body, assignment(ctx, lhs_name=var, rhs=rhs), "child_of")
        for obs in observes:
            call_expr = _numpyro_sample(
                ctx, name=obs.var, family=obs.morphism, args=obs.args,
                obs_name=obs.var,
            )
            ctx.e(body, call_expr, "child_of")

        return sb.build()


def _partition(
    module: Module, target: str
) -> tuple[ProgramDecl, list[ObjectDecl]]:
    program: ProgramDecl | None = None
    objects: list[ObjectDecl] = []
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
            continue
        else:
            raise UnsupportedConstruct(target, [str(stmt.kind)])
    if program is None:
        raise UnsupportedConstruct(
            target, ["no program_decl: nothing to transpile"]
        )
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
        elif isinstance(step, ReturnStep):
            continue
        else:
            raise UnsupportedConstruct(target, [f"step:{step.kind}"])
    return samples, observes


def _ignorable(stmt: Statement) -> bool:
    return str(stmt.kind) in {"export_decl", "let_decl"}


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
