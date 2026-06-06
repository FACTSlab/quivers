"""Pyro backend: QVR Module → Python source under the `python`
tree-sitter grammar.

Structurally identical to the NumPyro backend: ``def model(<observed>
=None): ...`` with ``pyro.sample("<var>", <dist>[, obs=<var>])`` and
``pyro.distributions.<Family>(...)`` for distributions.
"""

from __future__ import annotations

import didactic.api as dx
import panproto

from quivers.dsl.ast_nodes import LetStep, Module, ReturnStep, ScoreStep
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
from quivers.transpile.backends.numpyro import (
    _emit_python_let_step,
    _emit_python_return,
    _emit_python_score_step,
    _partition,
    _program_steps,
)
from quivers.transpile.backends._resolve import (
    build_let_table,
    build_morphism_table,
    resolve_step_dist,
)


# Pyro carries the same distribution surface as NumPyro; share the map.
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
    "InverseWishart": "InverseWishart",
    "MatrixNormal": "MultivariateNormal",
    "GP": "MultivariateNormal",
    "Horseshoe": "Normal",
}


class _PyroWalker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        proto = target_protocol("python")
        sb = proto.schema()
        ctx = PyCtx(sb)

        ctx.v("mod", "module")
        program, _ = _partition(module, "qvr-pyro")
        samples, observes = _program_steps(program, "qvr-pyro")

        body = ctx.v(ctx.fresh("body"), "block")
        func = function_def(
            ctx, name="model",
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
                family_registry=family_set, target="qvr-pyro",
            )
            for var in sam.vars:
                rhs = _pyro_sample(ctx, name=var, family=resolved.family,
                                   args=resolved.args, obs_name=None)
                ctx.e(body, assignment(ctx, lhs_name=var, rhs=rhs), "child_of")
        for body_step in program.draws:
            if isinstance(body_step, LetStep):
                _emit_python_let_step(ctx, body, body_step)
            elif isinstance(body_step, ScoreStep):
                _emit_python_score_step(
                    ctx, body, body_step,
                    factor_namespace=("pyro",),
                )
        for obs in observes:
            resolved = resolve_step_dist(
                obs.morphism, obs.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-pyro",
            )
            call_expr = _pyro_sample(ctx, name=obs.var, family=resolved.family,
                                     args=resolved.args, obs_name=obs.var)
            ctx.e(body, call_expr, "child_of")

        _emit_python_return(ctx, body, tuple(program.return_vars))

        return sb.build()


def _pyro_sample(
    ctx: PyCtx,
    *,
    name: str,
    family: str,
    args: tuple[str | float, ...] | None,
    obs_name: str | None,
) -> str:
    dist_class = _FAMILIES.get(family)
    if dist_class is None:
        raise UnsupportedConstruct("qvr-pyro", [f"family:{family}"])
    dist_callee = attribute(ctx, ("pyro", "distributions", dist_class))
    dist_args = tuple(arg_expr(ctx, a) for a in (args or ()))
    dist_call = call(ctx, dist_callee, positional=dist_args)
    sample_callee = attribute(ctx, ("pyro", "sample"))
    positional = (string_literal(ctx, name), dist_call)
    keyword: tuple[tuple[str, str], ...] = ()
    if obs_name is not None:
        keyword = (("obs", identifier(ctx, obs_name)),)
    return call(ctx, sample_callee, positional=positional, keyword=keyword)


@dx.codegen.emitter("qvr-pyro")
class PyroEmitter:
    file_extension: str = "py"
    grammar: str = "python"
    support: frozenset[str] = STAN_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            f"qvr-pyro emits instances, not classes; got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-pyro", module, allow=STAN_LIKE)
        return realize(module, grammar="python", transform=_PyroWalker())


__all__ = ["PyroEmitter"]
