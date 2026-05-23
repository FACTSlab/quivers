"""Turing.jl backend: QVR Module → Julia source under the `julia`
tree-sitter grammar.

Output shape:

    @model function model(y)
        theta ~ Beta(2, 2)
        y ~ Bernoulli(theta)
    end
"""

from __future__ import annotations

import didactic.api as dx
import panproto

from quivers.dsl.ast_nodes import Module
from quivers.transpile._api import STAN_LIKE, UnsupportedConstruct, unsupported_for
from quivers.transpile._pipeline import (
    SchemaTransform,
    realize,
    target_protocol,
)
from quivers.transpile.backends._juliahelpers import (
    JlCtx,
    arg,
    call,
    function_def,
    ident,
    macro_call,
    tilde_assignment,
)
from quivers.transpile.backends.numpyro import _partition, _program_steps


_FAMILIES: dict[str, str] = {
    "Normal": "Normal", "HalfNormal": "truncated", "Cauchy": "Cauchy",
    "HalfCauchy": "truncated", "Bernoulli": "Bernoulli", "Beta": "Beta",
    "Categorical": "Categorical", "Dirichlet": "Dirichlet",
    "Exponential": "Exponential", "Gamma": "Gamma",
    "InverseGamma": "InverseGamma", "Laplace": "Laplace",
    "LogNormal": "LogNormal", "MultivariateNormal": "MvNormal",
    "Pareto": "Pareto", "StudentT": "TDist", "Uniform": "Uniform",
    "Weibull": "Weibull",
}


class _TuringWalker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        proto = target_protocol("julia")
        sb = proto.schema()
        ctx = JlCtx(sb)

        ctx.v("src", "source_file")
        program, _ = _partition(module, "qvr-turing")
        samples, observes = _program_steps(program, "qvr-turing")

        body = ctx.v(ctx.fresh("body"), "block")
        for sam in samples:
            for var in sam.vars:
                rhs = _dist(ctx, family=sam.morphism, args=sam.args)
                ctx.e(body, tilde_assignment(ctx, ident(ctx, var), rhs))
        for obs in observes:
            rhs = _dist(ctx, family=obs.morphism, args=obs.args)
            ctx.e(body, tilde_assignment(ctx, ident(ctx, obs.var), rhs))

        fn = function_def(
            ctx, name="model",
            params=tuple(o.var for o in observes),
            body_vid=body,
        )
        mc = macro_call(ctx, "model", fn)
        ctx.e("src", mc)
        return sb.build()


def _dist(
    ctx: JlCtx,
    *,
    family: str,
    args: tuple[str | float, ...] | None,
) -> str:
    dist_name = _FAMILIES.get(family)
    if dist_name is None:
        raise UnsupportedConstruct("qvr-turing", [f"family:{family}"])
    return call(ctx, ident(ctx, dist_name),
                positional=tuple(arg(ctx, a) for a in (args or ())))


@dx.codegen.emitter("qvr-turing")
class TuringEmitter:
    file_extension: str = "jl"
    grammar: str = "julia"
    support: frozenset[str] = STAN_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            f"qvr-turing emits instances, not classes; got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-turing", module, allow=STAN_LIKE)
        return realize(module, grammar="julia", transform=_TuringWalker())


__all__ = ["TuringEmitter"]
