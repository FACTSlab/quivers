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
from quivers.transpile.backends._resolve import (
    build_let_table,
    build_morphism_table,
    resolve_step_dist,
)


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
        morphisms = build_morphism_table(module)
        lets = build_let_table(module)
        family_set = frozenset(_FAMILIES)

        body = ctx.v(ctx.fresh("body"), "block")
        for sam in samples:
            resolved = resolve_step_dist(
                sam.morphism, sam.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-turing",
            )
            for var in sam.vars:
                rhs = _dist(ctx, family=resolved.family, args=resolved.args)
                ctx.e(body, tilde_assignment(ctx, ident(ctx, var), rhs))
        for obs in observes:
            resolved = resolve_step_dist(
                obs.morphism, obs.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-turing",
            )
            rhs = _dist(ctx, family=resolved.family, args=resolved.args)
            ctx.e(body, tilde_assignment(ctx, ident(ctx, obs.var), rhs))

        # `return <var>` clause: emit a return_statement at the end of
        # the model body for every return_var. Tuple returns become a
        # Julia tuple_expression.
        if program.return_vars:
            ret = ctx.v(ctx.fresh("ret"), "return_statement")
            if len(program.return_vars) == 1:
                ctx.e(ret, ident(ctx, program.return_vars[0]))
            else:
                tup = ctx.v(ctx.fresh("tup"), "tuple_expression")
                for var in program.return_vars:
                    ctx.e(tup, ident(ctx, var))
                ctx.e(ret, tup)
            ctx.e(body, ret)

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
