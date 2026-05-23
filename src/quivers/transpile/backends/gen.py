"""Gen.jl backend: QVR Module → Julia source under the `julia`
tree-sitter grammar.

Output shape:

    @gen function model(y)
        theta = @trace(beta(2, 2), :theta)
        y = @trace(bernoulli(theta), :y)
    end

Gen.jl wraps each random choice in ``@trace(distribution, address)``;
the trace address is the variable's name as a Julia symbol literal.
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
)
from quivers.transpile.backends.numpyro import _partition, _program_steps


# Gen.jl distribution constructors (lowercase by convention).
_FAMILIES: dict[str, str] = {
    "Normal": "normal", "Bernoulli": "bernoulli", "Beta": "beta",
    "Categorical": "categorical", "Dirichlet": "dirichlet",
    "Exponential": "exponential", "Gamma": "gamma", "Cauchy": "cauchy",
    "Laplace": "laplace", "LogNormal": "lognormal", "Uniform": "uniform",
    "Poisson": "poisson", "Geometric": "geometric",
}


def _assignment(ctx: JlCtx, lhs: str, rhs: str) -> str:
    """Build ``<lhs> = <rhs>`` (regular Julia assignment, not ``~``)."""
    # Tree-sitter Julia represents `x = y` as `assignment` with two
    # children. The compound_assignment_expression kind is for things
    # like `x ~ y` and `x += y`.
    asn = ctx.v(ctx.fresh("asn"), "assignment")
    ctx.e(asn, lhs)
    ctx.e(asn, ctx.v(ctx.fresh("eq"), "operator"))
    # The equals operator literal-value
    op_id = list(ctx._sb.build().vertices)  # noqa: SLF001  # unused; placeholder
    del op_id
    ctx.e(asn, rhs)
    return asn


# Gen.jl uses regular `=` so we need an `assignment` vertex with a
# `=` operator child. To avoid `_assignment` complexity, emit
# `compound_assignment_expression` with operator `=` since both forms
# share the same internal shape for tree-sitter Julia.
def _eq_assignment(ctx: JlCtx, lhs: str, rhs: str) -> str:
    asn = ctx.v(ctx.fresh("ca"), "compound_assignment_expression")
    op = ctx.v(ctx.fresh("op"), "operator")
    ctx.lit(op, "=")
    ctx.e(asn, lhs)
    ctx.e(asn, op)
    ctx.e(asn, rhs)
    return asn


def _symbol_literal(ctx: JlCtx, name: str) -> str:
    """Build a Julia symbol literal ``:<name>``."""
    # Tree-sitter Julia represents `:foo` as `quote_expression`
    # containing an `identifier`. The colon is implicit in the kind.
    q = ctx.v(ctx.fresh("qe"), "quote_expression")
    ctx.e(q, ident(ctx, name))
    return q


def _trace_call(
    ctx: JlCtx,
    *,
    family: str,
    args: tuple[str | float, ...] | None,
    address: str,
) -> str:
    """Build ``@trace(<dist>(args), :<address>)``."""
    dist_name = _FAMILIES.get(family)
    if dist_name is None:
        raise UnsupportedConstruct("qvr-gen", [f"family:{family}"])
    dist = call(ctx, ident(ctx, dist_name),
                positional=tuple(arg(ctx, a) for a in (args or ())))
    inner = call(ctx, ident(ctx, "trace"),
                 positional=(dist, _symbol_literal(ctx, address)))
    return macro_call(ctx, "trace", inner)


class _GenWalker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        proto = target_protocol("julia")
        sb = proto.schema()
        ctx = JlCtx(sb)

        ctx.v("src", "source_file")
        program, _ = _partition(module, "qvr-gen")
        samples, observes = _program_steps(program, "qvr-gen")

        body = ctx.v(ctx.fresh("body"), "block")
        for sam in samples:
            for var in sam.vars:
                rhs = _trace_call(ctx, family=sam.morphism,
                                  args=sam.args, address=var)
                ctx.e(body, _eq_assignment(ctx, ident(ctx, var), rhs))
        for obs in observes:
            rhs = _trace_call(ctx, family=obs.morphism,
                              args=obs.args, address=obs.var)
            ctx.e(body, _eq_assignment(ctx, ident(ctx, obs.var), rhs))

        fn = function_def(
            ctx, name="model",
            params=tuple(o.var for o in observes),
            body_vid=body,
        )
        mc = macro_call(ctx, "gen", fn)
        ctx.e("src", mc)
        return sb.build()


@dx.codegen.emitter("qvr-gen")
class GenEmitter:
    file_extension: str = "jl"
    grammar: str = "julia"
    support: frozenset[str] = STAN_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            f"qvr-gen emits instances, not classes; got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-gen", module, allow=STAN_LIKE)
        return realize(module, grammar="julia", transform=_GenWalker())


__all__ = ["GenEmitter"]
