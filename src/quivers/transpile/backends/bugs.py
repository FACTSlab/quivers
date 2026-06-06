"""BUGS backend: QVR Module → BUGS source under the `bugs` tree-sitter
grammar.

Output shape:

    model {
        theta ~ dbeta(2, 2)
        y ~ dbern(theta)
    }

BUGS / WinBUGS / OpenBUGS share a `dnorm` / `dbern` / `dbeta` /
`dgamma` / ... distribution naming convention. Each ``sample`` /
``observe`` statement compiles to one ``stochastic_relation`` of the
form ``variable ~ distribution_call``.
"""

from __future__ import annotations

import didactic.api as dx
import panproto

from quivers.dsl.ast_nodes import LetStep, Module, ScoreStep
from quivers.transpile.backends._letexpr_bugs import render_let_expr_bugs
from quivers.transpile._api import STAN_LIKE, UnsupportedConstruct, unsupported_for
from quivers.transpile._pipeline import (
    SchemaTransform,
    realize,
    target_protocol,
)
from quivers.transpile.backends.numpyro import _partition, _program_steps
from quivers.transpile.backends._resolve import (
    build_let_table,
    build_morphism_table,
    resolve_step_dist,
)


# QVR family → BUGS distribution name. BUGS dialects vary slightly;
# WinBUGS / OpenBUGS / classic-BUGS all accept the d-prefix forms here.
_FAMILIES: dict[str, str] = {
    "Normal": "dnorm", "HalfNormal": "dnorm",
    "HalfCauchy": "dt",
    "Bernoulli": "dbern", "Beta": "dbeta",
    "Categorical": "dcat", "Dirichlet": "ddirch",
    "Exponential": "dexp", "Gamma": "dgamma", "Cauchy": "dt",
    "Uniform": "dunif", "Pareto": "dpar", "LogNormal": "dlnorm",
    "MultivariateNormal": "dmnorm", "Wishart": "dwish",
    "Chi2": "dchisqr",
    "GP": "dmnorm", "MatrixNormal": "dmnorm",
    "Horseshoe": "dnorm",
}


class _Ctx:
    def __init__(self, sb: panproto.SchemaBuilder) -> None:
        self._sb = sb
        self._n = 0

    def fresh(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}_{self._n}"

    def v(self, vid: str, kind: str) -> str:
        self._sb.vertex(vid, kind)
        return vid

    def e(self, src: str, tgt: str, kind: str = "child_of") -> None:
        self._sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)


def _ident(ctx: _Ctx, text: str) -> str:
    vid = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(vid, text)
    return vid


def _number(ctx: _Ctx, value: float) -> str:
    vid = ctx.v(ctx.fresh("num"), "number")
    text = str(int(value)) if value == int(value) else repr(value)
    ctx.lit(vid, text)
    return vid


def _arg(ctx: _Ctx, raw: str | float) -> str:
    if isinstance(raw, str):
        return _ident(ctx, raw)
    return _number(ctx, raw)


def _stochastic_relation(
    ctx: _Ctx,
    *,
    var: str,
    family: str,
    args: tuple[str | float, ...] | None,
    grammar: str,
) -> str:
    """Build ``<var> ~ <dist>(<args...>)`` as a stochastic_relation."""
    dist_name = _FAMILIES.get(family)
    if dist_name is None:
        raise UnsupportedConstruct(f"qvr-{grammar}", [f"family:{family}"])
    sr = ctx.v(ctx.fresh("sr"), "stochastic_relation")
    var_id = _ident(ctx, var)
    ctx.e(sr, var_id, "variable")
    dc = ctx.v(ctx.fresh("dc"), "distribution_call")
    dc_name = _ident(ctx, dist_name)
    ctx.e(dc, dc_name, "name")
    al = ctx.v(ctx.fresh("al"), "argument_list")
    for a in args or ():
        ctx.e(al, _arg(ctx, a))
    ctx.e(dc, al, "arguments")
    ctx.e(sr, dc, "distribution")
    return sr


def _build(module: Module, grammar: str) -> panproto.Schema:
    proto = target_protocol(grammar)
    sb = proto.schema()
    ctx = _Ctx(sb)

    ctx.v("src", "source_file")
    program, _ = _partition(module, f"qvr-{grammar}")
    samples, observes = _program_steps(program, f"qvr-{grammar}")
    morphisms = build_morphism_table(module)
    lets = build_let_table(module)
    family_set = frozenset(_FAMILIES)

    block = ctx.v(ctx.fresh("mb"), "model_block")
    ctx.e("src", block)
    for sam in samples:
        resolved = resolve_step_dist(
            sam.morphism, sam.args,
            morphisms=morphisms, lets=lets,
            family_registry=family_set, target=f"qvr-{grammar}",
        )
        for var in sam.vars:
            sr = _stochastic_relation(
                ctx, var=var, family=resolved.family, args=resolved.args,
                grammar=grammar,
            )
            ctx.e(block, sr)
    for obs in observes:
        resolved = resolve_step_dist(
            obs.morphism, obs.args,
            morphisms=morphisms, lets=lets,
            family_registry=family_set, target=f"qvr-{grammar}",
        )
        sr = _stochastic_relation(
            ctx, var=obs.var, family=resolved.family, args=resolved.args,
            grammar=grammar,
        )
        ctx.e(block, sr)
    for body_step in program.draws:
        if isinstance(body_step, LetStep):
            # BUGS / JAGS deterministic relation: `<name> <- <expr>`.
            asn = ctx.v(ctx.fresh("dr"), "deterministic_relation")
            ident_v = ctx.v(ctx.fresh("id"), "identifier")
            ctx.lit(ident_v, body_step.name)
            ctx.e(asn, ident_v)
            arrow = ctx.v(ctx.fresh("op"), "operator")
            ctx.lit(arrow, "<-")
            ctx.e(asn, arrow)
            ctx.e(asn, render_let_expr_bugs(ctx, body_step.value))
            ctx.e(block, asn)
        elif isinstance(body_step, ScoreStep):
            # ScoreStep in BUGS / JAGS uses the "zeros trick":
            # declare a dummy zero observation with a Poisson rate
            # equal to `-<score>` so the log-likelihood contribution
            # is `-(-<score>) = <score>`. This is the canonical
            # idiom in the BUGS book for arbitrary log-density
            # factors. First we bind the score value as a
            # deterministic relation, then we tie the trick.
            asn = ctx.v(ctx.fresh("dr"), "deterministic_relation")
            ident_v = ctx.v(ctx.fresh("id"), "identifier")
            ctx.lit(ident_v, body_step.name)
            ctx.e(asn, ident_v)
            arrow = ctx.v(ctx.fresh("op"), "operator")
            ctx.lit(arrow, "<-")
            ctx.e(asn, arrow)
            ctx.e(asn, render_let_expr_bugs(ctx, body_step.value))
            ctx.e(block, asn)
            # `__zero_<name> ~ dpois(-(<name>))` -- requires the
            # caller to declare `__zero_<name>` as observed at 0.
            sr = ctx.v(ctx.fresh("sr"), "stochastic_relation")
            lhs = ctx.v(ctx.fresh("id"), "identifier")
            ctx.lit(lhs, f"_zero_{body_step.name}")
            ctx.e(sr, lhs)
            tilde = ctx.v(ctx.fresh("op"), "operator")
            ctx.lit(tilde, "~")
            ctx.e(sr, tilde)
            dpois = ctx.v(ctx.fresh("call"), "function_call")
            fn = ctx.v(ctx.fresh("fn"), "identifier")
            ctx.lit(fn, "dpois")
            ctx.e(dpois, fn)
            neg = ctx.v(ctx.fresh("u"), "unary_expression")
            op = ctx.v(ctx.fresh("op"), "operator")
            ctx.lit(op, "-")
            ctx.e(neg, op)
            score_id = ctx.v(ctx.fresh("id"), "identifier")
            ctx.lit(score_id, body_step.name)
            ctx.e(neg, score_id)
            ctx.e(dpois, neg)
            ctx.e(sr, dpois)
            ctx.e(block, sr)

    return sb.build()


class _BugsWalker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        return _build(module, "bugs")


@dx.codegen.emitter("qvr-bugs")
class BugsEmitter:
    file_extension: str = "bugs"
    grammar: str = "bugs"
    support: frozenset[str] = STAN_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            f"qvr-bugs emits instances, not classes; got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-bugs", module, allow=STAN_LIKE)
        return realize(module, grammar="bugs", transform=_BugsWalker())


__all__ = ["BugsEmitter"]
