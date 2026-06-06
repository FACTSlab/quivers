"""WebPPL backend: QVR Module → JavaScript source under the
`javascript` tree-sitter grammar.

Output shape:

    function model(y){
      var theta = sample(Beta({a: 2, b: 2}));
      observe(Bernoulli({p: theta}), y);
      return y;
    }

WebPPL is a JavaScript subset with `sample` / `observe` / `factor` /
`Infer` primitives. Distributions take a single object literal (e.g.
``Beta({a: 2, b: 2})``), so each QVR family gets a positional →
keyword mapping.
"""

from __future__ import annotations

import didactic.api as dx
import panproto

from quivers.dsl.ast_nodes import LetStep, Module, ReturnStep
from quivers.transpile.backends._letexpr_javascript import (
    render_let_expr_javascript,
)
from quivers.transpile._api import CHURCH_LIKE, UnsupportedConstruct, unsupported_for
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


# QVR family → (WebPPL distribution constructor name, positional arg names).
_FAMILIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "Normal":       ("Gaussian", ("mu", "sigma")),
    # HalfNormal in QVR takes a single `sigma` argument (the half is
    # the support restriction, mu is fixed at 0). WebPPL's Gaussian
    # is the unrestricted normal; we emit it with mu=0 (kw not in args)
    # via the one-argument signature.
    "HalfNormal":   ("Gaussian", ("sigma",)),
    "HalfCauchy":   ("Cauchy", ("scale",)),
    "Beta":         ("Beta", ("a", "b")),
    "Bernoulli":    ("Bernoulli", ("p",)),
    "Categorical":  ("Categorical", ("ps",)),
    "Dirichlet":    ("Dirichlet", ("alpha",)),
    "Exponential":  ("Exponential", ("a",)),
    "Gamma":        ("Gamma", ("shape", "scale")),
    "Cauchy":       ("Cauchy", ("location", "scale")),
    "Laplace":      ("Laplace", ("location", "scale")),
    "LogNormal":    ("LogNormal", ("mu", "sigma")),
    "MultivariateNormal": ("MultivariateGaussian", ("mu", "cov")),
    # StudentT canonical parameterization is (nu, mu, sigma).
    "StudentT":     ("StudentT", ("nu", "mu", "sigma")),
    "Uniform":      ("Uniform", ("a", "b")),
    "GP":           ("MultivariateGaussian", ("mu", "cov")),
    "MatrixNormal": ("MultivariateGaussian", ("mu", "cov")),
    "Horseshoe":    ("Gaussian", ("mu", "sigma")),
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

    def e(self, src: str, tgt: str, kind: str) -> None:
        self._sb.edge(src, tgt, kind)

    def lit(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)


def _ident(ctx: _Ctx, text: str) -> str:
    vid = ctx.v(ctx.fresh("id"), "identifier")
    ctx.lit(vid, text)
    return vid


def _prop_ident(ctx: _Ctx, text: str) -> str:
    vid = ctx.v(ctx.fresh("pid"), "property_identifier")
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


def _object_literal(ctx: _Ctx, entries: tuple[tuple[str, str], ...]) -> str:
    """Build ``{k1: v1, k2: v2, ...}``."""
    obj = ctx.v(ctx.fresh("obj"), "object")
    for key, value_vid in entries:
        pair = ctx.v(ctx.fresh("pair"), "pair")
        ctx.e(pair, _prop_ident(ctx, key), "key")
        ctx.e(pair, value_vid, "value")
        ctx.e(obj, pair, "child_of")
    return obj


def _call(ctx: _Ctx, callee: str, positional: tuple[str, ...]) -> str:
    """Build ``callee(arg1, arg2, ...)``."""
    c = ctx.v(ctx.fresh("call"), "call_expression")
    args = ctx.v(ctx.fresh("args"), "arguments")
    ctx.e(c, callee, "function")
    ctx.e(c, args, "arguments")
    for pid in positional:
        ctx.e(args, pid, "child_of")
    return c


class _WebPPLWalker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        proto = target_protocol("javascript")
        sb = proto.schema()
        ctx = _Ctx(sb)

        ctx.v("prog", "program")
        program, _ = _partition(module, "qvr-webppl")
        samples, observes = _program_steps(program, "qvr-webppl")
        morphisms = build_morphism_table(module)
        lets = build_let_table(module)
        family_set = frozenset(_FAMILIES)

        # function model(obs1, obs2, ...) { body }
        fn = ctx.v(ctx.fresh("fn"), "function_declaration")
        fn_name = _ident(ctx, "model")
        params = ctx.v(ctx.fresh("ps"), "formal_parameters")
        body = ctx.v(ctx.fresh("body"), "statement_block")
        ctx.e(fn, fn_name, "name")
        ctx.e(fn, params, "parameters")
        ctx.e(fn, body, "body")
        for obs in observes:
            ctx.e(params, _ident(ctx, obs.var), "child_of")
        ctx.e("prog", fn, "child_of")

        for sam in samples:
            resolved = resolve_step_dist(
                sam.morphism, sam.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-webppl",
            )
            for var in sam.vars:
                rhs = _dist_call(ctx, "sample",
                                 family=resolved.family, args=resolved.args)
                # var theta = sample(Beta({...}));
                decl = ctx.v(ctx.fresh("vd"), "variable_declaration")
                d = ctx.v(ctx.fresh("dr"), "variable_declarator")
                ctx.e(d, _ident(ctx, var), "name")
                ctx.e(d, rhs, "value")
                ctx.e(decl, d, "child_of")
                ctx.e(body, decl, "child_of")

        for obs in observes:
            resolved = resolve_step_dist(
                obs.morphism, obs.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-webppl",
            )
            dist = _dist_call_inner(ctx, family=resolved.family,
                                    args=resolved.args)
            obs_call = _call(ctx, _ident(ctx, "observe"),
                             positional=(dist, _ident(ctx, obs.var)))
            stmt = ctx.v(ctx.fresh("es"), "expression_statement")
            ctx.e(stmt, obs_call, "child_of")
            ctx.e(body, stmt, "child_of")

        for let_step in program.draws:
            if isinstance(let_step, LetStep):
                decl = ctx.v(ctx.fresh("vd"), "variable_declaration")
                d = ctx.v(ctx.fresh("dr"), "variable_declarator")
                ctx.e(d, _ident(ctx, let_step.name), "name")
                ctx.e(d, render_let_expr_javascript(ctx, let_step.value), "value")
                ctx.e(decl, d, "child_of")
                ctx.e(body, decl, "child_of")

        _emit_webppl_return(ctx, body, tuple(program.return_vars))
        return sb.build()


def _emit_webppl_return(
    ctx: _Ctx, body_vid: str, return_vars: tuple[str, ...]
) -> None:
    """Emit ``return <var>;`` as a `return_statement` inside ``body_vid``.

    WebPPL's grammar is JavaScript; multi-var returns use a JS array
    expression (`return [a, b];``).
    """
    if not return_vars:
        return
    rs = ctx.v(ctx.fresh("ret"), "return_statement")
    if len(return_vars) == 1:
        ctx.e(rs, _ident(ctx, return_vars[0]), "child_of")
    else:
        arr = ctx.v(ctx.fresh("arr"), "array")
        for var in return_vars:
            ctx.e(arr, _ident(ctx, var), "child_of")
        ctx.e(rs, arr, "child_of")
    ctx.e(body_vid, rs, "child_of")
    return rs


def _dist_call(
    ctx: _Ctx,
    fn_name: str,
    *,
    family: str,
    args: tuple[str | float, ...] | None,
) -> str:
    """Build ``<fn_name>(<Family>({k:v, ...}))``."""
    dist = _dist_call_inner(ctx, family=family, args=args)
    return _call(ctx, _ident(ctx, fn_name), positional=(dist,))


def _dist_call_inner(
    ctx: _Ctx,
    *,
    family: str,
    args: tuple[str | float, ...] | None,
) -> str:
    spec = _FAMILIES.get(family)
    if spec is None:
        raise UnsupportedConstruct("qvr-webppl", [f"family:{family}"])
    dist_name, keys = spec
    raw = tuple(args or ())
    if len(raw) != len(keys):
        msg = (
            f"family:{family} expects {len(keys)} args ({keys}); "
            f"got {len(raw)}"
        )
        raise UnsupportedConstruct("qvr-webppl", [msg])
    entries = tuple((k, _arg(ctx, v)) for k, v in zip(keys, raw))
    obj = _object_literal(ctx, entries)
    return _call(ctx, _ident(ctx, dist_name), positional=(obj,))


@dx.codegen.emitter("qvr-webppl")
class WebPPLEmitter:
    file_extension: str = "js"
    grammar: str = "javascript"
    support: frozenset[str] = CHURCH_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            f"qvr-webppl emits instances, not classes; got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-webppl", module, allow=CHURCH_LIKE)
        return realize(module, grammar="javascript", transform=_WebPPLWalker())


__all__ = ["WebPPLEmitter"]
