"""Church backend: QVR Module → Scheme source under the `scheme`
tree-sitter grammar.

Output shape:

    (define (model y)
      (let* ((theta (sample (beta 2 2))))
        (observe (bernoulli theta) y)
        y))

Scheme's tree-sitter grammar exposes only four vertex kinds (``list``,
``number``, ``symbol``, ``program``), so every Church construct boils
down to nested ``list`` vertices with ``symbol`` and ``number`` leaves.
"""

from __future__ import annotations

import didactic.api as dx
import panproto

from quivers.dsl.ast_nodes import Module
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


# QVR family → Church / Scheme distribution constructor symbol.
_FAMILIES: dict[str, str] = {
    "Normal": "gaussian", "HalfNormal": "gaussian", "Cauchy": "cauchy",
    "Bernoulli": "flip", "Beta": "beta", "Categorical": "categorical",
    "Dirichlet": "dirichlet", "Exponential": "exponential",
    "Gamma": "gamma", "Uniform": "uniform", "Pareto": "pareto",
    "LogNormal": "lognormal", "StudentT": "student-t",
    "MultivariateNormal": "multivariate-gaussian",
    "GP": "multivariate-gaussian", "MatrixNormal": "multivariate-gaussian",
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


def _sym(ctx: _Ctx, text: str) -> str:
    vid = ctx.v(ctx.fresh("sym"), "symbol")
    ctx.lit(vid, text)
    return vid


def _num(ctx: _Ctx, value: float) -> str:
    vid = ctx.v(ctx.fresh("num"), "number")
    text = str(int(value)) if value == int(value) else repr(value)
    ctx.lit(vid, text)
    return vid


def _list(ctx: _Ctx, children: tuple[str, ...]) -> str:
    """Build a parenthesised ``list`` vertex containing ``children``."""
    lst = ctx.v(ctx.fresh("lst"), "list")
    for c in children:
        ctx.e(lst, c)
    return lst


def _arg(ctx: _Ctx, raw: str | float) -> str:
    if isinstance(raw, str):
        return _sym(ctx, raw)
    return _num(ctx, raw)


def _dist(
    ctx: _Ctx,
    *,
    family: str,
    args: tuple[str | float, ...] | None,
) -> str:
    """Build ``(family arg1 arg2 ...)``."""
    sym = _FAMILIES.get(family)
    if sym is None:
        raise UnsupportedConstruct("qvr-church", [f"family:{family}"])
    return _list(ctx, (_sym(ctx, sym), *(_arg(ctx, a) for a in (args or ()))))


class _ChurchWalker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        proto = target_protocol("scheme")
        sb = proto.schema()
        ctx = _Ctx(sb)

        ctx.v("prog", "program")
        program, _ = _partition(module, "qvr-church")
        samples, observes = _program_steps(program, "qvr-church")
        morphisms = build_morphism_table(module)
        lets = build_let_table(module)
        family_set = frozenset(_FAMILIES)

        # body = sequence of (define theta ...) followed by (observe ...) calls
        # plus a final reference to the returned variable.
        body_forms: list[str] = []
        for sam in samples:
            resolved = resolve_step_dist(
                sam.morphism, sam.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-church",
            )
            for var in sam.vars:
                # (define <var> (sample (<family> args...)))
                define = _list(ctx, (
                    _sym(ctx, "define"),
                    _sym(ctx, var),
                    _list(ctx, (_sym(ctx, "sample"),
                                _dist(ctx, family=resolved.family,
                                      args=resolved.args))),
                ))
                body_forms.append(define)
        for obs in observes:
            resolved = resolve_step_dist(
                obs.morphism, obs.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-church",
            )
            # (observe (<family> args...) <var>)
            obs_form = _list(ctx, (
                _sym(ctx, "observe"),
                _dist(ctx, family=resolved.family, args=resolved.args),
                _sym(ctx, obs.var),
            ))
            body_forms.append(obs_form)
        return_vars = program.return_vars or ()
        for rv in return_vars:
            body_forms.append(_sym(ctx, rv))

        # Top-level: (define (model <obs...>) <body_forms...>)
        signature = _list(ctx, (
            _sym(ctx, "model"),
            *(_sym(ctx, o.var) for o in observes),
        ))
        top_define = _list(ctx, (_sym(ctx, "define"), signature, *body_forms))
        ctx.e("prog", top_define)

        return sb.build()


@dx.codegen.emitter("qvr-church")
class ChurchEmitter:
    file_extension: str = "scm"
    grammar: str = "scheme"
    support: frozenset[str] = CHURCH_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            f"qvr-church emits instances, not classes; got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-church", module, allow=CHURCH_LIKE)
        return realize(module, grammar="scheme", transform=_ChurchWalker())


__all__ = ["ChurchEmitter"]
