"""NumPyro backend: QVR Module → Python source under the `python`
tree-sitter grammar, idiomatic for NumPyro.

The output is a single ``def model(<observed>=None): ...`` function
whose body is a sequence of ``numpyro.sample("<var>", <dist>)``
assignments for latents and ``numpyro.sample("<var>", <dist>, obs=<var>)``
for observations. Distributions live in
``numpyro.distributions.<Family>`` (fully qualified throughout to avoid
emitting imports).

Vertex kinds match Python's tree-sitter grammar:
``module``, ``function_definition``, ``parameters``, ``default_parameter``,
``block``, ``assignment``, ``call``, ``argument_list``,
``keyword_argument``, ``attribute``, ``identifier``, ``string``,
``string_start``, ``string_content``, ``string_end``, ``float``,
``integer``, ``none``.
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


# QVR family name → NumPyro distribution class name.
_FAMILIES: dict[str, str] = {
    "Normal": "Normal",
    "HalfNormal": "HalfNormal",
    "Cauchy": "Cauchy",
    "HalfCauchy": "HalfCauchy",
    "Bernoulli": "Bernoulli",
    "Beta": "Beta",
    "Categorical": "Categorical",
    "Dirichlet": "Dirichlet",
    "Exponential": "Exponential",
    "Gamma": "Gamma",
    "InverseGamma": "InverseGamma",
    "Laplace": "Laplace",
    "LogNormal": "LogNormal",
    "MultivariateNormal": "MultivariateNormal",
    "Pareto": "Pareto",
    "StudentT": "StudentT",
    "Uniform": "Uniform",
    "Weibull": "Weibull",
    "Gumbel": "Gumbel",
    "Chi2": "Chi2",
    "ContinuousBernoulli": "ContinuousBernoulli",
    "Wishart": "Wishart",
    "InverseWishart": "InverseWishart",
    "MatrixNormal": "MatrixNormal",
}


class _NumPyroWalker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        proto = target_protocol("python")
        sb = proto.schema()
        ctx = _PyCtx(sb)

        ctx.v("mod", "module")
        # Partition AST
        program: ProgramDecl | None = None
        objects: list[ObjectDecl] = []
        for stmt in module.statements:
            if isinstance(stmt, ProgramDecl):
                if program is not None:
                    raise UnsupportedConstruct(
                        "qvr-numpyro",
                        ["multiple program_decl: numpyro backend transpiles one"],
                    )
                program = stmt
            elif isinstance(stmt, ObjectDecl):
                objects.append(stmt)
            elif _ignorable(stmt):
                continue
            else:
                raise UnsupportedConstruct("qvr-numpyro", [str(stmt.kind)])

        if program is None:
            raise UnsupportedConstruct(
                "qvr-numpyro", ["no program_decl: nothing to transpile"]
            )

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
                raise UnsupportedConstruct("qvr-numpyro", [f"step:{step.kind}"])

        # Build def model(<obs1>=None, <obs2>=None, ...): <body>
        func = ctx.v("fn", "function_definition")
        fname = _identifier(ctx, "model")
        params = ctx.v("ps", "parameters")
        body = ctx.v("body", "block")
        ctx.e(func, fname, "name")
        ctx.e(func, params, "parameters")
        ctx.e(func, body, "body")
        ctx.e("mod", func, "child_of")

        for obs in observes:
            dp = ctx.v(ctx.fresh("dp"), "default_parameter")
            dp_name = _identifier(ctx, obs.var)
            dp_val = ctx.v(ctx.fresh("none"), "none")
            ctx.literal(dp_val, "None")
            ctx.e(dp, dp_name, "name")
            ctx.e(dp, dp_val, "value")
            ctx.e(params, dp, "child_of")

        for sam in samples:
            for var in sam.vars:
                # var = numpyro.sample("var", numpyro.distributions.<Family>(args))
                stmt = _assignment(
                    ctx,
                    lhs_name=var,
                    rhs=_numpyro_sample(
                        ctx, name=var, family=sam.morphism,
                        args=sam.args, obs_name=None,
                        target="qvr-numpyro",
                    ),
                )
                ctx.e(body, stmt, "child_of")

        for obs in observes:
            # numpyro.sample("var", numpyro.distributions.<Family>(args), obs=var)
            call_expr = _numpyro_sample(
                ctx, name=obs.var, family=obs.morphism,
                args=obs.args, obs_name=obs.var,
                target="qvr-numpyro",
            )
            ctx.e(body, call_expr, "child_of")

        return sb.build()


class _PyCtx:
    """Owns a SchemaBuilder + fresh-id counter for Python emission."""

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

    def literal(self, vid: str, text: str) -> None:
        self._sb.constraint(vid, "literal-value", text)


def _ignorable(stmt: Statement) -> bool:
    return str(stmt.kind) in {"export_decl", "let_decl"}


def _identifier(ctx: _PyCtx, text: str) -> str:
    vid = ctx.v(ctx.fresh("id"), "identifier")
    ctx.literal(vid, text)
    return vid


def _string_literal(ctx: _PyCtx, text: str) -> str:
    s = ctx.v(ctx.fresh("s"), "string")
    start = ctx.v(ctx.fresh("ss"), "string_start")
    ctx.literal(start, '"')
    content = ctx.v(ctx.fresh("sc"), "string_content")
    ctx.literal(content, text)
    end = ctx.v(ctx.fresh("se"), "string_end")
    ctx.literal(end, '"')
    ctx.e(s, start, "child_of")
    ctx.e(s, content, "child_of")
    ctx.e(s, end, "child_of")
    return s


def _number_literal(ctx: _PyCtx, value: float) -> str:
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        vid = ctx.v(ctx.fresh("int"), "integer")
        ctx.literal(vid, str(int(value)))
    else:
        vid = ctx.v(ctx.fresh("flt"), "float")
        ctx.literal(vid, repr(float(value)))
    return vid


def _attribute(ctx: _PyCtx, obj_chain: tuple[str, ...]) -> str:
    """Build a nested attribute access ``a.b.c.d``.

    Tree-sitter Python represents `a.b.c` as
    ``attribute(object: attribute(object: identifier 'a', attribute: 'b'),
                attribute: 'c')``. Built left-recursively from the chain.
    """
    if len(obj_chain) < 2:
        msg = f"_attribute needs at least 2 names; got {obj_chain!r}"
        raise ValueError(msg)
    current = _identifier(ctx, obj_chain[0])
    for attr_name in obj_chain[1:]:
        attr = ctx.v(ctx.fresh("attr"), "attribute")
        attr_id = _identifier(ctx, attr_name)
        ctx.e(attr, current, "object")
        ctx.e(attr, attr_id, "attribute")
        current = attr
    return current


def _call(
    ctx: _PyCtx,
    function: str,
    positional: tuple[str, ...] = (),
    keyword: tuple[tuple[str, str], ...] = (),
) -> str:
    """Build a `call` vertex with positional args and keyword args.

    ``function`` is the vertex id of the callee (e.g. an `attribute` or
    `identifier`). ``positional`` / ``keyword`` carry already-emitted
    expression vertex ids.
    """
    call = ctx.v(ctx.fresh("call"), "call")
    args = ctx.v(ctx.fresh("args"), "argument_list")
    ctx.e(call, function, "function")
    ctx.e(call, args, "arguments")
    for pid in positional:
        ctx.e(args, pid, "child_of")
    for name, vid in keyword:
        kw = ctx.v(ctx.fresh("kw"), "keyword_argument")
        kw_name = _identifier(ctx, name)
        ctx.e(kw, kw_name, "name")
        ctx.e(kw, vid, "value")
        ctx.e(args, kw, "child_of")
    return call


def _assignment(ctx: _PyCtx, *, lhs_name: str, rhs: str) -> str:
    asn = ctx.v(ctx.fresh("asn"), "assignment")
    lhs = _identifier(ctx, lhs_name)
    ctx.e(asn, lhs, "left")
    ctx.e(asn, rhs, "right")
    return asn


def _arg_expr(ctx: _PyCtx, raw: str | float) -> str:
    if isinstance(raw, str):
        return _identifier(ctx, raw)
    return _number_literal(ctx, raw)


def _numpyro_sample(
    ctx: _PyCtx,
    *,
    name: str,
    family: str,
    args: tuple[str | float, ...] | None,
    obs_name: str | None,
    target: str,
) -> str:
    """Build ``numpyro.sample("<name>", numpyro.distributions.<Family>(args), obs=<obs>)``."""
    dist_class = _FAMILIES.get(family)
    if dist_class is None:
        raise UnsupportedConstruct(target, [f"family:{family}"])

    # numpyro.distributions.<Family>(args)
    dist_callee = _attribute(ctx, ("numpyro", "distributions", dist_class))
    dist_args = tuple(_arg_expr(ctx, a) for a in (args or ()))
    dist_call = _call(ctx, dist_callee, positional=dist_args)

    # numpyro.sample("name", dist_call[, obs=...])
    sample_callee = _attribute(ctx, ("numpyro", "sample"))
    positional = (_string_literal(ctx, name), dist_call)
    keyword: tuple[tuple[str, str], ...] = ()
    if obs_name is not None:
        keyword = (("obs", _identifier(ctx, obs_name)),)
    return _call(ctx, sample_callee, positional=positional, keyword=keyword)


@dx.codegen.emitter("qvr-numpyro")
class NumPyroEmitter:
    """NumPyro backend, registered as the ``qvr-numpyro`` emitter."""

    file_extension: str = "py"
    grammar: str = "python"
    support: frozenset[str] = STAN_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            "qvr-numpyro emits instances (a parsed Module), not classes; "
            f"got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-numpyro", module, allow=STAN_LIKE)
        return realize(module, grammar="python", transform=_NumPyroWalker())


__all__ = ["NumPyroEmitter"]
