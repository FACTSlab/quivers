"""Edward2 backend: QVR Module → Python source under the `python`
tree-sitter grammar.

Output shape: a plain ``def model(<observed>=None): ...`` whose body
constructs Edward2 random variables via
``edward2.<Family>(args, name="<name>")``. Observations are surfaced
by carrying them as default-None parameters; conditioning is the
caller's responsibility (Edward2's interceptor mechanism rather than
an `obs=` keyword).
"""

from __future__ import annotations

import didactic.api as dx
import panproto

from quivers.dsl.ast_nodes import Module, ReturnStep
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
    string_literal,
)
from quivers.transpile.backends.numpyro import (
    _emit_python_return,
    _partition,
    _program_steps,
)
from quivers.transpile.backends._resolve import (
    build_let_table,
    build_morphism_table,
    resolve_step_dist,
)


_FAMILIES: dict[str, str] = {
    "Normal": "Normal", "HalfNormal": "HalfNormal", "Cauchy": "Cauchy",
    "HalfCauchy": "HalfCauchy", "Bernoulli": "Bernoulli", "Beta": "Beta",
    "Categorical": "Categorical", "Dirichlet": "Dirichlet",
    "Exponential": "Exponential", "Gamma": "Gamma",
    "InverseGamma": "InverseGamma", "Laplace": "Laplace",
    "LogNormal": "LogNormal", "MultivariateNormalDiag": "MultivariateNormalDiag",
    "MultivariateNormal": "MultivariateNormalFullCovariance",
    "Pareto": "Pareto", "StudentT": "StudentT", "Uniform": "Uniform",
    "Wishart": "Wishart",
    "MatrixNormal": "MatrixNormalLinearOperator",
    "GP": "GaussianProcess",
}


class _Edward2Walker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        proto = target_protocol("python")
        sb = proto.schema()
        ctx = PyCtx(sb)

        ctx.v("mod", "module")
        program, _ = _partition(module, "qvr-edward2")
        samples, observes = _program_steps(program, "qvr-edward2")
        morphisms = build_morphism_table(module)
        lets = build_let_table(module)
        family_set = frozenset(_FAMILIES)

        body = ctx.v(ctx.fresh("body"), "block")
        func = function_def(
            ctx, name="model",
            default_params=tuple(o.var for o in observes),
            body_vid=body,
        )
        ctx.e("mod", func, "child_of")

        for sam in samples:
            resolved = resolve_step_dist(
                sam.morphism, sam.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-edward2",
            )
            for var in sam.vars:
                rhs = _ed_rv(ctx, name=var, family=resolved.family,
                             args=resolved.args)
                ctx.e(body, assignment(ctx, lhs_name=var, rhs=rhs), "child_of")
        for obs in observes:
            resolved = resolve_step_dist(
                obs.morphism, obs.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-edward2",
            )
            rhs = _ed_rv(ctx, name=obs.var, family=resolved.family,
                         args=resolved.args)
            ctx.e(body, assignment(ctx, lhs_name=obs.var, rhs=rhs), "child_of")

        _emit_python_return(ctx, body, tuple(program.return_vars))

        return sb.build()


def _ed_rv(
    ctx: PyCtx,
    *,
    name: str,
    family: str,
    args: tuple[str | float, ...] | None,
) -> str:
    """Build ``edward2.<Family>(args, name="<name>")``."""
    dist_class = _FAMILIES.get(family)
    if dist_class is None:
        raise UnsupportedConstruct("qvr-edward2", [f"family:{family}"])
    callee = attribute(ctx, ("edward2", dist_class))
    positional = tuple(arg_expr(ctx, a) for a in (args or ()))
    keyword = (("name", string_literal(ctx, name)),)
    return call(ctx, callee, positional=positional, keyword=keyword)


@dx.codegen.emitter("qvr-edward2")
class Edward2Emitter:
    file_extension: str = "py"
    grammar: str = "python"
    support: frozenset[str] = STAN_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            f"qvr-edward2 emits instances, not classes; got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-edward2", module, allow=STAN_LIKE)
        return realize(module, grammar="python", transform=_Edward2Walker())


__all__ = ["Edward2Emitter"]
