"""PyMC backend: QVR Module → Python source under the `python`
tree-sitter grammar.

Output shape:

    with pymc.Model() as model:
        theta = pymc.Beta("theta", 2.0, 2.0)
        y = pymc.Bernoulli("y", theta, observed=y)

Each sample becomes a named PyMC RV inside a single ``with
pymc.Model() as model:`` context. Observed variables use the
``observed=<var>`` keyword instead of an obs-keyword sample.
"""

from __future__ import annotations

import didactic.api as dx
import panproto

from quivers.dsl.ast_nodes import LetStep, Module, ScoreStep
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
    with_statement,
)
from quivers.transpile.backends.numpyro import (
    _emit_python_let_step,
    _emit_python_return,
    _emit_python_score_step,
)
from quivers.transpile.backends.numpyro import _partition, _program_steps
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
    "LogNormal": "LogNormal", "MvNormal": "MvNormal",
    "MultivariateNormal": "MvNormal", "Pareto": "Pareto",
    "StudentT": "StudentT", "Uniform": "Uniform", "Weibull": "Weibull",
    "Gumbel": "Gumbel", "ChiSquared": "ChiSquared", "Chi2": "ChiSquared",
    "Wishart": "Wishart",
    "MatrixNormal": "MatrixNormal",
    "GP": "MvNormal",
    "Horseshoe": "Normal",
}


class _PyMCWalker(SchemaTransform):
    def forward(self, module: Module) -> panproto.Schema:  # type: ignore[override]
        proto = target_protocol("python")
        sb = proto.schema()
        ctx = PyCtx(sb)

        ctx.v("mod", "module")
        program, _ = _partition(module, "qvr-pymc")
        samples, observes = _program_steps(program, "qvr-pymc")
        morphisms = build_morphism_table(module)
        lets = build_let_table(module)
        family_set = frozenset(_FAMILIES)

        # Wrap the `with pymc.Model() as model: ...` block in a
        # `def model_fn(y=None): ...` function so we can attach a
        # module-level return clause. PyMC programs are commonly
        # wrapped this way (a single function that constructs the
        # model and returns it / the latent variables of interest).
        fn_body = ctx.v(ctx.fresh("fbody"), "block")
        model_call = call(ctx, attribute(ctx, ("pymc", "Model")))
        with_body = ctx.v(ctx.fresh("wbody"), "block")
        ws = with_statement(ctx, expression=model_call, alias="model",
                            body_vid=with_body)
        ctx.e(fn_body, ws, "child_of")

        for sam in samples:
            resolved = resolve_step_dist(
                sam.morphism, sam.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-pymc",
            )
            for var in sam.vars:
                rhs = _pymc_rv(ctx, name=var, family=resolved.family,
                               args=resolved.args, observed_name=None)
                ctx.e(with_body, assignment(ctx, lhs_name=var, rhs=rhs),
                      "child_of")
        for body_step in program.draws:
            if isinstance(body_step, LetStep):
                _emit_python_let_step(ctx, with_body, body_step)
            elif isinstance(body_step, ScoreStep):
                _emit_python_score_step(
                    ctx, with_body, body_step,
                    factor_namespace=("pymc",),
                    factor_fn="Potential",
                )
        for obs in observes:
            resolved = resolve_step_dist(
                obs.morphism, obs.args,
                morphisms=morphisms, lets=lets,
                family_registry=family_set, target="qvr-pymc",
            )
            rhs = _pymc_rv(ctx, name=obs.var, family=resolved.family,
                           args=resolved.args, observed_name=obs.var)
            ctx.e(with_body, assignment(ctx, lhs_name=obs.var, rhs=rhs),
                  "child_of")

        # Trailing `return <vars>` inside the function body (outside
        # the `with` block).
        if program.return_vars:
            _emit_python_return(ctx, fn_body, tuple(program.return_vars))

        fn = function_def(
            ctx, name="build_model",
            default_params=tuple(o.var for o in observes),
            body_vid=fn_body,
        )
        ctx.e("mod", fn, "child_of")

        return sb.build()


def _pymc_rv(
    ctx: PyCtx,
    *,
    name: str,
    family: str,
    args: tuple[str | float, ...] | None,
    observed_name: str | None,
) -> str:
    """Build ``pymc.<Family>("<name>", args, observed=<observed>)``."""
    dist_class = _FAMILIES.get(family)
    if dist_class is None:
        raise UnsupportedConstruct("qvr-pymc", [f"family:{family}"])
    callee = attribute(ctx, ("pymc", dist_class))
    positional = (string_literal(ctx, name),) + tuple(
        arg_expr(ctx, a) for a in (args or ())
    )
    keyword: tuple[tuple[str, str], ...] = ()
    if observed_name is not None:
        keyword = (("observed", identifier(ctx, observed_name)),)
    return call(ctx, callee, positional=positional, keyword=keyword)


@dx.codegen.emitter("qvr-pymc")
class PyMCEmitter:
    file_extension: str = "py"
    grammar: str = "python"
    support: frozenset[str] = STAN_LIKE

    def emit_class(self, cls: object) -> bytes:
        raise NotImplementedError(
            f"qvr-pymc emits instances, not classes; got cls={cls!r}"
        )

    def emit_instance(self, module: Module) -> bytes:  # type: ignore[override]
        unsupported_for("qvr-pymc", module, allow=STAN_LIKE)
        return realize(module, grammar="python", transform=_PyMCWalker())


__all__ = ["PyMCEmitter"]
