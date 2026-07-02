"""End-to-end DSL tests for the compositional measure-algebra
surface in QVR. Three layers:

1. Grammar — the new `family_call_arg` and `list_arg` productions
   parse and walk to the right tagged
   [`DrawArg`][quivers.dsl.ast_nodes.DrawArg] AST shapes.
2. Compiler dispatch — operator families (`PointMass`, `Restrict`,
   `Pushforward`, `Mixture`, `Independent`, `Normalize`) build the
   right composite distribution at observe / sample sites.
3. Sugar — `TruncatedNormal`, `HalfNormal`, etc. desugar to the
   canonical operator form at parse time; the resulting log-density
   is identical to the explicit operator-form program.
"""

from __future__ import annotations

import torch

from quivers.dsl import loads
from quivers.dsl.ast_nodes import (
    DrawArgDist,
    DrawArgList,
    DrawArgName,
    DrawArgScalar,
    ObserveStep,
)
from quivers.dsl.compiler.sugar import desugar_step
from quivers.dsl.parser import parse
from quivers.inference.trace import trace


# ---------------------------------------------------------------------------
# Parser walks the new node kinds to the right AST classes
# ---------------------------------------------------------------------------


def _first_observe(src: str) -> ObserveStep:
    ast = parse(src)
    for s in ast.statements:
        if hasattr(s, "draws"):
            for d in s.draws:
                if isinstance(d, ObserveStep):
                    return d
    raise AssertionError("no observe step found")


def test_parse_distribution_call_arg() -> None:
    src = """
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Restrict(Normal(0.0, 1.0), 0.0, 1.0)
    return y
export m
"""
    obs = _first_observe(src)
    assert obs.morphism == "Restrict"
    assert obs.args is not None
    assert isinstance(obs.args[0], DrawArgDist)
    assert obs.args[0].family == "Normal"
    assert isinstance(obs.args[1], DrawArgScalar)


def test_parse_list_arg() -> None:
    src = """
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Mixture([0.3, 0.7], [PointMass(0.0), Poisson(2.0)])
    return y
export m
"""
    obs = _first_observe(src)
    assert obs.morphism == "Mixture"
    assert obs.args is not None
    weights, components = obs.args
    assert isinstance(weights, DrawArgList)
    assert all(isinstance(w, DrawArgScalar) for w in weights.items)
    assert isinstance(components, DrawArgList)
    assert all(isinstance(c, DrawArgDist) for c in components.items)


def test_parse_nested_operator_calls() -> None:
    src = """
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Mixture([0.5, 0.5], [PointMass(0.0), Pushforward(Normal(0.0, 1.0), Exp)])
    return y
export m
"""
    obs = _first_observe(src)
    assert obs.morphism == "Mixture"
    components = obs.args[1]
    assert components.items[1].family == "Pushforward"
    push_args = components.items[1].args
    assert push_args[0].family == "Normal"
    # `Exp` without `()` is a bijector name (an identifier); the
    # compiler resolves it via the bijector registry at build time.
    assert isinstance(push_args[1], DrawArgName)
    assert push_args[1].text == "Exp"


# ---------------------------------------------------------------------------
# Operator dispatch — distributions trace to a finite log-joint
# ---------------------------------------------------------------------------


def test_pointmass_observe() -> None:
    program = loads("""
object Resp : FinSet 3
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- PointMass(0.0)
    return y
export m
""")
    y = torch.zeros(3)
    tr = trace(program.morphism, torch.zeros(3, 1), observations={"y": y})
    assert tr.log_joint is not None
    assert torch.isfinite(tr.log_joint).all()


def test_restrict_observe() -> None:
    program = loads("""
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Restrict(Normal(0.0, 1.0), 0.0, 1.0)
    return y
export m
""")
    y = torch.tensor([0.3, 0.5, 0.7, 0.2])
    tr = trace(program.morphism, torch.zeros(4, 1), observations={"y": y})
    assert tr.log_joint is not None
    assert torch.isfinite(tr.log_joint).all()


def test_pushforward_lognormal() -> None:
    program = loads("""
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Pushforward(Normal(0.0, 1.0), Exp)
    return y
export m
""")
    y = torch.tensor([0.5, 1.0, 2.0, 1.5])
    tr = trace(program.morphism, torch.zeros(4, 1), observations={"y": y})
    assert tr.log_joint is not None
    assert torch.isfinite(tr.log_joint).all()


def test_mixture_observe() -> None:
    program = loads("""
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Mixture([0.3, 0.7], [PointMass(0.0), Poisson(2.0)])
    return y
export m
""")
    y = torch.tensor([0.0, 1.0, 2.0, 0.0])
    tr = trace(program.morphism, torch.zeros(4, 1), observations={"y": y})
    assert tr.log_joint is not None
    assert torch.isfinite(tr.log_joint).all()


def test_mixture_pushforward_composition() -> None:
    program = loads("""
object Resp : FinSet 3
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Mixture([0.5, 0.5], [Pushforward(Normal(0.0, 1.0), Exp), Pushforward(Normal(1.0, 0.5), Exp)])
    return y
export m
""")
    y = torch.tensor([1.0, 2.0, 0.5])
    tr = trace(program.morphism, torch.zeros(3, 1), observations={"y": y})
    assert tr.log_joint is not None
    assert torch.isfinite(tr.log_joint).all()


# ---------------------------------------------------------------------------
# Sugar table desugars to operator form with identical log-density
# ---------------------------------------------------------------------------


def _trace_logjoint(src: str, y: torch.Tensor) -> torch.Tensor:
    program = loads(src)
    torch.manual_seed(0)
    tr = trace(
        program.morphism,
        torch.zeros(y.shape[0], 1),
        observations={"y": y},
    )
    return tr.log_joint


def test_truncated_normal_equals_restrict_normal() -> None:
    y = torch.tensor([0.3, 0.5, 0.7, 0.2])
    sugar = _trace_logjoint(
        """
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- TruncatedNormal(0.0, 1.0, 0.0, 1.0)
    return y
export m
""",
        y,
    )
    operator = _trace_logjoint(
        """
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Restrict(Normal(0.0, 1.0), 0.0, 1.0)
    return y
export m
""",
        y,
    )
    torch.testing.assert_close(sugar, operator, atol=1e-6, rtol=1e-6)


def test_half_normal_equals_restrict_normal_at_zero() -> None:
    y = torch.tensor([0.5, 1.0, 1.5, 0.2])
    sugar = _trace_logjoint(
        """
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- HalfNormal(2.0)
    return y
export m
""",
        y,
    )
    operator = _trace_logjoint(
        """
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Restrict(Normal(0.0, 2.0), 0.0)
    return y
export m
""",
        y,
    )
    torch.testing.assert_close(sugar, operator, atol=1e-6, rtol=1e-6)


def test_half_cauchy_equals_restrict_cauchy_at_zero() -> None:
    y = torch.tensor([0.5, 1.0, 1.5, 0.2])
    sugar = _trace_logjoint(
        """
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- HalfCauchy(2.5)
    return y
export m
""",
        y,
    )
    operator = _trace_logjoint(
        """
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Restrict(Cauchy(0.0, 2.5), 0.0)
    return y
export m
""",
        y,
    )
    torch.testing.assert_close(sugar, operator, atol=1e-6, rtol=1e-6)


def test_sugar_desugar_step_idempotent() -> None:
    """`desugar_step` must be idempotent: applying it twice yields the
    same step, because the first application produces operator-form
    morphisms and the table is keyed on sugar names only.
    """
    obs = ObserveStep(
        var="y",
        morphism="TruncatedNormal",
        args=(
            DrawArgScalar(value=0.0),
            DrawArgScalar(value=1.0),
            DrawArgScalar(value=0.0),
            DrawArgScalar(value=1.0),
        ),
    )
    once = desugar_step(obs)
    twice = desugar_step(once)
    assert once.morphism == "Restrict"
    assert twice.morphism == "Restrict"
    assert once.args == twice.args


def test_sugar_with_variable_arg_passes_through() -> None:
    """When a sugar family is called with a free variable (e.g.
    `HalfNormal(sigma)`), the desugaring leaves the step unchanged so
    the dedicated inline family entry handles it via the variable-
    binding path.
    """
    obs = ObserveStep(
        var="y",
        morphism="HalfNormal",
        args=(DrawArgName(text="sigma"),),
    )
    out = desugar_step(obs)
    assert out.morphism == "HalfNormal"
    assert out.args == obs.args


# ---------------------------------------------------------------------------
# Existing-surface compatibility (0.14.0 surface keeps working)
# ---------------------------------------------------------------------------


def test_zip_via_existing_inline_family() -> None:
    program = loads("""
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- ZeroInflatedPoisson(pi, rate)
    return y
export m
""")
    y = torch.tensor([0, 1, 2, 0]).long()
    pi = torch.tensor([0.3, 0.3, 0.3, 0.3])
    rate = torch.tensor([1.5, 1.5, 1.5, 1.5])
    tr = trace(
        program.morphism,
        torch.zeros(4, 1),
        observations={"y": y, "pi": pi, "rate": rate},
    )
    assert tr.log_joint is not None
    assert torch.isfinite(tr.log_joint).all()


def test_truncated_normal_with_literal_args_matches_explicit_operator() -> None:
    """All-literal TruncatedNormal call must desugar to Restrict and
    produce the same log-density. Free-variable TruncatedNormal calls
    keep the existing 0.14.0 inline family entry as the fallback.
    """
    y = torch.tensor([0.5])
    sugar = _trace_logjoint(
        """
object Resp : FinSet 1
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- TruncatedNormal(0.0, 1.0, -2.0, 2.0)
    return y
export m
""",
        y,
    )
    operator = _trace_logjoint(
        """
object Resp : FinSet 1
program m : Resp -> Resp
    sample foo <- Normal(0.0, 1.0)
    observe y : Resp <- Restrict(Normal(0.0, 1.0), -2.0, 2.0)
    return y
export m
""",
        y,
    )
    torch.testing.assert_close(sugar, operator, atol=1e-6, rtol=1e-6)
