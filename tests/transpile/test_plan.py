"""The plan is derived from the checked module, not from the source steps."""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from quivers.dsl.ast_nodes.let_expressions import LetExprBinOp, LetExprVar
from quivers.transpile.ir import (
    CSIntegerInterval,
    DimStatic,
    IRArgBroadcast,
    IRArgRef,
    IRCall,
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRObserve,
    IRReturn,
    IRSample,
    IRScore,
    LetExprAffineMap,
)
from quivers.transpile.plan import Lower

REGRESSION = """\
object Resp : FinSet 5
program prog : Resp -> Resp
    sample alpha <- Normal(0.0, 5.0)
    sample beta <- Normal(0.0, 5.0)
    let mu = alpha + beta * x
    score penalty = -0.5 * alpha * alpha
    observe y : Resp <- Normal(mu, 1.0)
    return alpha
export prog
"""

GROUPED = """\
object Item : FinSet 8
object Comp : FinSet 4
program prog : Item -> Item
    sample probs <- Dirichlet(1.0) [over=Comp]
    marginalize z : Comp <- Categorical(probs) [over=Item, reduction=logsumexp]
        observe r : Item <- Normal(0.0, 1.0) [via=idx]
    return probs
export prog
"""

RELAXED = """\
object Resp : FinSet 6
object Out : Real 1
program prog : Out -> Out
    sample p <- Beta(2.0, 2.0)
    sample rate <- Gamma(2.0, 1.0)
    marginalize z : Resp <- ContinuousBernoulli(p)
        let gated = z * rate
        observe y : Resp <- Poisson(gated)
    return rate
export prog
"""

KERNEL = """\
object Obs : Real 2
morphism kernel : Obs -> Obs [role=kernel] ~ Normal
program prog : Obs -> Obs
    sample x <- kernel
    return x
export prog
"""


def _kinds(body: tuple[object, ...]) -> list[str]:
    """The node kinds of a plan body, in order."""
    return [type(node).__name__ for node in body]


def test_the_plan_reads_every_statement_off_the_computation() -> None:
    ir = Lower().forward(parse(REGRESSION))
    assert ir.name == "prog"
    assert ir.module.module
    assert _kinds(ir.body) == [
        "IRSample",
        "IRSample",
        "IRDeterministic",
        "IRScore",
        "IRObserve",
        "IRReturn",
    ]
    sample, _, mu, penalty, observe, returned = ir.body
    assert isinstance(sample, IRSample)
    assert sample.name == "alpha" and sample.family == "Normal"
    assert sample.arg_names == ("loc", "scale")
    assert isinstance(mu, IRDeterministic)
    assert mu.plate.batch_dims == (DimStatic(size=5, name="Resp"),)
    assert isinstance(penalty, IRScore)
    assert penalty.name == "penalty"
    assert isinstance(observe, IRObserve)
    assert observe.args == (IRArgRef(name="mu", indices=()), observe.args[1])
    assert observe.plate.batch_dims == (DimStatic(size=5, name="Resp"),)
    assert isinstance(returned, IRReturn)
    assert returned.names == ("alpha",)
    assert [(item.name, type(item.constraint).__name__) for item in ir.inputs] == [
        ("y", "CSReal"),
        ("x", "CSReal"),
    ]
    assert all(isinstance(item, IRDataInput) for item in ir.inputs)
    assert ir.inputs[1].plate.batch_dims == (DimStatic(size=5, name="Resp"),)


def test_a_grouped_marginalization_states_its_fibration_on_the_observation() -> None:
    ir = Lower().forward(parse(GROUPED))
    probs, marginal, _ = ir.body
    assert isinstance(probs, IRSample)
    assert probs.args == (
        IRArgBroadcast(value=probs.args[0].value, target_shape=(4,)),  # type: ignore[union-attr]
    )
    assert isinstance(marginal, IRMarginalize)
    assert marginal.latent == "z"
    assert marginal.plate.batch_dims == (DimStatic(size=8, name="Item"),)
    assert marginal.constraint == CSIntegerInterval(lower=0, upper=1)
    (observe,) = marginal.scope
    assert isinstance(observe, IRObserve)
    assert observe.via == "idx"
    fibration = next(item for item in ir.inputs if item.name == "idx")
    assert fibration.constraint == CSIntegerInterval(lower=0, upper=7)


def test_a_relaxed_latent_marginalizes_one_atom_pair_per_element() -> None:
    ir = Lower().forward(parse(RELAXED))
    marginal = next(node for node in ir.body if isinstance(node, IRMarginalize))
    assert marginal.family == "ContinuousBernoulli"
    assert marginal.plate.batch_dims == (DimStatic(size=6, name="Resp"),)
    gated, observe = marginal.scope
    assert isinstance(gated, IRDeterministic)
    assert gated.plate.batch_dims == (DimStatic(size=6, name="Resp"),)
    assert isinstance(observe, IRObserve)
    assert observe.via is None


def test_a_kernel_morphism_binds_its_heads_before_the_draw() -> None:
    ir = Lower().forward(parse(KERNEL))
    assert _kinds(ir.body) == [
        "IRDeterministic",
        "IRDeterministic",
        "IRDeterministic",
        "IRSample",
        "IRReturn",
    ]
    location, raw, scale, draw, _ = ir.body
    assert isinstance(location, IRDeterministic)
    assert location.name == "x_loc"
    assert isinstance(location.expr, LetExprAffineMap)
    assert location.expr.transform == "identity"
    assert isinstance(raw, IRDeterministic) and raw.name == "x_scale_raw"
    assert isinstance(scale, IRDeterministic) and scale.name == "x_scale"
    assert isinstance(draw, IRSample)
    assert draw.args == (
        IRArgRef(name="x_loc", indices=()),
        IRArgRef(name="x_scale", indices=()),
    )
    names = [item.name for item in ir.inputs]
    assert names == ["obs", "kernel_param_weight", "kernel_param_bias"]
    weight = ir.inputs[1]
    assert [
        dim.size for dim in weight.plate.batch_dims if isinstance(dim, DimStatic)
    ] == [4, 2]


def test_an_unknown_let_call_is_refused_under_its_kind() -> None:
    source = REGRESSION.replace("alpha + beta * x", "helper(alpha)")
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(source), target="stan")
    assert caught.value.kinds == ["let:call:unknown:helper"]


def test_a_program_template_call_is_refused_under_its_kind() -> None:
    source = """\
object School : FinSet 3
object Effect : Real 1
program effects(spread : Real, K : FinSet) : K -> Effect
    sample z : K <- Normal(0.0, 1.0)
    return z
program prog : School -> Effect
    sample theta <- effects(0.5, School)
    return theta
export prog
"""
    with pytest.raises(UnsupportedConstruct) as caught:
        Lower().forward(parse(source))
    assert caught.value.kinds[0].startswith("family:effects")


CALLS = """\
object Obs : FinSet 4

define shift(x : Real, k : Real) : Real !{} =
    return x + k

define noisy(x : Real) : Real !{random} =
    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))
    return y

instance random : Random

program prog : Obs -> Obs
    sample a <- Normal(0.0, 1.0)
    let b <- shift(a, 2.0)
    let c <- noisy(b)
    observe y : Obs <- Normal(c, 0.5)
    return c
export prog
"""


def test_a_pure_call_is_inlined_and_an_effectful_call_is_a_call() -> None:
    ir = Lower().forward(parse(CALLS))
    assert _kinds(ir.body) == [
        "IRSample",
        "IRDeterministic",
        "IRCall",
        "IRObserve",
        "IRReturn",
    ]
    _, shifted, call, observe, _ = ir.body
    assert isinstance(shifted, IRDeterministic)
    assert shifted.name == "b"
    assert isinstance(shifted.expr, LetExprBinOp)
    assert shifted.expr.op == "+"
    assert isinstance(call, IRCall)
    assert call.name == "c"
    assert call.callee == "noisy"
    assert call.static_arguments == ()
    assert call.arguments == (LetExprVar(name="b"),)
    assert call.random_instance.startswith("qiec:effect-instance:")
    assert call.score_instance.startswith("qiec:effect-instance:")
    assert call.plate.batch_dims == ()
    assert isinstance(observe, IRObserve)
    assert observe.args[0] == IRArgRef(name="c", indices=())
