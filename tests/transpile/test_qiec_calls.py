"""What each target emits for a program that calls a module computation.

The plan carries a call the elaboration did not inline as an
[`IRCall`][quivers.transpile.ir.IRCall]. Every host-language target
places it in the model body against the callee's generated entry point
with a native operation table bound once per body; Stan defines a pure
callee as a user-defined function and calls it; BUGS and JAGS refuse the
call. These are text-level checks; the probe tier scores the same
programs against the reference machine.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile

DYNAMIC = ("pyro", "numpyro", "pymc", "edward2", "turing", "gen", "webppl", "church")

EFFECTFUL = """\
object Obs : FinSet 4

define noisy(x : Real) : Real !{random, score} =
    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))
    perform score.add(weight(-0.25 * y * y))
    return y

instance random : Random
instance score : Score

program prog : Obs -> Obs
    sample a <- Normal(0.0, 1.0)
    let c <- noisy(a)
    let d <- noisy(c)
    observe y : Obs <- Normal(d, 0.5)
    return d
export prog
"""

RECURSIVE = """\
object Obs : FinSet 4

define scale(x : Real, k : Int) : Real !{} =
    if k <= 0 then
        return x
    else
        let rest <- scale(x * 2.0, k - 1)
        return rest

program prog : Obs -> Obs
    sample a <- Normal(0.0, 1.0)
    let c <- scale(a, 3)
    observe y : Obs <- Normal(c, 0.5)
    return c
export prog
"""

SCORE = """\
object Obs : FinSet 4
program prog : Obs -> Obs
    sample x <- Normal(0.0, 1.0)
    score penalty = x * x
    return x
export prog
"""

CASE_TABLE = """\
object Class : FinSet 3
object Obs : FinSet 4
program prog : Obs -> Obs
    sample p <- Beta(1.0, 1.0)
    let leaf = factor cls : Class in { 0 -> log(p), 1 -> log(1.0 - p), 2 -> 0.0, }
    let head = leaf[0]
    observe y : Obs <- Normal(head, 0.5)
    return p
export prog
"""


@pytest.mark.parametrize("target", DYNAMIC)
def test_a_dynamic_target_calls_the_helper_under_one_operation_table(
    target: str,
) -> None:
    output = transpile(parse(EFFECTFUL), target=target).decode()
    # Both calls read the callee's entry point.
    calls = r"\(qiec_noisy " if target == "church" else r"qiec_noisy\("
    assert len(re.findall(calls, output)) >= 2
    # The table is bound once per body, at its first call, and read by
    # every call; the bridge defines the table's constructor once.
    table = (
        "_qvr-qiec-native-operations"
        if target == "church"
        else "_qvr_qiec_native_operations"
    )
    assert output.count(f"{table} ") + output.count(f"{table}(") == 2, output.count(
        table
    )


def test_static_targets_refuse_an_effectful_call_under_their_kinds() -> None:
    for target, kind in (
        ("bugs", "call:graph:noisy"),
        ("jags", "call:graph:noisy"),
        ("stan", "qiec:capability:perform:noisy"),
    ):
        with pytest.raises(UnsupportedConstruct) as caught:
            transpile(parse(EFFECTFUL), target=target)
        assert kind in caught.value.kinds
        assert "no statement that runs a computation" in str(caught.value) or (
            "perform" in str(caught.value)
        )


def test_stan_defines_a_recursive_pure_callee_as_a_function() -> None:
    output = transpile(parse(RECURSIVE), target="stan").decode()
    assert "real qiec_scale(real qv_x, int qv_k)" in output
    assert "if ((qv_k <= 0))" in output
    assert "qiec_scale((qv_x * 2.0), (qv_k - 1))" in output
    assert "real c = qiec_scale(a,3);" in output


@pytest.mark.requires_tool("stanc")
def test_stanc_accepts_the_recursive_function(tmp_path: pathlib.Path) -> None:
    stanc = shutil.which("stanc")
    assert stanc is not None
    source = tmp_path / "recursive.stan"
    source.write_bytes(transpile(parse(RECURSIVE), target="stan"))
    completed = subprocess.run(
        [stanc, "--o", str(tmp_path / "recursive.hpp"), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_bugs_and_jags_refuse_a_recursive_callee() -> None:
    for target in ("bugs", "jags"):
        with pytest.raises(UnsupportedConstruct) as caught:
            transpile(parse(RECURSIVE), target=target)
        assert "call:graph:scale" in caught.value.kinds


def test_stan_refuses_a_primitive_it_cannot_spell() -> None:
    source = RECURSIVE.replace(
        "let rest <- scale(x * 2.0, k - 1)\n        return rest",
        "let rest <- scale(x * 2.0, k - 1)\n        let n = int(rest)\n        return real(n)",
    )
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(source), target="stan")
    assert "qiec:stan:primitive:real_to_int:scale" in caught.value.kinds
    assert "no spelling in a Stan user-defined function" in str(caught.value)


def test_edward2_and_gen_trace_a_score_as_a_factor_choice() -> None:
    edward2 = transpile(parse(SCORE), target="edward2").decode()
    assert "class _QvrFactorDistribution" in edward2
    assert '_qvr_qiec_factor("penalty" ,penalty)' in edward2
    gen = transpile(parse(SCORE), target="gen").decode()
    assert "struct QvrFactorDist" in gen
    assert "@trace(_qvr_qiec_factor(penalty) , :qvr_factor => :penalty)" in gen
    church = transpile(parse(SCORE), target="church").decode()
    assert "(define penalty(* x x))(factor penalty)" in church


def test_gen_marginalizes_through_a_traced_factor() -> None:
    source = """\
object Item : FinSet 8
object Comp : FinSet 4
program prog : Item -> Item
    sample probs <- Dirichlet(1.0) [over=Comp]
    marginalize z : Comp <- Categorical(probs) [over=Item, reduction=logsumexp]
        observe r : Item <- Normal(0.0, 1.0) [via=idx]
    return probs
export prog
"""
    output = transpile(parse(source), target="gen").decode()
    assert "Distributions.logpdf.(Normal(0, 1) , r)" in output
    assert (
        "@trace(_qvr_qiec_factor(sum(__marg_z) ) , :qvr_factor => :__marg_z)" in output
    )


def test_graph_targets_write_a_plated_case_table_cell_by_cell() -> None:
    for target in ("bugs", "jags"):
        output = transpile(parse(CASE_TABLE), target=target).decode()
        assert "leaf[1] <- log(p)" in output
        assert "leaf[2] <- log(1-p)" in output
        assert "leaf[3] <- 0" in output
        assert "leaf[m_Class] <- c(" not in output


def test_julia_targets_read_a_scalar_binding_as_one_value() -> None:
    source = """\
object Resp : FinSet 5
object K : FinSet 3
program prog : Resp -> Resp
    sample mu : K <- Normal(0.0, 1.0)
    let m0 = mu[0]
    observe y : Resp <- Normal(m0, 0.5)
    return mu
export prog
"""
    turing = transpile(parse(source), target="turing").decode()
    assert "m0 = mu[1]" in turing
    assert "y ~ filldist(Normal(m0, 0.5) , 5)" in turing
    gen = transpile(parse(source), target="gen").decode()
    assert "@trace(normal(m0, 0.5) , (:y, m_Resp))" in gen
