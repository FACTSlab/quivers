"""Program-plan regressions for checked collection computations."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from quivers.dsl import parse
from quivers.transpile import transpile


_DYNAMIC_TARGETS = (
    "pyro",
    "numpyro",
    "pymc",
    "edward2",
    "turing",
    "gen",
    "webppl",
    "church",
)

_ROOT = Path(__file__).resolve().parents[2]

_TRAVERSE_PROGRAM = """\
define shift(value : Real, offset : Real) : Real !{} =
    return value + offset

object Row : FinSet 3
object Value : Real 1

program shifted : Row -> Value
    sample offset <- Normal(0.0, 1.0)
    let locations <- traverse(
        [1.0, 2.0, 3.0],
        value -> shift(value, offset),
    )
    observe y : Row <- Normal(locations, 1.0)
    return offset

export shifted
"""

_INT_PROGRAM = """\
object Row : FinSet 1
object Value : Real 1

program cast_choice : Row -> Value
    let choice = int(observed_choice)
    return observed_choice

export cast_choice
"""


@pytest.mark.parametrize("target", _DYNAMIC_TARGETS)
def test_program_traverse_flattens_into_the_dynamic_target_plan(target: str) -> None:
    """A fixed traversal is ordinary sequenced QIEC, including in a program."""
    output = transpile(parse(_TRAVERSE_PROGRAM), target=target)
    assert output


@pytest.mark.parametrize(
    ("target", "spelling"),
    (
        ("pyro", "torch.as_tensor(observed_choice).long()"),
        ("numpyro", "jnp.int32(observed_choice)"),
        ("pymc", "pymc.pytensorf.pt.cast(observed_choice"),
        ("edward2", "tf.cast(observed_choice,tf.int32)"),
    ),
)
def test_python_targets_render_real_to_int(target: str, spelling: str) -> None:
    """The checked ``real_to_int`` primitive keeps its integer result type."""
    output = transpile(parse(_INT_PROGRAM), target=target).decode()
    ast.parse(output)
    assert spelling in output


@pytest.mark.parametrize("target", ("pyro", "numpyro"))
def test_lexical_uncertainty_rsa_transpiles_to_python_targets(target: str) -> None:
    """The released collection/RSA example crosses the complete boundary."""
    source = (_ROOT / "docs/examples/qiec/lexical-uncertainty-rsa.qvr").read_text()
    output = transpile(parse(source), target=target).decode()
    ast.parse(output)
    assert "qiec_specificity_listener" in output
    if target == "pyro":
        import pyro
        import torch

        namespace = {"pyro": pyro, "torch": torch}
        exec(output, namespace)
        model = pyro.condition(
            namespace["model"], data={"raw_alpha": torch.tensor(0.0)}
        )
        choices = torch.tensor([1.0] * 141 + [0.0] * 59)
        trace = pyro.poutine.trace(model).get_trace(
            chose_some_not_all=choices,
        )
        trace.compute_log_prob()
        assert float(trace.log_prob_sum()) == pytest.approx(-122.23291779, abs=1e-6)
        posterior = trace.nodes["_RETURN"]["value"]
        assert tuple(float(value) for value in posterior) == pytest.approx(
            (0.29586356, 0.70413644), abs=1e-6
        )
