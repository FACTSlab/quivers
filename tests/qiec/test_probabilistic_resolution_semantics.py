"""The literature example realizes the published finite resolution update."""

from __future__ import annotations

from pathlib import Path

import pytest

from quivers.dsl import Compiler, parse
from quivers.qiec import invoke_entry, run_named


_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = (
    _ROOT / "docs/examples/qiec/probabilistic-resolution-semantics.qvr"
).read_text()


@pytest.fixture(scope="module")
def module():
    compiler = Compiler(
        parse(_SOURCE),
        module_name="probabilistic_resolution_semantics",
        file_path="probabilistic-resolution-semantics.qvr",
    )
    compiler.compile()
    assert compiler.qiec_module is not None
    return compiler.qiec_module


def test_effect_surface_is_indexed_open_and_locally_handled(module) -> None:
    """The example exercises the effect features claimed by its documentation."""
    interface = next(item for item in module.effects if item.ref.name == "Resolve")
    assert [parameter.name for parameter in interface.telescope] == ["p"]

    unresolved = next(item for item in module.instances if item.name == "unresolved")
    request = next(item for item in module.computations if item.name == "request_loose")
    assert request.type.effects.lookup(unresolved.entry.instance) is not None
    assert request.type.effects.tail is not None
    assert request.type.effects.tail.proves_lacks(unresolved.entry.instance)

    grades = {
        handler.name: handler.clauses[0].grade.value for handler in module.handlers
    }
    assert grades == {"condition_nonexact": "omega", "select_exact": "1"}

    approximate = run_named(module, "approximate_likelihood", (480, 500))
    exact = run_named(module, "exact_likelihood", (500, 500))
    assert approximate.value == pytest.approx(0.25)
    assert exact.value == pytest.approx(1.0)
    assert [event.event for event in approximate.trace].count("resumption.invoked") == 2
    assert [event.event for event in exact.trace].count("resumption.invoked") == 1
    assert [event.event for event in approximate.trace].count("instance.allocated") == 1


def test_common_ground_updates_match_equations_49_through_53(module) -> None:
    """Uniform world priors expose the paper's semantic factors directly."""
    uniform = (1.0 / 7.0,) * 7
    approximate_run = run_named(module, "approximate_five", (uniform,))
    bare_run = run_named(module, "bare_five", (uniform,))
    approximate = approximate_run.value
    bare = bare_run.value

    assert approximate == pytest.approx(
        (0.0, 1.0 / 14.0, 2.0 / 7.0, 2.0 / 7.0, 2.0 / 7.0, 1.0 / 14.0, 0.0)
    )
    assert bare == pytest.approx((0.0, 0.05, 0.20, 0.50, 0.20, 0.05, 0.0))

    exported = invoke_entry(module, "hear_approximately_five", (uniform,))
    assert exported.entry.kind == "program"
    assert exported.value == pytest.approx(approximate)
    assert [event.event for event in exported.result.trace].count(
        "instance.allocated"
    ) == 7
    assert [event.event for event in exported.result.trace].count(
        "resumption.invoked"
    ) == 14
    assert [event.event for event in bare_run.trace].count("instance.allocated") == 14
    assert [event.event for event in bare_run.trace].count("resumption.invoked") == 21
