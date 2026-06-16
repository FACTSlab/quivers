"""Tier 6: composition fixtures (real-world programs) × every backend.

For each fixture in the inference-benchmark corpus
(`tests/benchmarks/models/*.qvr`) and every registered backend, the
test asserts a specific outcome per cell:

- Cells listed in `_KNOWN_COMPOSITION_GAPS` MUST raise
  `UnsupportedConstruct`. The walker fix is tracked outside the
  table; when fixed, the test FAILS so the operator removes the
  gap entry.
- All other cells MUST transpile to non-empty bytes.

Any silent UnsupportedConstruct (cell raises but isn't in the gap
table) is a real test failure; any unexpected silent empty-bytes
return is a real test failure.
"""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from tests.transpile.fixtures import _load


_BACKENDS = sorted(available_targets())


# Per-(fixture, backend) cells where the walker is known not to
# transpile the composition fixture today, with the reason. The
# entry pair is (fixture_stem, backend). When the walker grows the
# missing case, the strict test below fires so the entry is removed.
_KNOWN_COMPOSITION_GAPS: dict[tuple[str, str], str] = {
    # bayes_linear_regression / correlated_regression / etc. all
    # have `let mu = ...` (LetStep) and `observe ... <- ...` over a
    # composed expression; the walker doesn't emit LetStep yet.
    # Every Stan-family backend hits this on the same fixtures.
    **{
        (fixture, backend): (
            f"walker raises on LetStep / let-bound observation arg in "
            f"{fixture}"
        )
        for fixture in (
            "bayes_linear_regression",
            "correlated_regression",
            "eight_schools_noncentered",
            "neal_funnel",
            "normal_inverse_gamma",
        )
        for backend in (
            "stan", "numpyro", "pyro", "pymc",
            "church", "webppl", "turing", "bugs", "jags",
        )
    },
}


def _composition_fixtures() -> list[_load.Fixture]:
    return _load.load_compositions()


@pytest.mark.parametrize(
    "fixture", _composition_fixtures(), ids=lambda f: f.name
)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_composition_backend_cell(
    fixture: _load.Fixture, backend: str
) -> None:
    """Strict expectation per cell.

    Cells in `_KNOWN_COMPOSITION_GAPS` MUST raise with a
    `:`-tagged construct kind. Cells not in the gap table MUST
    transpile to non-empty bytes. A walker that closes a gap fires
    a strict test failure, so the gap entry is removed.
    """
    module = parse(fixture.source)
    cell = (fixture.name, backend)
    expected_gap = _KNOWN_COMPOSITION_GAPS.get(cell)

    if expected_gap is not None:
        try:
            output = transpile(module, target=backend)
        except UnsupportedConstruct as exc:
            kinds = exc.kinds
            assert any(":" in k for k in kinds), (
                f"{cell}: walker raised but error kinds {kinds!r} have "
                f"no `:`-tagged construct reference; the gap is no "
                f"longer the one described in `_KNOWN_COMPOSITION_GAPS`. "
                f"Update or remove the entry."
            )
        else:
            pytest.fail(
                f"{cell}: walker no longer raises "
                f"`UnsupportedConstruct`; gap closed. Remove the entry "
                f"from `_KNOWN_COMPOSITION_GAPS`. "
                f"(output: {output[:120]!r})"
            )
    else:
        output = transpile(module, target=backend)
        assert output, (
            f"{cell}: transpile returned empty bytes (expected non-empty "
            f"or an explicit `UnsupportedConstruct`)"
        )
