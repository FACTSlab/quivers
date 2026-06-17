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
#
# The rigor-swarm closure of LetStep / let-bound-observation gaps on
# numpyro / pyro / pymc / church / turing / bugs / jags (commit
# series after b043265) collapsed most of the table. The 14 entries
# that remain divide into two classes:
#
# 1. `return:undeclared:<var>` on Stan for fixtures that return a
#    let-bound deterministic. The Stan generated-quantities aliaser
#    needs the IR-walker to declare the let-bound name as a
#    transformed-parameter before the return statement; the walker
#    declares it inline at use site, so the generated-quantities
#    pass cannot find a declared shape.
#
# 2. `family:<F>:<backend>` for families a target legitimately does
#    not ship: Stan has no TruncatedNormal native, Gen / Church /
#    BUGS / JAGS / WebPPL have no InverseGamma / TruncatedNormal.
#    Each of these is a family_meta-level decision (a target_name
#    can be added if we add the corresponding renderer-side
#    construction recipe) rather than a renderer-walker bug.
_KNOWN_COMPOSITION_GAPS: dict[tuple[str, str], str] = {
    ("bayes_linear_regression", "stan"): (
        "return:undeclared:mu -- Stan return-alias needs let-bound "
        "name declared as transformed_parameter before return"
    ),
    ("correlated_regression", "stan"): (
        "return:undeclared:mu -- same"
    ),
    ("eight_schools_noncentered", "stan"): (
        "return:undeclared:theta -- same"
    ),
    ("normal_inverse_gamma", "bugs"): (
        "family:InverseGamma: no BUGS target name (BUGS has no native "
        "InverseGamma; a `dgamma`-on-precision recipe could close it)"
    ),
    ("normal_inverse_gamma", "church"): (
        "family:InverseGamma:church (Church has no InverseGamma)"
    ),
    ("normal_inverse_gamma", "gen"): (
        "family:InverseGamma: no Gen.jl target name"
    ),
    ("normal_inverse_gamma", "jags"): (
        "family:no-target-name:InverseGamma"
    ),
    ("normal_inverse_gamma", "webppl"): (
        "family:no-webppl-target:InverseGamma"
    ),
    ("truncated_normal_recovery", "bugs"): (
        "family:TruncatedNormal: no BUGS target name"
    ),
    ("truncated_normal_recovery", "church"): (
        "family:TruncatedNormal:church"
    ),
    ("truncated_normal_recovery", "gen"): (
        "family:TruncatedNormal: no Gen.jl target name"
    ),
    ("truncated_normal_recovery", "jags"): (
        "family:no-target-name:TruncatedNormal"
    ),
    ("truncated_normal_recovery", "stan"): (
        "family:no-stan-target:TruncatedNormal -- Stan has no native "
        "TruncatedNormal; the Stan idiom is `T[lo, hi]` on a "
        "vanilla `Normal` LHS declaration"
    ),
    ("truncated_normal_recovery", "webppl"): (
        "family:no-webppl-target:TruncatedNormal"
    ),
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
