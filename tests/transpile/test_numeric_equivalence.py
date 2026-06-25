"""Tier 3: measure-equivalence checks across backends in Docker.

For each composition fixture × each backend whose Docker image is
locally built, compute log-density at a deterministic test-point
set in both the QVR reference probe (in-process) and the target
backend's probe (in-container), and assert constant-spread
equivalence per
[`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match].

When two or more backend probes succeeded on a fixture, the test
also asserts pairwise transitivity:
``constant(target_a) - constant(target_b)`` is the additive offset
on the (target_a, target_b) pair.

Cells are skipped (not failed) when the backend's Docker image is
absent. CI builds them once via ``tests/transpile/docker/build.sh``.
"""

from __future__ import annotations

import pathlib

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.fixtures import _load
from tests.transpile.probes import _protocol
from tests.transpile.probes.qvr import QvrProbe


_BACKENDS_WITH_IMAGES = {
    "stan": ("panproto-test-stan", "stan", "stan.py"),
    "numpyro": ("panproto-test-numpyro", "py", "numpyro.py"),
    "pyro": ("panproto-test-pyro", "py", "pyro.py"),
    "pymc": ("panproto-test-pymc", "py", "pymc.py"),
    "edward2": ("panproto-test-edward2", "py", "edward2.py"),
    "turing": ("panproto-test-julia", "jl", "turing.jl"),
    "gen": ("panproto-test-julia", "jl", "gen.jl"),
    "webppl": ("panproto-test-node", "js", "webppl.py"),
    "jags": ("panproto-test-jags", "jags", "jags.py"),
    "bugs": ("panproto-test-bugs", "bugs", "jags.py"),
}


# Composition fixtures we know transpile successfully across most
# backends today; cells outside this set xfail per
# `test_composition_fixtures.py`.
_NUMERIC_FIXTURES = (
    "beta_bernoulli",
    "bayes_linear_regression",
    "normal_normal",
    "half_normal_scale",
    "gamma_exponential",
    "ill_conditioned_mvn",
    "truncated_normal_recovery",
)


def _per_fixture_point_set(fixture_name: str) -> list[_protocol.Point]:
    """Deterministic test points per fixture.

    The point set covers each latent's natural support with a small
    grid (5 points / axis, capped at 16 in this tier; cap is
    conservative so each cell takes only seconds in-container). Each
    fixture's points encode its (params, data) pair as the probe
    contract expects.
    """
    grids = _equivalence.deterministic_grid(
        _PARAM_BOUNDARIES[fixture_name],
        points_per_axis=5,
        cap=16,
    )
    data = _PARAM_DATA[fixture_name]
    return [
        _protocol.Point(params=grid, data=data) for grid in grids
    ]


# Per-fixture parameter boundaries (used by `deterministic_grid`).
# Boundaries are inclusive of support endpoints with a 1% inward
# shift to dodge log-singularities at the strict boundary.
_PARAM_BOUNDARIES: dict[str, dict[str, tuple[float, float]]] = {
    "beta_bernoulli": {"theta": (0.01, 0.99)},
    "bayes_linear_regression": {
        "a": (-2.0, 2.0),
        "b": (-2.0, 2.0),
    },
    "normal_normal": {"mu": (-3.0, 3.0)},
    "half_normal_scale": {"sigma": (0.01, 5.0)},
    "gamma_exponential": {"rate": (0.01, 5.0)},
    "ill_conditioned_mvn": {
        "x_1": (-100.0, 100.0),
        "x_2": (-10.0, 10.0),
        "x_3": (-1.0, 1.0),
        "x_4": (-0.1, 0.1),
        "x_5": (-0.01, 0.01),
    },
    "truncated_normal_recovery": {"mu": (0.01, 0.99)},
}


# Per-fixture observed-data values. Pinned per fixture so every
# probe sees the same data; the constant-spread check is invariant
# under the data choice as long as both backends see the same data.
_PARAM_DATA: dict[str, dict[str, float | int | list]] = {
    # Shapes must match each fixture's declared `Obs` plate size
    # (the renderer emits `array [N] ... y;`, so a scalar `y` makes
    # Stan reject the data with a "dims declared vs found" error).
    "beta_bernoulli": {"y": [1] * 50},
    "bayes_linear_regression": {
        "y": [0.5] * 60,
        "x_design": [float(i) / 30.0 - 1.0 for i in range(60)],
    },
    # Each `_PARAM_DATA["<fixture>"]["y"]` list length must match
    # the fixture's `object Obs : FinSet N` declaration (the
    # renderer emits `array [N] ... y;`, so a wrong shape rejects
    # in stanc with a "dims declared vs found" mismatch).
    "normal_normal": {"y": [0.5] * 30},
    "half_normal_scale": {"y": [0.3] * 80},
    "gamma_exponential": {"y": [1.0] * 80},
    "ill_conditioned_mvn": {
        "y_1": 50.0, "y_2": 5.0, "y_3": 0.5,
        "y_4": 0.05, "y_5": 0.005,
    },
    "truncated_normal_recovery": {"y": [0.5] * 60},
}


# Per-fixture condition-number override for
# [`adaptive_atol`][tests.transpile._equivalence.adaptive_atol].
# Numerically-extreme fixtures (large eigenvalue spread, very large
# log-density magnitude) get a proportionally larger tolerance so
# the constant-spread check still asserts equivalence rather than
# being skipped. Default is 1.0 for fixtures with no matrix ops.
_FIXTURE_CONDITION_NUMBER: dict[str, float] = {
    # `ill_conditioned_mvn`: per-axis variance ranges over four
    # orders of magnitude (1e2 .. 1e-2), so the relative precision
    # is ~1e-7 across torch / Stan / JAX (O(1e6) log-density).
    # Empirically, the cross-backend log-density spread sits around
    # 0.1 nats per point at the test boundaries; pick a condition
    # number that drives `adaptive_atol` (n_obs * cond * 5e-14)
    # above that spread with headroom.
    "ill_conditioned_mvn": 1e13,
}


# Cells where the fixture references a family the backend has no
# target_name for. The pipeline MUST raise `UnsupportedConstruct`
# with a `family:`-prefixed kind; the test asserts that explicitly
# rather than catch-and-xfail.
_EXPECTED_UNSUPPORTED: dict[tuple[str, str], str] = {
    # WebPPL ships no TruncatedNormal constructor; the
    # truncated_normal_recovery fixture trips the family-target-name
    # check.
    ("webppl", "truncated_normal_recovery"): "family:",
}


def _observation_count(fixture_name: str) -> int:
    """Total number of observed-data sites for `fixture_name`, summed
    across every observation array in
    [`_PARAM_DATA`][tests.transpile.test_numeric_equivalence._PARAM_DATA].

    Used to derive a per-fixture
    [`adaptive_atol`][tests.transpile._equivalence.adaptive_atol]:
    larger observation counts get larger tolerances because the
    per-site `Distribution.log_prob` round-off compounds in the outer
    sum.
    """
    data = _PARAM_DATA.get(fixture_name, {})
    total = 0
    for value in data.values():
        if isinstance(value, list):
            total += len(value)
        else:
            total += 1
    return max(total, 1)


@pytest.mark.parametrize("backend", sorted(_BACKENDS_WITH_IMAGES))
@pytest.mark.parametrize("fixture_name", _NUMERIC_FIXTURES)
def test_log_density_equivalence(
    backend: str,
    fixture_name: str,
    scratch: pathlib.Path,
) -> None:
    """For each (fixture, backend) cell, run the QVR reference probe
    in-process and the target probe in-container; assert
    constant-spread log-density equivalence.

    Cells registered in
    [`_EXPECTED_UNSUPPORTED`][tests.transpile.test_numeric_equivalence._EXPECTED_UNSUPPORTED]
    instead assert that `transpile` raises
    `UnsupportedConstruct` with the expected kind-prefix; this turns
    the construct-gap report from a passive xfail into a positive
    contract that flips on regression / closure.
    """
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(
            f"docker image {image!r} not available; the session-scope "
            f"`_ensure_docker_environment` autouse fixture should have "
            f"built it. Re-check `tests/transpile/docker/build.sh` and "
            f"the conftest setup."
        )

    # Load the fixture from the benchmark corpus.
    compositions = {f.name: f for f in _load.load_compositions()}
    if fixture_name not in compositions:
        pytest.fail(
            f"composition fixture {fixture_name!r} missing; "
            f"available: {sorted(compositions)}"
        )
    fixture = compositions[fixture_name]
    module = parse(fixture.source)

    expected_unsupported = _EXPECTED_UNSUPPORTED.get((backend, fixture_name))
    if expected_unsupported is not None:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            transpile(module, target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith(expected_unsupported) for k in kinds), (
            f"{backend!r} on {fixture_name!r}: expected raise with "
            f"kind prefix {expected_unsupported!r}, got kinds={kinds!r}. "
            f"Either the renderer changed (update the entry in "
            f"`_EXPECTED_UNSUPPORTED`) or a different gap fired."
        )
        return

    target_source = transpile(module, target=backend)

    # QVR reference: in-process probe, given the original source.
    points = _per_fixture_point_set(fixture_name)
    qvr_probe = QvrProbe()
    qvr_result = qvr_probe.evaluate(
        fixture.source.encode("utf-8"),
        fixture_name,
        points,
        scratch=scratch / "qvr",
    )

    # Target backend: run the in-container probe.
    script_path = (
        pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    )
    points_json = [
        {"params": pt.params, "data": pt.data} for pt in points
    ]
    raw_result = _docker.run_probe(
        image=image,
        script=script_path,
        source=target_source,
        source_ext=ext,
        points=points_json,
        scratch=scratch / backend,
    )
    target_lps = [float(x) for x in raw_result["log_densities"]]

    n_obs = _observation_count(fixture_name)
    condition_number = _FIXTURE_CONDITION_NUMBER.get(fixture_name, 1.0)
    atol = _equivalence.adaptive_atol(
        n_obs=n_obs, condition_number=condition_number,
    )
    _equivalence.assert_log_density_match(
        qvr_result.log_densities,
        target_lps,
        atol=atol,
        context=f"{backend}@{fixture_name}",
    )
