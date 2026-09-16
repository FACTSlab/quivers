"""A program calling an effectful computation scores the same joint everywhere.

The `steps/call_step` fixture draws inside a helper under a site name
and scores a weight there. Every dynamic target renders the call
through its host runtime, with the helper's draw a site of the model
and its weight a term of the joint, so each target's log-density at a
clamped point must agree with the reference machine's up to a
constant. The static targets have no run-time call: BUGS and JAGS
refuse the call, and Stan refuses the helper's effects.
"""

from __future__ import annotations

import pathlib

import pytest

from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.qiec.program_runtime import run_program
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.fixtures import _load
from tests.transpile.probes import _protocol
from tests.transpile.probes.church import ChurchProbe
from tests.transpile.test_numeric_equivalence import _BACKENDS_WITH_IMAGES

_FIXTURE = "call_step"
_DATA: dict[str, list[float]] = {"y": [0.8, 0.7, 0.9, 1.0]}
_BOUNDARIES: dict[str, tuple[float, float]] = {
    "a": (-2.0, 2.0),
    "noise": (0.5, 3.5),
}
_DYNAMIC = ("pyro", "numpyro", "pymc", "edward2", "turing", "gen", "webppl")
_REFUSED: dict[str, str] = {
    "bugs": "call:graph:noisy",
    "jags": "call:graph:noisy",
    "stan": "qiec:capability:perform:noisy",
}


def _source() -> str:
    """The fixture's QVR text."""
    fixture = next(item for item in _load.load_steps() if item.name == _FIXTURE)
    return fixture.source


def _points() -> list[_protocol.Point]:
    """The clamped points, a grid over the model's and the helper's sites."""
    grids = _equivalence.deterministic_grid(_BOUNDARIES, points_per_axis=4, cap=16)
    points: list[_protocol.Point] = []
    for grid in grids:
        params: dict[str, float | int | list[float] | list[int]] = dict(grid)
        data: dict[str, float | int | list[float] | list[int]] = dict(_DATA)
        points.append(_protocol.Point(params=params, data=data))
    return points


def _reference(points: list[_protocol.Point]) -> list[float]:
    """The reference machine's joint at each point, every site replayed."""
    module = lower_qvr_to_qiec(parse(_source()), file_path=f"{_FIXTURE}.qvr")
    densities: list[float] = []
    for point in points:
        result = run_program(
            module,
            "prog",
            data={"y": tuple(_DATA["y"])},
            sites=dict(point.params),
        )
        densities.append(float(result.log_joint))
    return densities


@pytest.mark.parametrize("backend", _DYNAMIC)
def test_a_called_helper_scores_the_reference_joint(
    backend: str, scratch: pathlib.Path
) -> None:
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.image_available(image):
        raise RuntimeError(f"docker image {image!r} not available")
    points = _points()
    source = transpile(parse(_source()), target=backend)
    script = pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    raw = _docker.run_probe(
        image=image,
        script=script,
        source=source,
        source_ext=ext,
        points=[{"params": pt.params, "data": pt.data} for pt in points],
        scratch=scratch / backend,
    )
    _equivalence.assert_log_density_match(
        _reference(points),
        [float(value) for value in raw["log_densities"]],
        atol=_equivalence.adaptive_atol(n_obs=len(_DATA["y"])),
        context=f"{backend}@{_FIXTURE}",
    )


@pytest.mark.requires_probe("church")
def test_church_scores_a_called_helper_by_site_name(scratch: pathlib.Path) -> None:
    points = _points()
    source = transpile(parse(_source()), target="church")
    result = ChurchProbe().evaluate(
        source, _FIXTURE, points, scratch=scratch / "church"
    )
    _equivalence.assert_log_density_match(
        _reference(points),
        result.log_densities,
        atol=_equivalence.adaptive_atol(n_obs=len(_DATA["y"])),
        context=f"church@{_FIXTURE}",
    )


@pytest.mark.parametrize("backend", sorted(_REFUSED))
def test_static_targets_refuse_the_call_under_its_kind(backend: str) -> None:
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(_source()), target=backend)
    assert _REFUSED[backend] in caught.value.kinds
