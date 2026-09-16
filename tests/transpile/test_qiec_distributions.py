"""First-class distributions score identically on every host runtime.

A QIEC ``log_prob(Family(...), x)`` is stable data before lowering; each
dynamic target spells the family in its own library with its own
parameterization. These tests run the same constructions through every
host and hold each host's log density to the torch density the semantic
registry is checked against, so a target's spelling of a parameter (a
rate against a scale, a complemented probability, a shifted support)
cannot drift from the family's meaning.
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
from collections.abc import Sequence

import pytest
import torch

from quivers.dsl import parse
from quivers.transpile import transpile
from quivers.transpile.family_meta import FAMILY_META
from quivers.transpile.family_spelling import can_spell
from tests.transpile._tools import require_scheme
from tests.transpile._docker import image_available, run_probe_script

_SCRIPTS = pathlib.Path(__file__).parent / "probes" / "_scripts"

SCHEME_EXECUTABLE = require_scheme()


class Case:
    """One family construction scored at several points.

    Parameters
    ----------
    entry : str
        The QIEC computation's name.
    family : str
        The family constructed.
    parameter : str
        The QVR type of the scored point.
    construction : str
        The QVR expression constructing the family.
    arguments : dict[str, object]
        The construction's parameters as torch keyword arguments.
    points : Sequence[object]
        The values scored, as the QIEC entry point receives them.
    """

    def __init__(
        self,
        entry: str,
        family: str,
        parameter: str,
        construction: str,
        arguments: dict[str, object],
        points: Sequence[object],
    ) -> None:
        self.entry = entry
        self.family = family
        self.parameter = parameter
        self.construction = construction
        self.arguments = arguments
        self.points = points

    def source(self) -> str:
        """The QVR computation scoring the construction.

        Returns
        -------
        str
            ``define <entry>(x : <parameter>) : LogWeight !{} = ...``.
        """
        return (
            f"define {self.entry}(x : {self.parameter}) : LogWeight !{{}} =\n"
            f"    return log_prob({self.construction}, x)\n"
        )

    def expected(self, point: object) -> float:
        """The registry's density at one point.

        Parameters
        ----------
        point : object
            A scored value.

        Returns
        -------
        float
            The torch log density.
        """
        distribution = FAMILY_META[self.family].distribution_class(
            **{
                name: torch.as_tensor(value, dtype=torch.float64)
                for name, value in self.arguments.items()
            }
        )
        return float(distribution.log_prob(torch.as_tensor(point, dtype=torch.float64)))


CASES: tuple[Case, ...] = (
    Case(
        "normal",
        "Normal",
        "Real",
        "Normal(0.5, 2.0)",
        {"loc": 0.5, "scale": 2.0},
        (0.0, 1.7, -3.2),
    ),
    Case(
        "half_normal",
        "HalfNormal",
        "Real",
        "HalfNormal(1.5)",
        {"scale": 1.5},
        (0.3, 2.5),
    ),
    Case(
        "gamma",
        "Gamma",
        "Real",
        "Gamma(2.0, 3.0)",
        {"concentration": 2.0, "rate": 3.0},
        (0.4, 1.3),
    ),
    Case(
        "exponential",
        "Exponential",
        "Real",
        "Exponential(1.5)",
        {"rate": 1.5},
        (0.2, 2.0),
    ),
    Case(
        "beta",
        "Beta",
        "Real",
        "Beta(2.0, 3.0)",
        {"concentration1": 2.0, "concentration0": 3.0},
        (0.25, 0.7),
    ),
    Case(
        "uniform",
        "Uniform",
        "Real",
        "Uniform(-1.0, 2.0)",
        {"low": -1.0, "high": 2.0},
        (0.0, 1.5),
    ),
    Case(
        "student_t",
        "StudentT",
        "Real",
        "StudentT(4.0, 0.5, 2.0)",
        {"df": 4.0, "loc": 0.5, "scale": 2.0},
        (0.0, 3.0),
    ),
    Case(
        "cauchy",
        "Cauchy",
        "Real",
        "Cauchy(0.5, 2.0)",
        {"loc": 0.5, "scale": 2.0},
        (0.0, 4.0),
    ),
    Case(
        "log_normal",
        "LogNormal",
        "Real",
        "LogNormal(0.5, 0.8)",
        {"loc": 0.5, "scale": 0.8},
        (0.7, 3.0),
    ),
    Case(
        "laplace",
        "Laplace",
        "Real",
        "Laplace(0.5, 2.0)",
        {"loc": 0.5, "scale": 2.0},
        (0.0, 3.0),
    ),
    Case(
        "weibull",
        "Weibull",
        "Real",
        "Weibull(2.0, 1.5)",
        {"scale": 2.0, "concentration": 1.5},
        (0.5, 2.5),
    ),
    Case(
        "bernoulli",
        "Bernoulli",
        "Bool",
        "Bernoulli(0.3)",
        {"probs": 0.3},
        (True, False),
    ),
    Case(
        "categorical",
        "Categorical",
        "Int",
        "Categorical([0.2, 0.3, 0.5])",
        {"probs": [0.2, 0.3, 0.5]},
        (0, 2),
    ),
    Case("poisson", "Poisson", "Int", "Poisson(2.5)", {"rate": 2.5}, (0, 3)),
    Case(
        "binomial",
        "Binomial",
        "Int",
        "Binomial(10, 0.3)",
        {"total_count": 10, "probs": 0.3},
        (2, 7),
    ),
    Case(
        "negative_binomial",
        "NegativeBinomial",
        "Int",
        "NegativeBinomial(5.0, 0.4)",
        {"total_count": 5.0, "probs": 0.4},
        (0, 4),
    ),
    Case("geometric", "Geometric", "Int", "Geometric(0.3)", {"probs": 0.3}, (0, 3)),
    Case(
        "dirichlet",
        "Dirichlet",
        "Tensor[Real]([3])",
        "Dirichlet([1.0, 2.0, 3.0])",
        {"concentration": [1.0, 2.0, 3.0]},
        ([0.2, 0.3, 0.5], [0.6, 0.1, 0.3]),
    ),
    Case(
        "mvn",
        "MultivariateNormal",
        "Tensor[Real]([2])",
        "MultivariateNormal([0.0, 1.0], [[2.0, 0.5], [0.5, 1.0]])",
        {"loc": [0.0, 1.0], "covariance_matrix": [[2.0, 0.5], [0.5, 1.0]]},
        ([0.3, 0.8], [-1.0, 2.0]),
    ),
)

_BACKENDS: dict[str, tuple[str, str]] = {
    "pyro": ("panproto-test-pyro", "python"),
    "numpyro": ("panproto-test-numpyro", "python"),
    "pymc": ("panproto-test-pymc", "python"),
    "edward2": ("panproto-test-edward2", "python"),
    "turing": ("panproto-test-julia", "julia"),
    "gen": ("panproto-test-julia", "julia"),
}

_EXTENSIONS = {"python": "py", "julia": "jl"}

_TOLERANCES = {
    "pyro": 1e-4,
    "numpyro": 1e-4,
    "edward2": 1e-4,
    "pymc": 1e-7,
    "turing": 1e-9,
    "gen": 1e-9,
}


def _cases_for(target: str) -> tuple[Case, ...]:
    """The cases a target can spell.

    Parameters
    ----------
    target : str
        The dynamic target.

    Returns
    -------
    tuple[Case, ...]
        Every case whose family the target spells.
    """
    return tuple(case for case in CASES if can_spell(target, case.family))


def _source(cases: Sequence[Case]) -> str:
    """One QVR module holding every case's computation.

    Parameters
    ----------
    cases : Sequence[Case]
        The cases.

    Returns
    -------
    str
        The module text, with a site-valued computation appended.
    """
    return "".join(case.source() for case in cases) + (
        'define label() : Site[Real] !{} =\n    return site("x")\n'
    )


def _calls(cases: Sequence[Case]) -> list[list[object]]:
    """The entry-point calls a probe makes, one per point.

    Parameters
    ----------
    cases : Sequence[Case]
        The cases.

    Returns
    -------
    list[list[object]]
        ``[entry, [argument]]`` pairs, the label call last.
    """
    calls: list[list[object]] = [
        [case.entry, [point]] for case in cases for point in case.points
    ]
    calls.append(["label", []])
    return calls


def _check(cases: Sequence[Case], results: Sequence[object], tolerance: float) -> None:
    """Hold a probe's results to the registry densities.

    Parameters
    ----------
    cases : Sequence[Case]
        The cases, in the order their calls were made.
    results : Sequence[object]
        The probe's results.
    tolerance : float
        The relative tolerance; single-precision hosts round more.
    """
    expected = [
        (case.entry, point, case.expected(point))
        for case in cases
        for point in case.points
    ]
    assert len(results) == len(expected) + 1
    for (entry, point, density), actual in zip(expected, results, strict=False):
        assert math.isfinite(density), (entry, point)
        assert actual == pytest.approx(density, rel=tolerance, abs=tolerance), (
            entry,
            point,
            density,
            actual,
        )
    assert results[-1] == "x"


def _run_in_docker(
    target: str, tmp_path: pathlib.Path, cases: Sequence[Case]
) -> list[object]:
    """Transpile the cases for ``target`` and run their entry points in its image.

    Parameters
    ----------
    target : str
        The dynamic target.
    tmp_path : pathlib.Path
        The bind-mounted scratch directory.
    cases : Sequence[Case]
        The cases.

    Returns
    -------
    list[object]
        The probe's results.
    """
    image, language = _BACKENDS[target]
    extension = _EXTENSIONS[language]
    (tmp_path / f"source.{extension}").write_bytes(
        transpile(parse(_source(cases)), target=target)
    )
    (tmp_path / "calls.json").write_text(json.dumps(_calls(cases)))
    (tmp_path / "backend.txt").write_text(target)
    results = run_probe_script(
        image=image, script=_SCRIPTS / f"qiec_{language}.{extension}", scratch=tmp_path
    )
    assert isinstance(results, list)
    return results


@pytest.mark.parametrize("target", sorted(_BACKENDS))
def test_every_host_scores_the_registry_density(
    target: str, tmp_path: pathlib.Path
) -> None:
    image, _ = _BACKENDS[target]
    if not image_available(image):
        pytest.fail(
            f"probe image {image} is not built; run tests/transpile/docker/build.sh"
        )
    cases = _cases_for(target)
    results = _run_in_docker(target, tmp_path, cases)
    # Pyro, NumPyro, and Edward2 score in single precision; PyMC's own
    # closed-form densities carry single-precision constants.
    _check(cases, results, _TOLERANCES[target])


def test_church_scores_the_registry_density(tmp_path: pathlib.Path) -> None:
    cases = _cases_for("church")
    script = tmp_path / "score.scm"
    displays = "".join(
        f"(display (qiec_{case.entry} {_scheme(point)})) (newline)\n"
        for case in cases
        for point in case.points
    )
    script.write_bytes(
        transpile(parse(_source(cases)), target="church")
        + (displays + "(display (qiec_label)) (newline)\n").encode()
    )
    completed = subprocess.run(
        [SCHEME_EXECUTABLE, "--script", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = completed.stdout.strip().splitlines()
    results: list[object] = [float(line) for line in lines[:-1]]
    results.append(lines[-1])
    _check(cases, results, 1e-9)


def _scheme(point: object) -> str:
    """A scored point as Scheme source.

    Parameters
    ----------
    point : object
        The point.

    Returns
    -------
    str
        A Boolean, number, or tuple record literal.
    """
    if isinstance(point, bool):
        return "#t" if point else "#f"
    if isinstance(point, list):
        return (
            "(_qvr-qiec-tuple (list " + " ".join(_scheme(item) for item in point) + "))"
        )
    return repr(point)
