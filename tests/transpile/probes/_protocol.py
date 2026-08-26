"""Per-backend log-density probe Protocol.

Each entry in [`tests/transpile/probes/`][tests.transpile.probes]
implements [`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for one target language. The probe takes the transpiled source bytes
and a sequence of (parameter, data) points, and returns the
backend's log-density evaluation at each point.

The Tier 3 numeric-equivalence test feeds the same point set to the
QVR reference probe and each available target probe; the
[`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match]
helper then enforces the constant-spread contract.

A probe reports two independent things about a program, because a
QVR program declared `prog : A -> B` denotes a Markov kernel from `A`
to `B` and a joint density does not determine one. The first is
`log p(theta, y)`, the measure over `(latents, observations)`. The
second is the program's **exported value**: what its `return` clause
carries into `B`. Two programs can share the first and differ in the
second, so a renderer validated on log-density alone is validated on
half its obligation, and
[`test_export_equivalence`][tests.transpile.test_export_equivalence]
is the tier that holds the other half.

The export channel is opt-in per call. The harness writes the QVR
program's return-variable names, in declaration order, to
`/io/export_names.json` beside the point set; an out-of-process probe
that finds the file reads the exported value out of its target's own
return surface (a model function's `return`, a Stan
`generated quantities` alias, the second element of `Gen.assess`) and
reports one entry per name per point. A probe that cannot produce one
raises rather than reporting a shorter vector, because a silently
missing export is exactly the defect the tier exists to catch.

Probes that need an out-of-process runtime (Stan via cmdstanpy,
Julia via PyJulia, etc.) launch a Docker container per call when
the runtime is not importable in-process; the [`Image`][.] and
[`run_in_container`][.] helpers in
[`tests/transpile/_docker.py`][tests.transpile._docker] centralise
the container-launch boilerplate.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pathlib


@dataclasses.dataclass(frozen=True)
class Point:
    """One test point: a (latent-parameter, data) pair.

    Both ``params`` and ``data`` are flat name-to-Python-float (or
    Python-int) maps. Vector / matrix values are passed as flat
    Python lists in row-major order; the probe reshapes per backend.
    The shape per name is declared once per fixture in
    `<fixture_name>.shapes.json`; the probe loads that alongside
    the point set.
    """

    params: dict[str, float | int | list[float] | list[int]]
    """Latent parameter values at this point. Keys are the QVR sample
    variable names."""

    data: dict[str, float | int | list[float] | list[int]]
    """Observed data values at this point. Keys are the QVR observe
    variable names."""


#: One exported value as it crosses the probe boundary: a scalar, or
#: an arbitrarily nested tuple of scalars in row-major order. Vector
#: and matrix exports keep their nesting, so an element the emitted
#: program placed in the wrong slot stays in the wrong slot here
#: instead of being flattened into agreement.
ExportValue = float | int | tuple["ExportValue", ...]


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    """Result of evaluating one fixture across a point set."""

    backend: str
    """Backend name, e.g. ``"stan"`` or ``"qvr"``."""

    fixture: str
    """Fixture stem, e.g. ``"beta_bernoulli"``."""

    log_densities: list[float]
    """Log p(θ, y) at each point in the input sequence (same order)."""

    metadata: dict[str, str]
    """Free-form additional info (image name, version, runtime version
    string, ...). Surfaced in failure messages."""

    exports: tuple[tuple[ExportValue, ...], ...] = ()
    """The program's exported value at each point, in the same order
    as ``log_densities``; one inner entry per name the program's
    `return` clause declares, in declaration order.

    Empty when the caller did not ask for the export channel. An
    empty tuple therefore means "not requested", never "the program
    exports nothing": a program with no `return` clause is not
    scheduled through this channel at all, and a probe that was asked
    for an export it cannot produce raises instead of returning
    nothing."""


@runtime_checkable
class LogDensityProbe(Protocol):
    """Backend probe: turns a transpiled fixture into log-density
    values at a sequence of test points.

    Implementations live one per file under
    [`tests/transpile/probes/`][tests.transpile.probes]. Each probe
    advertises:

    - its ``backend`` name (matches
      [`quivers.transpile.available_targets`][quivers.transpile.available_targets]
      for non-QVR probes; ``"qvr"`` for the reference probe);
    - an ``available`` boolean / method, gated on the runtime being
      importable / a binary being on PATH / a Docker image being
      present.
    """

    backend: str

    def available(self) -> bool:
        """True iff the probe can run on the current host.

        Implementations check `shutil.which` for binaries,
        `importlib.util.find_spec` for Python modules, and
        `docker image inspect <tag>` for container images.
        """
        ...

    def evaluate(
        self,
        source: bytes,
        fixture_name: str,
        points: list[Point],
        *,
        scratch: pathlib.Path,
    ) -> ProbeResult:
        """Compute log-density at each point.

        Parameters
        ----------
        source
            The transpiled source bytes for this backend (already
            decoded from the transpile pipeline; the probe writes it
            to its own filename under ``scratch`` as needed).
        fixture_name
            Fixture stem; used for diagnostic messages and to look
            up any `<fixture>.shapes.json` companion file.
        points
            Test points; the same sequence is fed to every probe
            running this fixture.
        scratch
            Per-test scratch directory; the probe owns this fully.
            Persisted between calls only within one pytest test (the
            harness creates one scratch dir per `(fixture, backend)`
            cell).

        Returns
        -------
        ProbeResult
            Log-density values in the same order as ``points``, and
            the program's exported value per point when the caller
            asked for the export channel.
        """
        ...


__all__ = [
    "ExportValue",
    "LogDensityProbe",
    "Point",
    "ProbeResult",
]
