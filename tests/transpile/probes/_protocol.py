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
            Log-density values in the same order as ``points``.
        """
        ...


__all__ = [
    "LogDensityProbe",
    "Point",
    "ProbeResult",
]
