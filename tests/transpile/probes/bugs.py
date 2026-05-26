"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the BUGS backend.

BUGS-family distributions share the JAGS surface for the
QVR-supported subset (`dnorm`, `dbern`, `dbeta`, `dgamma`, ...).
There is no canonical BUGS Python runtime; the practical interpreter
is `multibugs` or `OpenBUGS`, neither of which exposes a
programmable log-density to Python. The probe therefore delegates
to the JAGS interpreter on the BUGS-syntax source: JAGS accepts
straight-BUGS programs by design (it was built as a free
implementation of the BUGS language). When `jags` and `pyjags` are
available, this probe loads the BUGS source through them and reads
the joint log-density via the deviance monitor — the same path the
JAGS probe takes.

Available iff `pyjags` is importable and the `jags` binary is on
PATH. If either is missing, `available()` returns False.
"""

from __future__ import annotations

import dataclasses
import pathlib

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult
from tests.transpile.probes.jags import JagsProbe


@dataclasses.dataclass(frozen=True)
class BugsProbe:
    backend: str = "bugs"
    _jags: JagsProbe = dataclasses.field(default_factory=JagsProbe)

    def available(self) -> bool:
        return self._jags.available()

    def evaluate(
        self,
        source: bytes,
        fixture_name: str,
        points: list[Point],
        *,
        scratch: pathlib.Path,
    ) -> ProbeResult:
        # JAGS accepts the BUGS-language source as-is; reuse the
        # JAGS probe's compile + sample-at-init path.
        result = self._jags.evaluate(
            source, fixture_name, points, scratch=scratch
        )
        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=result.log_densities,
            metadata={
                **result.metadata,
                "interpreted_via": "jags (BUGS-syntax compatible)",
            },
        )


_PROBE: LogDensityProbe = BugsProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["BugsProbe"]
