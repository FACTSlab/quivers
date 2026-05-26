"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the Church backend.

Church (the original MIT-Scheme-based language) has no maintained
runtime that exposes a programmable log-density for an arbitrary
`(sample, observe)` program; the canonical interpreters (`church`,
`bher`) all wrap the program in a query rather than expose
joint-density evaluation. The structural and external-syntax tiers
exercise the `qvr-church` output; the numeric tier omits it.

The probe accordingly returns `available() == False` unconditionally
so the numeric-equivalence test layer skips the `(*, church)` cell
with a clear reason.
"""

from __future__ import annotations

import dataclasses
import pathlib

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


@dataclasses.dataclass(frozen=True)
class ChurchProbe:
    backend: str = "church"

    def available(self) -> bool:
        """Church has no production runtime exposing a programmable
        log-density. The probe is unavailable by design; the
        numeric-equivalence test layer skips `(*, church)` cells.
        """
        return False

    def evaluate(
        self,
        source: bytes,
        fixture_name: str,
        points: list[Point],
        *,
        scratch: pathlib.Path,
    ) -> ProbeResult:
        raise RuntimeError(
            "ChurchProbe is unavailable by design (no Church runtime "
            "exposes a programmable joint log-density). Tier 3 tests "
            "must skip the `(*, church)` cell."
        )


_PROBE: LogDensityProbe = ChurchProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["ChurchProbe"]
