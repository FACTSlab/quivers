"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the Gen.jl backend.

Uses Julia via [`juliacall`][juliacall]: evals the transpiled
`@gen function model(...) ... end` source, then calls
`Gen.assess(model, args, constraints)` at each test point.
`Gen.assess` returns `(weight, retval)`; the weight is the joint
log-density of the supplied `constraints` (which we set to the
test point's combined params + data).

Available iff `juliacall` is importable AND the user's Julia
environment has the `Gen` package installed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


@dataclasses.dataclass(frozen=True)
class GenProbe:
    backend: str = "gen"

    def available(self) -> bool:
        if importlib.util.find_spec("juliacall") is None:
            return False
        try:
            from juliacall import Main as jl
            jl.seval("using Gen")
        except Exception:  # noqa: BLE001
            return False
        return True

    def evaluate(
        self,
        source: bytes,
        fixture_name: str,
        points: list[Point],
        *,
        scratch: pathlib.Path,
    ) -> ProbeResult:
        if not self.available():
            raise RuntimeError(
                "juliacall + Gen.jl not installed; GenProbe.available() "
                "returned False but evaluate() was called anyway"
            )
        from juliacall import Main as jl

        jl.seval("using Gen, Distributions")
        jl.seval(source.decode("utf-8"))
        model = jl.model

        log_densities: list[float] = []
        for pt in points:
            args = tuple(pt.data[name] for name in sorted(pt.data))
            constraints = jl.Gen.choicemap()
            for k, v in {**pt.params, **pt.data}.items():
                jl.Gen.set_value_b(constraints, jl.Symbol(k), v)
            weight, _ = jl.Gen.assess(model, args, constraints)
            log_densities.append(float(weight))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": "juliacall + Gen.jl"},
        )


_PROBE: LogDensityProbe = GenProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["GenProbe"]
