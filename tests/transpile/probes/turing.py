"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the Turing.jl backend.

Invokes Julia via [`juliacall`][juliacall]: instantiates a `Main`
module, imports `Turing`, `Distributions`, `LinearAlgebra`, evals
the transpiled `@model function model(...) ... end` source, and
calls `Turing.logjoint(model_callable, theta)` at each test point.

Available iff `juliacall` is importable AND the user's Julia
environment has the `Turing` package installed. The probe gates on
both: `available()` False means the cell is skipped.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


@dataclasses.dataclass(frozen=True)
class TuringProbe:
    backend: str = "turing"

    def available(self) -> bool:
        if importlib.util.find_spec("juliacall") is None:
            return False
        try:
            from juliacall import Main as jl
            jl.seval("using Turing")
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
                "juliacall + Turing.jl not installed; "
                "TuringProbe.available() returned False but evaluate() "
                "was called anyway"
            )
        from juliacall import Main as jl

        jl.seval("using Turing, Distributions, LinearAlgebra")
        jl.seval(source.decode("utf-8"))
        model_factory = jl.model  # the @model macro produces a callable

        log_densities: list[float] = []
        for pt in points:
            # Each observed value is passed as a model argument.
            # The transpiled `@model function model(<obs>)` expects
            # one positional arg per observed variable.
            args = tuple(pt.data[name] for name in sorted(pt.data))
            model_instance = model_factory(*args)
            theta = jl.NamedTuple(**{
                k: _wrap(jl, v) for k, v in pt.params.items()
            })
            lp = jl.Turing.logjoint(model_instance, theta)
            log_densities.append(float(lp))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": "juliacall + Turing.jl"},
        )


def _wrap(jl, value):
    """Wrap a Python scalar / list as a Julia value via juliacall."""
    if isinstance(value, (int, float)):
        return value
    return jl.collect(value)


_PROBE: LogDensityProbe = TuringProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["TuringProbe"]
