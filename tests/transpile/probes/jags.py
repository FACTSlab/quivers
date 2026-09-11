"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the JAGS backend.

Invokes the `jags` binary via [`pyjags`][pyjags]: compiles the
transpiled model, supplies the observed data and clamped latent
initial values, and reads the model's joint log-density via
JAGS's `marglik` deviance monitor at zero sampling iterations.

Available iff `pyjags` is importable and the `jags` binary is on
PATH.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import shutil

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


@dataclasses.dataclass(frozen=True)
class JagsProbe:
    backend: str = "jags"

    def available(self) -> bool:
        return (
            importlib.util.find_spec("pyjags") is not None
            and shutil.which("jags") is not None
        )

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
                "pyjags / jags binary not installed; "
                "JagsProbe.available() returned False but evaluate() "
                "was called anyway"
            )
        import pyjags

        scratch.mkdir(parents=True, exist_ok=True)
        model_path = scratch / f"{fixture_name}.jags"
        model_path.write_bytes(source)

        log_densities: list[float] = []
        for pt in points:
            jags_model = pyjags.Model(
                file=str(model_path),
                data={k: _as_jags(v) for k, v in pt.data.items()},
                init=[{k: _as_jags(v) for k, v in pt.params.items()}],
                chains=1,
                adapt=0,
                threads=1,
                progress_bar=False,
            )
            # The model's log-density at the supplied init is read
            # via JAGS's `deviance` monitor at iteration zero. The
            # joint log-density is (-deviance / 2) under the
            # convention deviance = -2 * log p(θ, y).
            samples = jags_model.sample(1, vars=["deviance"], thin=1)
            dev = float(samples["deviance"][0, 0, 0])
            log_densities.append(-dev / 2)

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": f"pyjags + jags {shutil.which('jags')}"},
        )


def _as_jags(value):
    """Convert a Python scalar / list to a numpy array (pyjags expects
    numpy)."""
    import numpy as np

    return np.asarray(value)


_PROBE: LogDensityProbe = JagsProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["JagsProbe"]
