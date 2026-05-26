"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the PyMC backend.

Runs the transpiled ``with pm.Model() as model: ...`` source by
`exec`ing the bytes; the resulting PyMC `Model` object's
``compile_logp()`` returns a closure mapping the point dict to a
joint log-density value.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


@dataclasses.dataclass(frozen=True)
class PyMCProbe:
    backend: str = "pymc"

    def available(self) -> bool:
        return importlib.util.find_spec("pymc") is not None

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
                "pymc not installed; PyMCProbe.available() returned "
                "False but evaluate() was called anyway"
            )
        import pymc

        log_densities: list[float] = []
        for pt in points:
            # Each point gets its own model build because the
            # transpiled output bakes the observed values in through
            # the `observed=` keyword on each RV. A fresh `exec`
            # against a namespace seeded with the point's data
            # rebuilds the model with that point's observations.
            namespace: dict[str, object] = {
                "pymc": pymc,
                **{k: _as_array(v) for k, v in pt.data.items()},
            }
            exec(source.decode("utf-8"), namespace)  # noqa: S102
            model = namespace.get("model")
            if not isinstance(model, pymc.Model):
                raise RuntimeError(
                    f"pymc probe: transpiled source for {fixture_name!r} "
                    f"did not produce a pymc.Model in `model`"
                )
            param_dict = {k: _as_array(v) for k, v in pt.params.items()}
            logp_fn = model.compile_logp()
            log_densities.append(float(logp_fn(param_dict)))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": f"pymc {pymc.__version__}"},
        )


def _as_array(value):
    import numpy as np

    if isinstance(value, (int, float)):
        return np.asarray(value)
    return np.asarray(value)


_PROBE: LogDensityProbe = PyMCProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["PyMCProbe"]
