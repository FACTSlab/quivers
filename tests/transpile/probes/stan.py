"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the Stan backend.

Compiles the transpiled `.stan` source via
[`cmdstanpy.CmdStanModel`][cmdstanpy.CmdStanModel] and evaluates the
joint log-density at each test point through
[`model.log_prob(params, data)`][cmdstanpy.CmdStanModel.log_prob].

The cmdstanpy install carries its own `stanc3` binary, so the probe
is in-process even though Stan itself is a compiled-source language.
Available iff `cmdstanpy` is importable AND cmdstanpy can find a
Stan install (it auto-installs to `~/.cmdstan/` on first use).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


@dataclasses.dataclass(frozen=True)
class StanProbe:
    backend: str = "stan"

    def available(self) -> bool:
        if importlib.util.find_spec("cmdstanpy") is None:
            return False
        import cmdstanpy
        try:
            cmdstanpy.cmdstan_path()
        except (ValueError, OSError):
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
                "cmdstanpy not installed or Stan not found; "
                "StanProbe.available() returned False but evaluate() "
                "was called anyway"
            )
        import cmdstanpy

        scratch.mkdir(parents=True, exist_ok=True)
        stan_path = scratch / f"{fixture_name}.stan"
        stan_path.write_bytes(source)
        model = cmdstanpy.CmdStanModel(stan_file=str(stan_path))

        log_densities: list[float] = []
        for pt in points:
            data_json = scratch / f"{fixture_name}.data.json"
            data_json.write_text(json.dumps(pt.data))
            # `model.log_prob(params, data)` returns a list of log
            # densities (one per parameter set passed); we pass a
            # single point.
            lp_value = model.log_prob(
                params=pt.params,
                data=str(data_json),
            )
            # cmdstanpy returns a numpy scalar or 1-element array.
            log_densities.append(float(lp_value))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": f"cmdstanpy {cmdstanpy.__version__}"},
        )


_PROBE: LogDensityProbe = StanProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["StanProbe"]
