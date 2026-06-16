"""In-container JAGS / BUGS probe.

Reads /io/source.jags (or source.bugs), compiles via pyjags, and
extracts the joint log-density via the ``deviance`` monitor at the
supplied point. Contract: joint log-density = ``-deviance / 2``.

JAGS evaluates the joint log-density at a fixed point by treating
every supplied value as observed. The probe puts both the QVR
``params`` (latent parameters) and ``data`` (observed nodes) into
the pyjags ``data`` dict; every named node in either dict becomes a
fixed observed node, so the deviance JAGS computes is the joint
log-density of the model evaluated at the requested point.

pyjags' built-in ``Model.__init__`` only auto-loads the ``dic``
module; the deviance monitor type lives in ``dic.so`` but the
``base`` and ``bugs`` modules supply the basic monitor factories
(`Trace`, `Mean`) and the standard distributions / samplers that
``dnorm`` / ``dbeta`` / ``dbern`` etc. resolve to, so the probe
loads all three explicitly before constructing the model.
"""
import json
import os
import pathlib

import numpy as np
import pyjags


pyjags.load_module("basemod")
pyjags.load_module("bugs")
pyjags.load_module("dic")


def _arr(value):
    return np.asarray(value)


def main() -> None:
    io = pathlib.Path("/io")
    ext = os.environ.get("FIXTURE_EXT", "jags")
    source_path = io / f"source.{ext}"
    points = json.loads((io / "points.json").read_text())

    log_densities = []
    for pt in points:
        # Combine params + data: JAGS treats every supplied node as
        # observed, so the deviance includes the joint log-density of
        # the latent parameters AND the observed data nodes.
        merged: dict[str, np.ndarray] = {}
        for k, v in pt.get("data", {}).items():
            merged[k] = _arr(v)
        for k, v in pt.get("params", {}).items():
            merged[k] = _arr(v)
        model = pyjags.Model(
            file=str(source_path),
            data=merged,
            chains=1,
            adapt=0,
            threads=1,
            progress_bar=False,
        )
        # `trace` monitor records the value at each iteration; with
        # every model variable fixed as observed there is nothing for
        # the sampler to update, so the single-iteration trace value
        # is exactly the joint log-density (times -2).
        model.console.setMonitor("deviance", 1, "trace")
        model.console.update(1)
        dumped = model.console.dumpMonitors("trace", True)
        dev_arr = dumped["deviance"]
        dev = float(np.asarray(dev_arr).flatten()[0])
        log_densities.append(-dev / 2)

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
