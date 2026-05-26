"""In-container JAGS / BUGS probe.

Reads /io/source.jags (or source.bugs), compiles via pyjags, and
extracts the joint log-density via the `deviance` monitor at the
supplied init point. The contract: joint log-density = -deviance / 2.
"""
import json
import os
import pathlib

import numpy as np
import pyjags


def _arr(value):
    return np.asarray(value)


def main() -> None:
    io = pathlib.Path("/io")
    ext = os.environ.get("FIXTURE_EXT", "jags")
    source_path = io / f"source.{ext}"
    points = json.loads((io / "points.json").read_text())

    log_densities = []
    for pt in points:
        data = {k: _arr(v) for k, v in pt.get("data", {}).items()}
        init = {k: _arr(v) for k, v in pt.get("params", {}).items()}
        model = pyjags.Model(
            file=str(source_path),
            data=data,
            init=[init],
            chains=1,
            adapt=0,
            threads=1,
            progress_bar=False,
        )
        samples = model.sample(1, vars=["deviance"], thin=1)
        dev = float(samples["deviance"][0, 0, 0])
        log_densities.append(-dev / 2)

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
