"""In-container PyMC probe."""
import json
import pathlib

import numpy as np
import pymc


def _arr(value):
    return np.asarray(value)


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())

    log_densities = []
    for pt in points:
        ns = {
            "pymc": pymc,
            **{k: _arr(v) for k, v in pt.get("data", {}).items()},
        }
        exec(source, ns)  # noqa: S102
        model = ns["model"]
        if not isinstance(model, pymc.Model):
            raise RuntimeError("transpiled source did not produce a pymc.Model")
        params = {k: _arr(v) for k, v in pt.get("params", {}).items()}
        logp_fn = model.compile_logp()
        log_densities.append(float(logp_fn(params)))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
