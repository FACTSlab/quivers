"""In-container NumPyro probe.

Reshapes each `Point` per `/io/shapes.json` (when present) so a
gallery example whose data tensor was declared with a multi-dim
shape sees `jnp.array(value).shape` match what the model expects.
"""
import json
import pathlib

import jax
# Match QVR's torch defaults (float64) so the constant-spread
# comparison is not dominated by float32 round-off in the JAX side.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpyro
from numpyro.infer.util import log_density

from _reshape import load_tables, reshape_point


def _arr(value):
    return jnp.array(value)


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)

    ns = {"numpyro": numpyro, "jnp": jnp}
    exec(source, ns)  # noqa: S102
    model = ns["model"]

    log_densities = []
    for pt in points:
        reshaped = reshape_point(pt, shapes, dtypes)
        data_kw = {k: _arr(v) for k, v in reshaped.get("data", {}).items()}
        param_dict = {
            k: _arr(v) for k, v in reshaped.get("params", {}).items()
        }
        lp, _ = log_density(model, (), data_kw, param_dict)
        log_densities.append(float(lp))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
