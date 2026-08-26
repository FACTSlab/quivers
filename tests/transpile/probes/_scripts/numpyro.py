"""In-container NumPyro probe.

Reshapes each `Point` per `/io/shapes.json` (when present) so a
gallery example whose data tensor was declared with a multi-dim
shape sees `jnp.array(value).shape` match what the model expects.

When `/io/export_names.json` is present the probe also reports the
program's exported value at each point. NumPyro's export surface is
the model function's own `return`, so the probe substitutes every
latent with the point's clamped value and calls the model: the value
that comes back is exactly what a downstream NumPyro user would get,
and a renderer that returns the wrong quantity (or none at all) is
visible here and nowhere in the log-density.
"""
import json
import pathlib

import jax
# Match QVR's torch defaults (float64) so the constant-spread
# comparison is not dominated by float32 round-off in the JAX side.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpyro  # noqa: E402
from numpyro.handlers import seed, substitute  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from _reshape import (  # noqa: E402
    export_payload,
    load_export_names,
    load_tables,
    reshape_point,
)


def _arr(value):
    return jnp.array(value)


def _returned(model, data_kw, param_dict):
    """The model's return value with every latent clamped.

    `substitute` pins each latent to the point's value; `seed` is
    still required because NumPyro raises on an unseeded `sample`
    even when every site is substituted. The clamps make the draw
    irrelevant: nothing the generator produces reaches the return
    value.
    """
    clamped = seed(
        substitute(model, param_dict), jax.random.PRNGKey(0),
    )
    return clamped(**data_kw)


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)
    export_names = load_export_names(io)

    ns = {"numpyro": numpyro, "jnp": jnp}
    exec(source, ns)  # noqa: S102
    model = ns["model"]

    log_densities = []
    exports = []
    for pt in points:
        reshaped = reshape_point(pt, shapes, dtypes)
        data_kw = {k: _arr(v) for k, v in reshaped.get("data", {}).items()}
        param_dict = {
            k: _arr(v) for k, v in reshaped.get("params", {}).items()
        }
        lp, _ = log_density(model, (), data_kw, param_dict)
        log_densities.append(float(lp))
        if export_names:
            exports.append(
                export_payload(
                    export_names, _returned(model, data_kw, param_dict),
                )
            )

    result = {"log_densities": log_densities}
    if export_names:
        result["exports"] = exports
    (io / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
