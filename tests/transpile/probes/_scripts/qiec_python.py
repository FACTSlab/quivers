"""In-container probe for QIEC entry points of a generated Python module.

Reads ``/io/source.py`` and ``/io/calls.json``, a list of
``[entry, arguments]`` pairs, calls each ``qiec_<entry>`` with its
arguments, and writes the results to ``/io/result.json`` as plain
numbers. The backend is read from ``/io/backend.txt`` and fixes the
imports the generated module expects in scope and how a host scalar is
read back as a number.
"""

import json
import pathlib

BACKEND = pathlib.Path("/io/backend.txt").read_text().strip()
namespace: dict[str, object] = {}

if BACKEND == "pyro":
    import pyro
    import torch

    namespace.update({"pyro": pyro, "torch": torch})

    def as_number(value):
        return value.item() if isinstance(value, torch.Tensor) else value

elif BACKEND == "numpyro":
    import jax.numpy as jnp
    import numpyro

    namespace.update({"numpyro": numpyro, "jnp": jnp})

    def as_number(value):
        return float(value) if isinstance(value, jnp.ndarray) else value

elif BACKEND == "pymc":
    import numpy as np
    import pymc

    namespace.update({"pymc": pymc, "np": np})

    def as_number(value):
        evaluate = getattr(value, "eval", None)
        return float(evaluate()) if callable(evaluate) else value

elif BACKEND == "edward2":
    import edward2
    import tensorflow as tf
    import tensorflow_probability as tfp

    namespace.update({"edward2": edward2, "tf": tf, "tfp": tfp})

    def as_number(value):
        return float(value.numpy()) if isinstance(value, tf.Tensor) else value

else:
    raise SystemExit(f"unknown backend {BACKEND!r}")


def as_value(value):
    """Turn a JSON argument into the host value a QIEC parameter takes.

    Lists are tensors, which QIEC represents as nested tuples.
    """
    if isinstance(value, list):
        return tuple(as_value(item) for item in value)
    return value


exec(pathlib.Path("/io/source.py").read_text(), namespace)  # noqa: S102
calls = json.loads(pathlib.Path("/io/calls.json").read_text())
results = []
for entry, arguments in calls:
    function = namespace[f"qiec_{entry}"]
    results.append(as_number(function(*(as_value(item) for item in arguments))))
pathlib.Path("/io/result.json").write_text(json.dumps(results))
