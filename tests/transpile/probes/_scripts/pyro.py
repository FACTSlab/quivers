"""In-container Pyro probe.

Reshapes each `Point` per `/io/shapes.json` (when present) so a
gallery example whose data tensor was declared with a multi-dim
shape sees `torch.tensor(value).shape` match what the model
expects.
"""
import json
import pathlib

import pyro
import torch

from _reshape import load_tables, reshape_point


def _tensor(value):
    if isinstance(value, (int, float)):
        return torch.tensor(float(value))
    return torch.tensor(value, dtype=torch.float64)


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)

    ns = {"pyro": pyro, "torch": torch}
    exec(source, ns)  # noqa: S102
    model = ns["model"]

    log_densities = []
    for pt in points:
        reshaped = reshape_point(pt, shapes, dtypes)
        data_kw = {
            k: _tensor(v) for k, v in reshaped.get("data", {}).items()
        }
        param_dict = {
            k: _tensor(v) for k, v in reshaped.get("params", {}).items()
        }
        conditioned = pyro.condition(model, data=param_dict)
        traced = pyro.poutine.trace(conditioned).get_trace(**data_kw)
        log_densities.append(float(traced.log_prob_sum()))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
