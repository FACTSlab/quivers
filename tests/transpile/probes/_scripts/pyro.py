"""In-container Pyro probe."""
import json
import pathlib

import pyro
import torch


def _tensor(value):
    if isinstance(value, (int, float)):
        return torch.tensor(float(value))
    return torch.tensor([float(v) for v in value])


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())

    ns = {"pyro": pyro, "torch": torch}
    exec(source, ns)  # noqa: S102
    model = ns["model"]

    log_densities = []
    for pt in points:
        data_kw = {k: _tensor(v) for k, v in pt.get("data", {}).items()}
        param_dict = {k: _tensor(v) for k, v in pt.get("params", {}).items()}
        conditioned = pyro.condition(model, data=param_dict)
        traced = pyro.poutine.trace(conditioned).get_trace(**data_kw)
        log_densities.append(float(traced.log_prob_sum()))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
