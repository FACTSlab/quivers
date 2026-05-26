"""In-container Stan probe.

Reads `/io/source.stan` + `/io/points.json`; compiles the model via
cmdstanpy and writes `/io/result.json` with the log-density at each
point.

Copied into the container at run time by the test harness.
"""
import json
import pathlib

import cmdstanpy


def main() -> None:
    io = pathlib.Path("/io")
    source = io / "source.stan"
    points = json.loads((io / "points.json").read_text())

    model = cmdstanpy.CmdStanModel(stan_file=str(source))
    log_densities = []
    for pt in points:
        # cmdstanpy expects a dict mapping declared `parameters` to
        # values and a separate `data` dict. Both come from `points`.
        params = pt.get("params", {})
        data = pt.get("data", {})
        lp = model.log_prob(params=params, data=data)
        log_densities.append(float(lp))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
