"""In-container Stan probe.

Reads `/io/source.stan` + `/io/points.json` (and optional
`/io/shapes.json` + `/io/dtypes.json`); compiles the model via
cmdstanpy and writes `/io/result.json` with the log-density at each
point.

Each point's `params` and `data` dicts arrive as flat row-major
lists; the model declares them with multi-dim shapes. The
`_reshape` helper rebuilds nested lists per `/io/shapes.json`
(and casts to int / float per `/io/dtypes.json`) so cmdstanpy
sees the shape Stan declared.

Copied into the container at run time by the test harness.
"""
import json
import pathlib

import cmdstanpy

from _reshape import load_tables, reshape_point


def main() -> None:
    io = pathlib.Path("/io")
    source = io / "source.stan"
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)

    model = cmdstanpy.CmdStanModel(stan_file=str(source))
    log_densities = []
    for pt in points:
        reshaped = reshape_point(pt, shapes, dtypes)
        params = reshaped.get("params", {})
        data = reshaped.get("data", {})
        # `jacobian=False` returns the constrained-space log
        # density (the model-statement contribution alone), without
        # adding the change-of-variables Jacobian Stan uses
        # internally to lift constrained parameters into
        # unconstrained-sampler space. QVR's trace evaluates in
        # constrained space too; comparing constrained-vs-constrained
        # avoids a theta-dependent Jacobian term that would
        # otherwise leak into the spread and violate the
        # constant-spread contract.
        lp_df = model.log_prob(
            params=params, data=data, jacobian=False
        )
        log_densities.append(float(lp_df["lp__"].iloc[0]))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
