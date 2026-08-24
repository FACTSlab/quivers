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

from _reshape import (
    index_input_names,
    load_tables,
    reshape_point,
    shift_index_inputs,
)


def main() -> None:
    io = pathlib.Path("/io")
    source = io / "source.stan"
    source_text = source.read_text()
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)
    # Stan arrays are 1-based; every 0-based covariate the model
    # subscripts must be lifted before it reaches cmdstanpy.
    index_names = index_input_names(source_text, dtypes)

    model = cmdstanpy.CmdStanModel(stan_file=str(source))
    log_densities = []
    for pt in points:
        reshaped = shift_index_inputs(
            reshape_point(pt, shapes, dtypes), index_names,
        )
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
