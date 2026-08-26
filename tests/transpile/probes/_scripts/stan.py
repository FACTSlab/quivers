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

When `/io/export_names.json` is present the probe also reports the
program's exported value at each point. Stan has no program-level
return, so the renderer exposes each returned name as a
``generated quantities`` alias ``<name>_value``, and the probe reads
that alias through CmdStan's own generated-quantities machinery: a
one-draw ``fixed_param`` run initialised at the point leaves every
parameter exactly where the point put it and evaluates the
``generated quantities`` block against those values. The log-density
still comes from ``log_prob``; the two calls agree on the point
because ``inits`` and ``params`` are the same dict.

Copied into the container at run time by the test harness.
"""
import json
import pathlib

import cmdstanpy

from _reshape import (
    as_nested,
    index_input_names,
    load_export_names,
    load_tables,
    reshape_point,
    shift_index_inputs,
)


def _generated_quantities(
    model: cmdstanpy.CmdStanModel,
    params: dict,
    data: dict,
    export_names: list,
) -> list:
    """Read each ``<name>_value`` generated quantity at the point.

    ``fixed_param`` is CmdStan's sampler for a model whose parameters
    do not move: it writes one draw at the supplied ``inits`` and
    evaluates the ``generated quantities`` block against them. That
    makes the reported export a deterministic function of the point,
    with no transition between the value the harness supplied and the
    value the block saw. ``seed`` is fixed so a model whose generated
    quantities happened to draw would still be reproducible; nothing
    in the exports the renderer emits does.

    A missing alias means the emit dropped the program's return
    clause, which raises rather than reporting a shorter export
    vector.
    """
    fit = model.sample(
        data=data,
        inits=params,
        chains=1,
        iter_warmup=0,
        iter_sampling=1,
        adapt_engaged=False,
        fixed_param=True,
        seed=1,
        show_progress=False,
    )
    variables = fit.stan_variables()
    values = []
    for name in export_names:
        alias = f"{name}_value"
        if alias not in variables:
            msg = (
                f"stan probe: the emitted model declares no {alias!r} "
                f"generated quantity, so it exposes nothing for the "
                f"QVR program's exported {name!r}. Model variables: "
                f"{sorted(variables)}"
            )
            raise RuntimeError(msg)
        values.append(as_nested(variables[alias][0]))
    return values


def main() -> None:
    io = pathlib.Path("/io")
    source = io / "source.stan"
    source_text = source.read_text()
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)
    export_names = load_export_names(io)
    # Stan arrays are 1-based; every 0-based covariate the model
    # subscripts must be lifted before it reaches cmdstanpy.
    index_names = index_input_names(source_text, dtypes)

    model = cmdstanpy.CmdStanModel(stan_file=str(source))
    log_densities = []
    exports = []
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
        if export_names:
            exports.append(
                _generated_quantities(model, params, data, export_names)
            )

    result = {"log_densities": log_densities}
    if export_names:
        result["exports"] = exports
    (io / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
