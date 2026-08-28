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

One more wire-form correction rides alongside the zero-based
subscripts the shared helper lifts: a categorical *outcome* the
renderer declares as `int <lower = 1 , upper = K>` is one-based in
Stan while the point payload codes it from zero, so it is lifted too.

A `simplex`-typed parameter needs one more marshalling step. Stan
unconstrains a constrained-space input through
`stan::math::simplex_free`, which rejects a row that misses summing
to one by more than its own constraint tolerance, and the wire form
carries each row as float32-rounded doubles. Every simplex-declared
name is therefore rescaled by its own row sum before it reaches
cmdstanpy, and a row too far from one to be float32 rounding raises
instead.

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
from __future__ import annotations

import json
import pathlib
import re
from typing import TYPE_CHECKING, cast

import cmdstanpy

from _reshape import (
    as_nested,
    index_input_names,
    load_export_names,
    load_tables,
    reshape_point,
    shift_index_inputs,
)

if TYPE_CHECKING:
    Number = int | float
    NestedNumber = Number | list["NestedNumber"]
    PointSection = dict[str, NestedNumber]


# A `simplex[K]` declaration, with or without an enclosing
# `array[N]`. The size expression is whatever Stan admits there (a
# literal, a data variable), so the name is all this pattern needs.
_SIMPLEX_DECL_RE = re.compile(
    r"\bsimplex\s*\[[^\]]*\]\s*([A-Za-z_][A-Za-z0-9_]*)"
)

# An `int` data declaration carrying both a lower bound of one and an
# upper bound: `array [200] int <lower = 1 , upper = 200> w;`. That
# pair is how the Stan renderer spells the support of a categorical
# *outcome*, whose codes Stan counts from one. A bare `int <lower = 1>`
# is a positivity constraint on a count and is deliberately not
# matched.
_ONE_BASED_DECL_RE = re.compile(
    r"\bint\s*<\s*lower\s*=\s*1\s*,\s*upper\s*=[^>]*>\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)"
)


# How far a wire-form simplex row may sit from summing to one before
# the probe calls it a defect rather than a rounding artefact. The
# point payload carries each row as float32-rounded doubles, so a
# K-component row misses one by at most a few multiples of float32
# epsilon (6e-08 on the 200-component rows of the gallery's LDA
# topics). Anything above this floor is a wrong value, not a wrong
# representation, and renormalising it would turn a mis-shipped
# parameter into a silently valid one.
_SIMPLEX_SUM_TOLERANCE = 1e-5


def _one_based_outcome_names(
    source_text: str, dtypes: dict[str, str],
) -> set[str]:
    """Names the emitted model declares on a one-based support.

    A QVR `Categorical` observation carries codes in `0..K-1`, and so
    does the point payload; Stan's `categorical_lpmf` counts its
    outcomes from one, and the renderer declares the receiving data
    array as `int <lower = 1 , upper = K>` to say so. The wire form
    therefore has to be lifted by one before it reaches cmdstanpy,
    the same marshalling
    [`shift_index_inputs`][tests.transpile.probes._scripts._reshape.shift_index_inputs]
    performs for a zero-based covariate the model subscripts.

    Reading the declaration rather than guessing from the value keeps
    a count observation (`int <lower = 0>`) and a Bernoulli response
    (`int <lower = 0 , upper = 1>`) untouched: neither is declared on
    a one-based support, and neither is an index.

    The dtype table gates the lift, exactly as it gates
    [`index_input_names`][tests.transpile.probes._scripts._reshape.index_input_names]:
    a caller that ships no `/io/dtypes.json` is one that already hands
    the probe values in the target's own convention, and shifting them
    again would move every code off by one.
    """
    return {
        name
        for name in _ONE_BASED_DECL_RE.findall(source_text)
        if dtypes.get(name) == "int"
    }


def _simplex_parameter_names(source_text: str) -> set[str]:
    """Names the emitted model declares with a `simplex` type."""
    return set(_SIMPLEX_DECL_RE.findall(source_text))


def _renormalise_rows(
    name: str, value: list[NestedNumber],
) -> list[NestedNumber]:
    """Scale every innermost row of ``value`` to sum to exactly one.

    Stan validates a constrained-space `simplex` input through
    `stan::math::simplex_free`, which rejects a row whose components
    miss one by more than its own constraint tolerance. The point
    payload reaches the container as float32-rounded doubles, so a row
    that *is* a simplex in the reference's arithmetic is not one in
    Stan's, and the run aborts before any density is computed.
    Rescaling the row by its own sum is the smallest correction that
    restores the constraint: it moves each component by at most the
    float32 rounding already present in the wire form.

    A row further from one than `_SIMPLEX_SUM_TOLERANCE` raises. That
    is the case where the harness shipped the wrong values for a
    simplex-typed name, and normalising it would manufacture a valid
    simplex out of a wrong one.
    """
    if value and isinstance(value[0], list):
        return [
            _renormalise_rows(name, row)
            for row in cast("list[list[NestedNumber]]", value)
        ]
    row = cast("list[Number]", value)
    total = float(sum(row))
    if abs(total - 1.0) > _SIMPLEX_SUM_TOLERANCE:
        msg = (
            f"stan probe: the point's {name!r} row sums to {total!r}, "
            f"which is further from one than the float32 rounding of "
            f"the wire form can explain "
            f"(tolerance {_SIMPLEX_SUM_TOLERANCE}). The model declares "
            f"{name!r} as a simplex, so this row is not the value the "
            f"reference scored."
        )
        raise ValueError(msg)
    return [float(component) / total for component in row]


def _renormalise_simplex_params(
    params: PointSection, simplex_names: set[str],
) -> PointSection:
    """Return ``params`` with every simplex-typed entry rescaled."""
    return {
        name: (
            _renormalise_rows(name, value)
            if name in simplex_names and isinstance(value, list)
            else value
        )
        for name, value in params.items()
    }


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
    # Stan counts array subscripts and categorical outcome codes from
    # one; every zero-based covariate the model subscripts and every
    # outcome the model declares on a one-based support must be lifted
    # before it reaches cmdstanpy.
    index_names = (
        index_input_names(source_text, dtypes)
        | _one_based_outcome_names(source_text, dtypes)
    )
    simplex_names = _simplex_parameter_names(source_text)

    model = cmdstanpy.CmdStanModel(stan_file=str(source))
    log_densities = []
    exports = []
    for pt in points:
        reshaped = shift_index_inputs(
            reshape_point(pt, shapes, dtypes), index_names,
        )
        params = _renormalise_simplex_params(
            reshaped.get("params", {}), simplex_names,
        )
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
