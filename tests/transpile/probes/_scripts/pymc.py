"""In-container PyMC probe.

The transpiled source defines a top-level ``build_model(<data>=None)``
function: the renderer wraps the ``pymc.Model`` block in a builder so
the per-call data inputs land in the function signature. The probe
instantiates one [`pymc.Model`][pymc.Model] per point with that
point's data, then evaluates the joint log-density in
constrained-space to match the QVR reference probe's convention.

Constrained-space evaluation rationale:
[`Model.compile_logp`][pymc.Model.compile_logp] expects values in
the unconstrained (transformed) space and adds the inverse-transform
Jacobian by default; the QVR reference computes log-densities in the
constrained space (the natural support, e.g. ``[0, 1]`` for
``Beta``) without a Jacobian term. The probe therefore sums
[`pymc.logp`][pymc.logp] over every free and observed RV directly,
substituting each free RV with its constrained-space value from
``Point.params`` and each observed RV with its data from
``Point.data``. The constant-spread tolerance in the harness then
absorbs the per-fixture normalisation offset between QVR's and
PyMC's internal conventions.
"""
import json
import pathlib

import numpy as np
import pymc
import pytensor
import pytensor.tensor as pt


def _arr(value):
    return np.asarray(value)


def _build_model(source: str, data_kw: dict) -> pymc.Model:
    """Execute the transpiled source and call ``build_model(**data)``.

    The renderer emits ``def build_model(<input1>=None, ...): ...``
    with one keyword argument per [`IRDataInput`][quivers.transpile.ir.IRDataInput].
    The probe locates that function in the executed namespace and
    invokes it with the per-point data dict, returning the constructed
    [`pymc.Model`][pymc.Model] instance.
    """
    ns = {"pymc": pymc, "np": np}
    exec(source, ns)  # noqa: S102
    builder = ns.get("build_model")
    if builder is None:
        msg = (
            "transpiled PyMC source missing top-level `build_model` "
            "function; available names: "
            + ", ".join(sorted(k for k in ns if not k.startswith("_")))
        )
        raise RuntimeError(msg)
    model = builder(**data_kw)
    if not isinstance(model, pymc.Model):
        msg = (
            "transpiled `build_model` did not return a `pymc.Model`; "
            f"got {type(model).__name__}"
        )
        raise RuntimeError(msg)
    return model


def _joint_logp_constrained(
    model: pymc.Model,
    params: dict,
) -> float:
    """Sum ``pymc.logp(rv, value)`` over every free and observed RV.

    Free RVs receive their constrained-space value from ``params``;
    observed RVs use their attached observation tensor. Free-RV
    references inside observed-RV expressions are substituted with
    their constrained values via
    [`graph_replace`][pytensor.graph.replace.graph_replace] so the
    resulting graph has no free symbolic inputs.
    """
    substitutions: dict[object, object] = {}
    logp_terms: list[object] = []

    for rv in model.free_RVs:
        if rv.name not in params:
            msg = (
                f"pymc probe: missing param for free RV {rv.name!r}; "
                f"available: {sorted(params)}"
            )
            raise RuntimeError(msg)
        value = pt.as_tensor(params[rv.name]).astype(rv.dtype)
        logp_terms.append(pymc.logp(rv, value).sum())
        substitutions[rv] = value

    for rv in model.observed_RVs:
        obs_value = model.rvs_to_values[rv]
        logp_terms.append(pymc.logp(rv, obs_value).sum())

    total = logp_terms[0]
    for term in logp_terms[1:]:
        total = total + term

    if substitutions:
        total = pytensor.graph.replace.graph_replace(total, substitutions)

    return float(total.eval())


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.py").read_text()
    points = json.loads((io / "points.json").read_text())

    log_densities = []
    for pt_record in points:
        data_kw = {
            k: _arr(v) for k, v in pt_record.get("data", {}).items()
        }
        params = {
            k: _arr(v) for k, v in pt_record.get("params", {}).items()
        }
        model = _build_model(source, data_kw)
        log_densities.append(_joint_logp_constrained(model, params))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
