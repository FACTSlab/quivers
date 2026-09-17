"""Scoring emitted Python targets in this process.

The Pyro and PyMC emissions run under the libraries the ``targets`` extra
installs, so a test can score an emitted model at a clamped point without
a container: the same conditioning the probe images apply, applied here.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pymc
import pyro
import pytensor.graph.replace
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable
import torch

type Host = float | int | list[float] | list[int] | list[list[float]]
"""A number or a nested list, as a point carries it."""


def _tensor(value: Host) -> torch.Tensor:
    """A point value as a double tensor.

    Parameters
    ----------
    value : Host
        The value.

    Returns
    -------
    torch.Tensor
        The tensor, a scalar for a number.
    """
    if isinstance(value, (int, float)):
        return torch.tensor(float(value), dtype=torch.float64)
    return torch.tensor(value, dtype=torch.float64)


def pyro_log_density(
    source: bytes,
    data: Mapping[str, Host],
    params: Mapping[str, Host],
    *,
    integer_data: frozenset[str] = frozenset(),
) -> float:
    """Score an emitted Pyro model at a clamped point.

    Parameters
    ----------
    source : bytes
        The emitted module.
    data : Mapping[str, Host]
        The model's data arguments by name.
    params : Mapping[str, Host]
        The value of every sample site, by name.
    integer_data : frozenset[str]
        The data arguments carrying indices, passed as long tensors.

    Returns
    -------
    float
        The trace's ``log_prob_sum`` under the conditioning.
    """
    namespace: dict[str, object] = {"pyro": pyro, "torch": torch}
    exec(source.decode("utf-8"), namespace)  # noqa: S102
    model = namespace["model"]
    data_kw = {
        name: (
            torch.tensor(value, dtype=torch.long)
            if name in integer_data
            else _tensor(value)
        )
        for name, value in data.items()
    }
    conditioned = pyro.condition(model, data={k: _tensor(v) for k, v in params.items()})
    traced = pyro.poutine.trace(conditioned).get_trace(**data_kw)
    return float(traced.log_prob_sum())


def pymc_log_density(
    source: bytes,
    data: Mapping[str, Host],
    params: Mapping[str, Host],
) -> float:
    """Score an emitted PyMC model at a clamped point.

    The joint is the sum of ``pymc.logp`` over every free variable at its
    given constrained value and every observed variable at its data,
    plus every potential.

    Parameters
    ----------
    source : bytes
        The emitted module.
    data : Mapping[str, Host]
        The model's data arguments by name.
    params : Mapping[str, Host]
        The value of every free variable, by name.

    Returns
    -------
    float
        The joint log density.

    Raises
    ------
    RuntimeError
        If the emission defines no ``build_model`` or a free variable
        is given no value.
    """
    namespace: dict[str, object] = {"pymc": pymc, "np": np}
    exec(source.decode("utf-8"), namespace)  # noqa: S102
    builder = namespace.get("build_model")
    if builder is None:
        raise RuntimeError("the PyMC emission defines no `build_model`")
    model = builder(**{name: np.asarray(value) for name, value in data.items()})  # type: ignore[operator]
    substitutions: dict[TensorVariable, TensorVariable] = {}
    terms: list[TensorVariable] = []
    for rv in model.free_RVs:
        if rv.name not in params:
            raise RuntimeError(f"free variable {rv.name!r} is given no value")
        value = pt.as_tensor(np.asarray(params[rv.name])).astype(rv.dtype)
        terms.append(pymc.logp(rv, value).sum())
        substitutions[rv] = value
    for rv in model.observed_RVs:
        terms.append(pymc.logp(rv, model.rvs_to_values[rv]).sum())
    for potential in model.potentials:
        terms.append(potential.sum())
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    if substitutions:
        total = pytensor.graph.replace.graph_replace(total, substitutions, strict=False)
    return float(total.eval())


__all__ = ["Host", "pymc_log_density", "pyro_log_density"]
