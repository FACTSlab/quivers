"""Conversion from quivers fits to `xarray.DataTree` (the
ArviZ 1.x replacement for the legacy ``InferenceData`` container).

The MCMC sampler hands back [`quivers.inference.MCMCResult`][quivers.inference.MCMCResult]
records whose ``samples`` dict already follows the
``(num_chains, num_samples, *site_shape)`` shape convention ArviZ
expects.  This module repackages those tensors as numpy arrays under
the canonical ArviZ group names (``posterior``, ``sample_stats``,
``posterior_predictive``, ``log_likelihood``, ``observed_data``) plus
the user-supplied ``coords`` / ``dims`` for nice axis labels in
plots.
"""

from __future__ import annotations

from typing import Mapping

import arviz as az
import torch
import xarray as xr

from quivers.inference.mcmc.driver import MCMCResult


def _tensor_to_numpy(t: torch.Tensor):
    """Detach a tensor, move to CPU, and convert to numpy.  The
    detach is necessary because samples may carry an autograd
    graph (e.g. when produced by SVI's ``Predictive`` rather than
    NUTS).
    """
    return t.detach().cpu().numpy()


def to_datatree(
    posterior: MCMCResult,
    *,
    observed_data: Mapping[str, torch.Tensor] | None = None,
    posterior_predictive: Mapping[str, torch.Tensor] | None = None,
    log_likelihood: Mapping[str, torch.Tensor] | None = None,
    constant_data: Mapping[str, torch.Tensor] | None = None,
    coords: Mapping[str, list] | None = None,
    dims: Mapping[str, list[str]] | None = None,
) -> xr.DataTree:
    """Convert an `MCMCResult` into an ArviZ-style DataTree.

    Parameters
    ----------
    posterior : MCMCResult
        Sampler output.  ``posterior.samples[name]`` carries an
        ``(num_chains, num_samples, *site_shape)`` tensor; each
        becomes a posterior variable.  ``log_densities`` populates
        ``sample_stats/lp``; ``acceptance_rates`` and
        ``divergence_counts`` populate ``sample_stats``.
    observed_data : Mapping[str, torch.Tensor], optional
        Site name to observed tensor (the original data used at
        fit time).  Becomes the ``observed_data`` group.
    posterior_predictive : Mapping[str, torch.Tensor], optional
        Site name to posterior-predictive draws of shape
        ``(num_chains, num_samples, *site_shape)``.  Becomes the
        ``posterior_predictive`` group.
    log_likelihood : Mapping[str, torch.Tensor], optional
        Site name to per-observation log-likelihood of shape
        ``(num_chains, num_samples, *obs_shape)``.  Becomes the
        ``log_likelihood`` group; required for
        `arviz.loo` / `arviz.waic`.
    constant_data : Mapping[str, torch.Tensor], optional
        Site name to fixed covariate tensor (e.g. design matrix).
        Becomes the ``constant_data`` group.
    coords : Mapping[str, list], optional
        Coordinate values per named axis (e.g.
        ``{"Verb": ["eat", "drink", "run"]}``).  Forwarded to ArviZ.
    dims : Mapping[str, list[str]], optional
        Per-site axis names (e.g. ``{"beta": ["Verb"]}``).
        Forwarded to ArviZ.

    Returns
    -------
    xr.DataTree
        Canonical ArviZ DataTree consumable by every plotting and
        diagnostic function in the ArviZ 1.x API.
    """
    data: dict[str, dict] = {}

    posterior_group: dict = {}
    for name, t in posterior.samples.items():
        posterior_group[name] = _tensor_to_numpy(t)
    data["posterior"] = posterior_group

    sample_stats_group: dict = {
        "lp": _tensor_to_numpy(posterior.log_densities),
    }
    # acceptance_rate and diverging are per-chain scalars in
    # quivers; ArviZ expects per-draw arrays.  We broadcast.
    n_chains = posterior.num_chains
    n_samples = posterior.num_samples
    sample_stats_group["mean_acceptance_per_chain"] = (
        posterior.acceptance_rates.detach()
        .cpu()
        .numpy()
        .reshape(n_chains, 1)
        .repeat(n_samples, axis=1)
    )
    sample_stats_group["total_divergences_per_chain"] = (
        posterior.divergence_counts.detach()
        .cpu()
        .numpy()
        .reshape(n_chains, 1)
        .repeat(n_samples, axis=1)
    )
    data["sample_stats"] = sample_stats_group

    if observed_data:
        data["observed_data"] = {
            name: _tensor_to_numpy(t) for name, t in observed_data.items()
        }
    if posterior_predictive:
        data["posterior_predictive"] = {
            name: _tensor_to_numpy(t) for name, t in posterior_predictive.items()
        }
    if log_likelihood:
        data["log_likelihood"] = {
            name: _tensor_to_numpy(t) for name, t in log_likelihood.items()
        }
    if constant_data:
        data["constant_data"] = {
            name: _tensor_to_numpy(t) for name, t in constant_data.items()
        }

    return az.from_dict(
        data,
        coords=dict(coords) if coords is not None else None,
        dims=dict(dims) if dims is not None else None,
    )
