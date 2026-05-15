"""Model comparison via :func:`arviz.compare`.

Thin wrapper around the canonical ArviZ entry point with quivers-
typed inputs.  No information-criterion math lives here; ArviZ's
implementations of PSIS-LOO, WAIC, stacking, and pseudo-BMA+ are
the source of truth.  See [the ArviZ team's *Exploratory Analysis
of Bayesian Models* textbook](https://arviz-devs.github.io/EABM/)
for the methodology.
"""

from __future__ import annotations

from typing import Literal, Mapping

import arviz as az
import xarray as xr


def compare(
    fits: Mapping[str, xr.DataTree],
    *,
    method: Literal["stacking", "BB-pseudo-BMA", "pseudo-BMA"] = "stacking",
    var_name: str | None = None,
    reference: str | None = None,
) -> object:
    """Rank candidate models by expected log predictive density.

    Delegates to :func:`arviz.compare`, which computes PSIS-LOO
    via :func:`arviz.loo` on each fit's ``log_likelihood`` group
    and combines the resulting :class:`~arviz.stats.ELPDData` records
    into a ranked comparison table.

    Parameters
    ----------
    fits : Mapping[str, xr.DataTree]
        Per-model fit, each a DataTree produced by
        :func:`~quivers.diagnostics.to_datatree`.  Every fit must
        carry a ``log_likelihood`` group; without it
        :func:`arviz.loo` cannot compute elpd.
    method : "stacking", "BB-pseudo-BMA", or "pseudo-BMA"
        Stacking weight estimator.  Default ``"stacking"`` follows
        [Yao, Vehtari, Simpson, Gelman 2018](https://doi.org/10.1214/17-BA1091).
    var_name : str, optional
        Name of the observed variable in ``log_likelihood`` to
        compare on; required when a fit's ``log_likelihood`` group
        carries multiple variables.
    reference : str, optional
        Fit name to use as the reference for elpd-difference
        comparisons.  Default is the top-ranked model.

    Returns
    -------
    pandas.DataFrame
        ArviZ ranking table with columns ``rank, elpd_loo, p_loo,
        se, weight, ...`` and one row per model.
    """
    return az.compare(
        dict(fits),
        method=method,
        var_name=var_name,
        reference=reference,
    )
