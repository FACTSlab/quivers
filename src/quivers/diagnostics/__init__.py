"""Adapter to the ArviZ Bayesian-analysis ecosystem.

Quivers' inference layer produces :class:`~quivers.inference.MCMCResult`
records and :class:`~quivers.inference.guides.base.Guide` fits.  The
canonical 2026 Bayesian-analysis library is [ArviZ](https://python.arviz.org/),
whose
[`xarray.DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html)-based
data model is the lingua franca for posterior-summary, diagnostic,
posterior-predictive, model-comparison, and calibration tooling.

The functions in this subpackage convert quivers fits into ArviZ
DataTrees and wrap the canonical ArviZ entry points
(:func:`arviz.compare`, :func:`arviz.loo`, :func:`arviz.hdi`,
:func:`arviz.plot_ppc`) with shape-aware, quivers-typed signatures.
No posterior-analysis primitives are reimplemented here; ArviZ owns
the analytics.
"""

from quivers.diagnostics.arviz_io import to_datatree
from quivers.diagnostics.comparison import compare
from quivers.diagnostics.predictive_checks import posterior_predictive_check

__all__ = [
    "to_datatree",
    "compare",
    "posterior_predictive_check",
]
