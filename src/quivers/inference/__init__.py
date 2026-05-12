"""Inference algorithms for probabilistic programs.

This module provides the inference layer for quivers: execution
tracing, conditioning on observations, the
:class:`~quivers.inference.registry.LatentRegistry` introspection
helper, automatic variational guide construction, variational
objectives, optimisation drivers, and posterior-predictive
sampling. MCMC and hybrid samplers land here as the relevant
modules grow in.

Submodules
----------
trace : Execution trace and sample site recording.
conditioning : Observation marking and ``condition()`` factory.
registry : :class:`LatentRegistry`, the per-site introspection
    helper every guide and MCMC kernel consumes.
guides : Variational guide families (mean-field, full-rank /
    low-rank Gaussian, normalising-flow, structured, mixture,
    Laplace). Re-exported here at top level.
elbo : Evidence lower bound (kept as a thin re-export of
    :class:`ELBO` from :mod:`quivers.inference.objectives` once
    objectives land).
svi : Stochastic variational inference training loop.
predictive : Posterior predictive sampling.
"""

from __future__ import annotations

from quivers.inference.trace import Trace, SampleSite, trace
from quivers.inference.conditioning import Conditioned, condition
from quivers.inference.registry import LatentRegistry, LatentSite
from quivers.inference.guides import Guide, AutoNormalGuide, AutoDeltaGuide
from quivers.inference.elbo import ELBO
from quivers.inference.svi import SVI
from quivers.inference.predictive import Predictive

__all__ = [
    "Trace",
    "SampleSite",
    "trace",
    "Conditioned",
    "condition",
    "LatentRegistry",
    "LatentSite",
    "Guide",
    "AutoNormalGuide",
    "AutoDeltaGuide",
    "ELBO",
    "SVI",
    "Predictive",
]
