"""Inference algorithms for probabilistic programs.

This module provides the inference layer for quivers, bridging the gap
between model specification (MonadicProgram) and posterior estimation.
It implements variational inference (SVI), execution tracing,
conditioning on observations, automatic guide construction, and
posterior predictive sampling.

Submodules
----------
trace : Execution trace and sample site recording
conditioning : Observation marking and conditioned models
guide : Variational families and automatic guide construction
elbo : Evidence lower bound computation
svi : Stochastic variational inference training loop
predictive : Posterior predictive sampling
"""

from __future__ import annotations

from quivers.inference.trace import Trace, SampleSite, trace
from quivers.inference.conditioning import Conditioned, condition
from quivers.inference.guide import Guide, AutoNormalGuide, AutoDeltaGuide
from quivers.inference.elbo import ELBO
from quivers.inference.svi import SVI
from quivers.inference.predictive import Predictive

__all__ = [
    "Trace",
    "SampleSite",
    "trace",
    "Conditioned",
    "condition",
    "Guide",
    "AutoNormalGuide",
    "AutoDeltaGuide",
    "ELBO",
    "SVI",
    "Predictive",
]
