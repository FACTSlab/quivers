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
    low-rank Gaussian, normalizing-flow, structured, mixture,
    Laplace). Re-exported here at top level.
estimators : Gradient estimators (reparameterized, sticking-the-
    landing, DReG, score-function).
objectives : Variational objectives (ELBO, IWAE, Rényi, VR-IWAE).
svi : Stochastic variational inference training loop.
predictive : Posterior predictive sampling.
"""

from __future__ import annotations

from quivers.inference.trace import Trace, SampleSite, trace
from quivers.inference.conditioning import Conditioned, condition
from quivers.inference.registry import LatentRegistry, LatentSite
from quivers.inference.guides import (
    AutoDeltaGuide,
    AutoIAFGuide,
    AutoLaplaceApproximation,
    AutoLowRankMultivariateNormalGuide,
    AutoMixtureGuide,
    AutoMultivariateNormalGuide,
    AutoNeuralSplineGuide,
    AutoNormalGuide,
    AutoNormalizingFlow,
    Guide,
)
from quivers.inference.estimators import (
    DoublyReparameterized,
    GradientEstimator,
    Reparameterized,
    ScoreFunction,
    StickingTheLanding,
)
from quivers.inference.objectives import (
    ELBO,
    IWAEBound,
    Objective,
    RenyiBound,
    VRIWAEBound,
)
from quivers.inference.mcmc import (
    HMCKernel,
    MCMC,
    MCMCKernel,
    MCMCResult,
    NUTSKernel,
)
from quivers.inference.svi import SVI
from quivers.inference.predictive import Predictive
from quivers.inference.dais import AutoDAIS
from quivers.inference.warmup import WarmupThenHMC

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
    "AutoMultivariateNormalGuide",
    "AutoLowRankMultivariateNormalGuide",
    "AutoDeltaGuide",
    "AutoLaplaceApproximation",
    "AutoNormalizingFlow",
    "AutoIAFGuide",
    "AutoNeuralSplineGuide",
    "AutoMixtureGuide",
    "MCMCKernel",
    "HMCKernel",
    "NUTSKernel",
    "MCMC",
    "MCMCResult",
    "AutoDAIS",
    "WarmupThenHMC",
    "Objective",
    "ELBO",
    "IWAEBound",
    "RenyiBound",
    "VRIWAEBound",
    "GradientEstimator",
    "Reparameterized",
    "StickingTheLanding",
    "DoublyReparameterized",
    "ScoreFunction",
    "SVI",
    "Predictive",
]
