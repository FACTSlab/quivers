"""MCMC kernels and driver.

Public surface (also re-exported from
:mod:`quivers.inference`):

* :class:`MCMCKernel` — ABC for Markov kernels on the flat
  unconstrained latent vector.
* :class:`HMCKernel` — Hamiltonian Monte Carlo with leapfrog
  integration, dual-averaging step-size adaptation, and Welford
  mass-matrix adaptation.
* :class:`NUTSKernel` — No-U-Turn Sampler with multinomial
  sampling.
* :class:`MCMC` — Chain orchestrator with warmup, parallel
  chains, and posterior diagnostics
  (split-:math:`\\hat R`, effective sample size).
* :class:`MCMCResult` — Posterior samples + per-chain
  diagnostics.
"""

from __future__ import annotations

from quivers.inference.mcmc.driver import MCMC, MCMCResult
from quivers.inference.mcmc.hmc import HMCKernel, NUTSKernel
from quivers.inference.mcmc.kernel import (
    KernelState,
    MCMCKernel,
    PotentialFn,
)

__all__ = [
    "MCMCKernel",
    "KernelState",
    "PotentialFn",
    "HMCKernel",
    "NUTSKernel",
    "MCMC",
    "MCMCResult",
]
