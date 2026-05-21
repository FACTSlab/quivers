"""MCMC kernels and driver.

Public surface (also re-exported from
`quivers.inference`):

* `MCMCKernel` — ABC for Markov kernels on the flat
  unconstrained latent vector.
* `HMCKernel` — Hamiltonian Monte Carlo with leapfrog
  integration, dual-averaging step-size adaptation, and Welford
  mass-matrix adaptation.
* `NUTSKernel` — No-U-Turn Sampler with multinomial
  sampling.
* `MCMC` — Chain orchestrator with warmup, parallel
  chains, and posterior diagnostics
  (split-:math:`\\hat R`, effective sample size).
* `MCMCResult` — Posterior samples + per-chain
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
