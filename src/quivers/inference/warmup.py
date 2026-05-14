"""Warmup-then-HMC composite sampler.

Pareto-dominates plain HMC on posteriors whose prior init places
chains far from the typical set (the canonical SuperTelicity-shape
case with constrained-support hierarchical priors). The composite
runs SVI to convergence on a chosen variational guide, then
initializes the HMC / NUTS chain at the guide's posterior mean and
adapts the mass matrix to the guide's posterior covariance — so
HMC's warmup is given a substantial head-start instead of starting
from scratch.

This is a two-phase orchestrator: it owns no kernel state of its
own, just a guide + an MCMC kernel + a driver. The two phases are
:meth:`fit_guide` (vanilla SVI) and :meth:`run_mcmc` (the warmup-
seeded MCMC chain); :meth:`run` calls both in sequence.
"""

from __future__ import annotations

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMC, MCMCResult
from quivers.inference.mcmc.kernel import MCMCKernel
from quivers.inference.objectives import ELBO, Objective
from quivers.inference.svi import SVI


class WarmupThenHMC:
    """Composite sampler: SVI warmup followed by HMC / NUTS chain.

    Parameters
    ----------
    guide : Guide
        Variational guide trained in the SVI warmup phase. Any
        :class:`Guide` subclass; :class:`AutoMultivariateNormalGuide`
        is the canonical pick because its learned Cholesky factor
        directly seeds the dense mass matrix of HMC's adaptation.
    kernel : MCMCKernel
        Markov kernel for the sampling phase (typically
        :class:`HMCKernel` or :class:`NUTSKernel`).
    svi_steps : int
        Number of SVI steps in the warmup phase.
    svi_lr : float
        Learning rate for the SVI optimizer. Default ``1e-2``.
    mcmc_warmup : int
        MCMC kernel's own warmup (on top of the SVI warmup; HMC
        still benefits from a few step-size-adaptation iterations
        because the SVI-mean's typical-set position differs from
        the maximum-density point).
    mcmc_samples : int
        Number of post-warmup MCMC samples per chain.
    num_chains : int
        Number of MCMC chains.
    objective : Objective
        SVI objective. Default :class:`ELBO`.
    """

    def __init__(
        self,
        guide: Guide,
        kernel: MCMCKernel,
        svi_steps: int,
        mcmc_warmup: int,
        mcmc_samples: int,
        num_chains: int = 2,
        svi_lr: float = 1e-2,
        objective: Objective | None = None,
    ) -> None:
        if svi_steps < 1:
            raise ValueError(f"WarmupThenHMC: svi_steps must be >= 1, got {svi_steps}")
        if mcmc_warmup < 0:
            raise ValueError(
                f"WarmupThenHMC: mcmc_warmup must be >= 0, got {mcmc_warmup}"
            )
        if mcmc_samples < 1:
            raise ValueError(
                f"WarmupThenHMC: mcmc_samples must be >= 1, got {mcmc_samples}"
            )
        self.guide = guide
        self.kernel = kernel
        self.svi_steps = svi_steps
        self.svi_lr = svi_lr
        self.mcmc_warmup = mcmc_warmup
        self.mcmc_samples = mcmc_samples
        self.num_chains = num_chains
        self.objective = objective if objective is not None else ELBO()

    def fit_guide(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> list[float]:
        """Run the SVI warmup phase and return the per-step losses."""
        optim = torch.optim.Adam(
            list(model.parameters()) + list(self.guide.parameters()),
            lr=self.svi_lr,
        )
        svi = SVI(model, self.guide, optim, self.objective)
        losses: list[float] = []
        for _ in range(self.svi_steps):
            losses.append(svi.step(x, observations))
        return losses

    def run_mcmc(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> MCMCResult:
        """Run the post-warmup MCMC phase. Chains are seeded from
        the fitted guide via :meth:`MCMC.run`'s ``init_strategy='guide'``."""
        driver = MCMC(
            kernel=self.kernel,
            num_warmup=self.mcmc_warmup,
            num_samples=self.mcmc_samples,
            num_chains=self.num_chains,
            init_strategy="guide",
        )
        return driver.run(model, x, observations, guide=self.guide)

    def run(
        self,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> tuple[list[float], MCMCResult]:
        """Convenience: SVI warmup, then MCMC sampling.

        Returns ``(svi_losses, mcmc_result)``.
        """
        losses = self.fit_guide(model, x, observations)
        result = self.run_mcmc(model, x, observations)
        return losses, result


__all__ = ["WarmupThenHMC"]
