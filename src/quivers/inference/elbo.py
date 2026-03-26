"""Evidence lower bound (ELBO) computation.

The ELBO is the central objective for variational inference:

    ELBO = E_q[log p(x, z) - log q(z)]

Maximizing the ELBO is equivalent to minimizing KL(q || p). This
module computes the negative ELBO (a loss to minimize) using
Monte Carlo estimation with multiple particles.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guide import Guide


class ELBO(nn.Module):
    """Compute the negative ELBO (loss to minimize).

    Parameters
    ----------
    num_particles : int
        Number of Monte Carlo samples for estimating the expectation.
    """

    def __init__(self, num_particles: int = 1) -> None:
        super().__init__()
        self.num_particles = num_particles

    def forward(
        self,
        model: MonadicProgram,
        guide: Guide,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute negative ELBO.

        Parameters
        ----------
        model : MonadicProgram
            The generative model.
        guide : Guide
            The variational guide.
        x : torch.Tensor
            Program input. Shape (batch, ...).
        observations : dict[str, torch.Tensor]
            Observed variable values.

        Returns
        -------
        torch.Tensor
            Scalar negative ELBO (averaged over batch and particles).
        """
        total = torch.tensor(0.0, device=x.device)

        for _ in range(self.num_particles):
            # sample latents from the guide
            latents = guide.rsample(x)

            # merge latents and observations for log-joint
            all_sites = {**latents, **observations}

            # model log-joint: log p(z, y_obs | x)
            model_lp = model.log_joint(x, all_sites)

            # guide log-prob: log q(z | x)
            guide_lp = guide.log_prob(x, latents)

            # elbo = E_q[log p - log q], loss = -elbo = E_q[log q - log p]
            total = total + (guide_lp - model_lp).mean()

        return total / self.num_particles
