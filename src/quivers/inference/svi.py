"""Stochastic Variational Inference (SVI) training loop.

SVI optimizes the ELBO by taking gradient steps on the guide and
model parameters. Each call to ``step`` performs one optimization
step and returns the loss value.
"""

from __future__ import annotations

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guide import Guide
from quivers.inference.elbo import ELBO


class SVI:
    """Stochastic Variational Inference optimizer.

    Parameters
    ----------
    model : MonadicProgram
        The generative model.
    guide : Guide
        The variational guide.
    optim : torch.optim.Optimizer
        Optimizer for both model and guide parameters.
    loss : ELBO
        The ELBO loss module.
    """

    def __init__(
        self,
        model: MonadicProgram,
        guide: Guide,
        optim: torch.optim.Optimizer,
        loss: ELBO,
    ) -> None:
        self.model = model
        self.guide = guide
        self.optim = optim
        self.loss = loss

    def step(
        self,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> float:
        """Perform a single SVI optimization step.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape (batch, ...).
        observations : dict[str, torch.Tensor]
            Observed variable values.

        Returns
        -------
        float
            The loss value for this step.
        """
        self.optim.zero_grad()
        loss_val = self.loss(self.model, self.guide, x, observations)
        loss_val.backward()
        self.optim.step()
        return loss_val.item()
