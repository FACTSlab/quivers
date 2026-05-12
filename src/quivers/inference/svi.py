"""Stochastic Variational Inference (SVI) training loop.

SVI optimises a :class:`~quivers.inference.objectives.Objective`
— the ELBO, IWAE, Rényi, or VR-IWAE bound — by taking gradient
steps on the guide and model parameters. The ``objective``
parameter accepts any :class:`Objective` subclass.
"""

from __future__ import annotations

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides import Guide
from quivers.inference.objectives import Objective


class SVI:
    """Stochastic Variational Inference optimiser.

    Parameters
    ----------
    model : MonadicProgram
        Generative model.
    guide : Guide
        Variational guide.
    optim : torch.optim.Optimizer
        Optimiser for both model and guide parameters.
    objective : Objective
        Variational objective (ELBO, IWAEBound, RenyiBound,
        VRIWAEBound, …).
    """

    def __init__(
        self,
        model: MonadicProgram,
        guide: Guide,
        optim: torch.optim.Optimizer,
        objective: Objective,
    ) -> None:
        self.model = model
        self.guide = guide
        self.optim = optim
        self.objective = objective

    def step(
        self,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> float:
        """One SVI step.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape ``(batch, ...)``.
        observations : dict[str, torch.Tensor]
            Observed variable values + host data (the
            non-site keys are exposed to the trace via the
            ``condition`` machinery).

        Returns
        -------
        float
            Scalar loss for this step.
        """
        self.optim.zero_grad()
        loss_val = self.objective(self.model, self.guide, x, observations)
        loss_val.backward()
        self.optim.step()
        return loss_val.item()
