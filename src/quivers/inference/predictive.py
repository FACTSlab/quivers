"""Posterior predictive sampling.

After training a guide via SVI, the ``Predictive`` class draws
posterior predictive samples by repeatedly sampling latents from
the guide and running the model forward.
"""

from __future__ import annotations

import torch

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guide import Guide
from quivers.inference.trace import trace


class Predictive:
    """Draw posterior predictive samples from a trained model + guide.

    Parameters
    ----------
    model : MonadicProgram
        The generative model.
    guide : Guide
        The trained variational guide.
    num_samples : int
        Number of posterior samples to draw.
    """

    def __init__(
        self,
        model: MonadicProgram,
        guide: Guide,
        num_samples: int = 100,
    ) -> None:
        self.model = model
        self.guide = guide
        self.num_samples = num_samples

    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Draw posterior predictive samples.

        For each of ``num_samples`` iterations, samples latents from
        the guide and traces the model with those latents as
        observations. Returns all site values stacked along a new
        leading dimension.

        Parameters
        ----------
        x : torch.Tensor
            Program input. Shape (batch, ...).
        observations : dict[str, torch.Tensor] or None
            Additional observed data to condition on.

        Returns
        -------
        dict[str, torch.Tensor]
            Each key is a site name, each value has shape
            (num_samples, batch, ...).
        """
        if observations is None:
            observations = {}

        collected: dict[str, list[torch.Tensor]] = {}

        for _ in range(self.num_samples):
            # sample latents from guide
            latents = self.guide.rsample(x)

            # merge with any fixed observations
            all_obs = {**latents, **observations}

            # trace the model with these values clamped
            tr = trace(self.model, x, observations=all_obs)

            for name, site in tr.sites.items():
                if site.is_deterministic:
                    continue

                if name not in collected:
                    collected[name] = []

                collected[name].append(site.value)

        # stack along a new leading dimension
        return {
            name: torch.stack(vals, dim=0)
            for name, vals in collected.items()
        }
