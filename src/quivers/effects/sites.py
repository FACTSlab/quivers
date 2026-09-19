"""Host sampleables: a torch morphism applied to an input, as a distribution.

A `MonadicProgram` step draws from a `ContinuousMorphism` at an input
tensor. The kernel's ``Random.sample`` request takes a sampleable value,
so the pair is wrapped as one: `TorchSampleable` offers the sampling
and scoring interface the prelude handlers use and keeps the morphism
and input a strategy or a trace may want to read.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from quivers.continuous.morphisms import ContinuousMorphism


@dataclass(frozen=True, slots=True)
class TorchSampleable:
    """A morphism applied to an input, offered to the kernel as a sampleable.

    Parameters
    ----------
    morphism
        The morphism the site draws from, under whatever parameters the
        run's ``Param`` handlers answered.
    input
        The input tensor the morphism is conditioned on, carrying the
        batch axis.
    declared
        The morphism as the program declares it, which a trace records
        and a strategy reads the family of; the drawing morphism itself
        when no parameter store rebinds it.
    """

    morphism: ContinuousMorphism
    input: torch.Tensor
    declared: ContinuousMorphism | None = None

    def __post_init__(self) -> None:
        """Default the declared morphism to the drawing one."""
        if self.declared is None:
            object.__setattr__(self, "declared", self.morphism)

    def rsample(self) -> torch.Tensor:
        """Draw a reparameterised sample.

        Returns
        -------
        torch.Tensor
            One draw per batch row.
        """
        return self.morphism.rsample(self.input)

    def sample(self) -> torch.Tensor:
        """Draw a sample.

        Returns
        -------
        torch.Tensor
            One draw per batch row.
        """
        return self.morphism.rsample(self.input)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Score a value under the morphism at the input.

        Parameters
        ----------
        value : torch.Tensor
            The value, one per batch row.

        Returns
        -------
        torch.Tensor
            The log density per batch row.
        """
        return self.morphism.log_prob(self.input, value)

    def parameters(self) -> dict[str, torch.Tensor]:
        """The distribution parameters the morphism computes at the input.

        Returns
        -------
        dict[str, torch.Tensor]
            The named parameters of the torch distribution the morphism
            builds, or ``loc`` and ``scale`` for a morphism that exposes
            only a location-scale pair.

        Raises
        ------
        TypeError
            If the morphism exposes neither a distribution builder nor a
            location-scale pair.
        """
        get_dist = getattr(self.morphism, "_get_dist", None)
        if callable(get_dist):
            distribution = get_dist(self.input)
            return {
                name: getattr(distribution, name)
                for name in distribution.arg_constraints
                if hasattr(distribution, name)
            }
        get_params = getattr(self.morphism, "_get_params", None)
        if callable(get_params):
            loc, scale = get_params(self.input)
            return {"loc": loc, "scale": scale}
        raise TypeError(
            f"{type(self.morphism).__name__} exposes no distribution parameters"
        )


__all__ = ["TorchSampleable"]
