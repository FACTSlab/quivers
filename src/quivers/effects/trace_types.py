"""Data types for execution traces.

`SampleSite` and `Trace` live here in a leaf module so both the
effect-handler interpreter and the thin `quivers.inference.trace`
wrapper import them without introducing a cycle. The effects
package produces traces; the inference package consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from quivers.continuous.morphisms import ContinuousMorphism


@dataclass
class SampleSite:
    """Record of a single sample site in a program trace.

    Holds a ``torch.Tensor`` per site; not a value type.

    Parameters
    ----------
    name : str
        Variable name bound at this site.
    morphism : ContinuousMorphism or None
        The distribution morphism (``None`` for let bindings).
    value : torch.Tensor
        The sampled or observed value.
    log_prob : torch.Tensor
        Log-density of the value under the morphism. Shape
        ``(batch,)``. Zero for let bindings.
    is_observed : bool
        Whether this site was clamped to an observed value.
    is_deterministic : bool
        Whether this is a deterministic let binding.
    """

    name: str
    morphism: ContinuousMorphism | None
    value: torch.Tensor
    log_prob: torch.Tensor
    is_observed: bool = False
    is_deterministic: bool = False


@dataclass
class Trace:
    """Complete execution trace of a monadic program.

    Mutable accumulator: ``sites`` grows as the program executes;
    not a value type.

    Parameters
    ----------
    sites : dict[str, SampleSite]
        All sample sites keyed by variable name.
    output : torch.Tensor or dict[str, torch.Tensor]
        The program's return value.
    log_joint : torch.Tensor
        Sum of log-densities across all stochastic sites. Shape
        ``(batch,)``.
    """

    sites: dict[str, SampleSite] = field(default_factory=dict)
    output: torch.Tensor | dict[str, torch.Tensor] | None = None
    log_joint: torch.Tensor | None = None

    @property
    def stochastic_sites(self) -> dict[str, SampleSite]:
        """Return only stochastic (non-deterministic) sites."""
        return {k: v for k, v in self.sites.items() if not v.is_deterministic}

    @property
    def latent_sites(self) -> dict[str, SampleSite]:
        """Return only latent (non-observed, non-deterministic) sites."""
        return {
            k: v
            for k, v in self.sites.items()
            if not v.is_observed and not v.is_deterministic
        }

    @property
    def observed_sites(self) -> dict[str, SampleSite]:
        """Return only observed sites."""
        return {k: v for k, v in self.sites.items() if v.is_observed}
