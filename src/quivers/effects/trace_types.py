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
from quivers.effects.sites import TorchSampleable

type SiteAddress = tuple[str, tuple[tuple[str, str | int], ...], tuple[int, ...]]
"""The kernel's dynamic address of one occurrence of a site."""


@dataclass
class SampleSite:
    """Record of a single site in a program trace.

    Holds a ``torch.Tensor`` per site; not a value type.

    Parameters
    ----------
    name : str
        Variable name bound at this site.
    morphism : ContinuousMorphism or None
        The distribution morphism (``None`` for let bindings and scores).
    value : torch.Tensor
        The sampled, observed, or computed value.
    log_prob : torch.Tensor
        The density the site contributed to the run's score, after every
        score transformer on the stack. Zero for let bindings and for
        intervened sites.
    is_observed : bool
        Whether this site was conditioned on an observed value.
    is_deterministic : bool
        Whether this is a let binding, a score step, or an intervened
        site.
    sampleable : TorchSampleable or None
        The sampleable the site's request carried, with the morphism and
        the input it was conditioned on; ``None`` for let bindings and
        scores.
    address : SiteAddress or None
        The kernel's dynamic address of the site's request, which is one
        identity for the run's trace and for every handler that answered
        it.
    metadata : dict[str, object]
        Annotations handlers attached to the site.
    """

    name: str
    morphism: ContinuousMorphism | None
    value: torch.Tensor
    log_prob: torch.Tensor
    is_observed: bool = False
    is_deterministic: bool = False
    sampleable: TorchSampleable | None = None
    address: SiteAddress | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Trace:
    """Complete execution trace of a monadic program.

    Mutable accumulator: ``sites`` grows as the program executes;
    not a value type.

    Parameters
    ----------
    sites : dict[str, SampleSite]
        All sites keyed by variable name.
    output : torch.Tensor or dict[str, torch.Tensor]
        The program's return value.
    log_joint : torch.Tensor
        Sum of log-densities across all sites. Shape ``(batch,)``.
    """

    sites: dict[str, SampleSite] = field(default_factory=dict)
    output: torch.Tensor | dict[str, torch.Tensor] | None = None
    log_joint: torch.Tensor | None = None

    @property
    def stochastic_sites(self) -> dict[str, SampleSite]:
        """The sites that are not deterministic.

        Returns
        -------
        dict[str, SampleSite]
            Every sampled or observed site.
        """
        return {k: v for k, v in self.sites.items() if not v.is_deterministic}

    @property
    def latent_sites(self) -> dict[str, SampleSite]:
        """The sites that are neither observed nor deterministic.

        Returns
        -------
        dict[str, SampleSite]
            Every sampled latent site.
        """
        return {
            k: v
            for k, v in self.sites.items()
            if not v.is_observed and not v.is_deterministic
        }

    @property
    def observed_sites(self) -> dict[str, SampleSite]:
        """The observed sites.

        Returns
        -------
        dict[str, SampleSite]
            Every site conditioned on an observation.
        """
        return {k: v for k, v in self.sites.items() if v.is_observed}


__all__ = ["SampleSite", "SiteAddress", "Trace"]
