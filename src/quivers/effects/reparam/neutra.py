"""NeuTra reparameterisation through a trained normalizing-flow guide.

`NeuTraReparam` warps the geometry a sampler sees through a trained
autoguide, after [Hoffman et al. (2019)](https://arxiv.org/abs/1903.03704).
"""

from __future__ import annotations

import torch

from quivers.effects.reparam.base import Reparam, SiteRequest
from quivers.inference.guides.base import Guide


class NeuTraReparam(Reparam):
    """Reparameterise a site through a trained normalizing-flow guide.

    Parameters
    ----------
    autoguide : Guide
        A trained `AutoIAFGuide`, or any `Guide` whose ``sample`` returns
        per-site values, whose draw at apply time has the site's
        support.
    """

    def __init__(self, autoguide: Guide) -> None:
        self.autoguide = autoguide

    def apply(self, site: SiteRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Rewrite one site.

        Parameters
        ----------
        site : SiteRequest
            The site.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The guide's draw for the site, or the site's own draw when
            the guide does not cover it, and its log density under the
            site's sampleable; scoring uses the original sampleable
            throughout, so a partially covered NeuTra still yields a
            well-defined joint.

        Raises
        ------
        KeyError
            If the guide covers the site but its draw does not carry it.
        """
        registry = getattr(self.autoguide, "registry", None)
        names = set(registry.names()) if registry is not None else set()
        if site.given is not None:
            y = site.given
        elif site.name not in names:
            y = site.sampleable.rsample()
        else:
            samples = self.autoguide.sample(site.sampleable.input)
            if site.name not in samples:
                raise KeyError(
                    f"NeuTraReparam: autoguide sample did not contain site "
                    f"'{site.name}'."
                )
            y = samples[site.name]
        return y, site.sampleable.log_prob(y)


__all__ = ["NeuTraReparam"]
