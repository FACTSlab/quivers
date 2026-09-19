"""Reparameterisation through a fixed bijector."""

from __future__ import annotations

import torch

from quivers.continuous.bijectors import Bijector
from quivers.effects.reparam.base import Reparam, SiteRequest


class TransformReparam(Reparam):
    """Reparameterise a sample site through a fixed bijector.

    The reparameterised base sample lives in the bijector's domain and
    the site's value in its codomain. The joint density remains that of
    the original distribution: the strategy scores the value under the
    site's own sampleable, and the bijector reshapes the sampling
    geometry a downstream sampler works in.

    Parameters
    ----------
    bijector : Bijector
        The measurable bijection ``b`` the site's geometry is reshaped
        through.
    """

    def __init__(self, bijector: Bijector) -> None:
        self.bijector = bijector

    def apply(self, site: SiteRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Rewrite one site.

        Parameters
        ----------
        site : SiteRequest
            The site.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The value, drawn from the site's sampleable unless given, and
            its log density under the sampleable.
        """
        y = site.given if site.given is not None else site.sampleable.rsample()
        return y, site.sampleable.log_prob(y)


__all__ = ["TransformReparam"]
