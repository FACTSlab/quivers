"""Non-centred parameterisation for location-scale families.

`LocScaleReparam` rewrites a site ``y ~ Normal(mu, sigma)`` into the
non-centred form ``y_raw ~ Normal(0, 1); y = mu + sigma * y_raw``.
The reparameterisation preserves the induced distribution on ``y``
exactly (the change-of-variable Jacobian cancels the affine scale),
while decoupling the strong ``(mu, sigma, y)`` posterior geometry
that gives HMC and NUTS the funnel-collapse pathology described in
[Betancourt and Girolami (2015)](https://arxiv.org/abs/1312.0906)
and isolated as a challenge to samplers in
[Neal (2003)](https://doi.org/10.1214/aos/1056562461).

The strategy is site-local: it reads the site's ``(loc, scale)``, draws
a base sample ``y_raw ~ Normal(0, 1)``, computes the deterministic image
``y = loc + scale * y_raw``, and scores ``y`` under the original
``Normal(loc, scale)``. Because change-of-variables through the affine
map has log-Jacobian ``log|scale|``, the reparameterised score
``log N(y_raw; 0, 1) - log|scale|`` equals ``log N(y; loc, scale)``
exactly, so a downstream inference that already respects the reparam
contract sees identical log-densities.
"""

from __future__ import annotations

import math

import torch

from quivers.effects.reparam.base import Reparam, SiteRequest


class LocScaleReparam(Reparam):
    """Non-centred rewrite for location-scale sample sites.

    The site's morphism must expose a ``_get_params(x)`` method returning
    ``(loc, scale)`` tensors of the same shape, the convention every
    ``ConditionalNormal``-shaped family in
    `quivers.continuous.families` follows.

    Parameters
    ----------
    centered : float
        Interpolation between fully centred (``1.0``) and fully
        non-centred (``0.0``) parameterisations. Values in between
        produce a partial reparam, matching the ``centered`` parameter
        of Pyro's
        [`LocScaleReparam`](https://docs.pyro.ai/en/stable/infer.reparam.html#pyro.infer.reparam.loc_scale.LocScaleReparam).

    Raises
    ------
    ValueError
        If ``centered`` is outside ``[0, 1]``.
    """

    def __init__(self, centered: float = 0.0) -> None:
        if not 0.0 <= centered <= 1.0:
            raise ValueError(
                f"LocScaleReparam: centered must be in [0, 1], got {centered}."
            )
        self.centered = float(centered)

    def apply(self, site: SiteRequest) -> tuple[torch.Tensor, torch.Tensor]:
        """Rewrite one site.

        Parameters
        ----------
        site : SiteRequest
            The site.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The value and its log density under ``Normal(loc, scale)``,
            summed over the trailing feature axis.

        Raises
        ------
        TypeError
            If the site's morphism exposes no ``_get_params``.
        """
        morphism = site.sampleable.morphism
        get_params = getattr(morphism, "_get_params", None)
        if get_params is None:
            raise TypeError(
                f"LocScaleReparam: site '{site.name}' morphism "
                f"{type(morphism).__name__} does not expose `_get_params(x)`; "
                f"LocScaleReparam requires a Normal-family morphism."
            )
        loc, scale = get_params(site.sampleable.input)
        # Partial centring interpolates the effective scale used to
        # push the base sample forward: ``centered=1`` recovers the
        # original sample-then-score path, ``centered=0`` the fully
        # non-centred rewrite.
        effective = scale.pow(self.centered)
        base_scale = scale / effective
        if site.given is None:
            eps = torch.randn_like(loc)
            y_raw = effective * eps
            y = loc + base_scale * y_raw
        else:
            y = site.given
        residual = (y - loc) / scale
        log_p = -0.5 * residual.pow(2) - scale.log() - 0.5 * math.log(2 * math.pi)
        return y, log_p.sum(dim=-1)


__all__ = ["LocScaleReparam"]
