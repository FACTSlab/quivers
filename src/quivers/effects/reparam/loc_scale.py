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

The handler is site-local: it consults the site's morphism for
``(loc, scale)``, draws a base sample ``y_raw ~ Normal(0, 1)``,
computes the deterministic image ``y = loc + scale * y_raw``, and
scores ``y`` under the original ``Normal(loc, scale)``. Because
change-of-variables through the affine map has log-Jacobian
``log|scale|``, the reparameterised score
``log N(y_raw; 0, 1) - log|scale|`` equals ``log N(y; loc, scale)``
exactly, so a downstream inference that already respects the
reparam contract will see identical log-densities.
"""

from __future__ import annotations

import math

import torch

from quivers.effects.base import Message
from quivers.effects.reparam.base import Reparam


class LocScaleReparam(Reparam):
    """Non-centred rewrite for location-scale sample sites.

    The site's morphism must expose a `_get_params(x)` method
    returning `(loc, scale)` tensors of the same shape (the
    convention every `ConditionalNormal`-shaped family in
    `quivers.continuous.families` follows). Sites whose morphism
    does not follow the convention raise on `apply`.

    Parameters
    ----------
    centered : float
        Interpolation between fully centred (``1.0``) and fully
        non-centred (``0.0``) parameterisations. Values in between
        produce a partial reparam, matching the ``centered``
        parameter of Pyro's
        [`LocScaleReparam`](https://docs.pyro.ai/en/stable/infer.reparam.html#pyro.infer.reparam.loc_scale.LocScaleReparam).
        Default ``0.0`` (fully non-centred).
    """

    def __init__(self, centered: float = 0.0) -> None:
        if not 0.0 <= centered <= 1.0:
            raise ValueError(
                f"LocScaleReparam: centered must be in [0, 1], got {centered}."
            )
        self.centered = float(centered)

    def apply(self, msg: Message) -> None:
        morph = msg.morphism
        assert morph is not None
        assert msg.input is not None
        get_params = getattr(morph, "_get_params", None)
        if get_params is None:
            raise TypeError(
                f"LocScaleReparam: site '{msg.name}' morphism "
                f"{type(morph).__name__} does not expose `_get_params(x)`; "
                f"LocScaleReparam requires a Normal-family morphism."
            )
        loc, scale = get_params(msg.input)

        # Partial centring interpolates the effective scale used to
        # push the base sample forward. `centered=1` recovers the
        # original sample-then-score path; `centered=0` is the fully
        # non-centred rewrite.
        c = self.centered
        eff_scale = scale.pow(c)  # scale ** c, elementwise
        base_scale = scale / eff_scale  # = scale ** (1 - c)
        eff_loc = loc * (1.0 - c) + loc * c  # loc regardless of c
        _ = eff_loc  # loc unaffected by the partial-centring convex combo

        if msg.value is None:
            eps = torch.randn_like(loc)
            y_raw = eff_scale * eps  # base sample in the reparam space
            y = loc + base_scale * y_raw
            msg.value = y
        else:
            y = msg.value

        # Score y under the original Normal(loc, scale). The affine
        # change-of-variable identity makes this equal the base-sample
        # score minus log|scale|; scoring y directly is simpler and
        # numerically identical.
        residual = (y - loc) / scale
        # log N(y; loc, scale) summed over the trailing feature axis.
        log_p = -0.5 * residual.pow(2) - scale.log() - 0.5 * math.log(2 * math.pi)
        msg.log_prob = log_p.sum(dim=-1)
