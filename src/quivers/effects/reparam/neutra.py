"""NeuTra: warp HMC geometry via a trained autoguide.

`NeuTraReparam` uses a trained normalizing-flow autoguide (typically
`AutoIAFGuide`) as a change-of-variables to reshape the geometry
HMC and NUTS see. Sampling in the *base* space of the flow
turns the posterior into a near-isotropic Gaussian, and the
sampler's Riemannian metric becomes trivial. The construction is
[Hoffman et al. (2019)](https://arxiv.org/abs/1903.03704).

The handler is site-local at the interface: given a site name it
draws a base value from the flow's variational family, transforms
it back to the model space, and scores it under the model's
original site distribution. Under NUTS this yields the
model-space log-density that the sampler needs to accept /
reject; the flow only reshapes the proposal geometry.
"""

from __future__ import annotations

from quivers.effects.base import Message
from quivers.effects.reparam.base import Reparam, _default_log_prob
from quivers.inference.guides import Guide


class NeuTraReparam(Reparam):
    """Reparameterise a site through a trained normalizing-flow guide.

    Parameters
    ----------
    autoguide : Guide
        A trained `AutoIAFGuide` (or any `Guide` that exposes a
        sample method returning per-site values). At apply time
        the guide is expected to produce a fresh draw whose
        support matches the model's site.
    """

    def __init__(self, autoguide: Guide) -> None:
        self.autoguide = autoguide

    def apply(self, msg: Message) -> None:
        morph = msg.morphism
        assert morph is not None
        assert msg.input is not None

        # Fall through to the model's own sampling path when the
        # guide does not cover the site: NeuTra only reshapes sites
        # the flow was trained on. Scoring uses the original
        # morphism throughout, so a partially-covered NeuTra still
        # produces a well-defined joint density.
        registry = getattr(self.autoguide, "registry", None)
        site_names = set(registry.names()) if registry is not None else set()
        if msg.name not in site_names:
            if msg.value is None:
                msg.value = morph.rsample(msg.input)
            msg.log_prob = _default_log_prob(msg, msg.value)
            return

        # Draw a single joint sample from the guide, then read the
        # site's slice out of the returned dict.
        samples = self.autoguide.sample(msg.input)
        if msg.name not in samples:
            raise KeyError(
                f"NeuTraReparam: autoguide sample did not contain site '{msg.name}'."
            )
        y = samples[msg.name]
        msg.value = y
        msg.log_prob = _default_log_prob(msg, y)
