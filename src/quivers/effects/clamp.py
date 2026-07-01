"""Clamp handler: pin named sample sites to fixed values.

`ClampHandler` (constructed via the `clamp` factory) rewrites
every sample site named in its data dict into an observed site
with the supplied value. The site's log-density is still scored
under the underlying distribution, so the joint density after
clamping is the posterior score up to normalisation. This is the
handler-stack analogue of Pyro's
[`pyro.poutine.condition`](https://docs.pyro.ai/en/stable/poutine.html#pyro.poutine.handlers.condition).

The name `clamp` avoids the collision with the top-level
[`quivers.inference.conditioning.condition`][quivers.inference.conditioning.condition]
factory, which returns a
[`Conditioned`][quivers.inference.conditioning.Conditioned] model
wrapper: a different abstraction covering the same intent through
a non-handler surface.
"""

from __future__ import annotations

import torch

from quivers.effects.base import EffectHandler, Message


class ClampHandler(EffectHandler):
    """Clamp sample sites to observed values.

    A sample site whose name appears in ``data`` is rewritten to an
    observe site: its value is set to ``data[name]``, its
    ``is_observed`` flag is set, and the interpreter falls back to
    the default ``morph.log_prob(inp, value)`` for the density.
    Other sites pass through untouched.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Site-name -> value bindings.
    """

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        self.data = data

    def _pyro_sample(self, msg: Message) -> None:
        if msg.name in self.data:
            msg.value = self.data[msg.name]
            msg.is_observed = True


def clamp(data: dict[str, torch.Tensor]) -> ClampHandler:
    """Return a `ClampHandler` that clamps the given sites."""
    return ClampHandler(data)
