"""Mask handler: element-wise gate on log-density contributions.

`MaskHandler` multiplies the per-batch log-density at every site by
a boolean or 0/1 mask tensor. The typical use is row-level opt-out
in a hierarchical model with heterogeneous missingness (see
[Pyro's `mask`](https://docs.pyro.ai/en/stable/poutine.html#pyro.poutine.handlers.mask)).
"""

from __future__ import annotations

import torch

from quivers.effects.base import EffectHandler, Message


class MaskHandler(EffectHandler):
    """Multiply every site's ``log_prob`` by a fixed mask tensor.

    The mask is broadcast against the site's log-density. A boolean
    mask acts as an element-wise switch; a float mask acts as an
    element-wise scale. Sites of every kind (sample, observe,
    score) are affected; let bindings already carry zero log-prob
    and are unchanged.

    Parameters
    ----------
    mask_tensor : torch.Tensor
        Broadcast-compatible mask.
    """

    def __init__(self, mask_tensor: torch.Tensor) -> None:
        self.mask_tensor = mask_tensor

    def _apply(self, msg: Message) -> None:
        if msg.log_prob is None:
            return
        mask = self.mask_tensor.to(dtype=msg.log_prob.dtype, device=msg.log_prob.device)
        msg.log_prob = msg.log_prob * mask

    def _pyro_post_sample(self, msg: Message) -> None:
        self._apply(msg)

    def _pyro_post_observe(self, msg: Message) -> None:
        self._apply(msg)

    def _pyro_post_score(self, msg: Message) -> None:
        self._apply(msg)


def mask(mask_tensor: torch.Tensor) -> MaskHandler:
    """Return a `MaskHandler` gated on the given mask tensor."""
    return MaskHandler(mask_tensor)
