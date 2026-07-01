"""Pearl's do-operator as an effect handler.

`DoHandler` replaces named sample sites with fixed values *without*
contributing their log-density to the joint. This is the
interventional semantics of
[Pearl (1995)](https://doi.org/10.1093/biomet/82.4.669) and
[Pearl (2009)](https://doi.org/10.1017/CBO9780511803161): removing
the incoming edges into the intervened variable and forcing its
value. Contrast with `condition`, which observes the site and still
scores it.
"""

from __future__ import annotations

import torch

from quivers.effects.base import EffectHandler, Message


class DoHandler(EffectHandler):
    """Perform Pearl's do-intervention on named sample sites.

    A site whose name appears in ``data`` is rewritten to a
    deterministic site with value ``data[name]`` and ``log_prob``
    equal to zero. Downstream sites see the intervened value; the
    intervened site itself no longer contributes to the joint,
    because under intervention its distribution is replaced by a
    point mass.

    Parameters
    ----------
    data : dict[str, torch.Tensor]
        Site-name -> intervened value bindings.
    """

    def __init__(self, data: dict[str, torch.Tensor]) -> None:
        self.data = data

    def _pyro_sample(self, msg: Message) -> None:
        if msg.name in self.data:
            val = self.data[msg.name]
            msg.value = val
            msg.log_prob = torch.zeros(
                val.shape[:1] if val.dim() > 0 else (1,),
                device=val.device,
            )
            msg.is_deterministic = True


def do(data: dict[str, torch.Tensor]) -> DoHandler:
    """Return a `DoHandler` that intervenes on the given sites."""
    return DoHandler(data)
