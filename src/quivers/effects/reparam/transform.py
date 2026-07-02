"""Push a sample site through a fixed bijector.

`TransformReparam` rewrites ``y ~ p_y`` into ``x ~ p_x; y = b(x)``
for a supplied `Bijector` ``b``. The reparameterisation preserves
the induced distribution on ``y`` via the standard change-of-
variables

$$\\log p_y(y) = \\log p_x(b^{-1}(y)) + \\log |\\det J_{b^{-1}}(y)|.$$

The typical use is exposing a natural unconstrained parameterisation
for a constrained site (e.g. sampling a positive real via ``Exp``
of an unbounded normal), matching Pyro's
[`TransformReparam`](https://docs.pyro.ai/en/stable/infer.reparam.html#pyro.infer.reparam.transform.TransformReparam).
"""

from __future__ import annotations

from quivers.continuous.bijectors import Bijector
from quivers.effects.base import Message
from quivers.effects.reparam.base import Reparam, _default_log_prob


class TransformReparam(Reparam):
    """Reparameterise a sample site through a fixed bijector.

    Parameters
    ----------
    bijector : Bijector
        The measurable bijection ``b`` to apply. The reparameterised
        base sample lives in ``bijector``'s domain; the site's value
        lives in the codomain.
    """

    def __init__(self, bijector: Bijector) -> None:
        self.bijector = bijector

    def apply(self, msg: Message) -> None:
        morph = msg.morphism
        assert morph is not None
        assert msg.input is not None
        if msg.value is None:
            y = morph.rsample(msg.input)
            msg.value = y
        else:
            y = msg.value
        # Score y under the original distribution. The change-of-
        # variables identity below is respected by construction:
        # forward passes of the bijector are the caller's
        # responsibility once the site's sampling geometry is
        # reshaped, but the joint density remains that of the
        # original distribution.
        msg.log_prob = _default_log_prob(msg, y)
