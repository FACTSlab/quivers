"""Scale handler: multiply every site's log-density by a scalar factor.

`ScaleHandler` is the subsampling correction: when a stochastic
gradient sees a mini-batch of size ``M`` drawn from a full dataset
of size ``N``, scaling the likelihood contribution by ``N / M``
recovers an unbiased estimate of the full-data ELBO (see
[Pyro's `scale`](https://docs.pyro.ai/en/stable/poutine.html#pyro.poutine.handlers.scale)
and the SVI derivations in
[Hoffman et al. (2013)](http://jmlr.org/papers/v14/hoffman13a.html)).
"""

from __future__ import annotations

from quivers.effects.base import EffectHandler, Message


class ScaleHandler(EffectHandler):
    """Multiply every site's ``log_prob`` by a fixed scalar factor.

    Sample, observe, and score sites are all scaled; let bindings
    carry zero log-prob and are unchanged. Applying scale outside
    condition inside mask (or any other order) composes: each
    handler rewrites the message in the order the stack sees it.

    Parameters
    ----------
    factor : float
        Multiplicative factor applied to ``log_prob``.
    """

    def __init__(self, factor: float) -> None:
        self.factor = float(factor)

    def _apply(self, msg: Message) -> None:
        if msg.log_prob is None:
            return
        msg.log_prob = msg.log_prob * self.factor

    def _pyro_post_sample(self, msg: Message) -> None:
        self._apply(msg)

    def _pyro_post_observe(self, msg: Message) -> None:
        self._apply(msg)

    def _pyro_post_score(self, msg: Message) -> None:
        self._apply(msg)


def scale(factor: float) -> ScaleHandler:
    """Return a `ScaleHandler` that rescales every log-density by ``factor``."""
    return ScaleHandler(factor)
