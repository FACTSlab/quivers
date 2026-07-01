"""Reparameterization strategies as effect handlers.

A `Reparam` handler intercepts sample sites and rewrites them
according to a bijection that preserves the induced distribution
while reshaping the geometry seen by downstream inference. The
canonical use case is HMC on funnel geometries: reparameterising
``y ~ Normal(0, exp(z))`` into ``y_raw ~ Normal(0, 1); y = exp(z) *
y_raw`` decouples the tight scale-location dependency that would
otherwise kill NUTS
([Betancourt and Girolami 2015](https://arxiv.org/abs/1312.0906)).

Each concrete reparam strategy implements `apply(msg)`, which
returns the transformed value the interpreter should install and
the transformed log-density contribution. The dispatch
`ReparamOrchestrator` maps site names to strategies so a user can
write

    with reparam({"theta": LocScaleReparam(), "z": NeuTraReparam(guide)}):
        samples = nuts.run(model, x, observations)

and have per-site strategies apply in one pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.effects.base import EffectHandler, Message


class Reparam(ABC):
    """Base class for site-level reparameterisation strategies.

    A strategy is a *policy*, not a handler. The
    `ReparamOrchestrator` handler dispatches site messages to the
    matching strategy and installs the strategy's output on the
    message. Subclasses override `apply`.
    """

    @abstractmethod
    def apply(self, msg: Message) -> None:
        """Rewrite ``msg`` in place.

        Implementations must:

        * set ``msg.value`` to the reparameterised value in the
          site's original space (so downstream sites see the
          intended value),
        * set ``msg.log_prob`` to the reparameterised log-density
          contribution that a downstream sampler should score
          against.
        """


class ReparamOrchestrator(EffectHandler):
    """Dispatch sample-site messages to per-site reparam strategies.

    Parameters
    ----------
    strategies : dict[str, Reparam]
        Site-name -> strategy dispatch table.
    """

    def __init__(self, strategies: dict[str, Reparam]) -> None:
        self.strategies = dict(strategies)

    def _pyro_sample(self, msg: Message) -> None:
        strategy = self.strategies.get(msg.name)
        if strategy is None:
            return
        strategy.apply(msg)


def reparam(strategies: dict[str, Reparam]) -> ReparamOrchestrator:
    """Return a `ReparamOrchestrator` for the given per-site strategies."""
    return ReparamOrchestrator(strategies)


def _default_sample(msg: Message) -> torch.Tensor:
    """Draw a sample from the site's underlying morphism.

    Used by reparam strategies that need a fresh draw from the
    original distribution (e.g. to construct the deterministic
    forward image of a reparameterised base sample).
    """
    assert msg.morphism is not None
    assert msg.input is not None
    return msg.morphism.rsample(msg.input)


def _default_log_prob(msg: Message, value: torch.Tensor) -> torch.Tensor:
    """Score ``value`` under the site's underlying morphism."""
    assert msg.morphism is not None
    assert msg.input is not None
    return msg.morphism.log_prob(msg.input, value)
