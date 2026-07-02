"""Lift handler: turn every learnable parameter into a sample site.

`LiftHandler` walks the underlying morphism's `nn.Parameter`
buffers and rewrites each into a sample site drawn from a Normal
prior. The result is a Bayesian version of a point-estimated model
(see [Pyro's `lift`](https://docs.pyro.ai/en/stable/poutine.html#pyro.poutine.handlers.lift)).

The current implementation records prior samples for each parameter
into the handler's ``sampled_params`` dict; wiring the samples back
into the morphism's forward pass requires a categorical-combinator
IR extension that turns parameter access itself into an effect
message. Until that IR lands, the handler exposes the sampled
values so a downstream trainer can consume them (e.g. as a
metaparameter perturbation) without silently pretending the model's
own parameters were replaced.
"""

from __future__ import annotations

import torch

from quivers.effects.base import EffectHandler, Message


class LiftHandler(EffectHandler):
    """Draw every reachable ``nn.Parameter`` from a Normal prior.

    On each program invocation, the handler samples one value per
    parameter site (using the parameter's current shape) from
    ``Normal(0, prior_scale)``. The samples are stored in
    ``sampled_params`` keyed by fully-qualified parameter name;
    outer handlers see the samples through post-hook messages
    named ``"param/<qualified-name>"``.

    Parameters
    ----------
    prior_scale : float
        Standard deviation of the Normal prior. Default ``1.0``.
    """

    def __init__(self, prior_scale: float = 1.0) -> None:
        self.prior_scale = float(prior_scale)
        self.sampled_params: dict[str, torch.Tensor] = {}
        self._activated_this_run: bool = False

    def _pyro_post_sample(self, msg: Message) -> None:
        # Sample a prior draw for every parameter of the site's
        # morphism the first time we see a sample-kind message.
        # `msg.morphism` is a `ContinuousMorphism` (`nn.Module`), so
        # `named_parameters` enumerates every leaf tensor with
        # requires_grad. Subsequent sites reuse the same draws.
        if self._activated_this_run or msg.morphism is None:
            return
        module = msg.morphism
        for qualified_name, param in module.named_parameters():
            key = f"{msg.name}.{qualified_name}"
            noise = torch.randn_like(param.data)
            self.sampled_params[key] = param.data + self.prior_scale * noise
        self._activated_this_run = True

    def reset(self) -> None:
        """Clear per-run state before the next execution."""
        self.sampled_params.clear()
        self._activated_this_run = False


def lift(prior_scale: float = 1.0) -> LiftHandler:
    """Return a `LiftHandler` with the given prior standard deviation."""
    return LiftHandler(prior_scale)
