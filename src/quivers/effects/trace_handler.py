"""Trace handler: records every site the program visits.

`TraceHandler` sits on the handler stack, snapshots every message
during the postprocess pass, and exposes the resulting `Trace` via
the `trace` attribute after the program finishes running. The thin
`quivers.inference.trace.trace` wrapper stacks this handler on and
returns its accumulated `Trace`.
"""

from __future__ import annotations

import torch

from quivers.effects.base import EffectHandler, Message
from quivers.effects.trace_types import SampleSite, Trace


class TraceHandler(EffectHandler):
    """Record every site visited during a program's execution.

    Produces a `Trace` whose ``sites`` dict is keyed by variable
    name, whose ``output`` is the program's return value, and whose
    ``log_joint`` is the sum of every non-let site's ``log_prob``.

    A `TraceHandler` is single-use: run the program under one
    instance, read `trace`, then discard.

    Attributes
    ----------
    trace : Trace
        Accumulator. `output` and `log_joint` are filled in by the
        caller after the program returns (see
        `quivers.inference.trace.trace`).
    """

    def __init__(self) -> None:
        self.trace: Trace = Trace()

    def _pyro_post_sample(self, msg: Message) -> None:
        self._record(msg)

    def _pyro_post_observe(self, msg: Message) -> None:
        self._record(msg)

    def _pyro_post_let(self, msg: Message) -> None:
        self._record(msg)

    def _pyro_post_score(self, msg: Message) -> None:
        self._record(msg)

    def _record(self, msg: Message) -> None:
        assert msg.value is not None
        assert msg.log_prob is not None
        self.trace.sites[msg.name] = SampleSite(
            name=msg.name,
            morphism=msg.morphism,
            value=msg.value,
            log_prob=msg.log_prob,
            is_observed=msg.is_observed,
            is_deterministic=msg.is_deterministic,
        )

    def total_log_joint(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sum every site's ``log_prob`` into the joint density.

        Sample and observe sites contribute their log-density; let
        bindings contribute the zero tensor the interpreter set on
        their message; score steps (compiled marginalize bodies)
        contribute their callable's return value, which the
        interpreter installed as the message's log-prob.

        The accumulator seeds from a scalar zero rather than a
        ``batch_size``-wide zero: the joint is the plain sum of the
        per-site log-densities, whose shape is the broadcast of the
        contributing sites. A replica-batched model whose every site
        carries a leading ``(batch,)`` axis broadcasts to ``(batch,)``,
        recovering the per-replica joint; a single-instance plate model
        whose sites each reduce over their own plate / event axes to a
        scalar (or a length-1 parameter-sample axis) sums to that
        scalar, so a shared prior contributes exactly once instead of
        being replicated across the response plate.
        """
        del batch_size  # the joint's shape follows from the sites
        total = torch.zeros((), device=device)
        for site in self.trace.sites.values():
            total = total + site.log_prob
        return total
