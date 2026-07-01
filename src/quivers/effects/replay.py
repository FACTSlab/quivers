"""Replay handler: feed values from a captured trace.

`ReplayHandler` reproduces the sample values recorded by a prior
`TraceHandler` run. It is the natural pair to `TraceHandler` for
gradient-through-samples SVI: sample once, replay against a
different program (or the same program with different parameters)
to score under the current model without redrawing the noise (see
[Pyro's `replay`](https://docs.pyro.ai/en/stable/poutine.html#pyro.poutine.handlers.replay)).
"""

from __future__ import annotations

from quivers.effects.base import EffectHandler, Message
from quivers.effects.trace_types import Trace


class ReplayHandler(EffectHandler):
    """Replace every sample site with the value recorded in ``trace``.

    A site whose name appears in ``trace.sites`` receives that
    site's ``value``; the interpreter then falls back to the
    default log-prob computation, so replay does not fix the
    site's density under the current model, only its value.

    Parameters
    ----------
    trace : Trace
        A trace whose site values will be replayed.
    """

    def __init__(self, trace: Trace) -> None:
        self._trace = trace

    def _pyro_sample(self, msg: Message) -> None:
        if msg.name in self._trace.sites:
            msg.value = self._trace.sites[msg.name].value


def replay(trace: Trace) -> ReplayHandler:
    """Return a `ReplayHandler` bound to the given `Trace`."""
    return ReplayHandler(trace)
