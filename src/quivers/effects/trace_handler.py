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


def _narrowest_shape(log_probs: list[torch.Tensor]) -> tuple[int, ...]:
    """The widest shape no site has to be replicated into.

    Shapes are right-aligned (a missing leading axis reads as extent
    1, exactly as broadcasting reads it) and the extent kept for each
    axis is the smallest any site carries there. Adding sites at that
    shape can only ever reduce a site, never repeat one.

    Parameters
    ----------
    log_probs : list of torch.Tensor
        Per-site log-densities that carry density. An empty list has
        no axis to agree on and gives the scalar shape.

    Returns
    -------
    tuple of int
        Target shape, right-aligned against every site.
    """
    if not log_probs:
        return ()
    rank = max(log_prob.dim() for log_prob in log_probs)
    extents: list[int] = []
    for axis in range(rank):
        smallest = None
        for log_prob in log_probs:
            offset = rank - log_prob.dim()
            here = 1 if axis < offset else int(log_prob.shape[axis - offset])
            smallest = here if smallest is None else min(smallest, here)
        assert smallest is not None
        extents.append(smallest)
    return tuple(extents)


def _reduce_to(log_prob: torch.Tensor, target: tuple[int, ...]) -> torch.Tensor:
    """Sum ``log_prob`` over every axis where it is wider than ``target``.

    The reduction keeps the axis at extent 1 rather than dropping it,
    so the result still lines up with ``target`` under the addition
    that follows and no term is repeated by that addition. An axis
    ``target`` does not reach at all belongs to a site of higher rank
    than any density-carrying one, and is summed away outright rather
    than kept, so it cannot widen the joint either.

    Parameters
    ----------
    log_prob : torch.Tensor
        One site's log-density.
    target : tuple of int
        Shape from
        [`_narrowest_shape`][quivers.effects.trace_handler._narrowest_shape],
        right-aligned against ``log_prob``.

    Returns
    -------
    torch.Tensor
        ``log_prob`` with its lane axes summed out.
    """
    rank = len(target)
    while log_prob.dim() > rank:
        log_prob = log_prob.sum(dim=0)
    offset = rank - log_prob.dim()
    axes = [
        axis - offset
        for axis in range(rank)
        if axis >= offset and int(log_prob.shape[axis - offset]) > target[axis]
    ]
    if not axes:
        return log_prob
    return log_prob.sum(dim=axes, keepdim=True)


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

        The accumulator uses the narrowest right-aligned shape shared
        by the nonzero site contributions. Wider axes are summed before
        addition so broadcasting cannot duplicate scalar terms. Zero
        contributions do not determine the target shape but are still
        added.

        Parameters
        ----------
        batch_size : int
            Leading extent of the program input. Not consulted: the
            joint's shape follows from the sites themselves, and a
            program whose sites reduce their own plate axes has a
            joint narrower than its input.
        device : torch.device
            Device the accumulator seeds on.

        Returns
        -------
        torch.Tensor
            The joint log-density.
        """
        del batch_size  # the joint's shape follows from the sites
        contributions = [site.log_prob for site in self.trace.sites.values()]
        target = _narrowest_shape(
            [lp for lp in contributions if not bool(torch.all(lp == 0))]
        )
        total = torch.zeros((), device=device)
        for log_prob in contributions:
            total = total + _reduce_to(log_prob, target)
        return total
