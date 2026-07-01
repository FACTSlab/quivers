"""Collapse handler: analytical marginalisation of conjugate site pairs.

`CollapseHandler` marks named site pairs as conjugate-collapsible.
When a downstream sampler asks for the marginal likelihood of the
observed child, the handler consults its registry of known
conjugate pairs and emits the closed-form marginal instead of the
sample-driven MC estimate. The pairs currently registered are the
standard Gaussian-Gaussian, Beta-Bernoulli, Gamma-Poisson, and
Dirichlet-Categorical conjugacies documented in
[Bernardo and Smith (1994)](https://doi.org/10.1002/9780470316870),
matching Pyro's `pyro.contrib.optim.multi_optim.collapse` surface.

The registered analytic solvers are stubs pending a combinator IR
extension that lets a handler intercept and rewrite a full
sub-program instead of a single site. The stub raises when a
downstream sampler asks for a collapsed density, which is the
intended behaviour under the "no fallbacks" policy: inference
algorithms should discover an unimplemented collapse during setup,
not at gradient time.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from quivers.effects.base import EffectHandler, Message


# Known conjugate pairs. Each entry maps (parent-family, child-family)
# to a solver: (parent_msg, child_msg) -> marginal log-density. The
# concrete solvers plug in once the combinator IR exposes per-family
# distribution parameters through the message channel.
_CONJUGATE_REGISTRY: dict[
    tuple[str, str],
    Callable[[Message, Message], torch.Tensor],
] = {}


def register_conjugate_pair(
    parent_family: str,
    child_family: str,
    solver: Callable[[Message, Message], torch.Tensor],
) -> None:
    """Register a closed-form marginal for a conjugate pair.

    Called at import time by family-specific modules that ship
    analytical collapse recipes.
    """
    _CONJUGATE_REGISTRY[(parent_family, child_family)] = solver


# Standard conjugate pair registrations. The solvers are placeholders
# that raise, per the "no fallbacks" discipline: any inference
# algorithm that opts into collapse must supply the analytic
# implementation instead of silently degrading to MC. See the module
# docstring for the rationale.


def _pending(name: str) -> Callable[[Message, Message], torch.Tensor]:
    def solver(parent_msg: Message, child_msg: Message) -> torch.Tensor:
        raise NotImplementedError(
            f"CollapseHandler: analytic solver for '{name}' is registered "
            f"but not yet implemented. Supply a concrete solver via "
            f"`quivers.effects.collapse.register_conjugate_pair`."
        )

    return solver


register_conjugate_pair("Normal", "Normal", _pending("Normal-Normal"))
register_conjugate_pair("Beta", "Bernoulli", _pending("Beta-Bernoulli"))
register_conjugate_pair("Gamma", "Poisson", _pending("Gamma-Poisson"))
register_conjugate_pair("Dirichlet", "Categorical", _pending("Dirichlet-Categorical"))


class CollapseHandler(EffectHandler):
    """Mark a program for analytical conjugate collapse.

    Attaching this handler declares intent: any downstream inference
    algorithm that respects collapse messages (a future NUTS pass,
    for instance) should look up the analytical marginal in the
    handler's registry instead of scoring the parent site
    stochastically. Handlers annotate each collapsed parent site
    with ``metadata["collapse"] = child_name`` during the process
    pass; the child site's density is left untouched so ordinary
    interpreters still produce a well-defined joint.
    """

    def __init__(self, pairs: dict[str, str] | None = None) -> None:
        """Parameters
        ----------
        pairs : dict[str, str] or None
            Parent-name -> child-name pairs to mark for collapse.
            ``None`` marks no site (a no-op handler that carries
            the registry for downstream introspection).
        """
        self.pairs = dict(pairs) if pairs is not None else {}
        self.registry = _CONJUGATE_REGISTRY

    def _pyro_sample(self, msg: Message) -> None:
        if msg.name in self.pairs:
            msg.metadata["collapse"] = self.pairs[msg.name]


def collapse(pairs: dict[str, str] | None = None) -> CollapseHandler:
    """Return a `CollapseHandler` that annotates the given pairs."""
    return CollapseHandler(pairs)
