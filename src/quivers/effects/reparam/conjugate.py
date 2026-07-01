"""Analytical conjugate collapse as a reparameterisation.

`ConjugateReparam` looks up the site's (parent-family, child-family)
in the `quivers.effects.collapse` registry and installs the closed-
form marginal in place of the sample-driven MC estimate. The
result is a Rao-Blackwellised estimator with strictly lower
variance for the site's contribution, matching Pyro's
[`ConjugateReparam`](https://docs.pyro.ai/en/stable/infer.reparam.html#pyro.infer.reparam.conjugate.ConjugateReparam).

The registry currently ships pending solvers for the standard
Normal-Normal, Beta-Bernoulli, Gamma-Poisson, and Dirichlet-
Categorical pairs (see `quivers.effects.collapse`). A caller that
enables `ConjugateReparam` for a site whose analytic solver has not
been supplied raises a `NotImplementedError` at apply time, per the
no-fallbacks policy: silent MC estimation would defeat the point of
requesting collapse.
"""

from __future__ import annotations

from quivers.effects.base import Message
from quivers.effects.collapse import _CONJUGATE_REGISTRY
from quivers.effects.reparam.base import Reparam, _default_log_prob


class ConjugateReparam(Reparam):
    """Rewrite a site's log-density via a registered conjugate marginal.

    Parameters
    ----------
    parent_family : str
        Name of the parent (prior) family.
    child_family : str
        Name of the child (likelihood) family.
    parent_msg : Message or None
        The parent site's message. Passed through to the solver.
        ``None`` when the caller supplies parent state through
        another channel (e.g. by name lookup at solver time).
    """

    def __init__(
        self,
        parent_family: str,
        child_family: str,
        parent_msg: Message | None = None,
    ) -> None:
        self.parent_family = parent_family
        self.child_family = child_family
        self.parent_msg = parent_msg

    def apply(self, msg: Message) -> None:
        key = (self.parent_family, self.child_family)
        solver = _CONJUGATE_REGISTRY.get(key)
        if solver is None:
            raise KeyError(
                f"ConjugateReparam: no analytic solver registered for "
                f"({self.parent_family}, {self.child_family}); use "
                f"`quivers.effects.collapse.register_conjugate_pair` to "
                f"install one."
            )
        if msg.value is None:
            morph = msg.morphism
            assert morph is not None
            assert msg.input is not None
            msg.value = morph.rsample(msg.input)
        parent = self.parent_msg if self.parent_msg is not None else msg
        msg.log_prob = solver(parent, msg)
        # Fall back to the direct score when the solver returned a
        # tensor of a different shape than expected: the collapse
        # contract requires (batch,)-shaped log-densities, and any
        # mismatch signals a solver bug the user should see.
        expected_shape = _default_log_prob(msg, msg.value).shape
        if msg.log_prob.shape != expected_shape:
            raise ValueError(
                f"ConjugateReparam: solver for {key} returned log-prob of "
                f"shape {tuple(msg.log_prob.shape)}; expected "
                f"{tuple(expected_shape)}."
            )
