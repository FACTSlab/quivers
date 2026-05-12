"""Mixture variational guide.

A finite mixture of component guides:

.. math::

    q(z) = \\sum_{k=1}^{K} \\pi_k\\, q_k(z),

with :math:`\\pi \\in \\Delta^{K-1}` parameterised by an
unconstrained logit vector. Sampling draws a component index
:math:`k \\sim \\mathrm{Categorical}(\\pi)` (Gumbel-Softmax-relaxed
during training so the choice is reparameterised) and then samples
:math:`z \\sim q_k`. Log-density uses ``logsumexp``:

.. math::

    \\log q(z) = \\mathrm{logsumexp}_k\\bigl(
        \\log \\pi_k + \\log q_k(z)\\bigr).

The component guides may be of any :class:`Guide` subclass. The
canonical use case is to wrap several :class:`AutoNormalGuide`
instances to recover a multimodal posterior that no single
unimodal guide can capture.

Gradients flow through both the component variational parameters
(via reparameterisation inside each component) and the mixture
logits (via the Gumbel-Softmax relaxation of the component pick).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from quivers.inference.guides.base import Guide


class AutoMixtureGuide(Guide):
    """Finite mixture variational guide.

    Parameters
    ----------
    components : list[Guide]
        Component guides. All components must share the same
        :class:`LatentRegistry` (i.e. be built against the same
        model + observed-name set).
    init_temperature : float
        Initial Gumbel-Softmax temperature. Default ``1.0``;
        anneal toward zero for sharper component selection.
    """

    def __init__(
        self,
        components: list[Guide],
        init_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if len(components) < 2:
            raise ValueError(
                f"AutoMixtureGuide: need at least 2 components, "
                f"got {len(components)}"
            )
        reference = components[0]
        ref_names = tuple(reference.registry.names)
        for i, comp in enumerate(components[1:], 1):
            comp_names = tuple(comp.registry.names)
            if comp_names != ref_names:
                raise ValueError(
                    f"AutoMixtureGuide: component {i} has different "
                    f"latent names {comp_names!r} than component 0 "
                    f"{ref_names!r}"
                )
        if init_temperature <= 0.0:
            raise ValueError(
                f"AutoMixtureGuide: init_temperature must be positive, "
                f"got {init_temperature}"
            )
        self._registry = reference.registry
        self.components = nn.ModuleList(components)
        self.mixture_logits = nn.Parameter(torch.zeros(len(components)))
        self._temperature: torch.Tensor
        self.register_buffer(
            "_temperature", torch.tensor(float(init_temperature))
        )

    @property
    def num_components(self) -> int:
        return len(self.components)

    @property
    def temperature(self) -> float:
        """Current Gumbel-Softmax temperature."""
        return float(self._temperature.item())

    def set_temperature(self, value: float) -> None:
        """Anneal the Gumbel-Softmax temperature."""
        if value <= 0.0:
            raise ValueError(
                f"AutoMixtureGuide.set_temperature: must be positive, "
                f"got {value}"
            )
        self._temperature.fill_(float(value))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reparameterised mixture draw via Gumbel-Softmax.

        Each call samples a Gumbel-Softmax weight vector
        :math:`w \\in \\Delta^{K-1}` and returns
        :math:`\\sum_k w_k \\cdot v^{(k)}` per site, where
        :math:`v^{(k)}` is component ``k``'s constrained-space
        sample. Because the constrained-space sites' supports are
        not in general convex (e.g. a Cholesky factor on
        :data:`torch.distributions.constraints.corr_cholesky`), the
        soft mixture can drift outside any single component's
        support during training; the categorical-pick fallback in
        :meth:`hard_rsample` returns a single component's sample
        for use at inference time.
        """
        gumbel_logits = (
            self.mixture_logits
            - torch.empty_like(self.mixture_logits).exponential_().log()
        )
        w = F.softmax(gumbel_logits / self._temperature, dim=-1)
        component_samples = [comp.rsample(x) for comp in self.components]

        result: dict[str, torch.Tensor] = {}
        for site_name in self._registry.names:
            stacked = torch.stack(
                [comp_samples[site_name] for comp_samples in component_samples],
                dim=0,
            )
            # Broadcast w against the stacked shape.
            broadcast_shape = (
                self.num_components,
            ) + (1,) * (stacked.dim() - 1)
            result[site_name] = (w.reshape(broadcast_shape) * stacked).sum(dim=0)
        return result

    def hard_rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Categorical-pick variant: sample a component index and
        return that component's draw verbatim. Use at inference
        time when soft-mixture interpolation would violate a
        support constraint."""
        probs = F.softmax(self.mixture_logits, dim=-1)
        k = int(torch.distributions.Categorical(probs=probs).sample().item())
        return self.components[k].rsample(x)

    # ------------------------------------------------------------------
    # Log-density
    # ------------------------------------------------------------------

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Mixture log-density via logsumexp over components."""
        log_pi = F.log_softmax(self.mixture_logits, dim=-1)
        component_log_probs = torch.stack(
            [comp.log_prob(x, sites) for comp in self.components], dim=0
        )
        return torch.logsumexp(
            log_pi.unsqueeze(-1) + component_log_probs, dim=0
        )

    @property
    def latent_names(self) -> list[str]:
        return list(self._registry.names)


__all__ = ["AutoMixtureGuide"]
