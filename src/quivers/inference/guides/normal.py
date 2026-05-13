"""Mean-field Normal variational guide.

:class:`AutoNormalGuide` factorizes the variational posterior as
a product of independent Normals — one per latent site in
unconstrained space — and pushes each through the site's
constrained-support bijector. This is the smallest and fastest
guide quivers ships; it works well when posterior correlations
are weak (a deliberately wide class of problems) and serves as
the warm-start for richer guides
(:class:`~quivers.inference.guides.multivariate_normal.AutoMultivariateNormal`,
:class:`~quivers.inference.guides.flow.AutoIAFGuide`, …).

The construction follows Pyro's ``AutoNormal``:

1. For each latent site :math:`v_i` with prior support
   :math:`\\mathrm{supp}(p_i) \\subseteq B_i`, maintain
   :math:`(\\mathrm{loc}_i, \\log\\mathrm{scale}_i) \\in
   \\mathbb{R}^{d_i} \\times \\mathbb{R}^{d_i}` where
   :math:`d_i = \\dim T_i^{-1}(B_i)` is the unconstrained-side
   event dimension.
2. Sample :math:`z_i \\sim \\mathcal{N}(\\mathrm{loc}_i, \\exp(\\log\\mathrm{scale}_i))`.
3. Return :math:`v_i = T_i(z_i)` where
   :math:`T_i = \\mathsf{biject\\_to}(\\mathrm{supp}(p_i))`.

Log-density is the change-of-variables identity:

.. math::

    \\log q(v) = \\sum_i \\Bigl[
        \\log\\mathcal{N}(z_i;\\, \\mathrm{loc}_i, \\mathrm{scale}_i)
        + \\log\\bigl|\\det J_{T_i^{-1}}(v_i)\\bigr|
    \\Bigr].

Plate latents (:class:`~quivers.continuous.bayesian.PlateDraw`)
are stored as ``(|A|, d_i)`` parameter tensors and sampled batch-
invariantly: the latent vector is a global model parameter shared
across every row of an observed plate, not replicated against the
program input's leading batch axis. This matches the model-side
:meth:`PlateDraw.rsample` convention and is the standard Pyro /
NumPyro plate semantic.
"""

from __future__ import annotations

import torch
import torch.distributions as D
import torch.nn as nn

from quivers.continuous.programs import MonadicProgram
from quivers.inference.guides.base import Guide
from quivers.inference.registry import LatentSite


class AutoNormalGuide(Guide):
    """Mean-field Normal guide with per-site constrained-support
    bijector.

    Parameters
    ----------
    model : MonadicProgram
        Generative model to build a guide for.
    observed_names : set[str]
        Variable names treated as observations (skipped in the
        guide; their values flow through the conditioning data
        dict at trace time).
    init_scale : float
        Initial scale (in unconstrained space) of every latent.
        Default ``0.1``; small enough to keep the guide near its
        prior at the start of optimisation.
    """

    def __init__(
        self,
        model: MonadicProgram,
        observed_names: set[str],
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self._registry = self.build_registry(model, observed_names)
        init_log_scale = float(torch.tensor(init_scale).log().item())

        for site in self._registry.sites.values():
            self.register_parameter(
                f"loc_{site.name}",
                nn.Parameter(torch.zeros(site.unconstrained_shape)),
            )
            self.register_parameter(
                f"log_scale_{site.name}",
                nn.Parameter(
                    torch.full(site.unconstrained_shape, init_log_scale)
                ),
            )

    # ------------------------------------------------------------------
    # Variational parameter access
    # ------------------------------------------------------------------

    def _loc(self, name: str) -> torch.Tensor:
        return getattr(self, f"loc_{name}")

    def _log_scale(self, name: str) -> torch.Tensor:
        return getattr(self, f"log_scale_{name}")

    def _scale(self, name: str) -> torch.Tensor:
        return self._log_scale(name).exp().clamp(min=1e-6)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_site(
        self, site: LatentSite, batch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z, v)`` for a single site: unconstrained
        Normal sample and the matching constrained value."""
        loc = self._loc(site.name)
        scale = self._scale(site.name)
        if site.is_plate:
            z = D.Normal(loc, scale).rsample()
        else:
            # Broadcast against the particle / batch axis.
            loc_b = loc.unsqueeze(0).expand(batch, *site.unconstrained_shape)
            scale_b = scale.unsqueeze(0).expand(batch, *site.unconstrained_shape)
            z = D.Normal(loc_b, scale_b).rsample()
        v = site.bijector(z)
        if (
            site.constrained_dim == 1
            and v.dim() >= 1
            and v.shape[-1] == 1
        ):
            v = v.squeeze(-1)
        return z, v

    def rsample(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Reparameterized mean-field Normal-then-bijector sample."""
        batch = x.shape[0]
        result: dict[str, torch.Tensor] = {}
        for site in self._registry.sites.values():
            _, v = self._sample_site(site, batch)
            result[site.name] = v
        return result

    # ------------------------------------------------------------------
    # Log-density
    # ------------------------------------------------------------------

    def log_prob(
        self,
        x: torch.Tensor,
        sites: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pushforward log-density at constrained values ``sites``.

        Uses the change-of-variables identity:

            log q(v) = log Normal(z; loc, scale) + log|det J_{T^{-1}}(v)|

        where ``z = bijector.inv(v)``. The plate / scalar shape
        dispatch matches :meth:`rsample`'s convention.
        """
        batch = x.shape[0]
        total = torch.zeros(batch, device=x.device)
        for site in self._registry.sites.values():
            if site.name not in sites:
                continue
            v = sites[site.name]
            if (
                site.constrained_dim == 1
                and v.dim() == (1 if site.is_plate else 1)
            ):
                v = v.unsqueeze(-1)
            z = site.bijector.inv(v)
            loc = self._loc(site.name)
            scale = self._scale(site.name)
            if site.is_plate:
                # Plate latent: single shared sample, scalar density
                # broadcast against the batch accumulator.
                log_q_z = D.Normal(loc, scale).log_prob(z)
                log_abs_det = site.bijector.inv.log_abs_det_jacobian(v, z)
                contribution = log_q_z.reshape(-1).sum() + log_abs_det.reshape(-1).sum()
                total = total + contribution
            else:
                loc_b = loc.unsqueeze(0).expand(batch, *site.unconstrained_shape)
                scale_b = scale.unsqueeze(0).expand(batch, *site.unconstrained_shape)
                log_q_z = D.Normal(loc_b, scale_b).log_prob(z)
                log_abs_det = site.bijector.inv.log_abs_det_jacobian(v, z)
                while log_q_z.dim() > 1:
                    log_q_z = log_q_z.sum(dim=-1)
                while log_abs_det.dim() > 1:
                    log_abs_det = log_abs_det.sum(dim=-1)
                total = total + log_q_z + log_abs_det
        return total

    @property
    def latent_names(self) -> list[str]:
        return list(self._registry.names)


__all__ = ["AutoNormalGuide"]
