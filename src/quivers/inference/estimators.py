"""Gradient estimators for variational objectives.

A :class:`GradientEstimator` is the strategy that takes (latent
samples, model log-density, guide log-density) and returns a
scalar loss whose ``backward()`` produces the chosen gradient
estimator. Different strategies trade variance against
applicability:

* :class:`Reparameterised` — pathwise gradient (the standard SVI
  reparameterisation trick). Lowest variance for reparameterisable
  families; requires ``rsample``.
* :class:`StickingTheLanding` — detaches the variational-parameter
  dependence in :math:`\\log q_\\phi(z)` so the gradient variance
  asymptotically vanishes as :math:`q \\to p^*`
  (Roeder-Wu-Duvenaud 2017,
  `doi:10.48550/arXiv.1703.09194 <https://doi.org/10.48550/arXiv.1703.09194>`_).
* :class:`DoublyReparameterised` — the DReG estimator for IWAE
  (Tucker-Lawson-Gu-Maddison 2019,
  `doi:10.48550/arXiv.1810.04152 <https://doi.org/10.48550/arXiv.1810.04152>`_).
  Removes the score-function term whose variance grows with the
  particle count :math:`K`.
* :class:`ScoreFunction` — REINFORCE / black-box VI. The
  fallback for non-reparameterisable sites (discrete latents,
  reject-sampled families). Highest variance; pair with a
  baseline whenever possible.

Estimators are *strategies* held by :class:`Objective`
implementations; they don't store any state themselves and
operate on tensors only. The :class:`Reparameterised` instance
is a singleton — every objective defaults to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class GradientEstimator(ABC):
    """Strategy for computing :math:`\\nabla_\\phi \\mathcal{L}` from
    samples + densities.

    Subclasses implement :meth:`negative_objective`: given the
    per-particle ``log_p`` and ``log_q`` tensors (and any
    estimator-specific auxiliary data), return the *negated*
    objective whose ``backward()`` produces the desired gradient
    estimator.
    """

    @abstractmethod
    def negative_objective(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        log_q_detached: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the scalar loss whose gradient is the chosen
        estimator.

        Parameters
        ----------
        log_p : torch.Tensor
            Model log-joint ``log p(z, y)`` at the sampled latents.
            Shape ``(K, batch)`` where ``K`` is the particle axis
            (``K = 1`` for plain ELBO) and ``batch`` is the
            program-input batch axis.
        log_q : torch.Tensor
            Guide log-density ``log q_phi(z)`` at the sampled
            latents. Same shape as ``log_p``. Gradients flow back
            to the variational parameters through this tensor.
        log_q_detached : torch.Tensor or None
            ``log q_{stop_grad(phi)}(z)`` — the guide log-density
            with the variational parameters detached from the
            autograd graph. Required by sticking-the-landing and
            DReG; ignored by the basic estimators.
        """
        ...


class Reparameterised(GradientEstimator):
    """Standard pathwise gradient.

    For the ELBO with ``num_particles = 1`` this is the textbook
    reparameterisation trick (Kingma-Welling 2013,
    `doi:10.48550/arXiv.1312.6114 <https://doi.org/10.48550/arXiv.1312.6114>`_).
    For higher ``num_particles`` it's the importance-weighted
    score function with reparameterised samples — i.e. the IWAE
    bound under the naive gradient.
    """

    def negative_objective(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        log_q_detached: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del log_q_detached
        # ELBO with K particles:
        #   L = (1/K) Σ_k [ log p(z_k) − log q(z_k) ]    (ELBO, K=1: standard ELBO)
        # IWAE with K particles:
        #   L = logsumexp_k [ log p(z_k) − log q(z_k) ] − log K
        # We don't decide which here — that's the Objective's
        # job. We return the per-particle term log p − log q
        # negated and averaged over (K, batch).
        diff = log_p - log_q
        return -(diff.mean())


class StickingTheLanding(GradientEstimator):
    """Roeder-Wu-Duvenaud 2017 sticking-the-landing estimator.

    Replaces ``log q(z)`` in the loss with
    ``log q_{detach(phi)}(z)``: the score is evaluated at the
    same sample but the variational parameters are detached
    from the autograd graph. The total derivative loses its
    direct dependence on the variational parameters through
    ``log q``, leaving only the indirect dependence through the
    sampled ``z``. As :math:`q \\to p^*` the latter vanishes and
    so does the gradient variance.

    Use when training with a guide that's already close to the
    true posterior — typically after a warm-up phase. May
    *increase* variance early in training when ``q`` is far
    from ``p``.
    """

    def negative_objective(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        log_q_detached: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if log_q_detached is None:
            raise RuntimeError(
                "StickingTheLanding: requires the caller to supply "
                "log_q_detached (log q evaluated with the variational "
                "parameters detached). The objective in use does not "
                "produce this — check that the Objective implementation "
                "supplies log_q_detached when this estimator is selected."
            )
        del log_q
        diff = log_p - log_q_detached
        return -(diff.mean())


class DoublyReparameterised(GradientEstimator):
    """Doubly-reparameterised IWAE gradient (Tucker-Lawson-Gu-
    Maddison 2019).

    Specialised for the IWAE bound at K particles. Reweights the
    per-particle terms so the variance no longer collapses as
    :math:`K \\to \\infty`. The objective itself is the standard
    IWAE bound; only the gradient is reweighted.

    The estimator's mathematical content is the *gradient*:

    .. math::

        \\nabla_\\phi \\mathcal{L}_{\\mathrm{IWAE}} \\;=\\;
        \\sum_k w_k^2 \\, \\nabla_\\phi
            \\bigl[\\log p(z_k) - \\log q_\\phi(z_k)\\bigr]

    where :math:`w_k = \\exp(\\log p_k - \\log q_k) /
    \\sum_j \\exp(\\log p_j - \\log q_j)`. Implementing this as a
    surrogate loss whose ``backward()`` yields the right gradient
    is the standard trick: detach the importance weights from the
    autograd graph and use them as a non-differentiable scaling
    on the per-particle reparameterised difference.
    """

    def negative_objective(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        log_q_detached: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if log_q_detached is None:
            raise RuntimeError(
                "DoublyReparameterised: requires the caller to supply "
                "log_q_detached (log q evaluated with the variational "
                "parameters detached). The IWAE objective produces this "
                "natively — confirm the objective is IWAEBound."
            )
        if log_p.dim() < 1:
            raise RuntimeError(
                "DoublyReparameterised: log_p must have a leading "
                f"particle axis (K, batch); got shape {tuple(log_p.shape)}"
            )
        # w_k = softmax_k (log p_k - log q_k) along the particle
        # axis (axis 0). Detach the weights so they don't
        # contribute to the gradient.
        log_w = log_p - log_q_detached
        weights = torch.softmax(log_w, dim=0).detach()
        # Surrogate loss whose gradient is the DReG estimator.
        surrogate = (weights**2) * (log_p - log_q)
        return -(surrogate.sum(dim=0).mean())


class ScoreFunction(GradientEstimator):
    """REINFORCE / black-box VI gradient
    (Ranganath-Gerrish-Blei 2014,
    `doi:10.48550/arXiv.1401.0118 <https://doi.org/10.48550/arXiv.1401.0118>`_).

    Uses the log-derivative identity instead of the
    reparameterisation trick. Required when sampling is not
    differentiable (discrete latents, hard-truncated families,
    accept-reject samplers). Variance is typically orders of
    magnitude higher than reparameterised — combine with a
    control-variate baseline whenever possible.
    """

    def negative_objective(
        self,
        log_p: torch.Tensor,
        log_q: torch.Tensor,
        log_q_detached: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del log_q_detached
        # Score-function estimator: ∇φ E_q[f(z)] = E_q[f(z) ∇φ log q(z)].
        # As a surrogate loss whose backward gives the right
        # gradient: detach f(z) so it contributes as a scalar
        # weight on the score ∇φ log q(z).
        f = (log_p - log_q).detach()
        return -((f * log_q).mean())


__all__ = [
    "GradientEstimator",
    "Reparameterised",
    "StickingTheLanding",
    "DoublyReparameterised",
    "ScoreFunction",
]
