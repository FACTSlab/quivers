"""Adaptation primitives for HMC / NUTS warmup.

Two primitives:

* :class:`DualAveraging` — Nesterov's dual averaging for adapting
  the step size to a target Metropolis acceptance probability
  (Hoffman-Gelman 2014, Algorithm 6,
  `doi:10.48550/arXiv.1111.4246
  <https://doi.org/10.48550/arXiv.1111.4246>`_).
* :class:`WelfordCovariance` — Welford-style online accumulator
  for the empirical covariance of the warmup samples; both the
  diagonal and dense forms are exposed because HMC mass matrices
  are typically one of those two shapes.

These primitives are *stateful*: they accumulate per-step samples
during warmup and freeze when the kernel stops adapting. They are
shared across HMC and NUTS, which is why they live in their own
module rather than being inlined into either kernel.
"""

from __future__ import annotations

import math
from typing import Callable

import torch


class DualAveraging:
    """Nesterov dual averaging for adaptive step-size selection.

    Maintains a running smoothed estimate of the log-step-size that
    drives the chain's average acceptance probability toward the
    user-supplied ``target_accept``.

    Algorithm parameters follow Hoffman-Gelman 2014 (Algorithm 6):
    :math:`\\gamma = 0.05`, :math:`t_0 = 10`, :math:`\\kappa = 0.75`.
    These are the defaults Stan uses; they're robust across a wide
    range of problems and rarely need tuning.

    Parameters
    ----------
    initial_step_size : float
        Starting step size. The first proposal uses ``log(10 * ε₀)``
        as the shrinkage anchor :math:`\\mu` per Algorithm 6.
    target_accept : float
        Target Metropolis acceptance probability. Default ``0.8``
        (NUTS's standard) — somewhat below the optimal 0.65 for HMC
        because NUTS averages acceptance over the tree.
    gamma : float
        Adaptation regularisation. Default ``0.05``.
    t0 : float
        Stabilisation offset. Default ``10.0``.
    kappa : float
        Decay exponent for the smoothed step size. Default ``0.75``.
    """

    def __init__(
        self,
        initial_step_size: float,
        target_accept: float = 0.8,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
    ) -> None:
        if initial_step_size <= 0:
            raise ValueError(
                f"DualAveraging: initial_step_size must be > 0, got {initial_step_size}"
            )
        if not 0.0 < target_accept < 1.0:
            raise ValueError(
                f"DualAveraging: target_accept must be in (0, 1), got {target_accept}"
            )
        self._mu = math.log(10.0 * initial_step_size)
        self._target_accept = target_accept
        self._gamma = gamma
        self._t0 = t0
        self._kappa = kappa
        self._log_step = math.log(initial_step_size)
        self._log_step_smoothed = 0.0
        self._h_bar = 0.0
        self._step_count = 0

    def step_size(self) -> float:
        """Current step size used by the *next* leapfrog proposal."""
        return math.exp(self._log_step)

    def smoothed_step_size(self) -> float:
        """Smoothed step size frozen after warmup ends."""
        return math.exp(self._log_step_smoothed)

    def update(self, acceptance_prob: float) -> None:
        """Update the step size given the latest Metropolis
        acceptance probability."""
        if not math.isfinite(acceptance_prob):
            acceptance_prob = 0.0
        acceptance_prob = max(0.0, min(1.0, acceptance_prob))
        self._step_count += 1
        m = float(self._step_count)
        eta_h = 1.0 / (m + self._t0)
        self._h_bar = (1.0 - eta_h) * self._h_bar + eta_h * (
            self._target_accept - acceptance_prob
        )
        self._log_step = self._mu - math.sqrt(m) / self._gamma * self._h_bar
        eta_step = m ** (-self._kappa)
        self._log_step_smoothed = (
            eta_step * self._log_step + (1.0 - eta_step) * self._log_step_smoothed
        )

    @property
    def step_count(self) -> int:
        return self._step_count


class WelfordCovariance:
    """Welford-style online covariance accumulator.

    Tracks the running mean and (co)variance of a stream of
    ``D``-dimensional vectors. Supports both diagonal and dense
    forms — diagonal sufficient for ill-scaled axis-aligned
    posteriors, dense needed when off-axis correlations matter.

    The dense form follows the standard pairwise update:

    .. math::

        M_n = M_{n-1} + (x_n - \\bar{x}_{n-1})(x_n - \\bar{x}_n)^\\top

    with :math:`\\bar{x}_n = \\bar{x}_{n-1} + (x_n - \\bar{x}_{n-1}) / n`.

    Parameters
    ----------
    dim : int
        Vector dimension.
    regularise : bool
        Apply Stan's regularisation
        :math:`(n / (n + 5)) \\cdot \\Sigma + 0.001 \\cdot (5 / (n + 5)) \\cdot I`
        to the final covariance. Default ``True``; mirrors Stan's
        warmup-end mass-matrix construction.
    diagonal : bool
        Track only the diagonal of the covariance (vector, not
        matrix). Default ``False``.
    """

    def __init__(
        self,
        dim: int,
        regularise: bool = True,
        diagonal: bool = False,
    ) -> None:
        if dim < 1:
            raise ValueError(f"WelfordCovariance: dim must be >= 1, got {dim}")
        self._dim = dim
        self._regularise = regularise
        self._diagonal = diagonal
        self._n = 0
        self._mean = torch.zeros(dim)
        self._m2: torch.Tensor
        if diagonal:
            self._m2 = torch.zeros(dim)
        else:
            self._m2 = torch.zeros(dim, dim)

    @property
    def n(self) -> int:
        return self._n

    @property
    def diagonal(self) -> bool:
        return self._diagonal

    def update(self, x: torch.Tensor) -> None:
        """Fold one vector into the running statistics."""
        if x.shape != (self._dim,):
            raise ValueError(
                f"WelfordCovariance.update: expected shape "
                f"({self._dim},); got {tuple(x.shape)}"
            )
        self._n += 1
        delta = x - self._mean
        self._mean = self._mean + delta / float(self._n)
        delta_post = x - self._mean
        if self._diagonal:
            self._m2 = self._m2 + delta * delta_post
        else:
            self._m2 = self._m2 + delta.unsqueeze(-1) @ delta_post.unsqueeze(0)

    def covariance(self) -> torch.Tensor:
        """Return the empirical covariance (regularised if
        configured)."""
        if self._n < 2:
            # Insufficient samples; return identity so the kernel
            # behaves as before adaptation kicked in.
            if self._diagonal:
                return torch.ones(self._dim)
            return torch.eye(self._dim)
        cov = self._m2 / float(self._n - 1)
        if self._regularise:
            n = float(self._n)
            shrink = n / (n + 5.0)
            jitter = 1e-3 * (5.0 / (n + 5.0))
            if self._diagonal:
                cov = shrink * cov + jitter * torch.ones(self._dim)
            else:
                cov = shrink * cov + jitter * torch.eye(self._dim)
        return cov

    def reset(self) -> None:
        """Discard accumulated statistics."""
        self._n = 0
        self._mean = torch.zeros(self._dim)
        if self._diagonal:
            self._m2 = torch.zeros(self._dim)
        else:
            self._m2 = torch.zeros(self._dim, self._dim)


def find_reasonable_step_size(
    log_density: Callable[[torch.Tensor], torch.Tensor],
    grad_log_density: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    initial_step: float = 1.0,
    target_log_accept: float = math.log(0.5),
) -> float:
    """Heuristic-by-doubling initial step size (Hoffman-Gelman 2014
    Algorithm 4).

    Starting from ``initial_step``, doubles or halves the step size
    until a single leapfrog step's Metropolis log-acceptance crosses
    ``target_log_accept`` (default :math:`\\log 0.5`). Returns the
    step size at the crossing.
    """
    eps = initial_step
    z = z0.clone()
    ld0 = log_density(z)
    g0 = grad_log_density(z)
    p = torch.randn_like(z)
    h0 = ld0 - 0.5 * (p * p).sum()
    # One leapfrog step at eps.
    z1 = z + eps * p + 0.5 * eps * eps * g0
    g1 = grad_log_density(z1)
    p1 = p + 0.5 * eps * (g0 + g1)
    ld1 = log_density(z1)
    h1 = ld1 - 0.5 * (p1 * p1).sum()
    log_accept = float(h1 - h0)
    if not math.isfinite(log_accept):
        log_accept = -float("inf")
    direction = 1 if log_accept > target_log_accept else -1
    for _ in range(100):
        eps = eps * (2.0**direction)
        if eps < 1e-10 or eps > 1e10:
            break
        z1 = z + eps * p + 0.5 * eps * eps * g0
        g1 = grad_log_density(z1)
        p1 = p + 0.5 * eps * (g0 + g1)
        ld1 = log_density(z1)
        h1 = ld1 - 0.5 * (p1 * p1).sum()
        new_log_accept = float(h1 - h0)
        if not math.isfinite(new_log_accept):
            new_log_accept = -float("inf")
        crossed = (direction == 1 and new_log_accept < target_log_accept) or (
            direction == -1 and new_log_accept > target_log_accept
        )
        log_accept = new_log_accept
        if crossed:
            break
    return eps


__all__ = [
    "DualAveraging",
    "WelfordCovariance",
    "find_reasonable_step_size",
]
