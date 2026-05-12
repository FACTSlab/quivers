"""Posterior comparison metrics for the benchmark suite.

The metrics here are sample-based — they consume a tensor of
posterior draws and compare against an analytical reference
(mean, variance, or full distribution). The reference can be
expressed as a torch.distributions.Distribution where appropriate.
"""

from __future__ import annotations

import math

import torch


def posterior_mean_error(
    samples: torch.Tensor,
    true_mean: torch.Tensor | float,
) -> float:
    """RMSE between the empirical mean of ``samples`` and the
    truth.

    Parameters
    ----------
    samples : torch.Tensor
        Shape ``(N,)`` or ``(N, D)``.
    true_mean : torch.Tensor or float
        Analytical posterior mean of matching shape.
    """
    if samples.dim() == 1:
        empirical = samples.mean()
        return float(torch.abs(empirical - float(true_mean)))
    empirical = samples.mean(dim=0)
    truth = (
        true_mean
        if isinstance(true_mean, torch.Tensor)
        else torch.tensor(true_mean)
    )
    return float(torch.linalg.vector_norm(empirical - truth) / math.sqrt(samples.shape[-1]))


def posterior_variance_error(
    samples: torch.Tensor, true_variance: float
) -> float:
    """Absolute error between the empirical sample variance and
    the truth."""
    empirical = float(samples.var(unbiased=True))
    return abs(empirical - float(true_variance))


def total_variation_1d(
    samples: torch.Tensor,
    reference_distribution: torch.distributions.Distribution,
    *,
    num_bins: int = 100,
    eval_range: tuple[float, float] | None = None,
) -> float:
    """Total-variation distance estimate between the histogram of
    ``samples`` and a reference :class:`torch.distributions.Distribution`.

    Operates on a 1-D parameter; for multivariate posteriors call
    coordinate-wise.
    """
    if samples.dim() != 1:
        raise ValueError(
            f"total_variation_1d: expected 1-D samples; got "
            f"{samples.dim()}-D"
        )
    if eval_range is None:
        low = float(samples.min()) - 1.0
        high = float(samples.max()) + 1.0
    else:
        low, high = eval_range
    edges = torch.linspace(low, high, num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (high - low) / num_bins
    counts = torch.histc(samples, bins=num_bins, min=low, max=high)
    empirical_density = counts / (counts.sum() * width)
    ref_density = torch.exp(reference_distribution.log_prob(centers))
    return 0.5 * float(torch.abs(empirical_density - ref_density).sum() * width)


def coverage(
    samples: torch.Tensor,
    true_value: torch.Tensor | float,
    level: float = 0.9,
) -> bool:
    """Whether the ``level``-credible interval (centred on the
    empirical median) covers ``true_value``."""
    if not 0.0 < level < 1.0:
        raise ValueError(
            f"coverage: level must be in (0, 1); got {level}"
        )
    lower = float(samples.quantile(0.5 - level / 2.0))
    upper = float(samples.quantile(0.5 + level / 2.0))
    return lower <= float(true_value) <= upper


def correlation_error(
    samples: torch.Tensor, true_correlation: float
) -> float:
    """For a 2-D posterior, the absolute error between the empirical
    correlation of the samples and the analytical correlation."""
    if samples.dim() != 2 or samples.shape[-1] != 2:
        raise ValueError(
            f"correlation_error: expected (N, 2) samples; got "
            f"{tuple(samples.shape)}"
        )
    mean = samples.mean(dim=0)
    centered = samples - mean
    cov = (centered.t() @ centered) / float(samples.shape[0] - 1)
    rho = float(cov[0, 1] / (cov[0, 0] * cov[1, 1]).sqrt())
    return abs(rho - true_correlation)


__all__ = [
    "posterior_mean_error",
    "posterior_variance_error",
    "total_variation_1d",
    "coverage",
    "correlation_error",
]
