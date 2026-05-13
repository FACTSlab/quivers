"""Posterior comparison metrics for the benchmark suite.

Metrics are split into two groups:

* **Sample-based** (``posterior_mean_error``, ``posterior_variance_error``,
  ``correlation_error``, ``total_variation_1d``, ``coverage``,
  ``wasserstein_2_1d``) — consume tensors of draws and compare
  against an analytical reference distribution / value.
* **Chain-based** (``split_r_hat``, ``effective_sample_size``) —
  consume ``(chain, draw, ...)``-shaped MCMC output and produce
  the standard mixing diagnostics from Vehtari et al. 2021,
  doi:10.1214/20-BA1221.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Sample-based moment / distance metrics
# ---------------------------------------------------------------------------


def posterior_mean_error(
    samples: torch.Tensor,
    true_mean: torch.Tensor | float,
) -> float:
    """L2 / RMSE error between the empirical mean of ``samples`` and the truth.

    For 1-D samples returns the absolute error in the mean; for
    ``(N, D)`` samples returns the per-dimension RMSE of the
    means.
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
    return float(
        torch.linalg.vector_norm(empirical - truth) / math.sqrt(samples.shape[-1])
    )


def posterior_variance_error(
    samples: torch.Tensor, true_variance: float
) -> float:
    """Absolute error between empirical variance and the truth."""
    empirical = float(samples.var(unbiased=True))
    return abs(empirical - float(true_variance))


def correlation_error(
    samples: torch.Tensor, true_correlation: float
) -> float:
    """For 2-D posterior samples, absolute error in the empirical
    correlation."""
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


def total_variation_1d(
    samples: torch.Tensor,
    reference_distribution: torch.distributions.Distribution,
    *,
    num_bins: int = 100,
    eval_range: tuple[float, float] | None = None,
) -> float:
    """Total-variation distance estimate between the histogram of
    ``samples`` and the reference distribution."""
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


def total_variation_grid(
    samples: torch.Tensor,
    ref_grid: torch.Tensor,
    ref_density: torch.Tensor,
) -> float:
    """Total-variation distance between empirical samples and a
    reference density evaluated on a dense grid.

    Useful when the reference posterior is computed by quadrature
    rather than carried by a :mod:`torch.distributions` object.
    """
    low = float(ref_grid.min())
    high = float(ref_grid.max())
    num_bins = ref_grid.shape[0]
    width = (high - low) / num_bins
    counts = torch.histc(samples, bins=num_bins, min=low, max=high)
    empirical_density = counts / (counts.sum() * width + 1e-30)
    p_normalised = ref_density / (ref_density.sum() * width + 1e-30)
    return 0.5 * float(torch.abs(empirical_density - p_normalised).sum() * width)


def coverage(
    samples: torch.Tensor,
    true_value: torch.Tensor | float,
    level: float = 0.9,
) -> bool:
    """Whether the central ``level``-credible interval covers ``true_value``."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"coverage: level must be in (0, 1); got {level}")
    lower = float(samples.quantile(0.5 - level / 2.0))
    upper = float(samples.quantile(0.5 + level / 2.0))
    return lower <= float(true_value) <= upper


def wasserstein_2_1d(
    samples_a: torch.Tensor, samples_b: torch.Tensor
) -> float:
    """1-D 2-Wasserstein distance via sorted-CDF matching.

    For two empirical distributions on ``\\mathbb{R}`` the 2-Wasserstein
    distance is the L2 distance between their sorted samples
    (after aligning quantiles). Cheap, exact for matched sample
    sizes; for mismatched sizes we interpolate to the smaller.
    """
    if samples_a.dim() != 1 or samples_b.dim() != 1:
        raise ValueError("wasserstein_2_1d: both inputs must be 1-D")
    n = min(samples_a.shape[0], samples_b.shape[0])
    quantiles = torch.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    qa = samples_a.quantile(quantiles)
    qb = samples_b.quantile(quantiles)
    return float(((qa - qb) ** 2).mean().sqrt())


def gaussian_kl(
    mu_a: float, var_a: float, mu_b: float, var_b: float
) -> float:
    """KL(N(mu_a, var_a) || N(mu_b, var_b)) — closed form.

    Useful when both the candidate and reference posteriors are
    Gaussian (Tier-1 Normal-Normal, Eight Schools mu).
    """
    return float(
        0.5
        * (
            math.log(var_b / var_a)
            + (var_a + (mu_a - mu_b) ** 2) / var_b
            - 1.0
        )
    )


# ---------------------------------------------------------------------------
# Chain-based MCMC diagnostics (Vehtari et al. 2021)
# ---------------------------------------------------------------------------


def _split_chains(samples: torch.Tensor) -> torch.Tensor:
    """Split each chain in half, doubling the chain count.

    Input: ``(num_chains, num_draws)`` or ``(num_chains, num_draws, ...)``.
    Output: ``(2 * num_chains, num_draws // 2, ...)``.
    """
    if samples.shape[1] < 4:
        raise ValueError(
            "split_r_hat: need at least 4 draws per chain to split"
        )
    half = samples.shape[1] // 2
    return torch.cat([samples[:, :half], samples[:, half : 2 * half]], dim=0)


def split_r_hat(samples: torch.Tensor) -> float:
    """Split-:math:`\\hat R` diagnostic (Vehtari et al. 2021).

    Input shape ``(num_chains, num_draws)`` (univariate) or
    ``(num_chains, num_draws, D)`` (multivariate; returns max-over-D).
    """
    if samples.dim() == 2:
        return _r_hat_univariate(_split_chains(samples))
    elif samples.dim() == 3:
        split = _split_chains(samples)
        rhats = [
            _r_hat_univariate(split[..., i].contiguous())
            for i in range(split.shape[-1])
        ]
        return max(rhats)
    raise ValueError(
        f"split_r_hat: samples must be 2-D or 3-D; got {samples.dim()}-D"
    )


def _r_hat_univariate(samples: torch.Tensor) -> float:
    """Compute split-R̂ on a ``(num_chains, num_draws)`` univariate tensor."""
    M, N = samples.shape
    if N < 2:
        return float("inf")
    chain_means = samples.mean(dim=1)
    chain_vars = samples.var(dim=1, unbiased=True)
    overall_mean = chain_means.mean()
    B = N * float(((chain_means - overall_mean) ** 2).sum() / (M - 1))
    W = float(chain_vars.mean())
    if W <= 0:
        return float("inf")
    var_plus = (N - 1) / N * W + B / N
    return float((var_plus / W) ** 0.5)


def effective_sample_size(samples: torch.Tensor) -> float:
    """Effective sample size via the Geyer initial monotone-sequence
    estimator (Vehtari et al. 2021 §3).

    Input shape ``(num_chains, num_draws)`` (univariate) or
    ``(num_chains, num_draws, D)`` (multivariate; returns
    min-over-D).
    """
    if samples.dim() == 2:
        return _ess_univariate(samples)
    elif samples.dim() == 3:
        return min(
            _ess_univariate(samples[..., i].contiguous())
            for i in range(samples.shape[-1])
        )
    raise ValueError(
        f"effective_sample_size: samples must be 2-D or 3-D; got "
        f"{samples.dim()}-D"
    )


def _ess_univariate(samples: torch.Tensor) -> float:
    """Effective sample size for ``(num_chains, num_draws)``."""
    M, N = samples.shape
    if N < 4:
        return float("nan")
    centered = samples - samples.mean(dim=1, keepdim=True)
    chain_vars = samples.var(dim=1, unbiased=True)
    var_avg = float(chain_vars.mean())
    if var_avg <= 0:
        return float("nan")
    # Autocorrelation via FFT for each chain, averaged.
    n_pad = 2 ** int(math.ceil(math.log2(2 * N)))
    fft = torch.fft.fft(centered, n=n_pad, dim=1)
    psd = fft * fft.conj()
    autocov = torch.fft.ifft(psd, dim=1).real[:, :N] / N
    rho = autocov.mean(dim=0) / var_avg
    # Geyer initial monotone sequence: pair up consecutive
    # autocorrelations and stop when the sum is non-positive.
    tau = 1.0
    for t in range(1, N // 2):
        pair = float(rho[2 * t - 1] + rho[2 * t]) if 2 * t < N else 0.0
        if pair <= 0:
            break
        tau += 2.0 * pair
    return float(M * N / tau)


__all__ = [
    "posterior_mean_error",
    "posterior_variance_error",
    "correlation_error",
    "total_variation_1d",
    "total_variation_grid",
    "coverage",
    "wasserstein_2_1d",
    "gaussian_kl",
    "split_r_hat",
    "effective_sample_size",
]
