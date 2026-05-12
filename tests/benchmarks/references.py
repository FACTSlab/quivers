"""Closed-form posterior references for the benchmark datasets.

Each reference function takes the generator's
:class:`BenchmarkData` and returns a small dict of analytical
moments (mean, variance, correlation, covariance) that the
benchmark tests assert their recovered posterior against. No
caching is needed because every shipped reference here is
analytical.
"""

from __future__ import annotations

import torch

from tests.benchmarks.datasets import BenchmarkData


def beta_bernoulli_reference(data: BenchmarkData) -> dict[str, float]:
    """``Beta(α₀ + Σy, β₀ + N − Σy)`` posterior moments."""
    alpha0 = float(data.true_params["alpha0"])
    beta0 = float(data.true_params["beta0"])
    N = int(data.true_params["N"])
    successes = float(data.observations["y"].sum())
    alpha = alpha0 + successes
    beta = beta0 + N - successes
    mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    return {"alpha": alpha, "beta": beta, "mean": mean, "variance": variance}


def normal_normal_reference(data: BenchmarkData) -> dict[str, float]:
    """Closed-form Normal-Normal posterior moments."""
    mu0 = float(data.true_params["mu0"])
    tau0 = float(data.true_params["tau0"])
    sigma = float(data.true_params["sigma"])
    N = int(data.true_params["N"])
    y_bar = float(data.observations["y"].mean())
    prec0 = 1.0 / (tau0 * tau0)
    prec_data = N / (sigma * sigma)
    post_prec = prec0 + prec_data
    mean = (prec0 * mu0 + prec_data * y_bar) / post_prec
    variance = 1.0 / post_prec
    return {"mean": mean, "variance": variance}


def correlated_regression_reference(
    data: BenchmarkData,
) -> dict[str, torch.Tensor | float]:
    """Linear-Gaussian posterior over ``(a, b)``."""
    x = data.observations["x_design"]
    y = data.observations["y"]
    sigma = float(data.true_params["sigma"])
    X = torch.stack([torch.ones_like(x), x], dim=1)
    sigma2 = sigma * sigma
    precision = torch.eye(2) + X.t() @ X / sigma2
    covariance = torch.linalg.inv(precision)
    mean = covariance @ (X.t() @ y / sigma2)
    correlation = float(
        covariance[0, 1] / (covariance[0, 0] * covariance[1, 1]).sqrt()
    )
    return {
        "mean": mean,
        "covariance": covariance,
        "correlation": correlation,
    }


__all__ = [
    "beta_bernoulli_reference",
    "normal_normal_reference",
    "correlated_regression_reference",
]
