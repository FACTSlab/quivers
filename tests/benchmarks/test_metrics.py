"""Unit tests for benchmark metric helpers.

The metrics in :mod:`tests.benchmarks.metrics` are used across
the Tier-1 / Tier-3 benchmarks; verifying them directly catches
regressions that would otherwise only surface as silent bench
failures.
"""

from __future__ import annotations

import math

import pytest
import torch

from tests.benchmarks.metrics import (
    correlation_error,
    coverage,
    posterior_mean_error,
    posterior_variance_error,
    total_variation_1d,
)


# ---------------------------------------------------------------------------
# posterior_mean_error
# ---------------------------------------------------------------------------


def test_posterior_mean_error_1d_returns_absolute_difference() -> None:
    samples = torch.tensor([1.0, 2.0, 3.0])  # mean 2.0
    assert math.isclose(posterior_mean_error(samples, 2.0), 0.0)
    assert math.isclose(posterior_mean_error(samples, 1.5), 0.5)


def test_posterior_mean_error_multivariate() -> None:
    samples = torch.tensor([[0.0, 0.0], [2.0, 2.0]])  # mean (1.0, 1.0)
    true_mean = torch.tensor([1.0, 1.0])
    assert math.isclose(
        posterior_mean_error(samples, true_mean), 0.0, abs_tol=1e-6
    )


def test_posterior_variance_error_returns_unsigned_difference() -> None:
    torch.manual_seed(0)
    samples = torch.randn(1000) * 2.0  # variance ~ 4.0
    err = posterior_variance_error(samples, 4.0)
    assert err < 0.5


# ---------------------------------------------------------------------------
# total_variation_1d
# ---------------------------------------------------------------------------


def test_total_variation_close_for_aligned_distributions() -> None:
    torch.manual_seed(0)
    ref = torch.distributions.Normal(0.0, 1.0)
    samples = ref.sample((5000,))
    tv = total_variation_1d(samples, ref, eval_range=(-5.0, 5.0))
    assert 0.0 <= tv <= 0.2, f"TV between sampled and exact Normal: {tv}"


def test_total_variation_large_for_mismatched_distributions() -> None:
    torch.manual_seed(0)
    sampling = torch.distributions.Normal(0.0, 1.0)
    reference = torch.distributions.Normal(5.0, 1.0)  # shifted
    samples = sampling.sample((5000,))
    tv = total_variation_1d(samples, reference, eval_range=(-5.0, 10.0))
    assert tv > 0.7, f"TV between shifted distributions: {tv}"


def test_total_variation_rejects_multivariate_samples() -> None:
    ref = torch.distributions.Normal(0.0, 1.0)
    samples = torch.randn(10, 2)
    with pytest.raises(ValueError, match="1-D samples"):
        total_variation_1d(samples, ref)


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_coverage_inside_interval_returns_true() -> None:
    torch.manual_seed(0)
    samples = torch.randn(1000)
    # The interval roughly [-1.96, 1.96] should cover 0.0.
    assert coverage(samples, 0.0, level=0.9) is True


def test_coverage_outside_interval_returns_false() -> None:
    torch.manual_seed(0)
    samples = torch.randn(1000)
    # 5.0 sits well outside any reasonable 90% CI for a standard
    # normal sample.
    assert coverage(samples, 5.0, level=0.9) is False


def test_coverage_rejects_invalid_level() -> None:
    samples = torch.randn(100)
    with pytest.raises(ValueError, match="level must be in"):
        coverage(samples, 0.0, level=1.5)
    with pytest.raises(ValueError, match="level must be in"):
        coverage(samples, 0.0, level=0.0)


# ---------------------------------------------------------------------------
# correlation_error
# ---------------------------------------------------------------------------


def test_correlation_error_returns_zero_on_known_correlation() -> None:
    torch.manual_seed(0)
    # Build a perfectly correlated 2D sample.
    base = torch.randn(2000)
    other = 2.0 * base + 0.05 * torch.randn(2000)
    samples = torch.stack([base, other], dim=-1)
    # Empirical correlation is very close to 1.0.
    err = correlation_error(samples, 1.0)
    assert err < 0.05


def test_correlation_error_rejects_wrong_shape() -> None:
    samples = torch.randn(10, 3)  # not (N, 2)
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        correlation_error(samples, 0.5)
