"""Synthetic benchmark suite for the quivers inference layer.

Each benchmark loads its model from
``tests/benchmarks/models/<name>.qvr`` and compares the inference
algorithm's recovered posterior against an analytical reference
(closed-form moments where conjugacy applies, quadrature on a
dense grid for constrained-support cases, cached NUTS-style
moments for Eight Schools).

Tier coverage:

* **Tier 1 — conjugate**: Beta-Bernoulli, Normal-Normal,
  Normal-Inverse-Gamma (joint mean / variance), Gamma-Exponential,
  Bayesian linear regression (well-conditioned). Mean error
  tolerances < 0.1 on the parameter of interest.
* **Tier 2 — hierarchical**: Eight Schools, centred and
  non-centred parameterisations (Rubin 1981, Gelman et al. 2013).
  Posterior mean / sd of ``mu``, ``tau`` and the per-school
  ``theta`` are compared against cached NUTS-derived moments.
* **Tier 3 — hard geometry**: correlated regression, Neal's
  funnel (Neal 2003), ill-conditioned product Gaussian. Tests
  algorithm-separation: AutoNormalGuide is expected to collapse
  off-diagonal correlations / fail the scale-of-scale geometry;
  AutoMVN / HMC recover.
* **Tier 6 — constrained support**: HalfNormal scale recovery,
  TruncatedNormal posterior recovery. Each sample must lie in
  the prior's constrained support; the guide-side bijector is
  exp / softplus / sigmoid.

The driver in :mod:`tests.benchmarks.runner` writes the full
algorithm × problem grid to ``docs/developer/inference-benchmarks.md``.
"""

from __future__ import annotations

from tests.benchmarks.datasets import (
    BenchmarkData,
    bayes_linear_regression,
    beta_bernoulli,
    correlated_regression,
    eight_schools_centred,
    eight_schools_noncentred,
    gamma_exponential,
    half_normal_scale,
    ill_conditioned_mvn,
    neal_funnel,
    normal_inverse_gamma,
    normal_normal,
    truncated_normal_recovery,
)
from tests.benchmarks.metrics import (
    correlation_error,
    coverage,
    effective_sample_size,
    gaussian_kl,
    posterior_mean_error,
    posterior_variance_error,
    split_r_hat,
    total_variation_1d,
    total_variation_grid,
    wasserstein_2_1d,
)
from tests.benchmarks.references import (
    bayes_linear_regression_reference,
    beta_bernoulli_reference,
    correlated_regression_reference,
    eight_schools_reference,
    gamma_exponential_reference,
    half_normal_scale_reference,
    ill_conditioned_mvn_reference,
    neal_funnel_reference,
    normal_inverse_gamma_reference,
    normal_normal_reference,
    truncated_normal_recovery_reference,
)

__all__ = [
    "BenchmarkData",
    # Tier 1
    "beta_bernoulli",
    "normal_normal",
    "normal_inverse_gamma",
    "gamma_exponential",
    "bayes_linear_regression",
    "beta_bernoulli_reference",
    "normal_normal_reference",
    "normal_inverse_gamma_reference",
    "gamma_exponential_reference",
    "bayes_linear_regression_reference",
    # Tier 2
    "eight_schools_centred",
    "eight_schools_noncentred",
    "eight_schools_reference",
    # Tier 3
    "correlated_regression",
    "neal_funnel",
    "ill_conditioned_mvn",
    "correlated_regression_reference",
    "neal_funnel_reference",
    "ill_conditioned_mvn_reference",
    # Tier 6
    "half_normal_scale",
    "truncated_normal_recovery",
    "half_normal_scale_reference",
    "truncated_normal_recovery_reference",
    # Metrics
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
