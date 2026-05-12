"""Synthetic benchmark suite for the quivers inference layer.

Each benchmark loads its model from
``tests/benchmarks/models/<name>.qvr`` (the canonical didactic-
model surface) and compares the inference algorithm's recovered
posterior against an analytical closed-form reference.

Tiers shipped:

* **Tier 1 — conjugate models** (``test_conjugate.py``):
  Beta-Bernoulli and Normal-Normal. Closed-form posterior moments;
  recovery tolerance bands tight (mean error < 0.1).
* **Tier 3 — hard posterior geometries** (``test_geometry.py``):
  near-collinear linear regression with a strongly correlated
  posterior. Tests the algorithm-separation contract — mean-field
  guides recover the marginal means but collapse the off-diagonal
  correlation, while AutoMVN and HMC recover both.

The full 6-tier grid from the inference plan is a separate
workstream. The shipped suite verifies the three algorithm-
dispatch decisions the plan turns on (conjugate-recovery cleanness,
correlation handling, end-to-end MCMC correctness); the remaining
tiers (multimodality, latent-variable models, constrained supports,
etc.) follow the same data-generator → reference → metric pattern.
"""

from __future__ import annotations

from tests.benchmarks.datasets import (
    BenchmarkData,
    beta_bernoulli,
    correlated_regression,
    normal_normal,
)
from tests.benchmarks.metrics import (
    correlation_error,
    coverage,
    posterior_mean_error,
    posterior_variance_error,
    total_variation_1d,
)
from tests.benchmarks.references import (
    beta_bernoulli_reference,
    correlated_regression_reference,
    normal_normal_reference,
)

__all__ = [
    "BenchmarkData",
    "beta_bernoulli",
    "normal_normal",
    "correlated_regression",
    "beta_bernoulli_reference",
    "normal_normal_reference",
    "correlated_regression_reference",
    "posterior_mean_error",
    "posterior_variance_error",
    "correlation_error",
    "total_variation_1d",
    "coverage",
]
