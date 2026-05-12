"""Data generators for the synthetic benchmark suite.

Each generator function loads its model from
``tests/benchmarks/models/<name>.qvr`` (the canonical didactic-
model surface for QVR) and returns a triple
``(model, observations, true_params)`` for the test code to
consume. There is no Python wrapper class around the model — the
``.qvr`` file *is* the model definition.

Data draws are deterministic in the supplied ``seed`` so the
benchmark numbers are reproducible across runs and platforms.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.dsl import loads


_MODELS_DIR = Path(__file__).parent / "models"


class BenchmarkData(NamedTuple):
    """Triple returned by every benchmark data generator."""

    model: ContinuousMorphism
    observations: dict[str, torch.Tensor]
    true_params: dict[str, float | torch.Tensor]


def _load_model(name: str):
    path = _MODELS_DIR / f"{name}.qvr"
    return loads(path.read_text()).morphism


# ---------------------------------------------------------------------------
# Beta-Bernoulli
# ---------------------------------------------------------------------------


def beta_bernoulli(
    true_theta: float = 0.7, seed: int = 0
) -> BenchmarkData:
    """Generate ``N`` Bernoulli observations at ``true_theta``."""
    model = _load_model("beta_bernoulli")
    g = torch.Generator().manual_seed(seed)
    # The model declares Obs : 50 so N is fixed at the model source.
    N = 50
    y = torch.bernoulli(torch.full((N,), true_theta), generator=g)
    return BenchmarkData(
        model=model,
        observations={"y": y},
        true_params={"theta": true_theta, "alpha0": 2.0, "beta0": 2.0, "N": N},
    )


# ---------------------------------------------------------------------------
# Normal-Normal
# ---------------------------------------------------------------------------


def normal_normal(
    true_mu: float = 1.5, sigma: float = 1.0, seed: int = 0
) -> BenchmarkData:
    """Generate ``N`` Normal(mu, sigma) observations."""
    model = _load_model("normal_normal")
    g = torch.Generator().manual_seed(seed)
    N = 30
    y = true_mu + sigma * torch.randn(N, generator=g)
    return BenchmarkData(
        model=model,
        observations={"y": y},
        true_params={
            "mu": true_mu,
            "sigma": sigma,
            "mu0": 0.0,
            "tau0": 1.0,
            "N": N,
        },
    )


# ---------------------------------------------------------------------------
# Correlated regression
# ---------------------------------------------------------------------------


def correlated_regression(
    true_a: float = 0.7,
    true_b: float = -0.5,
    correlation_strength: float = 0.95,
    sigma: float = 0.5,
    seed: int = 0,
) -> BenchmarkData:
    """Generate a near-collinear design and observations under the
    linear-Gaussian model.

    With ``correlation_strength`` close to 1 the design rows
    cluster near a constant, so the joint posterior over ``(a, b)``
    has a strong off-diagonal correlation. The analytical
    correlation can be computed from the design matrix; see
    :func:`tests.benchmarks.references.correlated_regression_reference`.
    """
    model = _load_model("correlated_regression")
    g_x = torch.Generator().manual_seed(seed + 1)
    g_y = torch.Generator().manual_seed(seed)
    N = 50
    z = torch.randn(N, generator=g_x)
    x_design = correlation_strength + (1.0 - correlation_strength) * z
    y = true_a + true_b * x_design + sigma * torch.randn(N, generator=g_y)
    return BenchmarkData(
        model=model,
        observations={"y": y, "x_design": x_design},
        true_params={
            "a": true_a,
            "b": true_b,
            "sigma": sigma,
            "correlation_strength": correlation_strength,
            "N": N,
            "x_design": x_design,
        },
    )


__all__ = [
    "BenchmarkData",
    "beta_bernoulli",
    "normal_normal",
    "correlated_regression",
]
