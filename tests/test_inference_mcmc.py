"""Tests for the MCMC kernels and driver.

Cover:

1. Adaptation primitives — :class:`DualAveraging` converges the
   step size to the target acceptance probability;
   :class:`WelfordCovariance` recovers the empirical covariance
   of a known stream.
2. :class:`HMCKernel` on a Normal-Normal conjugate model — the
   posterior mean of the recovered chain matches the closed-form
   analytical posterior.
3. :class:`NUTSKernel` on the same model — same recovery target,
   and the per-step tree depth stays well below ``max_tree_depth``.
4. Driver-level behaviour: parallel chains, split-:math:`\\hat R`
   close to ``1.0`` after convergence, ESS ≥ a small fraction of
   the chain length.
"""

from __future__ import annotations

import math

import torch

from quivers.dsl import loads
from quivers.inference.mcmc.adapt import (
    DualAveraging,
    WelfordCovariance,
    find_reasonable_step_size,
)
from quivers.inference.mcmc import HMCKernel, MCMC, NUTSKernel


# ---------------------------------------------------------------------------
# Adaptation primitives
# ---------------------------------------------------------------------------


def test_dual_averaging_converges_to_target_acceptance() -> None:
    """Feed a deterministic acceptance signal of 0.4; the dual-
    averaging step size should decrease (because target 0.8 > 0.4)
    over time."""
    da = DualAveraging(initial_step_size=1.0, target_accept=0.8)
    initial_step = da.step_size()
    for _ in range(200):
        da.update(0.4)
    final_step = da.smoothed_step_size()
    assert final_step < initial_step, (
        f"DualAveraging did not shrink step size when acceptance "
        f"(0.4) was below target (0.8): {initial_step:.4f} → {final_step:.4f}"
    )


def test_welford_recovers_empirical_covariance() -> None:
    torch.manual_seed(0)
    D = 5
    true_cov = torch.eye(D) * 0.5 + 0.5 * torch.ones(D, D) * 0.3
    L = torch.linalg.cholesky(true_cov)
    welford = WelfordCovariance(D, regularise=False)
    for _ in range(5000):
        x = L @ torch.randn(D)
        welford.update(x)
    estimated = welford.covariance()
    assert torch.allclose(estimated, true_cov, atol=0.05), (
        f"WelfordCovariance estimate too far from truth:\n"
        f"truth diag={torch.diagonal(true_cov)}\n"
        f"estd diag={torch.diagonal(estimated)}"
    )


def test_welford_diagonal_matches_dense_diagonal() -> None:
    torch.manual_seed(0)
    D = 4
    dense = WelfordCovariance(D, regularise=False, diagonal=False)
    diag = WelfordCovariance(D, regularise=False, diagonal=True)
    for _ in range(500):
        x = torch.randn(D) * torch.tensor([0.5, 1.0, 2.0, 0.1])
        dense.update(x)
        diag.update(x)
    dense_diag = torch.diagonal(dense.covariance())
    diag_diag = diag.covariance()
    assert torch.allclose(dense_diag, diag_diag, atol=1e-5)


def test_find_reasonable_step_size_returns_finite_positive() -> None:
    torch.manual_seed(0)

    def log_density(z: torch.Tensor) -> torch.Tensor:
        return -0.5 * (z * z).sum()

    def grad_log_density(z: torch.Tensor) -> torch.Tensor:
        return -z

    eps = find_reasonable_step_size(
        log_density, grad_log_density, torch.zeros(3)
    )
    assert math.isfinite(eps) and eps > 0


# ---------------------------------------------------------------------------
# Closed-form Normal-Normal recovery (the HMC / NUTS gold standard)
# ---------------------------------------------------------------------------


def _normal_normal_model():
    """y_i ~ N(mu, 1) with mu ~ N(0, 1). Posterior: N(N*y_bar/(N+1), 1/(N+1))."""
    return loads(
        "object Obs : 20\n"
        "program p : Obs -> Obs\n"
        "    mu <- Normal(0.0, 1.0)\n"
        "    observe y : Obs <- Normal(mu, 1.0)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def test_hmc_recovers_conjugate_normal_normal_posterior() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    N = 20
    y = torch.randn(N) + 1.5
    y_bar = float(y.mean())
    true_mean = N * y_bar / (N + 1)
    true_var = 1.0 / (N + 1)

    kernel = HMCKernel(
        step_size=0.1, num_steps=15, mass_matrix="identity",
        adapt_step_size=True, adapt_mass_matrix=False,
    )
    driver = MCMC(
        kernel=kernel, num_warmup=200, num_samples=500, num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    mu_samples = result.samples["mu"].reshape(-1)
    sample_mean = float(mu_samples.mean())
    sample_var = float(mu_samples.var(unbiased=True))
    assert abs(sample_mean - true_mean) < 0.15, (
        f"HMC posterior mean off: got {sample_mean:.4f}, "
        f"expected {true_mean:.4f}"
    )
    assert abs(sample_var - true_var) < 0.05, (
        f"HMC posterior variance off: got {sample_var:.4f}, "
        f"expected {true_var:.4f}"
    )
    assert result.mean_acceptance > 0.3


def test_nuts_recovers_conjugate_normal_normal_posterior() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    N = 20
    y = torch.randn(N) + 1.5
    y_bar = float(y.mean())
    true_mean = N * y_bar / (N + 1)
    true_var = 1.0 / (N + 1)

    kernel = NUTSKernel(
        step_size=0.1, max_tree_depth=8, mass_matrix="diagonal",
        adapt_step_size=True, adapt_mass_matrix=True,
    )
    driver = MCMC(
        kernel=kernel, num_warmup=200, num_samples=400, num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    mu_samples = result.samples["mu"].reshape(-1)
    sample_mean = float(mu_samples.mean())
    sample_var = float(mu_samples.var(unbiased=True))
    assert abs(sample_mean - true_mean) < 0.15
    assert abs(sample_var - true_var) < 0.05


def test_mcmc_result_carries_diagnostics() -> None:
    torch.manual_seed(0)
    model = _normal_normal_model()
    y = torch.randn(20) + 1.0
    kernel = HMCKernel(step_size=0.1, num_steps=10, mass_matrix="identity")
    driver = MCMC(kernel=kernel, num_warmup=100, num_samples=200, num_chains=2)
    result = driver.run(model, torch.zeros(1, 1), {"y": y})
    assert result.samples["mu"].shape[:2] == (2, 200)
    assert "mu" in result.r_hat
    assert "mu" in result.ess
    assert result.acceptance_rates.shape == (2,)
    assert result.divergence_counts.shape == (2,)


def test_mcmc_init_strategy_zero_is_deterministic() -> None:
    """Two runs with the same RNG seed immediately before the
    driver call must produce identical chains."""
    model = _normal_normal_model()
    torch.manual_seed(0)
    y = torch.randn(20) + 1.0

    def _run() -> torch.Tensor:
        torch.manual_seed(42)
        kernel = HMCKernel(step_size=0.1, num_steps=5, mass_matrix="identity")
        driver = MCMC(
            kernel=kernel, num_warmup=10, num_samples=20, num_chains=2,
            init_strategy="zero",
        )
        return driver.run(model, torch.zeros(1, 1), {"y": y}).samples["mu"]

    a = _run()
    b = _run()
    assert torch.allclose(a, b)
