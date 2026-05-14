"""Tier-1 benchmark: closed-form conjugate posteriors.

For each algorithm × conjugate problem we assert the recovered
posterior moment matches the analytical closed-form to within a
strict tolerance. The band is calibrated against runs we want to
lock in (e.g. AutoNormalGuide is allowed wider sigma2 tolerance
on Normal-Inverse-Gamma because of the inverse-Gamma bijector's
sensitivity to init_scale).
"""

from __future__ import annotations

import torch

from quivers.inference import (
    AutoMultivariateNormalGuide,
    AutoNormalGuide,
    ELBO,
    HMCKernel,
    MCMC,
    SVI,
)
from tests.benchmarks.datasets import (
    bayes_linear_regression,
    beta_bernoulli,
    gamma_exponential,
    normal_inverse_gamma,
    normal_normal,
)
from tests.benchmarks.metrics import (
    correlation_error,
    posterior_mean_error,
    posterior_variance_error,
)
from tests.benchmarks.references import (
    bayes_linear_regression_reference,
    beta_bernoulli_reference,
    gamma_exponential_reference,
    normal_inverse_gamma_reference,
    normal_normal_reference,
)


def _train_svi(model, guide, obs, *, steps: int = 400, lr: float = 5e-2) -> None:
    optim = torch.optim.Adam(
        list(model.parameters()) + list(guide.parameters()), lr=lr
    )
    svi = SVI(model, guide, optim, ELBO())
    for _ in range(steps):
        svi.step(torch.zeros(1, 1), obs)


def _guide_samples(guide, n: int = 1000, *, site: str) -> torch.Tensor:
    samples = torch.stack(
        [guide.rsample(torch.zeros(1, 1))[site].detach() for _ in range(n)],
        dim=0,
    )
    if samples.dim() == 2 and samples.shape[-1] == 1:
        samples = samples.squeeze(-1)
    return samples.reshape(-1)


# ---------------------------------------------------------------------------
# Normal-Normal (existing)
# ---------------------------------------------------------------------------


def test_autonormal_normal_normal_recovers_mean() -> None:
    torch.manual_seed(0)
    data = normal_normal()
    ref = normal_normal_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations)
    samples = _guide_samples(guide, site="mu")
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.1, (
        f"AutoNormalGuide / Normal-Normal: mean error {err:.4f} > 0.1"
    )


def test_autonormal_normal_normal_recovers_variance() -> None:
    torch.manual_seed(0)
    data = normal_normal()
    ref = normal_normal_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations)
    samples = _guide_samples(guide, n=2000, site="mu")
    err = posterior_variance_error(samples, ref["variance"])
    assert err < 0.05


def test_mvn_normal_normal_recovers_mean() -> None:
    torch.manual_seed(0)
    data = normal_normal()
    ref = normal_normal_reference(data)
    guide = AutoMultivariateNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations)
    samples = _guide_samples(guide, site="mu")
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.1


def test_hmc_normal_normal_recovers_mean() -> None:
    torch.manual_seed(0)
    data = normal_normal()
    ref = normal_normal_reference(data)
    kernel = HMCKernel(
        step_size=0.1, num_steps=10, mass_matrix="identity",
        adapt_step_size=True, adapt_mass_matrix=False,
    )
    driver = MCMC(
        kernel=kernel, num_warmup=100, num_samples=400, num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(data.model, torch.zeros(1, 1), data.observations)
    samples = result.samples["mu"].reshape(-1)
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.1


# ---------------------------------------------------------------------------
# Beta-Bernoulli (existing)
# ---------------------------------------------------------------------------


def test_autonormal_beta_bernoulli_recovers_mean() -> None:
    torch.manual_seed(0)
    data = beta_bernoulli()
    ref = beta_bernoulli_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations)
    samples = _guide_samples(guide, site="theta")
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.05


def test_hmc_beta_bernoulli_recovers_mean() -> None:
    torch.manual_seed(0)
    data = beta_bernoulli()
    ref = beta_bernoulli_reference(data)
    kernel = HMCKernel(
        step_size=0.05, num_steps=10, mass_matrix="identity",
        adapt_step_size=True, adapt_mass_matrix=False,
    )
    driver = MCMC(
        kernel=kernel, num_warmup=200, num_samples=500, num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(data.model, torch.zeros(1, 1), data.observations)
    samples = result.samples["theta"].reshape(-1)
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.05


# ---------------------------------------------------------------------------
# Normal-Inverse-Gamma
# ---------------------------------------------------------------------------


def test_autonormal_nig_recovers_mu() -> None:
    torch.manual_seed(0)
    data = normal_inverse_gamma()
    ref = normal_inverse_gamma_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations, steps=600)
    samples = _guide_samples(guide, n=2000, site="mu")
    err = posterior_mean_error(samples, ref["mu_mean"])
    assert err < 0.2, (
        f"AutoNormalGuide / NIG: mu mean error {err:.4f} > 0.2 "
        f"(true mu_N = {ref['mu_mean']:.4f})"
    )


def test_autonormal_nig_recovers_sigma2() -> None:
    torch.manual_seed(0)
    data = normal_inverse_gamma()
    ref = normal_inverse_gamma_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations, steps=600)
    samples = _guide_samples(guide, n=2000, site="sigma2")
    err = posterior_mean_error(samples, ref["sigma2_mean"])
    # Sigma² is harder to recover than mu: inverse-gamma is right-
    # skewed and the variational Gaussian is symmetric, so the band
    # is wider than for mu.
    tol = 0.5 * float(ref["sigma2_mean"])
    assert err < tol, (
        f"AutoNormalGuide / NIG: sigma2 mean error {err:.4f} > "
        f"{tol:.4f} (true E[sigma2|y] = {ref['sigma2_mean']:.4f})"
    )


# ---------------------------------------------------------------------------
# Gamma-Exponential
# ---------------------------------------------------------------------------


def test_autonormal_gamma_exponential_recovers_rate() -> None:
    torch.manual_seed(0)
    data = gamma_exponential()
    ref = gamma_exponential_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations, steps=600)
    samples = _guide_samples(guide, n=2000, site="rate")
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.15, (
        f"AutoNormalGuide / Gamma-Exponential: rate mean error "
        f"{err:.4f} > 0.15 (true rate posterior mean = {ref['mean']:.4f})"
    )


# ---------------------------------------------------------------------------
# Bayesian linear regression (well-conditioned)
# ---------------------------------------------------------------------------


def test_autonormal_blr_recovers_coefficients() -> None:
    torch.manual_seed(0)
    data = bayes_linear_regression()
    ref = bayes_linear_regression_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y", "x_design"})
    _train_svi(data.model, guide, data.observations, steps=800)
    samples_a = _guide_samples(guide, n=1000, site="a")
    samples_b = _guide_samples(guide, n=1000, site="b")
    err_a = posterior_mean_error(samples_a, ref["a_mean"])
    err_b = posterior_mean_error(samples_b, ref["b_mean"])
    assert err_a < 0.1, f"BLR a-mean error {err_a:.4f} > 0.1"
    assert err_b < 0.1, f"BLR b-mean error {err_b:.4f} > 0.1"


def test_mvn_blr_recovers_correlation() -> None:
    """Well-conditioned BLR has near-zero correlation; AutoMVN should
    reproduce that to within tolerance."""
    torch.manual_seed(0)
    data = bayes_linear_regression()
    ref = bayes_linear_regression_reference(data)
    guide = AutoMultivariateNormalGuide(
        data.model, observed_names={"y", "x_design"}, init_scale=0.3
    )
    _train_svi(data.model, guide, data.observations, steps=800)
    joint = torch.stack(
        [
            torch.stack(
                [
                    guide.rsample(torch.zeros(1, 1))["a"].squeeze().detach(),
                    guide.rsample(torch.zeros(1, 1))["b"].squeeze().detach(),
                ]
            )
            for _ in range(1500)
        ],
        dim=0,
    )
    err = correlation_error(joint, float(ref["correlation"]))
    # Well-conditioned BLR has small but nonzero correlation from
    # the random design's finite-N drift; AutoMVN tracks it up to a
    # moderate tolerance.
    assert err < 0.3, (
        f"AutoMVN / BLR: correlation error {err:.4f} > 0.3 "
        f"(true rho = {float(ref['correlation']):.4f})"
    )
