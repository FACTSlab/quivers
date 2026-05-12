"""Tier-1 benchmark: closed-form conjugate posteriors.

For each algorithm × conjugate problem we assert the recovered
posterior mean matches the analytical closed-form to within a
tier-1 tolerance band (RMSE < 0.1 for the parameter of interest).
The band is calibrated so:

* :class:`AutoNormalGuide` passes every Tier-1 problem cleanly
  (mean-field is near-exact for univariate conjugate posteriors).
* :class:`AutoMultivariateNormalGuide` and HMC also pass, with
  variance recovery on Normal-Normal tight to within 0.05.
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
from tests.benchmarks.datasets import beta_bernoulli, normal_normal
from tests.benchmarks.metrics import (
    posterior_mean_error,
    posterior_variance_error,
)
from tests.benchmarks.references import (
    beta_bernoulli_reference,
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
# Normal-Normal
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
    assert err < 0.05, (
        f"AutoNormalGuide / Normal-Normal: variance error {err:.4f} > 0.05"
    )


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
# Beta-Bernoulli
# ---------------------------------------------------------------------------


def test_autonormal_beta_bernoulli_recovers_mean() -> None:
    torch.manual_seed(0)
    data = beta_bernoulli()
    ref = beta_bernoulli_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations)
    samples = _guide_samples(guide, site="theta")
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.05, (
        f"AutoNormalGuide / Beta-Bernoulli: mean error {err:.4f} > 0.05"
    )


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
    assert err < 0.05, (
        f"HMC / Beta-Bernoulli: mean error {err:.4f} > 0.05"
    )
