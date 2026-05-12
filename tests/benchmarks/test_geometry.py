"""Tier-3 benchmark: hard posterior geometries (correlated MVN).

The ``correlated_regression`` data generator uses a near-collinear
design so the posterior over the two regression coefficients
``(a, b)`` has a strong off-diagonal correlation (analytical
correlation computed in :func:`correlated_regression_reference`).

This tier exercises the algorithm-dispatch decision the inference
plan is built around: mean-field VI is expected to fail to
recover the correlation (the variational posterior is diagonal),
while AutoMultivariateNormalGuide and HMC recover it.

Each test asserts an explicit pass-band per algorithm; the
:func:`test_autonormal_fails_to_recover_correlation` test is a
*capture* — it asserts the expected mean-field failure mode and
will fire if AutoNormalGuide ever stops behaving as mean-field.
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
from tests.benchmarks.datasets import correlated_regression
from tests.benchmarks.metrics import (
    correlation_error,
    posterior_mean_error,
)
from tests.benchmarks.references import correlated_regression_reference


def _train(model, guide, obs, steps: int = 600, lr: float = 5e-2) -> None:
    optim = torch.optim.Adam(
        list(model.parameters()) + list(guide.parameters()), lr=lr
    )
    svi = SVI(model, guide, optim, ELBO())
    for _ in range(steps):
        svi.step(torch.zeros(1, 1), obs)


def _joint_samples(guide, n: int = 1500) -> torch.Tensor:
    """Stack of ``(n, 2)`` joint draws of ``(a, b)``."""
    rows = []
    for _ in range(n):
        d = guide.rsample(torch.zeros(1, 1))
        a = d["a"].detach().reshape(-1)
        b = d["b"].detach().reshape(-1)
        rows.append(torch.stack([a[0], b[0]]))
    return torch.stack(rows, dim=0)


def test_autonormal_recovers_marginal_means() -> None:
    """Even a mean-field guide hits the marginal means of (a, b)
    within a generous tolerance."""
    torch.manual_seed(0)
    data = correlated_regression()
    ref = correlated_regression_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y", "x_design"})
    _train(data.model, guide, data.observations)
    samples = _joint_samples(guide)
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.2


def test_autonormal_fails_to_recover_correlation() -> None:
    """Capture test: AutoNormalGuide collapses the off-diagonal to
    ~0 regardless of the true posterior correlation. The error
    must stay above 0.3 — if it ever drops below, AutoNormalGuide
    has changed and we want to know."""
    torch.manual_seed(0)
    data = correlated_regression()
    ref = correlated_regression_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y", "x_design"})
    _train(data.model, guide, data.observations)
    samples = _joint_samples(guide)
    err = correlation_error(samples, float(ref["correlation"]))
    assert err > 0.3


def test_mvn_recovers_correlation() -> None:
    torch.manual_seed(0)
    data = correlated_regression()
    ref = correlated_regression_reference(data)
    guide = AutoMultivariateNormalGuide(
        data.model, observed_names={"y", "x_design"}, init_scale=0.3
    )
    _train(data.model, guide, data.observations, steps=800, lr=5e-2)
    samples = _joint_samples(guide)
    err = correlation_error(samples, float(ref["correlation"]))
    assert err < 0.15, (
        f"AutoMVN / CorrelatedMVN: correlation error {err:.4f} > 0.15 "
        f"(true rho = {float(ref['correlation']):.4f})"
    )


def test_hmc_recovers_correlation_and_means() -> None:
    torch.manual_seed(0)
    data = correlated_regression()
    ref = correlated_regression_reference(data)
    kernel = HMCKernel(
        step_size=0.1, num_steps=15, mass_matrix="diagonal",
        adapt_step_size=True, adapt_mass_matrix=True,
    )
    driver = MCMC(
        kernel=kernel, num_warmup=200, num_samples=600, num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(data.model, torch.zeros(1, 1), data.observations)
    a_samples = result.samples["a"].reshape(-1)
    b_samples = result.samples["b"].reshape(-1)
    joint = torch.stack([a_samples, b_samples], dim=-1)
    err = correlation_error(joint, float(ref["correlation"]))
    mean_err = posterior_mean_error(joint, ref["mean"])
    assert mean_err < 0.15
    assert err < 0.2
