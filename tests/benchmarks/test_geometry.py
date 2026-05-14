"""Tier-3 benchmarks: hard posterior geometries.

Coverage:

* **Correlated regression** — a near-collinear design produces a
  strongly-correlated posterior. Mean-field VI collapses the
  off-diagonal; full-rank MVN and HMC recover.
* **Neal's funnel** — the canonical scale-of-scale dependency
  (Neal 2003). Mean-field VI cannot fit the joint; HMC needs
  reparameterized mass adaptation.
* **Ill-conditioned product Gaussian** — per-dim posteriors with
  scales spanning four orders of magnitude. Stress for mass-matrix
  adaptation; AutoNormalGuide with appropriate init_scale handles it.
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
    correlated_regression,
    ill_conditioned_mvn,
    neal_funnel,
)
from tests.benchmarks.metrics import (
    correlation_error,
    posterior_mean_error,
)
from tests.benchmarks.references import (
    correlated_regression_reference,
    ill_conditioned_mvn_reference,
    neal_funnel_reference,
)


def _train(model, guide, obs, *, steps: int = 600, lr: float = 5e-2) -> None:
    optim = torch.optim.Adam(
        list(model.parameters()) + list(guide.parameters()), lr=lr
    )
    svi = SVI(model, guide, optim, ELBO())
    for _ in range(steps):
        svi.step(torch.zeros(1, 1), obs)


def _joint_samples(guide, sites: tuple[str, ...], n: int = 1500) -> torch.Tensor:
    rows = []
    for _ in range(n):
        d = guide.rsample(torch.zeros(1, 1))
        rows.append(
            torch.stack([d[s].detach().reshape(-1)[0] for s in sites])
        )
    return torch.stack(rows, dim=0)


# ---------------------------------------------------------------------------
# Correlated regression — existing
# ---------------------------------------------------------------------------


def test_autonormal_recovers_marginal_means() -> None:
    torch.manual_seed(0)
    data = correlated_regression()
    ref = correlated_regression_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y", "x_design"})
    _train(data.model, guide, data.observations)
    samples = _joint_samples(guide, ("a", "b"))
    err = posterior_mean_error(samples, ref["mean"])
    assert err < 0.2


def test_autonormal_fails_to_recover_correlation() -> None:
    """Capture: mean-field collapses the off-diagonal to ~0."""
    torch.manual_seed(0)
    data = correlated_regression()
    ref = correlated_regression_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"y", "x_design"})
    _train(data.model, guide, data.observations)
    samples = _joint_samples(guide, ("a", "b"))
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
    samples = _joint_samples(guide, ("a", "b"))
    err = correlation_error(samples, float(ref["correlation"]))
    assert err < 0.15, (
        f"AutoMVN / CorrelatedMVN: correlation error {err:.4f} > 0.15"
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


# ---------------------------------------------------------------------------
# Neal's funnel
# ---------------------------------------------------------------------------


def test_autonormal_funnel_captures_negative_bias() -> None:
    """Capture test for the funnel pathology: AutoNormalGuide's
    posterior over v has the right sign (negative, away from the
    prior mean of zero) but the magnitude is severely under-
    estimated. This documents the known mean-field failure on
    funnel-shaped posteriors — analogous to the
    ``test_autonormal_fails_to_recover_correlation`` capture for
    the correlated-regression case.
    """
    torch.manual_seed(0)
    data = neal_funnel()
    ref = neal_funnel_reference(data)
    guide = AutoNormalGuide(data.model, observed_names={"x"})
    _train(data.model, guide, data.observations, steps=1500, lr=2e-2)
    samples = torch.stack(
        [
            guide.rsample(torch.zeros(1, 1))["v"].detach().reshape(-1)[0]
            for _ in range(2000)
        ]
    )
    recovered_mean = float(samples.mean())
    target_mean = float(ref["v_mean"])
    assert recovered_mean < 0.0, (
        f"AutoNormal / Neal-funnel: posterior mean over v should be "
        f"negative; got {recovered_mean:.4f}"
    )
    # The capture is that the magnitude is much smaller than the
    # analytical target — at least 4x off.
    assert abs(recovered_mean) < 0.5 * abs(target_mean), (
        f"AutoNormal / Neal-funnel: posterior mean over v "
        f"{recovered_mean:.2f} is too close to analytical "
        f"{target_mean:.2f}; expected under-estimation"
    )


# ---------------------------------------------------------------------------
# Ill-conditioned product Gaussian
# ---------------------------------------------------------------------------


def test_autonormal_recovers_per_dim_means() -> None:
    """Each x_i has a distinct prior scale; AutoNormalGuide with a
    moderate init_scale recovers all five posterior means.

    The large-scale dimensions (x_1, x_2) require more SVI steps
    to march the variational mean from 0 to the data; we run long
    enough for all five to converge.
    """
    torch.manual_seed(0)
    data = ill_conditioned_mvn()
    ref = ill_conditioned_mvn_reference(data)
    guide = AutoNormalGuide(
        data.model,
        observed_names={f"y_{i + 1}" for i in range(5)},
        init_scale=0.3,
    )
    _train(data.model, guide, data.observations, steps=4000, lr=0.1)
    recovered = torch.stack(
        [
            torch.stack(
                [
                    guide.rsample(torch.zeros(1, 1))[f"x_{i + 1}"]
                    .detach()
                    .reshape(-1)[0]
                    for i in range(5)
                ]
            )
            for _ in range(800)
        ],
        dim=0,
    )
    err_per_dim = (recovered.mean(dim=0) - ref["mean"]).abs()
    # Tolerance per dim is 3 posterior sd's (allows for finite-SVI
    # imprecision on the tightest-scale dimensions where the
    # variational mean has very little gradient signal).
    post_sd = ref["variance"].sqrt()
    tol = 3.0 * post_sd
    # The widest-scale dimensions also need an absolute floor based
    # on the prior scale, since their post_sd is comparable to the
    # prior and finite SVI may leave error of the same magnitude.
    prior_scales = data.true_params["prior_scales"]
    assert isinstance(prior_scales, torch.Tensor)
    tol = torch.maximum(tol, 0.25 * prior_scales)
    fails = (err_per_dim > tol).nonzero().reshape(-1).tolist()
    assert not fails, (
        f"AutoNormal / ill-conditioned MVN: dims {fails} exceed tolerance; "
        f"errors = {err_per_dim.tolist()}, tolerances = {tol.tolist()}"
    )
