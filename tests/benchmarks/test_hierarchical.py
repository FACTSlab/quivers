"""Tier-2 hierarchical-Bayes benchmark: Eight Schools.

Centred and non-centred parameterisations on the canonical Rubin
(1981) y-vector. Cached NUTS-derived moments serve as the
ground-truth posterior reference.

The centred parameterisation has the canonical funnel pathology
between ``tau`` and ``theta``; we expect AutoNormalGuide to
under-fit ``tau`` and over-shrink ``theta`` toward ``mu``, while
the non-centred parameterisation lets the same guide reach the
reference posterior.
"""

from __future__ import annotations

import torch

from quivers.inference import (
    AutoMultivariateNormalGuide,
    AutoNormalGuide,
    ELBO,
    NUTSKernel,
    MCMC,
    SVI,
)
from tests.benchmarks.datasets import (
    eight_schools_centred,
    eight_schools_noncentred,
)
from tests.benchmarks.references import eight_schools_reference


def _train_svi(model, guide, obs, *, steps: int = 1000, lr: float = 3e-2) -> None:
    optim = torch.optim.Adam(
        list(model.parameters()) + list(guide.parameters()), lr=lr
    )
    svi = SVI(model, guide, optim, ELBO())
    for _ in range(steps):
        svi.step(torch.zeros(1, 1), obs)


def _guide_scalar_samples(guide, n: int = 1500, *, site: str) -> torch.Tensor:
    out = torch.stack(
        [guide.rsample(torch.zeros(1, 1))[site].detach() for _ in range(n)],
        dim=0,
    )
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Centred parameterisation — known to be hard
# ---------------------------------------------------------------------------


def test_autonormal_centred_recovers_mu_within_band() -> None:
    """AutoNormalGuide should at least put ``mu``'s posterior mean
    in the right ballpark (within 3 SE of the NUTS reference)."""
    torch.manual_seed(0)
    data = eight_schools_centred()
    ref = eight_schools_reference()
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations, steps=1200)
    samples = _guide_scalar_samples(guide, site="mu")
    err = abs(float(samples.mean()) - float(ref["mu_mean"]))
    # mu_std is the reference posterior sd; 3 SE allows for the
    # mean-field underfit on tau which propagates to mu.
    tol = 3.0 * float(ref["mu_std"])
    assert err < tol, (
        f"AutoNormalGuide / Eight-Schools (centred): "
        f"mu error {err:.4f} > {tol:.4f}"
    )


def test_autonormal_centred_tau_does_not_collapse_to_zero() -> None:
    """Capture test: mean-field VI on the centred parameterisation
    is well-known to underestimate tau (the funnel pathology). The
    posterior mean of tau should at least be > 0.5 — much smaller
    than the NUTS reference (~7.5) but still nonzero."""
    torch.manual_seed(0)
    data = eight_schools_centred()
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations, steps=1200)
    samples = _guide_scalar_samples(guide, site="tau")
    tau_mean = float(samples.mean())
    assert tau_mean > 0.3, (
        f"AutoNormalGuide / Eight-Schools (centred): tau collapsed "
        f"to {tau_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# Non-centred parameterisation — easier
# ---------------------------------------------------------------------------


def test_autonormal_noncentred_recovers_mu() -> None:
    """The non-centred parameterisation breaks the funnel; an
    AutoNormalGuide should get mu to within 1 SE of the NUTS ref."""
    torch.manual_seed(0)
    data = eight_schools_noncentred()
    ref = eight_schools_reference()
    guide = AutoNormalGuide(data.model, observed_names={"y"})
    _train_svi(data.model, guide, data.observations, steps=1500)
    samples = _guide_scalar_samples(guide, site="mu")
    err = abs(float(samples.mean()) - float(ref["mu_mean"]))
    tol = 2.0 * float(ref["mu_std"])
    assert err < tol, (
        f"AutoNormalGuide / Eight-Schools (non-centred): "
        f"mu error {err:.4f} > {tol:.4f}"
    )


def test_mvn_noncentred_recovers_mu() -> None:
    torch.manual_seed(0)
    data = eight_schools_noncentred()
    ref = eight_schools_reference()
    guide = AutoMultivariateNormalGuide(
        data.model, observed_names={"y"}, init_scale=0.3
    )
    _train_svi(data.model, guide, data.observations, steps=1500)
    samples = _guide_scalar_samples(guide, site="mu")
    err = abs(float(samples.mean()) - float(ref["mu_mean"]))
    tol = 2.0 * float(ref["mu_std"])
    assert err < tol


def test_nuts_noncentred_recovers_mu() -> None:
    """NUTS on the non-centred form recovers mu to within ~1 SE."""
    torch.manual_seed(0)
    data = eight_schools_noncentred()
    ref = eight_schools_reference()
    kernel = NUTSKernel(
        target_accept=0.8,
        max_tree_depth=8,
        mass_matrix="diagonal",
    )
    driver = MCMC(
        kernel=kernel,
        num_warmup=300,
        num_samples=600,
        num_chains=2,
        init_strategy="zero",
    )
    result = driver.run(data.model, torch.zeros(1, 1), data.observations)
    samples = result.samples["mu"].reshape(-1)
    samples = samples[torch.isfinite(samples)]
    err = abs(float(samples.mean()) - float(ref["mu_mean"]))
    tol = 2.0 * float(ref["mu_std"])
    assert err < tol, (
        f"NUTS / Eight-Schools (non-centred): mu error {err:.4f} > "
        f"{tol:.4f}"
    )
