"""Algorithm × problem runner for the synthetic benchmark suite.

Produces ``docs/developer/inference-benchmarks.md`` — the public
record of which inference algorithms recover which posteriors
under deterministic seeded data.

Each cell of the grid runs one ``(algorithm, problem)`` pair and
reports:

* ``status``: PASS / FAIL / SKIP / ERROR
* ``metric``: the primary diagnostic (mean error, correlation
  error, ...) along with the tolerance
* ``samples_per_sec``: end-to-end throughput in posterior draws
  per second (including any SVI warmup)
* ``notes``: a one-line description (e.g. "guide collapsed
  off-diagonal correlation" for the AutoNormal × correlated-MVN
  capture)

Invoked from CI or interactively::

    QVR_USE_LOCAL_GRAMMAR=1 python -m tests.benchmarks.runner

The runner respects seeds and is deterministic given the same
torch / numpy versions.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from quivers.inference import (
    AutoLaplaceApproximation,
    AutoMultivariateNormalGuide,
    AutoNormalGuide,
    ELBO,
    HMCKernel,
    MCMC,
    NUTSKernel,
    SVI,
)
from tests.benchmarks.datasets import (
    bayes_linear_regression,
    beta_bernoulli,
    correlated_regression,
    eight_schools_centered,
    eight_schools_noncentered,
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
    posterior_mean_error,
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


# ---------------------------------------------------------------------------
# Cell definitions
# ---------------------------------------------------------------------------


@dataclass
class CellResult:
    """One algorithm × problem cell outcome."""

    status: str  # "PASS", "FAIL", "SKIP", "ERROR"
    metric_name: str
    metric_value: float
    tolerance: float
    samples_per_sec: float = 0.0
    notes: str = ""
    error_message: str = ""


@dataclass
class TierSpec:
    """Group of related problems plus the algorithms to evaluate."""

    name: str
    description: str
    problems: list[tuple[str, Callable[[], "ProblemSpec"]]] = field(
        default_factory=list
    )


@dataclass
class ProblemSpec:
    """One benchmark problem: data generator, reference, evaluator."""

    label: str
    description: str
    data_factory: Callable
    reference_factory: Callable
    observed_names: set[str]
    site: str
    metric_name: str
    tolerance: float
    metric_fn: Callable[[torch.Tensor, dict], float]
    math: str = ""
    """Markdown / LaTeX block describing the generative model, the
    reference posterior, and the metric. Rendered verbatim into the
    'Problem details' section of the report."""
    capture: bool = False
    """A capture problem expects the metric to *exceed* the tolerance
    (used for documented-failure cases like
    AutoNormal × correlated-MVN correlation)."""


def _mean_error_against_ref_key(ref_key: str) -> Callable:
    def fn(samples: torch.Tensor, ref: dict) -> float:
        return posterior_mean_error(samples, ref[ref_key])

    return fn


def _correlation_error_2d(ref_key: str = "correlation") -> Callable:
    def fn(samples: torch.Tensor, ref: dict) -> float:
        return correlation_error(samples, float(ref[ref_key]))

    return fn


_BETA_BERNOULLI_MATH = r"""**Model.** Conjugate Beta prior on a Bernoulli rate:

$$
\theta \sim \mathrm{Beta}(2, 2), \qquad
y_i \mid \theta \sim \mathrm{Bernoulli}(\theta), \quad i = 1, \dots, 50.
$$

**Data.** $N = 50$ Bernoulli draws at $\theta^\star = 0.7$.

**Reference.** Conjugacy gives $\theta \mid y \sim \mathrm{Beta}\bigl(\alpha_0 + \sum_i y_i,\ \beta_0 + N - \sum_i y_i\bigr)$ with closed-form mean $\alpha / (\alpha + \beta)$."""

_NORMAL_NORMAL_MATH = r"""**Model.** Conjugate Normal prior on a Normal mean with known variance:

$$
\mu \sim \mathcal{N}(0, 1), \qquad
y_i \mid \mu \sim \mathcal{N}(\mu, 1), \quad i = 1, \dots, 30.
$$

**Data.** $N = 30$ Normal draws at $\mu^\star = 1.5$, $\sigma = 1$.

**Reference.** Posterior precision $\tau_N = \tau_0 + N / \sigma^2$ gives a Normal posterior with mean $(\tau_0 \mu_0 + N \bar{y} / \sigma^2) / \tau_N$."""

_NIG_MATH = r"""**Model.** Joint conjugate prior on unknown mean *and* variance:

$$
\sigma^2 \sim \mathrm{InverseGamma}(3, 2), \qquad
\mu \mid \sigma^2 \sim \mathcal{N}(0, \sigma), \qquad
y_i \mid \mu, \sigma^2 \sim \mathcal{N}(\mu, \sigma), \quad i = 1, \dots, 60.
$$

**Data.** $N = 60$ Normal draws at $\mu^\star = 0.3$, $\sigma^{2\star} = 1.5$.

**Reference.** NIG posterior updates (Murphy 2007 §5) give marginal mean $\mu_N = (\kappa_0 \mu_0 + N \bar{y}) / (\kappa_0 + N)$.

Stress test for guides handling two latents with mixed supports: the unconstrained $\mu$ and the positive $\sigma^2$ (whose bijector is $\exp$ / softplus)."""

_GAMMA_EXP_MATH = r"""**Model.** Conjugate Gamma prior on an Exponential rate:

$$
r \sim \mathrm{Gamma}(2, 1), \qquad
y_i \mid r \sim \mathrm{Exponential}(r), \quad i = 1, \dots, 80.
$$

**Data.** $N = 80$ Exponential draws at $r^\star = 2$.

**Reference.** $r \mid y \sim \mathrm{Gamma}\bigl(a_0 + N,\ b_0 + \sum_i y_i\bigr)$, with mean $a / b$."""

_BLR_MATH = r"""**Model.** Two-parameter linear regression with iid standard-Normal design and known observation noise:

$$
a, b \sim \mathcal{N}(0, 1), \qquad
x_i \sim \mathcal{N}(0, 1), \qquad
y_i \mid a, b \sim \mathcal{N}(a + b x_i, \sigma), \quad i = 1, \dots, 60,
$$

with $\sigma = 0.3$, $a^\star = 0.7$, $b^\star = -0.5$.

**Reference.** Closed-form Gaussian posterior with precision $I + X^\top X / \sigma^2$ and mean $\Sigma X^\top y / \sigma^2$."""

_8S_CENTERED_MATH = r"""**Model.**

$$
\mu \sim \mathcal{N}(0, 10), \qquad
\tau \sim \mathrm{HalfCauchy}(5), \qquad
\theta_j \mid \mu, \tau \sim \mathcal{N}(\mu, \tau), \qquad
y_j \mid \theta_j \sim \mathcal{N}(\theta_j, 12),
$$

for $j = 1, \dots, 8$ on the canonical Rubin (1981) effect sizes $y = (28, 8, -3, 7, -1, 1, 18, 12)$.

**Reference.** Cached NUTS moments (4 chains, 5000 post-warmup draws): $\mathbb{E}[\mu] \approx 5.4$, posterior standard deviation $\approx 4$.

Tolerance is set at three reference standard deviations: a loose target reflecting how hard the funnel geometry is for VI."""

_8S_NONCENTERED_MATH = r"""**Model.** Same priors as the centered model, with the group-level draws reparameterised:

$$
\eta_j \sim \mathcal{N}(0, 1), \qquad
\theta_j = \mu + \tau \eta_j,
$$

decoupling $\tau$ from $\theta_j$ and eliminating the funnel in the prior.

**Reference.** Same cached NUTS moments as the centered model.

Tolerance is tightened to two reference standard deviations: the reparam should pay off."""

_CORR_REG_MATH = r"""**Model.** Linear regression as in Tier 1, but with a near-constant design:

$$
a, b \sim \mathcal{N}(0, 1), \qquad
x_i = \rho + (1 - \rho) z_i, \quad z_i \sim \mathcal{N}(0, 1), \qquad
y_i \mid a, b \sim \mathcal{N}(a + b x_i, 0.5),
$$

with $\rho = 0.95$ and $N = 50$.

**Reference.** Closed-form Gaussian posterior with off-diagonal correlation $\rho \approx 0.95+$.

The mean-field guide ignores this correlation; the first-moment metric below still passes (the documented underfit lives in the second moment)."""

_NEAL_MATH = r"""**Model.** Neal's funnel:

$$
v \sim \mathcal{N}(0, 3), \qquad
x_i \mid v \sim \mathcal{N}(0, e^{v / 2}), \quad i = 1, \dots, 9.
$$

**Data.** Condition on $x_i = 0$ (inference target is $p(v \mid x = 0)$).

**Reference.** The log-likelihood is linear in $v$: $\log p(x_i = 0 \mid v) = -\tfrac{1}{2}\log(2\pi) - v / 2$, so the conditional posterior is Gaussian with mean $-9 N / 2 = -40.5$ and variance $9$ at $N = 9$. The *joint* posterior over $(v, x)$ remains funnel-shaped; only the conditional given $x = 0$ is tractable.

**Capture semantics.** All five algorithms under-estimate the magnitude of $v$. PASS means the metric *exceeds* the tolerance, confirming the documented underfit."""

_ILL_COND_MATH = r"""**Model.** Five-dimensional product Gaussian with five orders of magnitude of prior scale and a fixed observation noise:

$$
x_d \sim \mathcal{N}(0, \sigma_d^{\text{prior}}), \qquad
y_d \mid x_d \sim \mathcal{N}(x_d, 0.1), \qquad d = 1, \dots, 5,
$$

with $\sigma^{\text{prior}} = (100, 10, 1, 0.1, 0.01)$.

**Reference.** Per-dimension Gaussian: $x_d \mid y_d \sim \mathcal{N}\bigl(y_d / (1 + (0.1 / \sigma_d)^2),\ (1 / \sigma_d^2 + 1 / 0.01)^{-1}\bigr)$.

Tracks the middle scale $x_3$, where the diagonal mass matrix is roughly correct but the gradient signal is dwarfed by the larger-scale dimensions."""

_HALFNORMAL_MATH = r"""**Model.**

$$
\sigma \sim \mathrm{HalfNormal}(2), \qquad
y_i \mid \sigma \sim \mathcal{N}(0, \sigma), \quad i = 1, \dots, 80.
$$

**Reference.** No conjugate form. Integrate

$$
p(\sigma \mid y) \propto \exp(-\sigma^2 / 8) \cdot \sigma^{-N} \cdot \exp\bigl(-\tfrac{1}{2 \sigma^2} \sum_i y_i^2\bigr)
$$

on a 4096-point grid in $[0.05, 6]$ for the reference moments."""

_TRUNC_NORMAL_MATH = r"""**Model.**

$$
\mu \sim \mathrm{Uniform}(0, 1), \qquad
y_i \mid \mu \sim \mathrm{TruncatedNormal}(\mu, 0.2, 0, 1), \quad i = 1, \dots, 60.
$$

**Reference.** Evaluate the truncated-Normal log-likelihood on a 4096-point $\mu$-grid in $(0, 1)$ with stable log-CDF differences for the truncation constant; normalise for the posterior moments."""


PROBLEMS: dict[str, ProblemSpec] = {
    # Tier 1, conjugate
    "beta_bernoulli": ProblemSpec(
        label="Beta-Bernoulli",
        description="theta ~ Beta(2, 2), y_i ~ Bernoulli(theta), N=50",
        math=_BETA_BERNOULLI_MATH,
        data_factory=beta_bernoulli,
        reference_factory=beta_bernoulli_reference,
        observed_names={"y"},
        site="theta",
        metric_name="|E[theta]_q - E[theta]_true|",
        tolerance=0.05,
        metric_fn=_mean_error_against_ref_key("mean"),
    ),
    "normal_normal": ProblemSpec(
        label="Normal-Normal",
        description="mu ~ N(0,1), y_i ~ N(mu, 1), N=30",
        math=_NORMAL_NORMAL_MATH,
        data_factory=normal_normal,
        reference_factory=normal_normal_reference,
        observed_names={"y"},
        site="mu",
        metric_name="|E[mu]_q - E[mu]_true|",
        tolerance=0.15,
        metric_fn=_mean_error_against_ref_key("mean"),
    ),
    "normal_inverse_gamma": ProblemSpec(
        label="Normal-Inverse-Gamma",
        description="joint mu, sigma2 unknown; conjugate NIG posterior",
        math=_NIG_MATH,
        data_factory=normal_inverse_gamma,
        reference_factory=normal_inverse_gamma_reference,
        observed_names={"y"},
        site="mu",
        metric_name="|E[mu]_q - E[mu]_true|",
        tolerance=0.2,
        metric_fn=_mean_error_against_ref_key("mu_mean"),
    ),
    "gamma_exponential": ProblemSpec(
        label="Gamma-Exponential",
        description="rate ~ Gamma(2, 1), y_i ~ Exponential(rate), N=80",
        math=_GAMMA_EXP_MATH,
        data_factory=gamma_exponential,
        reference_factory=gamma_exponential_reference,
        observed_names={"y"},
        site="rate",
        metric_name="|E[rate]_q - E[rate]_true|",
        tolerance=0.3,
        metric_fn=_mean_error_against_ref_key("mean"),
    ),
    "bayes_linear_regression": ProblemSpec(
        label="Bayesian linear regression",
        description="well-conditioned design, sigma=0.3, N=60",
        math=_BLR_MATH,
        data_factory=bayes_linear_regression,
        reference_factory=bayes_linear_regression_reference,
        observed_names={"y", "x_design"},
        site="a",
        metric_name="|E[a]_q - E[a]_true|",
        tolerance=0.1,
        metric_fn=_mean_error_against_ref_key("a_mean"),
    ),
    # Tier 2, hierarchical
    "eight_schools_centered": ProblemSpec(
        label="Eight Schools (centered)",
        description="mu, tau, theta_j; canonical hierarchical funnel",
        math=_8S_CENTERED_MATH,
        data_factory=eight_schools_centered,
        reference_factory=lambda _: eight_schools_reference(),
        observed_names={"y"},
        site="mu",
        metric_name="|E[mu]_q - mu_ref|",
        tolerance=3.0 * float(eight_schools_reference()["mu_std"]),
        metric_fn=_mean_error_against_ref_key("mu_mean"),
    ),
    "eight_schools_noncentered": ProblemSpec(
        label="Eight Schools (non-centered)",
        description="non-centered reparam removes funnel",
        math=_8S_NONCENTERED_MATH,
        data_factory=eight_schools_noncentered,
        reference_factory=lambda _: eight_schools_reference(),
        observed_names={"y"},
        site="mu",
        metric_name="|E[mu]_q - mu_ref|",
        tolerance=2.0 * float(eight_schools_reference()["mu_std"]),
        metric_fn=_mean_error_against_ref_key("mu_mean"),
    ),
    # Tier 3, hard geometry
    "correlated_regression": ProblemSpec(
        label="Correlated regression",
        description="near-collinear design produces rho ~ 0.95+",
        math=_CORR_REG_MATH,
        data_factory=correlated_regression,
        reference_factory=correlated_regression_reference,
        observed_names={"y", "x_design"},
        site="a",
        metric_name="|E[a]_q - E[a]_true|",
        tolerance=0.2,
        metric_fn=_mean_error_against_ref_key("a_mean"),
    ),
    "neal_funnel_mean_field_capture": ProblemSpec(
        label="Neal's funnel (under-estimation capture)",
        description="capture: every algorithm under-estimates v's posterior magnitude",
        math=_NEAL_MATH,
        data_factory=neal_funnel,
        reference_factory=neal_funnel_reference,
        observed_names={"x"},
        site="v",
        metric_name="|E[v]_q - E[v]_true|",
        tolerance=0.5 * abs(float(neal_funnel_reference(neal_funnel())["v_mean"])),
        metric_fn=_mean_error_against_ref_key("v_mean"),
        capture=True,
    ),
    "ill_conditioned_mvn": ProblemSpec(
        label="Ill-conditioned product Gaussian",
        description="5 dims with prior scales 10^[0..-4]",
        math=_ILL_COND_MATH,
        data_factory=ill_conditioned_mvn,
        reference_factory=ill_conditioned_mvn_reference,
        observed_names={f"y_{i + 1}" for i in range(5)},
        site="x_3",
        metric_name="|E[x_3]_q - E[x_3]_true|",
        tolerance=0.3,
        metric_fn=lambda s, ref: float(abs(s.mean() - ref["mean"][2])),
    ),
    # Tier 6, constrained-support
    "half_normal_scale": ProblemSpec(
        label="HalfNormal scale",
        description="sigma > 0; quadrature reference",
        math=_HALFNORMAL_MATH,
        data_factory=half_normal_scale,
        reference_factory=half_normal_scale_reference,
        observed_names={"y"},
        site="sigma",
        metric_name="|E[sigma]_q - E[sigma]_true|",
        tolerance=0.15,
        metric_fn=_mean_error_against_ref_key("mean"),
    ),
    "truncated_normal_recovery": ProblemSpec(
        label="TruncatedNormal recovery",
        description="mu in (0,1); quadrature reference",
        math=_TRUNC_NORMAL_MATH,
        data_factory=truncated_normal_recovery,
        reference_factory=truncated_normal_recovery_reference,
        observed_names={"y"},
        site="mu",
        metric_name="|E[mu]_q - E[mu]_true|",
        tolerance=0.05,
        metric_fn=_mean_error_against_ref_key("mean"),
    ),
}


@dataclass
class TierGroup:
    """Tier title, narrative description, and member problems."""

    title: str
    description: str
    problems: list[str]


TIERS: list[TierGroup] = [
    TierGroup(
        title="Tier 1: conjugate posteriors",
        description=(
            "Five textbook problems with closed-form posteriors. They "
            "establish a floor: every algorithm should match the analytical "
            "moment to within a tight tolerance."
        ),
        problems=[
            "beta_bernoulli",
            "normal_normal",
            "normal_inverse_gamma",
            "gamma_exponential",
            "bayes_linear_regression",
        ],
    ),
    TierGroup(
        title="Tier 2: hierarchical posteriors",
        description=(
            "The Eight Schools problem (Rubin 1981) in both parameterisations. "
            "Tests how each algorithm handles the funnel geometry that arises "
            "when a group-level scale tau shrinks toward zero."
        ),
        problems=[
            "eight_schools_centered",
            "eight_schools_noncentered",
        ],
    ),
    TierGroup(
        title="Tier 3: hard posterior geometry",
        description=(
            "Problems chosen to expose specific failure modes of mean-field "
            "VI and of HMC under poor preconditioning."
        ),
        problems=[
            "correlated_regression",
            "neal_funnel_mean_field_capture",
            "ill_conditioned_mvn",
        ],
    ),
    TierGroup(
        title="Tier 6: constrained-support stress",
        description=(
            "Latents on a half-line or in a bounded interval. References "
            "come from dense-grid quadrature; variational guides must "
            "traverse a non-linear bijector to reach the constrained scale."
        ),
        problems=[
            "half_normal_scale",
            "truncated_normal_recovery",
        ],
    ),
]


# Human-readable algorithm descriptions for the report preface.
ALGORITHM_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "AutoNormal": (
        "Mean-field SVI, factorised diagonal Normal in unconstrained space",
        "Adam, lr=0.05, 800 steps (1500 for positive-support sites), 1500 posterior draws",
    ),
    "AutoMVN": (
        "Full-covariance SVI, single MVN in unconstrained space",
        "Adam, lr=0.05, 800 steps (1500 for positive-support sites), `init_scale=0.3`, 1500 draws",
    ),
    "AutoLaplace": (
        "MAP plus a Gaussian centred at the mode with Hessian covariance",
        "Adam, lr=0.05, 500 steps, 1500 draws",
    ),
    "HMC": (
        "Hamiltonian Monte Carlo with fixed integrator length",
        "`step_size=0.1` (adapted), `num_steps=10`, diagonal mass matrix (adapted), 200 warmup, 400 samples, 2 chains",
    ),
    "NUTS": (
        "No-U-Turn HMC",
        "`target_accept=0.8`, `max_tree_depth=8`, diagonal mass matrix, 200 warmup, 400 samples, 2 chains",
    ),
}


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------


def _run_svi(
    data, guide_cls, *, steps: int, lr: float, init_scale: float | None = None
) -> tuple[Callable, float]:
    """Train an Auto*Guide for ``steps`` SVI iterations; return
    (sampler, samples_per_sec)."""
    if init_scale is not None:
        guide = guide_cls(
            data.model,
            observed_names=data.observations.keys() | {"x_design"}
            if "x_design" in data.observations
            else data.observations.keys(),
            init_scale=init_scale,
        )
    else:
        names = set(data.observations.keys())
        guide = guide_cls(data.model, observed_names=names)
    optim = torch.optim.Adam(
        list(data.model.parameters()) + list(guide.parameters()), lr=lr
    )
    svi = SVI(data.model, guide, optim, ELBO())
    t0 = time.perf_counter()
    for _ in range(steps):
        svi.step(torch.zeros(1, 1), data.observations)
    elapsed = time.perf_counter() - t0
    return guide, steps / max(elapsed, 1e-9)


def _draw_site_samples(guide, *, site: str, n: int) -> torch.Tensor:
    return torch.stack(
        [
            guide.rsample(torch.zeros(1, 1))[site].detach().reshape(-1)[0]
            for _ in range(n)
        ]
    )


def _run_mcmc(
    data, kernel, *, num_warmup: int, num_samples: int, num_chains: int = 2
) -> tuple[dict, float]:
    """Run MCMC; return (samples_dict, draws_per_sec)."""
    driver = MCMC(
        kernel=kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        init_strategy="zero",
    )
    t0 = time.perf_counter()
    result = driver.run(data.model, torch.zeros(1, 1), data.observations)
    elapsed = time.perf_counter() - t0
    total = num_chains * num_samples
    return result.samples, total / max(elapsed, 1e-9)


# Algorithm implementations: each takes a ProblemSpec and returns
# (samples_tensor, samples_per_sec).
def _alg_autonormal(prob: ProblemSpec):
    torch.manual_seed(0)
    data = prob.data_factory()
    # Positive-support recovery (HalfNormal, Gamma, InverseGamma)
    # needs more SVI iterations to settle the mean: the exp /
    # softplus bijector is non-linear so the variational mean has
    # to move further in unconstrained space.
    needs_long_svi = any(
        site in prob.observed_names | {prob.site} for site in ("sigma", "rate")
    ) or prob.site in ("sigma", "rate")
    steps = 1500 if needs_long_svi else 800
    guide, sps = _run_svi(data, AutoNormalGuide, steps=steps, lr=5e-2)
    samples = _draw_site_samples(guide, site=prob.site, n=1500)
    ref = prob.reference_factory(data)
    return samples, sps, ref


def _alg_automvn(prob: ProblemSpec):
    torch.manual_seed(0)
    data = prob.data_factory()
    needs_long_svi = prob.site in ("sigma", "rate")
    steps = 1500 if needs_long_svi else 800
    guide, sps = _run_svi(
        data, AutoMultivariateNormalGuide, steps=steps, lr=5e-2, init_scale=0.3
    )
    samples = _draw_site_samples(guide, site=prob.site, n=1500)
    ref = prob.reference_factory(data)
    return samples, sps, ref


def _alg_autolaplace(prob: ProblemSpec):
    torch.manual_seed(0)
    data = prob.data_factory()
    guide, sps = _run_svi(data, AutoLaplaceApproximation, steps=500, lr=5e-2)
    samples = _draw_site_samples(guide, site=prob.site, n=1500)
    ref = prob.reference_factory(data)
    return samples, sps, ref


def _alg_hmc(prob: ProblemSpec):
    torch.manual_seed(0)
    data = prob.data_factory()
    kernel = HMCKernel(
        step_size=0.1,
        num_steps=10,
        mass_matrix="diagonal",
        adapt_step_size=True,
        adapt_mass_matrix=True,
    )
    samples_dict, sps = _run_mcmc(data, kernel, num_warmup=200, num_samples=400)
    samples = samples_dict[prob.site].reshape(-1)
    samples = samples[torch.isfinite(samples)]
    ref = prob.reference_factory(data)
    return samples, sps, ref


def _alg_nuts(prob: ProblemSpec):
    torch.manual_seed(0)
    data = prob.data_factory()
    kernel = NUTSKernel(
        target_accept=0.8,
        max_tree_depth=8,
        mass_matrix="diagonal",
    )
    samples_dict, sps = _run_mcmc(data, kernel, num_warmup=200, num_samples=400)
    samples = samples_dict[prob.site].reshape(-1)
    samples = samples[torch.isfinite(samples)]
    ref = prob.reference_factory(data)
    return samples, sps, ref


ALGORITHMS: dict[str, Callable[[ProblemSpec], tuple[torch.Tensor, float, dict]]] = {
    "AutoNormal": _alg_autonormal,
    "AutoMVN": _alg_automvn,
    "AutoLaplace": _alg_autolaplace,
    "HMC": _alg_hmc,
    "NUTS": _alg_nuts,
}


# ---------------------------------------------------------------------------
# Grid driver
# ---------------------------------------------------------------------------


def _evaluate(
    alg_name: str, alg_fn: Callable, prob_name: str, prob: ProblemSpec
) -> CellResult:
    """Run one cell; catch exceptions and surface them as ERROR cells."""
    try:
        samples, sps, ref = alg_fn(prob)
        if not torch.isfinite(samples).any():
            return CellResult(
                status="ERROR",
                metric_name=prob.metric_name,
                metric_value=float("nan"),
                tolerance=prob.tolerance,
                samples_per_sec=sps,
                error_message="all samples non-finite",
            )
        metric_value = prob.metric_fn(samples, ref)
        if prob.capture:
            # Capture problems pass when the metric *exceeds* the
            # tolerance (documented failure case).
            ok = metric_value > prob.tolerance
            notes = "capture: documented failure mode"
        else:
            ok = metric_value < prob.tolerance
            notes = ""
        return CellResult(
            status="PASS" if ok else "FAIL",
            metric_name=prob.metric_name,
            metric_value=metric_value,
            tolerance=prob.tolerance,
            samples_per_sec=sps,
            notes=notes,
        )
    except Exception as e:
        return CellResult(
            status="ERROR",
            metric_name=prob.metric_name,
            metric_value=float("nan"),
            tolerance=prob.tolerance,
            samples_per_sec=0.0,
            error_message=str(e).split("\n")[0][:180],
        )


def run_grid(
    *,
    algorithms: list[str] | None = None,
    problems: list[str] | None = None,
) -> dict[tuple[str, str], CellResult]:
    """Run the full algorithm × problem grid; return a dict keyed by
    ``(algorithm, problem)``."""
    algs = algorithms or list(ALGORITHMS)
    probs = problems or list(PROBLEMS)
    results = {}
    for alg_name in algs:
        for prob_name in probs:
            print(f"  [{alg_name} x {prob_name}]", flush=True)
            results[(alg_name, prob_name)] = _evaluate(
                alg_name, ALGORITHMS[alg_name], prob_name, PROBLEMS[prob_name]
            )
    return results


# ---------------------------------------------------------------------------
# Markdown emission
# ---------------------------------------------------------------------------


def _status_glyph(s: str) -> str:
    return {
        "PASS": "**PASS**",
        "FAIL": "FAIL",
        "ERROR": "ERROR",
        "SKIP": "skip",
    }.get(s, s)


def _accuracy_cell(res: CellResult | None) -> str:
    if res is None:
        return "n/a"
    if res.status in ("PASS", "FAIL"):
        if math.isnan(res.metric_value):
            return _status_glyph(res.status)
        return (
            f"{_status_glyph(res.status)} "
            f"`{res.metric_value:.3g} / {res.tolerance:.3g}`"
        )
    if res.status == "ERROR":
        return f"ERROR `{res.error_message[:60]}`"
    return res.status


def _throughput_cell(res: CellResult | None) -> str:
    if res is None or res.samples_per_sec <= 0.0:
        return "n/a"
    return f"{res.samples_per_sec:.1f}"


def emit_markdown(
    results: dict[tuple[str, str], CellResult],
    algorithms: list[str],
    *,
    out_path: Path,
) -> None:
    """Write a markdown report with one section per tier."""
    lines: list[str] = []
    lines.append("# Inference benchmark results")
    lines.append("")
    lines.append(
        "This page reports how each posterior inference algorithm shipped in "
        "[`quivers.inference`](../api/inference.md) recovers known posterior "
        "moments on a deterministic suite of synthetic problems. The grid is "
        "regenerated by `tests/benchmarks/runner.py` from the seeded data "
        "factories and analytical references in `tests/benchmarks/`."
    )
    lines.append("")

    lines.append("## What the suite tests")
    lines.append("")
    lines.append("Every benchmark is an `(algorithm, problem)` cell. A *problem* fixes:")
    lines.append("")
    lines.append(
        "1. A generative model written in QVR and loaded from `tests/benchmarks/models/*.qvr`."
    )
    lines.append(
        "2. A deterministic data generator (fixed `torch.manual_seed`) that "
        "produces the observations the model is conditioned on."
    )
    lines.append(
        "3. A *reference posterior* moment for one latent site, computed "
        "analytically (conjugate problems), by quadrature on a dense grid "
        "(constrained-support problems), or by a long cached NUTS run (Eight Schools)."
    )
    lines.append(
        r"4. A scalar metric (almost always $|\mathbb{E}_q[\cdot] - \mathbb{E}_{\text{ref}}[\cdot]|$) and a tolerance."
    )
    lines.append("")
    lines.append(
        "A *cell* runs the algorithm on the problem, draws posterior samples "
        "for the target site, and compares the recovered moment against the reference."
    )
    lines.append("")
    lines.append(
        "Throughput is reported as SVI iterations per second for the variational "
        "guides and as posterior draws per second (summed across chains) for the MCMC kernels."
    )
    lines.append("")

    lines.append("## Cell statuses")
    lines.append("")
    lines.append("- **PASS**: recovered moment is within tolerance of the reference.")
    lines.append("- **FAIL**: algorithm runs cleanly but the moment is outside tolerance.")
    lines.append(
        "- **ERROR**: algorithm raised during execution (NaN gradient, "
        "support-boundary explosion, divergent trajectory, etc.)."
    )
    lines.append(
        "- *capture* problems invert the convention: PASS means the metric "
        "*exceeds* the tolerance, confirming a documented failure mode."
    )
    lines.append("")
    lines.append(
        "Determinism: every cell calls `torch.manual_seed(0)` before constructing "
        "the problem, so the same `(algorithm, problem)` pair reproduces across runs "
        "given fixed PyTorch and NumPy versions."
    )
    lines.append("")

    lines.append("## Algorithms")
    lines.append("")
    lines.append(
        "All algorithms are evaluated on every problem. Hyperparameters are "
        "uniform across problems so that the grid measures the algorithms, "
        "not a per-problem tuning effort."
    )
    lines.append("")
    lines.append("| Algorithm | Family | Key hyperparameters |")
    lines.append("|---|---|---|")
    for alg_name in algorithms:
        if alg_name in ALGORITHM_DESCRIPTIONS:
            family, hp = ALGORITHM_DESCRIPTIONS[alg_name]
            lines.append(f"| `{alg_name}` | {family} | {hp} |")
        else:
            lines.append(f"| `{alg_name}` | (no description) | (no description) |")
    lines.append("")
    lines.append(
        "Variational guides operate in unconstrained space via the bijector "
        "attached to each latent's support, so positive-support and "
        "bounded-support sites are exercised through `exp` / `softplus` / "
        "`sigmoid` transforms rather than through constrained Gaussian families."
    )
    lines.append("")

    for tier in TIERS:
        lines.append(f"## {tier.title}")
        lines.append("")
        lines.append(tier.description)
        lines.append("")
        for prob_key in tier.problems:
            prob = PROBLEMS[prob_key]
            heading = f"### {prob.label}"
            if prob.capture:
                heading += " *(capture)*"
            lines.append(heading)
            lines.append("")
            if prob.math:
                lines.append(prob.math)
                lines.append("")
            lines.append(
                f"**Metric.** `{prob.metric_name}`, tolerance `{prob.tolerance:.4g}`."
            )
            lines.append("")

        lines.append("### Results")
        lines.append("")
        lines.append("Posterior accuracy (metric / tolerance):")
        lines.append("")
        header = "| Problem | " + " | ".join(algorithms) + " |"
        sep = "|---|" + "|".join(["---"] * len(algorithms)) + "|"
        lines.append(header)
        lines.append(sep)
        for prob_key in tier.problems:
            prob = PROBLEMS[prob_key]
            label = prob.label + (" *(capture)*" if prob.capture else "")
            cells = [label]
            for alg_name in algorithms:
                cells.append(_accuracy_cell(results.get((alg_name, prob_key))))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

        lines.append("Throughput (iters/s for SVI, draws/s for MCMC):")
        lines.append("")
        thr_sep = "|---|" + "|".join(["---:"] * len(algorithms)) + "|"
        lines.append(header)
        lines.append(thr_sep)
        for prob_key in tier.problems:
            prob = PROBLEMS[prob_key]
            label = prob.label + (" *(capture)*" if prob.capture else "")
            cells = [label]
            for alg_name in algorithms:
                cells.append(_throughput_cell(results.get((alg_name, prob_key))))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Reproducing the grid")
    lines.append("")
    lines.append("```bash")
    lines.append("QVR_USE_LOCAL_GRAMMAR=1 python -m tests.benchmarks.runner")
    lines.append("```")
    lines.append("")
    lines.append(
        "The runner accepts `--algorithms` and `--problems` flags for partial "
        "runs and writes the regenerated table back to this file by default. "
        "See `tests/benchmarks/runner.py` for the cell definitions and "
        "`tests/benchmarks/references.py` for the reference posteriors."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs"
        / "developer"
        / "inference-benchmarks.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS),
        help="Subset of algorithms to run.",
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        default=list(PROBLEMS),
        help="Subset of problems to run.",
    )
    args = parser.parse_args()

    print(
        f"Running {len(args.algorithms)} algorithms × "
        f"{len(args.problems)} problems = "
        f"{len(args.algorithms) * len(args.problems)} cells"
    )
    results = run_grid(algorithms=args.algorithms, problems=args.problems)

    pass_count = sum(1 for r in results.values() if r.status == "PASS")
    fail_count = sum(1 for r in results.values() if r.status == "FAIL")
    error_count = sum(1 for r in results.values() if r.status == "ERROR")
    print(
        f"\nDone: {pass_count} PASS, {fail_count} FAIL, "
        f"{error_count} ERROR / {len(results)} cells"
    )

    emit_markdown(results, args.algorithms, out_path=args.out)
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
