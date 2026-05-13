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
    problems: list[tuple[str, Callable[[], "ProblemSpec"]]] = field(default_factory=list)


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


PROBLEMS: dict[str, ProblemSpec] = {
    # Tier 1 — conjugate
    "beta_bernoulli": ProblemSpec(
        label="Beta-Bernoulli",
        description="theta ~ Beta(2, 2), y_i ~ Bernoulli(theta), N=50",
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
        data_factory=bayes_linear_regression,
        reference_factory=bayes_linear_regression_reference,
        observed_names={"y", "x_design"},
        site="a",
        metric_name="|E[a]_q - E[a]_true|",
        tolerance=0.1,
        metric_fn=_mean_error_against_ref_key("a_mean"),
    ),
    # Tier 2 — hierarchical
    "eight_schools_centred": ProblemSpec(
        label="Eight Schools (centred)",
        description="mu, tau, theta_j; canonical hierarchical funnel",
        data_factory=eight_schools_centred,
        reference_factory=lambda _: eight_schools_reference(),
        observed_names={"y"},
        site="mu",
        metric_name="|E[mu]_q - mu_ref|",
        tolerance=3.0 * float(eight_schools_reference()["mu_std"]),
        metric_fn=_mean_error_against_ref_key("mu_mean"),
    ),
    "eight_schools_noncentred": ProblemSpec(
        label="Eight Schools (non-centred)",
        description="non-centred reparam removes funnel",
        data_factory=eight_schools_noncentred,
        reference_factory=lambda _: eight_schools_reference(),
        observed_names={"y"},
        site="mu",
        metric_name="|E[mu]_q - mu_ref|",
        tolerance=2.0 * float(eight_schools_reference()["mu_std"]),
        metric_fn=_mean_error_against_ref_key("mu_mean"),
    ),
    # Tier 3 — hard geometry
    "correlated_regression": ProblemSpec(
        label="Correlated regression",
        description="near-collinear design produces rho ~ 0.95+",
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
        data_factory=ill_conditioned_mvn,
        reference_factory=ill_conditioned_mvn_reference,
        observed_names={f"y_{i + 1}" for i in range(5)},
        site="x_3",
        metric_name="|E[x_3]_q - E[x_3]_true|",
        tolerance=0.3,
        metric_fn=lambda s, ref: float(abs(s.mean() - ref["mean"][2])),
    ),
    # Tier 6 — constrained-support
    "half_normal_scale": ProblemSpec(
        label="HalfNormal scale",
        description="sigma > 0; quadrature reference",
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
        data_factory=truncated_normal_recovery,
        reference_factory=truncated_normal_recovery_reference,
        observed_names={"y"},
        site="mu",
        metric_name="|E[mu]_q - E[mu]_true|",
        tolerance=0.05,
        metric_fn=_mean_error_against_ref_key("mean"),
    ),
}


TIERS = [
    ("Tier 1 — Conjugate", [
        "beta_bernoulli",
        "normal_normal",
        "normal_inverse_gamma",
        "gamma_exponential",
        "bayes_linear_regression",
    ]),
    ("Tier 2 — Hierarchical", [
        "eight_schools_centred",
        "eight_schools_noncentred",
    ]),
    ("Tier 3 — Hard geometry", [
        "correlated_regression",
        "neal_funnel_mean_field_capture",
        "ill_conditioned_mvn",
    ]),
    ("Tier 6 — Constrained support", [
        "half_normal_scale",
        "truncated_normal_recovery",
    ]),
]


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
            data.model, observed_names=data.observations.keys() | {"x_design"}
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
        [guide.rsample(torch.zeros(1, 1))[site].detach().reshape(-1)[0] for _ in range(n)]
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
    guide, sps = _run_svi(data, AutoNormalGuide, steps=800, lr=5e-2)
    samples = _draw_site_samples(guide, site=prob.site, n=1500)
    ref = prob.reference_factory(data)
    return samples, sps, ref


def _alg_automvn(prob: ProblemSpec):
    torch.manual_seed(0)
    data = prob.data_factory()
    guide, sps = _run_svi(
        data, AutoMultivariateNormalGuide, steps=800, lr=5e-2, init_scale=0.3
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
        step_size=0.1, num_steps=10, mass_matrix="diagonal",
        adapt_step_size=True, adapt_mass_matrix=True,
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
        target_accept=0.8, max_tree_depth=8, mass_matrix="diagonal",
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
        "Algorithm × problem grid for the quivers inference layer."
    )
    lines.append(
        "Generated by ``tests/benchmarks/runner.py`` against the "
        "seeded synthetic suite in ``tests/benchmarks/``."
    )
    lines.append("")
    lines.append("## Reading the table")
    lines.append("")
    lines.append("- **PASS** — recovered posterior moment matches the reference within the listed tolerance.")
    lines.append("- **FAIL** — the algorithm runs but the recovered moment is outside tolerance; see the `metric` column for the diagnostic.")
    lines.append("- **ERROR** — the algorithm raised during execution (e.g. NaN gradient, support-boundary explosion).")
    lines.append("- Problems marked *capture* document expected failure modes: PASS means the metric exceeds the tolerance, confirming the known underfit.")
    lines.append("")
    lines.append("Throughput is iterations per second for SVI guides, posterior draws per second for MCMC kernels.")
    lines.append("")
    for tier_name, problem_keys in TIERS:
        lines.append(f"## {tier_name}")
        lines.append("")
        # Header.
        header = "| Problem | " + " | ".join(algorithms) + " |"
        sep = "|---|" + "|".join(["---"] * len(algorithms)) + "|"
        lines.append(header)
        lines.append(sep)
        for prob_key in problem_keys:
            prob = PROBLEMS[prob_key]
            row_label = prob.label
            if prob.capture:
                row_label += " *(capture)*"
            cells = [row_label]
            for alg_name in algorithms:
                res = results.get((alg_name, prob_key))
                if res is None:
                    cells.append("—")
                    continue
                if res.status in ("PASS", "FAIL"):
                    if math.isnan(res.metric_value):
                        body = _status_glyph(res.status)
                    else:
                        body = (
                            f"{_status_glyph(res.status)}<br>"
                            f"`{res.metric_value:.3g} / {res.tolerance:.3g}`"
                            f"<br>{res.samples_per_sec:.1f}/s"
                        )
                elif res.status == "ERROR":
                    body = f"ERROR<br>`{res.error_message[:60]}`"
                else:
                    body = res.status
                cells.append(body)
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        # Per-problem descriptions.
        lines.append("### Problem details")
        lines.append("")
        for prob_key in problem_keys:
            prob = PROBLEMS[prob_key]
            lines.append(f"- **{prob.label}** — {prob.description}. Metric: `{prob.metric_name}`, tolerance: `{prob.tolerance:.4g}`.")
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
