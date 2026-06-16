"""Three-way log-density agreement: QVR vs target backend vs
hand-derived analytic.

Tier 3 (`test_numeric_equivalence.py`) compares QVR-side log-density
against transpiled-target log-density. If both sides share a bug
that biases the joint by the same amount at every test point, the
constant-spread check passes spuriously. The fix is to add an
independent third witness: the hand-derived joint log-density for
a fixture whose joint sits in closed form.

For each fixture below:

1. The fixture's joint `log p(theta, y)` is in closed form, written
   here as a small Python function that builds the prior /
   likelihood factors directly with `torch.distributions` (so the
   analytic expression is the *same* primitive QVR's trace
   accumulates, eliminating the family-formula degree of freedom).
2. The deterministic grid of (theta, y) points is shared across the
   three probes: the QVR reference probe (in-process), the Stan
   probe (in-container via cmdstanpy), and the NumPyro probe
   (in-container via `numpyro.infer.util.log_density`).
3. Pairwise constant-spread agreement is asserted between every
   available pairing: `(QVR, analytic)`, `(Stan, analytic)`,
   `(NumPyro, analytic)`. A failure on any pair pinpoints whether
   the discrepancy is QVR-side or transpile-side.

The probes that need an image are skipped (not failed) when the
image is absent; the analytic / QVR pair still runs unconditionally.
"""

from __future__ import annotations

import math
import pathlib

import pytest
import torch
import torch.distributions as D

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _equivalence
from tests.transpile.fixtures import _load
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import QvrProbe


_DOUBLE = torch.float64
"""All analytic / QVR-probe evaluations use float64. The
constant-spread tolerance in
[`_equivalence._DEFAULT_ATOL`][tests.transpile._equivalence] is
5e-4; per-fixture spreads sit comfortably below that floor in
double precision."""


# ---------------------------------------------------------------------------
# Analytic joint densities. Each function returns
# `log p(theta, y)` at a single (params, data) point as a float.
# ---------------------------------------------------------------------------


def _analytic_beta_bernoulli(
    params: dict[str, float], data: dict[str, list[float]]
) -> float:
    """`log p(theta, y) = log Beta(theta; 2, 2) + sum_i log Bernoulli(y_i; theta)`.

    Beta(2, 2) prior + Bernoulli(theta) likelihood over a vector of
    binary observations.
    """
    theta = torch.tensor(params["theta"], dtype=_DOUBLE)
    y = torch.tensor(data["y"], dtype=_DOUBLE)
    prior = D.Beta(
        torch.tensor(2.0, dtype=_DOUBLE),
        torch.tensor(2.0, dtype=_DOUBLE),
    )
    likelihood = D.Bernoulli(probs=theta)
    lp = prior.log_prob(theta) + likelihood.log_prob(y).sum()
    return float(lp.item())


def _analytic_normal_normal(
    params: dict[str, float], data: dict[str, list[float]]
) -> float:
    """`log p(mu, y) = log Normal(mu; 0, 1) + sum_i log Normal(y_i; mu, 1)`.

    Normal(0, 1) prior on `mu`, Normal(mu, 1) likelihood with known
    unit sigma.
    """
    mu = torch.tensor(params["mu"], dtype=_DOUBLE)
    y = torch.tensor(data["y"], dtype=_DOUBLE)
    prior = D.Normal(
        torch.tensor(0.0, dtype=_DOUBLE),
        torch.tensor(1.0, dtype=_DOUBLE),
    )
    likelihood = D.Normal(mu, torch.tensor(1.0, dtype=_DOUBLE))
    lp = prior.log_prob(mu) + likelihood.log_prob(y).sum()
    return float(lp.item())


def _analytic_gamma_exponential(
    params: dict[str, float], data: dict[str, list[float]]
) -> float:
    """`log p(rate, y) = log Gamma(rate; 2, 1) + sum_i log Exp(y_i; rate)`.

    Gamma(concentration=2, rate=1) prior + Exponential(rate)
    likelihood; conjugate posterior is
    `Gamma(2 + N, 1 + sum(y))`.
    """
    rate = torch.tensor(params["rate"], dtype=_DOUBLE)
    y = torch.tensor(data["y"], dtype=_DOUBLE)
    prior = D.Gamma(
        torch.tensor(2.0, dtype=_DOUBLE),
        torch.tensor(1.0, dtype=_DOUBLE),
    )
    likelihood = D.Exponential(rate)
    lp = prior.log_prob(rate) + likelihood.log_prob(y).sum()
    return float(lp.item())


def _analytic_half_normal_scale(
    params: dict[str, float], data: dict[str, list[float]]
) -> float:
    """`log p(sigma, y) = log HalfNormal(sigma; 2.0) + sum_i log Normal(y_i; 0, sigma)`.

    HalfNormal(scale=2) prior on the positive-support latent;
    Normal(0, sigma) likelihood (a scale model with known zero mean).
    """
    sigma = torch.tensor(params["sigma"], dtype=_DOUBLE)
    y = torch.tensor(data["y"], dtype=_DOUBLE)
    prior = D.HalfNormal(torch.tensor(2.0, dtype=_DOUBLE))
    likelihood = D.Normal(torch.tensor(0.0, dtype=_DOUBLE), sigma)
    lp = prior.log_prob(sigma) + likelihood.log_prob(y).sum()
    return float(lp.item())


def _analytic_truncated_normal_recovery(
    params: dict[str, float], data: dict[str, list[float]]
) -> float:
    """`log p(mu, y) = log Uniform(mu; 0, 1) + sum_i log TruncN(y_i; mu, 0.2, 0, 1)`.

    Uniform(0, 1) prior; TruncatedNormal(mu, 0.2) likelihood
    truncated to (0, 1). The closed-form truncated-normal
    log-density at `y` with parent `Normal(mu, sigma)` and support
    `(a, b)` is

    `log phi((y - mu)/sigma) - log sigma - log(Phi((b - mu)/sigma)
    - Phi((a - mu)/sigma))`

    where `phi` / `Phi` are the standard-Normal pdf / cdf. We
    construct it via the explicit ratio so the analytic side is
    *not* simply a wrapper around `torch.distributions.Normal` with
    a truncation bolt-on; it is the textbook expression.
    """
    mu = torch.tensor(params["mu"], dtype=_DOUBLE)
    y = torch.tensor(data["y"], dtype=_DOUBLE)
    sigma = torch.tensor(0.2, dtype=_DOUBLE)
    lo = torch.tensor(0.0, dtype=_DOUBLE)
    hi = torch.tensor(1.0, dtype=_DOUBLE)
    # Uniform(0, 1).log_prob = 0 on its support; mu is enforced
    # inside (0, 1) by the deterministic grid (lo+1%, hi-1%).
    prior_lp = torch.tensor(0.0, dtype=_DOUBLE)
    base = D.Normal(
        torch.tensor(0.0, dtype=_DOUBLE),
        torch.tensor(1.0, dtype=_DOUBLE),
    )
    z = (y - mu) / sigma
    log_phi = base.log_prob(z)
    cdf_hi = 0.5 * (1.0 + torch.erf((hi - mu) / (sigma * math.sqrt(2.0))))
    cdf_lo = 0.5 * (1.0 + torch.erf((lo - mu) / (sigma * math.sqrt(2.0))))
    log_norm = torch.log(cdf_hi - cdf_lo)
    log_pdf = log_phi - torch.log(sigma) - log_norm
    lp = prior_lp + log_pdf.sum()
    return float(lp.item())


# ---------------------------------------------------------------------------
# Per-fixture grids + observed data.
# ---------------------------------------------------------------------------


_PARAM_BOUNDARIES: dict[str, dict[str, tuple[float, float]]] = {
    "beta_bernoulli": {"theta": (0.05, 0.95)},
    "normal_normal": {"mu": (-3.0, 3.0)},
    "gamma_exponential": {"rate": (0.05, 5.0)},
    "half_normal_scale": {"sigma": (0.05, 5.0)},
    "truncated_normal_recovery": {"mu": (0.05, 0.95)},
}


# Observation lists; lengths match each fixture's `object Obs : FinSet N`.
# Discrete-likelihood fixtures (Bernoulli) pin `y` as Python ints
# because Stan declares the data array as `int <lower = 0, upper =
# 1> y;`; cmdstanpy rejects float-valued entries for int arrays.
_PARAM_DATA: dict[str, dict[str, list[float] | list[int]]] = {
    "beta_bernoulli": {"y": [1] * 50},
    "normal_normal": {"y": [0.5] * 30},
    "gamma_exponential": {"y": [1.0] * 80},
    "half_normal_scale": {"y": [0.3] * 80},
    "truncated_normal_recovery": {"y": [0.5] * 60},
}


_ANALYTIC: dict[
    str,
    "object",
] = {
    "beta_bernoulli": _analytic_beta_bernoulli,
    "normal_normal": _analytic_normal_normal,
    "gamma_exponential": _analytic_gamma_exponential,
    "half_normal_scale": _analytic_half_normal_scale,
    "truncated_normal_recovery": _analytic_truncated_normal_recovery,
}


# Each backend's (image-tag, source-file-extension, in-container
# script name) triple, taken from `test_numeric_equivalence.py`.
_BACKENDS: dict[str, tuple[str, str, str]] = {
    "stan": ("panproto-test-stan", "stan", "stan.py"),
    "numpyro": ("panproto-test-numpyro", "py", "numpyro.py"),
}


_FIXTURES = tuple(_PARAM_BOUNDARIES)


def _points_for(fixture_name: str) -> list[Point]:
    """Deterministic grid of (params, data) points for ``fixture_name``."""
    grids = _equivalence.deterministic_grid(
        _PARAM_BOUNDARIES[fixture_name],
        points_per_axis=5,
        cap=16,
    )
    data = _PARAM_DATA[fixture_name]
    return [Point(params=g, data=data) for g in grids]


def _analytic_log_densities(
    fixture_name: str, points: list[Point]
) -> list[float]:
    """Evaluate the hand-derived joint at every grid point."""
    f = _ANALYTIC[fixture_name]
    return [f(pt.params, pt.data) for pt in points]


def _qvr_log_densities(
    source: str, fixture_name: str, points: list[Point], scratch: pathlib.Path
) -> list[float]:
    """Run the QVR reference probe under float64."""
    prior = torch.get_default_dtype()
    torch.set_default_dtype(_DOUBLE)
    try:
        probe = QvrProbe()
        result = probe.evaluate(
            source.encode("utf-8"),
            fixture_name,
            points,
            scratch=scratch / "qvr",
        )
    finally:
        torch.set_default_dtype(prior)
    return list(result.log_densities)


def _target_log_densities(
    backend: str,
    fixture_source: bytes,
    points: list[Point],
    scratch: pathlib.Path,
) -> list[float] | None:
    """Run the docker-backed probe for ``backend``; return ``None``
    when the image is absent."""
    image, ext, script_name = _BACKENDS[backend]
    if not _docker.image_available(image):
        return None
    script_path = (
        pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    )
    points_json = [{"params": pt.params, "data": pt.data} for pt in points]
    raw = _docker.run_probe(
        image=image,
        script=script_path,
        source=fixture_source,
        source_ext=ext,
        points=points_json,
        scratch=scratch / backend,
    )
    return [float(x) for x in raw["log_densities"]]


@pytest.mark.parametrize("fixture_name", _FIXTURES)
def test_three_way_closed_form_agreement(
    fixture_name: str,
    scratch: pathlib.Path,
) -> None:
    """For each fixture: QVR, Stan, NumPyro, and analytic all agree
    on the joint log-density up to a constant.

    Each pairwise agreement is asserted independently so the failure
    message identifies which leg disagreed. Backends whose Docker
    image is absent are skipped per-backend; the QVR / analytic leg
    runs unconditionally.
    """
    compositions = {f.name: f for f in _load.load_compositions()}
    if fixture_name not in compositions:
        pytest.fail(
            f"composition fixture {fixture_name!r} missing; available: "
            f"{sorted(compositions)}"
        )
    fixture = compositions[fixture_name]
    module = parse(fixture.source)

    points = _points_for(fixture_name)

    lp_analytic = _analytic_log_densities(fixture_name, points)
    lp_qvr = _qvr_log_densities(
        fixture.source, fixture_name, points, scratch
    )

    # QVR vs analytic: pure-Python pair, runs unconditionally. A
    # nonzero spread here is a QVR-side bug (trace accumulation,
    # bijector, or family construction) since the analytic uses the
    # same `torch.distributions.<Family>.log_prob`.
    _equivalence.assert_log_density_match(
        lp_qvr,
        lp_analytic,
        context=f"qvr@{fixture_name} vs analytic",
    )

    # Per-backend: transpile, run, compare to analytic. Each
    # backend can independently skip (image absent) or xfail
    # (transpile UnsupportedConstruct on that target).
    any_target_ran = False
    for backend in sorted(_BACKENDS):
        try:
            target_source = transpile(module, target=backend)
        except UnsupportedConstruct:
            continue
        target_lps = _target_log_densities(
            backend, target_source, points, scratch
        )
        if target_lps is None:
            continue
        any_target_ran = True
        _equivalence.assert_log_density_match(
            target_lps,
            lp_analytic,
            context=f"{backend}@{fixture_name} vs analytic",
        )

    if not any_target_ran:
        pytest.skip(
            f"no backend image available for {fixture_name!r}; "
            f"QVR vs analytic still passed"
        )
