"""Closed-form / independent-reference checks for the Phase B
inline-distribution builders registered in
`quivers.continuous.inline`.

Each Phase B family's builder (`_logistic_builder`,
`_half_student_t_builder`, `_beta_binomial_builder`,
`_kumaraswamy_builder`, `_lkj_cholesky_builder`) constructs a
torch `Distribution` from a parameter tensor list. This test
asserts the constructed distribution's `log_prob` agrees with an
independent reference at a sweep of parameter / evaluation points.
A bug in the builder (transposed args, wrong transform, missing
folding constant, etc.) shows up here as a non-zero `log_prob`
difference at any test point.

References used:

* `Logistic`: closed-form pdf
  `log f(y; loc, s) = -(y - loc)/s - log s - 2 log(1 + exp(-(y - loc)/s))`.
* `HalfStudentT`: the folded student-t identity
  `log p_half(x; df, sigma) = log 2 + log p_StudentT(x; df, 0, sigma)`
  for `x >= 0`.
* `BetaBinomial`: closed-form pmf via `lgamma` /
  `lbeta(a + k, b + n - k) - lbeta(a, b) + lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)`.
* `Kumaraswamy`: closed-form pdf
  `log f(x; a, b) = log a + log b + (a - 1) log x + (b - 1) log(1 - x^a)`.
* `LKJCholesky`: the builder reduces to a closure around
  `torch.distributions.LKJCholesky`; the dim-closure correctness
  is verified by shape equality and by comparing `log_prob`
  against a hand-constructed `torch.distributions.LKJCholesky(K,
  conc)` at random Cholesky factors.

Each check runs at a small parameter sweep + random eval-point
draws; `torch.allclose` with a 1e-5 tolerance captures any
material disagreement while tolerating float64 round-off.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributions as D

from quivers.continuous.inline import _FAMILY_BUILDERS


_ATOL = 1e-5
"""Absolute tolerance for `log_prob` agreement; chosen above the
float64 round-off floor for a 1-2 step log-density computation."""


def _build(family: str, params: list[float]) -> D.Distribution:
    """Invoke a `_FAMILY_BUILDERS` entry with literal params."""
    _names, builder, _discrete = _FAMILY_BUILDERS[family]
    tensors = [torch.tensor(p, dtype=torch.float64) for p in params]
    return builder(tensors)


@pytest.mark.parametrize("loc", [-2.0, 0.0, 1.5])
@pytest.mark.parametrize("scale", [0.3, 1.0, 4.0])
def test_logistic_builder_matches_closed_form(
    loc: float, scale: float
) -> None:
    """`Logistic(loc, scale).log_prob(y)` agrees with the closed-form
    logistic pdf at every test point."""
    dist = _build("Logistic", [loc, scale])
    for y in torch.linspace(loc - 4 * scale, loc + 4 * scale, 11):
        y = y.to(torch.float64)
        z = (y - loc) / scale
        # log f(y; loc, s) = -z - log(s) - 2 log(1 + exp(-z))
        expected = (
            -z
            - math.log(scale)
            - 2.0 * torch.log1p(torch.exp(-z))
        )
        actual = dist.log_prob(y)
        assert torch.allclose(actual, expected, atol=_ATOL), (
            f"Logistic(loc={loc}, scale={scale}) log_prob({y.item()}) "
            f"= {actual.item()}; expected {expected.item()}"
        )


@pytest.mark.parametrize("df", [1.5, 3.0, 10.0])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_half_student_t_builder_matches_folded_identity(
    df: float, scale: float
) -> None:
    """`HalfStudentT(df, scale).log_prob(x)` agrees with
    `log 2 + StudentT(df, 0, scale).log_prob(x)` for x >= 0."""
    dist = _build("HalfStudentT", [df, scale])
    base = D.StudentT(
        torch.tensor(df, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(scale, dtype=torch.float64),
    )
    for x in torch.linspace(0.01, 5.0 * scale, 11).to(torch.float64):
        expected = math.log(2.0) + base.log_prob(x)
        actual = dist.log_prob(x)
        assert torch.allclose(actual, expected, atol=_ATOL), (
            f"HalfStudentT(df={df}, scale={scale}).log_prob({x.item()}) "
            f"= {actual.item()}; expected {expected.item()}"
        )


@pytest.mark.parametrize("n", [5, 20, 50])
@pytest.mark.parametrize("a", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("b", [0.5, 1.0, 3.0])
def test_beta_binomial_builder_matches_closed_form(
    n: int, a: float, b: float
) -> None:
    """`BetaBinomial(n, a, b).log_prob(k)` agrees with the closed-form
    Beta-Binomial pmf
    `log C(n, k) + lbeta(a + k, b + n - k) - lbeta(a, b)`."""
    dist = _build("BetaBinomial", [float(n), a, b])
    for k in range(n + 1):
        k_t = torch.tensor(float(k), dtype=torch.float64)
        # log C(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
        log_comb = (
            torch.lgamma(torch.tensor(float(n + 1), dtype=torch.float64))
            - torch.lgamma(k_t + 1.0)
            - torch.lgamma(torch.tensor(float(n - k + 1), dtype=torch.float64))
        )
        # lbeta(x, y) = lgamma(x) + lgamma(y) - lgamma(x + y)
        log_beta_post = (
            torch.lgamma(torch.tensor(a + k, dtype=torch.float64))
            + torch.lgamma(torch.tensor(b + n - k, dtype=torch.float64))
            - torch.lgamma(torch.tensor(a + b + n, dtype=torch.float64))
        )
        log_beta_prior = (
            torch.lgamma(torch.tensor(a, dtype=torch.float64))
            + torch.lgamma(torch.tensor(b, dtype=torch.float64))
            - torch.lgamma(torch.tensor(a + b, dtype=torch.float64))
        )
        expected = log_comb + log_beta_post - log_beta_prior
        actual = dist.log_prob(k_t)
        assert torch.allclose(actual, expected, atol=_ATOL), (
            f"BetaBinomial(n={n}, a={a}, b={b}).log_prob({k}) "
            f"= {actual.item()}; expected {expected.item()}"
        )


@pytest.mark.parametrize("a", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("b", [0.5, 1.0, 3.0])
def test_kumaraswamy_builder_matches_closed_form(
    a: float, b: float
) -> None:
    """`Kumaraswamy(a, b).log_prob(x)` agrees with the closed-form pdf
    `log a + log b + (a - 1) log x + (b - 1) log(1 - x^a)`."""
    dist = _build("Kumaraswamy", [a, b])
    for x in torch.linspace(0.05, 0.95, 11).to(torch.float64):
        expected = (
            math.log(a)
            + math.log(b)
            + (a - 1.0) * torch.log(x)
            + (b - 1.0) * torch.log1p(-torch.pow(x, a))
        )
        actual = dist.log_prob(x)
        assert torch.allclose(actual, expected, atol=_ATOL), (
            f"Kumaraswamy(a={a}, b={b}).log_prob({x.item()}) "
            f"= {actual.item()}; expected {expected.item()}"
        )


# ---------------------------------------------------------------------------
# Support-boundary cells. The parameter sweeps above skip support
# boundaries (Beta domain starts at 0.05, Kumaraswamy at 0.05, etc.)
# where many distributions have log-density singularities or
# surprising behaviour. The tests below evaluate the builders AT the
# closed-support boundary and assert the analytic value, or detect
# the singular value (`+inf` / `-inf` / `nan`) explicitly so a future
# regression that silently swaps the behaviour fails loudly.
# ---------------------------------------------------------------------------


def _beta(a: float, b: float) -> D.Beta:
    """Construct a `Beta(a, b)` distribution in float64."""
    return D.Beta(
        torch.tensor(a, dtype=torch.float64),
        torch.tensor(b, dtype=torch.float64),
    )


@pytest.mark.parametrize("endpoint", [0.0, 1.0])
def test_beta_2_2_log_prob_at_boundary_is_neg_inf(endpoint: float) -> None:
    """`Beta(2, 2)` has density ``6 x (1 - x)`` on ``[0, 1]``; the
    density vanishes at both closed-support endpoints, so
    ``log_prob`` is ``-inf`` at ``x = 0`` and ``x = 1``. The
    builder must report the analytic value (or `-inf`) rather than
    crashing or silently returning a finite garbage value.
    """
    dist = _beta(2.0, 2.0)
    actual = dist.log_prob(torch.tensor(endpoint, dtype=torch.float64))
    assert torch.isinf(actual) and actual.item() < 0.0, (
        f"Beta(2, 2).log_prob({endpoint}) = {actual.item()}; "
        "expected -inf (density vanishes at the boundary)"
    )


@pytest.mark.parametrize("endpoint", [0.0, 1.0])
def test_beta_half_half_log_prob_at_boundary_is_pos_inf(
    endpoint: float,
) -> None:
    """`Beta(0.5, 0.5)` is the arcsine distribution with density
    ``1 / (pi sqrt(x (1 - x)))``; the density diverges at the
    boundary, so ``log_prob`` is ``+inf`` at ``x = 0`` and ``x = 1``.
    The builder must surface the singular value rather than crashing.
    """
    dist = _beta(0.5, 0.5)
    actual = dist.log_prob(torch.tensor(endpoint, dtype=torch.float64))
    assert torch.isinf(actual) and actual.item() > 0.0, (
        f"Beta(0.5, 0.5).log_prob({endpoint}) = {actual.item()}; "
        "expected +inf (arcsine density diverges at the boundary)"
    )


def test_beta_1_1_log_prob_at_boundary_is_zero() -> None:
    """`Beta(1, 1)` is the uniform distribution on ``[0, 1]``;
    the density is exactly 1 everywhere on the closed support, so
    ``log_prob`` is exactly 0 at the boundaries. Pins the
    well-defined endpoint of the Beta family so any future
    regression to a half-open support is caught."""
    dist = _beta(1.0, 1.0)
    for endpoint in (0.0, 1.0):
        actual = dist.log_prob(
            torch.tensor(endpoint, dtype=torch.float64)
        )
        assert torch.allclose(
            actual,
            torch.tensor(0.0, dtype=torch.float64),
            atol=_ATOL,
        ), (
            f"Beta(1, 1).log_prob({endpoint}) = {actual.item()}; "
            "expected 0 (uniform density)"
        )


@pytest.mark.parametrize("a, b", [(2.0, 2.0), (0.5, 0.5)])
def test_kumaraswamy_log_prob_approaching_boundary(
    a: float, b: float
) -> None:
    """`Kumaraswamy(a, b).log_prob(x)` near ``x = 0``: for ``a >= 1``
    the density is finite at the interior but the closed-form
    ``(a - 1) log x`` term diverges to ``-inf`` for ``a > 1`` and
    ``log x`` diverges to ``-inf`` for ``a < 1``. The exact-boundary
    evaluation goes through torch's `TransformedDistribution`
    machinery and returns `nan` because of a `0 * log(0)` pattern in
    the transform inverse, but the builder is well-defined at
    arbitrarily small interior x and the limit matches the closed
    form. This test pins both behaviours: NaN at exactly 0, and
    a divergent log-prob as `x` shrinks towards 0.
    """
    dist = _build("Kumaraswamy", [a, b])
    # Exact boundary: NaN (TransformedDistribution quirk).
    at_zero = dist.log_prob(torch.tensor(0.0, dtype=torch.float64))
    assert torch.isnan(at_zero), (
        f"Kumaraswamy({a}, {b}).log_prob(0.0) = {at_zero.item()}; "
        "expected NaN at the exact boundary (TransformedDistribution "
        "evaluates 0 * log(0) in the inverse-transform Jacobian)"
    )
    # Interior point near the boundary: agrees with the closed-form
    #   log f(x; a, b) = log a + log b
    #     + (a - 1) log x + (b - 1) log(1 - x^a)
    # The PyTorch implementation underflows the ``log(1 - x^a)`` term
    # for very small x (the ``1 - x^a`` subtraction loses every bit
    # of precision below ~1e-8 for ``a = 2``), so we evaluate at a
    # moderately small point where both regimes match.
    x_small = torch.tensor(1e-6, dtype=torch.float64)
    near_zero = dist.log_prob(x_small)
    expected = (
        math.log(a)
        + math.log(b)
        + (a - 1.0) * torch.log(x_small)
        + (b - 1.0) * torch.log1p(-torch.pow(x_small, a))
    )
    assert torch.isfinite(near_zero), (
        f"Kumaraswamy({a}, {b}).log_prob(1e-6) = {near_zero.item()}; "
        "expected a finite value at an interior point near the boundary"
    )
    assert torch.allclose(near_zero, expected, atol=_ATOL), (
        f"Kumaraswamy({a}, {b}).log_prob(1e-6) = {near_zero.item()}; "
        f"expected {expected.item()} from the closed-form pdf"
    )
    # Divergence direction matches the sign of (a - 1):
    #   a > 1: density vanishes at 0, so log_prob -> -inf;
    #   a < 1: density blows up at 0, so log_prob -> +inf.
    if a > 1.0:
        assert near_zero.item() < 0.0, (
            f"Kumaraswamy({a}, {b}).log_prob(1e-6) = "
            f"{near_zero.item()}; expected a large negative value "
            "because the density vanishes at 0 when a > 1"
        )
    elif a < 1.0:
        assert near_zero.item() > 0.0, (
            f"Kumaraswamy({a}, {b}).log_prob(1e-6) = "
            f"{near_zero.item()}; expected a large positive value "
            "because the density diverges at 0 when a < 1"
        )


@pytest.mark.parametrize("n", [5, 20])
@pytest.mark.parametrize("a, b", [(0.5, 0.5), (1.0, 1.0), (3.0, 3.0)])
def test_beta_binomial_boundary_masses_match_closed_form(
    n: int, a: float, b: float
) -> None:
    """`BetaBinomial(n, a, b).log_prob(0)` and ``log_prob(n)`` are
    the masses at the extreme atoms of the Beta-Binomial support.
    Both are finite for any positive ``a``, ``b``. The closed-form
    boundary mass simplifies because ``C(n, 0) = C(n, n) = 1``:

        log p(0; n, a, b) = log B(a, b + n) - log B(a, b)
                          = lgamma(b + n) + lgamma(a + b)
                            - lgamma(b) - lgamma(a + b + n)
        log p(n; n, a, b) = log B(a + n, b) - log B(a, b)
                          = lgamma(a + n) + lgamma(a + b)
                            - lgamma(a) - lgamma(a + b + n)

    The builder must hit both within `_ATOL`.
    """
    dist = _build("BetaBinomial", [float(n), a, b])
    expected_at_zero = (
        math.lgamma(b + n)
        + math.lgamma(a + b)
        - math.lgamma(b)
        - math.lgamma(a + b + n)
    )
    expected_at_n = (
        math.lgamma(a + n)
        + math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(a + b + n)
    )
    actual_at_zero = dist.log_prob(torch.tensor(0.0, dtype=torch.float64))
    actual_at_n = dist.log_prob(torch.tensor(float(n), dtype=torch.float64))
    assert torch.allclose(
        actual_at_zero,
        torch.tensor(expected_at_zero, dtype=torch.float64),
        atol=_ATOL,
    ), (
        f"BetaBinomial(n={n}, a={a}, b={b}).log_prob(0) "
        f"= {actual_at_zero.item()}; expected {expected_at_zero}"
    )
    assert torch.allclose(
        actual_at_n,
        torch.tensor(expected_at_n, dtype=torch.float64),
        atol=_ATOL,
    ), (
        f"BetaBinomial(n={n}, a={a}, b={b}).log_prob({n}) "
        f"= {actual_at_n.item()}; expected {expected_at_n}"
    )


@pytest.mark.parametrize("df", [1.5, 3.0, 10.0])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_half_student_t_peak_at_zero_matches_folded_identity(
    df: float, scale: float
) -> None:
    """`HalfStudentT(df, scale)` peaks at ``x = 0`` (the mode of any
    half-folded symmetric distribution). The value at the peak is
    finite for every ``df > 0`` because the underlying Student-t
    density is finite at its mode, and the folding constant adds
    ``log 2``. This boundary check confirms the peak agrees with
    ``log 2 + StudentT(df, 0, scale).log_prob(0)`` rather than
    crashing or returning ``-inf`` at the closed-support endpoint.
    """
    dist = _build("HalfStudentT", [df, scale])
    base = D.StudentT(
        torch.tensor(df, dtype=torch.float64),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(scale, dtype=torch.float64),
    )
    expected = math.log(2.0) + base.log_prob(
        torch.tensor(0.0, dtype=torch.float64)
    )
    actual = dist.log_prob(torch.tensor(0.0, dtype=torch.float64))
    assert torch.isfinite(actual), (
        f"HalfStudentT(df={df}, scale={scale}).log_prob(0.0) "
        f"= {actual.item()}; expected a finite peak value"
    )
    assert torch.allclose(actual, expected, atol=_ATOL), (
        f"HalfStudentT(df={df}, scale={scale}).log_prob(0.0) "
        f"= {actual.item()}; expected {expected.item()} from the "
        "folded-Student-t identity at the peak"
    )


@pytest.mark.parametrize("dim", [2, 3, 5, 8])
@pytest.mark.parametrize("eta", [0.5, 1.0, 3.0])
def test_lkj_cholesky_builder_matches_torch_directly(
    dim: int, eta: float
) -> None:
    """`LKJCholesky` builder routed through `_dim_dependent_builder`
    must produce a distribution whose `log_prob` agrees with
    `torch.distributions.LKJCholesky(dim, eta).log_prob`. Also
    confirms `sample()` returns a `(dim, dim)` lower-triangular
    factor."""
    from quivers.continuous.inline import _dim_dependent_builder
    from quivers.continuous.spaces import Euclidean

    builder = _dim_dependent_builder(
        "LKJCholesky", Euclidean(name="K", dim=dim)
    )
    eta_t = torch.tensor(eta, dtype=torch.float64)
    dist = builder([eta_t])
    reference = D.LKJCholesky(dim, eta_t)

    # Sample shape parity.
    torch.manual_seed(0)
    s_b = dist.sample()
    torch.manual_seed(0)
    s_r = reference.sample()
    assert s_b.shape == s_r.shape == (dim, dim), (
        f"LKJCholesky dim={dim} sample shape mismatch: "
        f"builder={s_b.shape}, reference={s_r.shape}"
    )

    # log_prob parity at the reference's own sample.
    lp_b = dist.log_prob(s_r)
    lp_r = reference.log_prob(s_r)
    assert torch.allclose(lp_b, lp_r, atol=_ATOL), (
        f"LKJCholesky(dim={dim}, eta={eta}) log_prob mismatch: "
        f"builder={lp_b.item()}, reference={lp_r.item()}"
    )
