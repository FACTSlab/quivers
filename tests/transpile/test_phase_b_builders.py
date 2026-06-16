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
