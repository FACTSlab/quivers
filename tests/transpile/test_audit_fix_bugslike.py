"""Regression tests for audit-confirmed transpile correctness fixes on
the BUGS, JAGS, WebPPL, and Church backends.

Each test transpiles a model exercising one distribution family and
asserts the emitted distribution call is now correct: right name, right
argument order, and right per-target parameterisation (precision vs
rate, scale vs rate, ...). Assertions run on emitted text so they stay
fast and deterministic.

The contract each fix preserves is that the emitted program's joint
log-density equals the QVR model's up to an additive constant.
"""

from __future__ import annotations

from quivers.dsl.parser import parse
from quivers.transpile import transpile


def _scalar_model(step: str) -> str:
    """A one-sample program exercising a single family."""
    return (
        "object O:FinSet 8\n"
        "program g:O->O\n"
        f"    {step}\n"
        "    return t\n"
        "export g"
    )


def _emit(step: str, target: str) -> str:
    return transpile(parse(_scalar_model(step)), target=target).decode()


def _dense(text: str) -> str:
    """Strip whitespace so substring assertions ignore layout."""
    return "".join(text.split())


# ---------------------------------------------------------------------------
# StudentT -> dt(mu, tau, k): JAGS / BUGS Student-t is
# location-precision-df parameterised, but torch carries (df, loc,
# scale). The renderer must reorder to (loc, 1/scale^2, df).
# ---------------------------------------------------------------------------


def test_jags_studentt_dt_location_precision_df() -> None:
    out = _dense(_emit("sample t <- StudentT(3.0,1.0,2.0)", "jags"))
    assert "dt(1,1/(2*2),3)" in out
    # The old bug emitted the torch arg order verbatim.
    assert "dt(3,1,2)" not in out


def test_bugs_studentt_dt_location_precision_df() -> None:
    out = _dense(_emit("sample t <- StudentT(3.0,1.0,2.0)", "bugs"))
    assert "dt(1,1/(2*2),3)" in out
    assert "dt(3,1,2)" not in out


# ---------------------------------------------------------------------------
# Laplace -> ddexp(mu, tau): JAGS / BUGS double-exponential is
# rate-parameterised (tau = 1/scale), not precision (1/scale^2).
# ---------------------------------------------------------------------------


def test_jags_laplace_ddexp_rate() -> None:
    out = _dense(_emit("sample t <- Laplace(0.0,2.0)", "jags"))
    assert "ddexp(0,1/2)" in out
    # The old bug emitted the precision 1/(scale*scale).
    assert "ddexp(0,1/(2*2))" not in out


def test_bugs_laplace_ddexp_rate() -> None:
    out = _dense(_emit("sample t <- Laplace(0.0,2.0)", "bugs"))
    assert "ddexp(0,1/2)" in out
    assert "ddexp(0,1/(2*2))" not in out


# ---------------------------------------------------------------------------
# JAGS / BUGS families that were already correct must stay correct: the
# per-family transform override for Laplace must not perturb Normal,
# Cauchy, or HalfCauchy (all precision-parameterised).
# ---------------------------------------------------------------------------


def test_jags_normal_stays_precision() -> None:
    out = _dense(_emit("sample t <- Normal(1.0,2.0)", "jags"))
    assert "dnorm(1,1/(2*2))" in out


def test_jags_cauchy_stays_precision_df_one() -> None:
    out = _dense(_emit("sample t <- Cauchy(1.0,2.0)", "jags"))
    assert "dt(1,1/(2*2),1)" in out


# ---------------------------------------------------------------------------
# WebPPL Gamma is scale-parameterised (scale = 1/rate); the torch rate
# must be reciprocated into the scale slot.
# ---------------------------------------------------------------------------


def test_webppl_gamma_rate_reciprocated_to_scale() -> None:
    out = _dense(_emit("sample t <- Gamma(2.0,5.0)", "webppl"))
    assert "shape:2" in out
    assert "scale:1/5" in out


def test_webppl_exponential_rate_unchanged() -> None:
    # WebPPL's Exponential({a}) is rate-parameterised; no reciprocal.
    out = _dense(_emit("sample t <- Exponential(3.0)", "webppl"))
    assert "Exponential({a:3})" in out


# ---------------------------------------------------------------------------
# WebPPL lacks LogNormal / StudentT / Weibull / NegativeBinomial; the
# renderer grafts a runtime helper defining each so the emitted call
# resolves instead of referencing an undefined name.
# ---------------------------------------------------------------------------


def test_webppl_lognormal_grafts_runtime_helper() -> None:
    out = _emit("sample t <- LogNormal(0.0,1.0)", "webppl")
    assert "var LogNormal = function" in out
    assert _dense("sample(LogNormal({") in _dense(out)


def test_webppl_studentt_grafts_runtime_helper() -> None:
    out = _emit("sample t <- StudentT(3.0,1.0,2.0)", "webppl")
    assert "var StudentT = function" in out
    assert _dense("sample(StudentT({") in _dense(out)


def test_webppl_weibull_grafts_runtime_helper() -> None:
    out = _emit("sample t <- Weibull(2.0,5.0)", "webppl")
    assert "var Weibull = function" in out
    assert _dense("sample(Weibull({") in _dense(out)


def test_webppl_negbinomial_grafts_runtime_helper() -> None:
    out = _emit("sample t <- NegativeBinomial(10.0,0.3)", "webppl")
    assert "var NegativeBinomial = function" in out
    assert _dense("sample(NegativeBinomial({") in _dense(out)


# ---------------------------------------------------------------------------
# WebPPL array-valued deterministic lets: a `let` combining an
# array-valued prior under scalar operators must broadcast elementwise
# via `_qvr_bcast` rather than coerce arrays to NaN.
# ---------------------------------------------------------------------------


_CHANGEPOINT = """object Step : FinSet 64

program cp : Step -> Step
    sample tau <- Uniform(0.0, 100.0)
    sample rate_before <- Gamma(2.0, 1.0)
    sample rate_after <- Gamma(2.0, 1.0)
    let s = sigmoid(20.0 * (t - tau))
    let rate = (1.0 - s) * rate_before + s * rate_after
    observe y : Step <- Poisson(rate)
    return tau

export cp"""


def test_webppl_array_valued_let_broadcasts() -> None:
    out = transpile(parse(_CHANGEPOINT), target="webppl").decode()
    dense = _dense(out)
    # The `rate` binding combines the array-valued `s` with scalar
    # rates; every operator must route through `_qvr_bcast`.
    assert '_qvr_bcast("+"' in dense
    assert '_qvr_bcast("*"' in dense
    assert '_qvr_bcast("-"' in dense
    # The broadcast helper is grafted above the model.
    assert "var _qvr_bcast = function" in out
    # The old bug emitted a bare scalar product of the array `s`.
    assert "(1-s)*rate_before" not in dense
