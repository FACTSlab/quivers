"""Regression tests for audit-confirmed Stan transpilation defects.

Each test transpiles a model exercising one family and asserts the
emitted Stan distribution call is now correct (right name plus right
argument order / values / parameterisation). Assertions are on the
emitted text, keyed off the repros in
`notes/transpile-audit-findings.md`.

Covered defects:

* `weibull-shape-scale-swapped`: Stan's `weibull(alpha, sigma)` is
  shape-first, so the QVR `(scale, concentration)` args are swapped.
* `negbinomial-stan-neg-binomial-2-unconverted`: Stan's
  `neg_binomial_2(mu, phi)` is mean / dispersion parameterised, so the
  QVR `(total_count, probs)` args are converted to
  `(total_count * probs / (1 - probs), total_count)`.
* `logitnormal-nonexistent-stan`: Stan ships no `logit_normal`, so the
  renderer grafts a `logit_normal_lpdf` / `_rng` runtime helper.
"""

from __future__ import annotations

from quivers.dsl.parser import parse
from quivers.transpile import transpile


def _stan(source: str) -> str:
    """Transpile a QVR source string to Stan and decode the bytes."""
    return transpile(parse(source), target="stan").decode()


def _nospace(text: str) -> str:
    """Drop spaces so assertions ignore the formatter's comma spacing."""
    return text.replace(" ", "")


_WEIBULL_SRC = """
object Obs : Real 4
morphism weibull_kernel : Obs -> Obs [role=kernel] ~ Weibull(1.5, 2.0)
program weibull_fixture : Obs -> Obs
    sample x <- weibull_kernel
    return x
export weibull_fixture
"""


def test_weibull_shape_scale_order() -> None:
    """`Weibull(scale=1.5, concentration=2.0)` must emit shape-first as
    `weibull(2, 1.5)`, not the transposed `weibull(1.5, 2)`."""
    out = _nospace(_stan(_WEIBULL_SRC))
    assert "weibull(2,1.5)" in out
    assert "weibull(1.5,2)" not in out


_NEGBIN_SRC = """
object Item : FinSet 21
object Out : FinSet 3
object Resp : FinSet 63

program negbin_regression : Resp -> Resp
    sample beta_0 : Out <- Normal(0.0, 5.0)
    sample beta_1 : Out <- Normal(0.0, 5.0)
    sample dispersion : Out <- Gamma(2.0, 0.5)

    let b0 = beta_0[out_idx]
    let b1 = beta_1[out_idx]
    let disp = dispersion[out_idx]
    let eta = b0 + b1 * x
    let mu = exp(eta)
    let probs = disp / (disp + mu)

    observe y : Resp <- NegativeBinomial(disp, probs)
    return beta_1
"""


def test_negbinomial_neg_binomial_2_conversion() -> None:
    """`NegativeBinomial(total_count, probs)` maps to Stan's mean /
    dispersion `neg_binomial_2(mu, phi)` with
    `mu = total_count * probs / (1 - probs)` and `phi = total_count`,
    not the identity `neg_binomial_2(total_count, probs)`."""
    out = _nospace(_stan(_NEGBIN_SRC))
    assert (
        "neg_binomial_2(disp[m_Resp]*probs[m_Resp]/(1-probs[m_Resp]),"
        "disp[m_Resp])" in out
    )
    # The uncorrected identity mapping must not survive.
    assert "neg_binomial_2(disp[m_Resp],probs[m_Resp])" not in out


_LOGITNORMAL_SRC = """
object Obs : FinSet 8
program logitnormal_fixture : Obs -> Obs
    sample theta <- LogitNormal(0.0, 1.0)
    return theta
export logitnormal_fixture
"""


def test_logitnormal_runtime_helper_grafted() -> None:
    """LogitNormal has no Stan built-in; the renderer grafts a
    `logit_normal_lpdf` helper (with the change-of-variables Jacobian)
    so `theta ~ logit_normal(0, 1)` resolves through Stan's
    `<family>_lpdf` lookup."""
    out = _stan(_LOGITNORMAL_SRC)
    assert (
        "real logit_normal_lpdf(real y, real mu, real sigma)" in out
    )
    assert "real logit_normal_rng(real mu, real sigma)" in out
    # The Jacobian term -log(y) - log1m(y) guards the density.
    assert "- log(y) - log1m(y)" in out
    assert _nospace("theta ~ logit_normal(0,1)") in _nospace(out)
