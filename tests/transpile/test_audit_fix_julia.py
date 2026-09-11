"""Regression tests for the Gen.jl and Turing.jl transpile-audit fixes.

Each test transpiles a model exercising one family to its Julia
target and asserts on the emitted source text that the distribution
call is now correct (right name plus right argument order / values).
The assertions key off the repros recorded in
`notes/transpile-audit-findings.md`; they stay on emitted text so the
suite is fast and deterministic.

Covered bugs (target in {gen, turing}):

* `weibull-shape-scale-swapped` (turing): Distributions.jl `Weibull`
  is shape-first; the QVR `(scale, concentration)` order must be
  swapped.
* `studentt-tdist-arity-locscale` (turing): location-scale StudentT
  emitted as an affine `loc + scale * TDist(df)`, not the invalid
  `TDist(df, loc, scale)`.
* `lkj-missing-dimension-arg` (turing): `LKJCholesky` carries its
  mandatory matrix dimension as the leading argument.
* `negbinomial-probs-not-complemented` (gen, turing): the probs slot
  is complemented (`1 - p`) for Distributions.jl's parameterization.
* `gamma-rate-as-scale` (gen): the QVR rate is reciprocated into
  Gen.jl's scale slot.
* `lognormal-undefined-symbol-gen` (gen): `lognormal` is grafted as a
  runtime `Gen.Distribution` helper rather than referencing an
  undefined name.
* `weibull-undefined-symbol-gen` (gen): `weibull` is grafted as a
  runtime helper mapping the QVR `(scale, concentration)` order to
  Distributions.jl's `Weibull(shape, scale)`.
"""

from __future__ import annotations

from quivers.dsl.parser import parse
from quivers.transpile import transpile


def _emit(source: str, target: str) -> str:
    return transpile(parse(source), target=target).decode("utf-8")


_WEIBULL_QVR = (
    "object Obs : Real 4\n"
    "morphism weibull_kernel : Obs -> Obs [role=kernel] ~ Weibull(1.5, 1.0)\n"
    "program weibull_fixture : Obs -> Obs\n"
    "    sample x <- weibull_kernel\n"
    "    return x\n"
    "export weibull_fixture\n"
)

_STUDENTT_QVR = (
    "object Obs : Real 4\n"
    "morphism studentt_kernel : Obs -> Obs [role=kernel] "
    "~ StudentT(5.0, 0.0, 1.0)\n"
    "program studentt_fixture : Obs -> Obs\n"
    "    sample x <- studentt_kernel\n"
    "    return x\n"
    "export studentt_fixture\n"
)

_LKJ_QVR = (
    "object Dim : FinSet 4\n"
    "program correlation_model : Dim -> Dim\n"
    "    sample eta <- HalfNormal(2.0)\n"
    "    sample chol : Dim <- LKJCholesky(eta)\n"
    "    return chol\n"
    "export correlation_model\n"
)

_NEGBIN_QVR = (
    "object Obs : FinSet 8\n"
    "program nb : Obs -> Obs\n"
    "    sample y <- NegativeBinomial(10.0, 0.3)\n"
    "    return y\n"
    "export nb\n"
)

_GAMMA_QVR = (
    "object O : FinSet 8\n"
    "program g : O -> O\n"
    "    sample t <- Gamma(2.0, 5.0)\n"
    "    return t\n"
    "export g\n"
)

_LOGNORMAL_QVR = (
    "object Obs : FinSet 8\n"
    "program lognormal_fixture : Obs -> Obs\n"
    "    sample theta <- LogNormal(0.0, 1.0)\n"
    "    return theta\n"
    "export lognormal_fixture\n"
)


# ---------------------------------------------------------------------------
# Turing.jl
# ---------------------------------------------------------------------------


def test_turing_weibull_shape_scale_swapped() -> None:
    """QVR `Weibull(scale=1.5, concentration=1.0)` emits
    Distributions.jl `Weibull(1, 1.5)` (shape first, then scale)."""
    out = _emit(_WEIBULL_QVR, "turing")
    assert "Weibull(1, 1.5)" in out
    # The pre-fix bug emitted the QVR order verbatim.
    assert "Weibull(1.5, 1)" not in out


def test_turing_studentt_affine_locscale() -> None:
    """Location-scale StudentT emits the affine `loc + scale * TDist(df)`
    rather than the invalid `TDist(df, loc, scale)`."""
    out = _emit(_STUDENTT_QVR, "turing")
    assert "0 + 1 * TDist(5)" in out
    assert "TDist(5, 0, 1)" not in out


def test_turing_lkj_prepends_dimension() -> None:
    """`LKJCholesky` carries its matrix dimension as the leading arg."""
    out = _emit(_LKJ_QVR, "turing")
    assert "LKJCholesky(4, eta)" in out
    assert "LKJCholesky(eta)" not in out


def test_turing_negbinomial_probs_complemented() -> None:
    """The NegativeBinomial probs slot is complemented (`1 .- p`)."""
    out = _emit(_NEGBIN_QVR, "turing")
    assert "NegativeBinomial(10, 1 .- 0.3)" in out
    # The pre-fix bug passed probs verbatim.
    assert "NegativeBinomial(10, 0.3)" not in out


# ---------------------------------------------------------------------------
# Gen.jl
# ---------------------------------------------------------------------------


def test_gen_gamma_rate_reciprocated() -> None:
    """Gen.jl's `gamma` is scale-parameterized; the QVR rate is
    reciprocated via `inv(rate)`."""
    out = _emit(_GAMMA_QVR, "gen")
    assert "gamma(2, inv(5)" in out
    assert "gamma(2, 5)" not in out


def test_gen_negbinomial_probs_complemented() -> None:
    """The NegativeBinomial probs slot is complemented (`1 - p`)."""
    out = _emit(_NEGBIN_QVR, "gen")
    assert "neg_binom(10, 1 - 0.3)" in out
    assert "neg_binom(10, 0.3)" not in out


def test_gen_lognormal_grafts_runtime_helper() -> None:
    """LogNormal grafts a `lognormal` runtime `Gen.Distribution` and
    calls it, rather than referencing an undefined Gen.jl name."""
    out = _emit(_LOGNORMAL_QVR, "gen")
    assert "struct LogNormalDist" in out
    assert "const lognormal = LogNormalDist()" in out
    assert "@trace(lognormal(0, 1)" in out


def test_gen_weibull_grafts_runtime_helper() -> None:
    """Weibull grafts a `weibull` runtime helper; the helper maps the
    QVR `(scale, concentration)` order to Distributions.jl's
    `Weibull(shape, scale)`."""
    out = _emit(_WEIBULL_QVR, "gen")
    assert "struct WeibullDist" in out
    assert "const weibull = WeibullDist()" in out
    # The emitted call preserves QVR order (scale, concentration); the
    # helper reorders internally to `Weibull(concentration, scale)`.
    assert "Distributions.Weibull(concentration, scale)" in out
    assert "@trace(weibull(1.5, 1)" in out
