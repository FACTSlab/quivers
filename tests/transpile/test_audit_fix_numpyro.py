"""Regression tests for audit-confirmed NumPyro transpile fixes.

Each test transpiles a model exercising one family whose NumPyro
emission was a confirmed correctness defect (wrong parameterisation,
missing mandatory argument, or a distribution NumPyro does not ship)
and asserts on the emitted source text that the call is now correct.

The defects, their repro fixtures, and the intended emission are
recorded in ``notes/transpile-audit-findings.md`` (the ``numpyro``
section).
"""

from __future__ import annotations

import pathlib

from quivers.dsl.parser import parse
from quivers.transpile import transpile


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _numpyro(source: str) -> str:
    """Transpile a QVR source string to NumPyro and decode it."""
    return transpile(parse(source), target="numpyro").decode()


def _numpyro_file(relpath: str) -> str:
    """Transpile a QVR file (relative to the repo root) to NumPyro."""
    return _numpyro((_REPO_ROOT / relpath).read_text())


# ---------------------------------------------------------------------------
# NegativeBinomial -> NegativeBinomial2(mean, concentration)
# ---------------------------------------------------------------------------


def test_negative_binomial_maps_to_nb2_mean_dispersion() -> None:
    """``NegativeBinomial(total_count, probs)`` must lower to NumPyro's
    mean / dispersion ``NegativeBinomial2``: ``mean = r*p/(1-p)`` and
    ``concentration = r``, never the identity ``(total_count, probs)``
    keywords."""
    out = _numpyro_file("docs/examples/source/negbin_regression.qvr")
    assert "NegativeBinomial2(mean=(disp*probs)/(1-probs)" in out
    assert "concentration=disp)" in out
    # The un-converted identity mapping must be gone.
    assert "NegativeBinomial2(total_count=" not in out
    assert "NegativeBinomial(" not in out.replace("NegativeBinomial2(", "")


def test_negative_binomial_scalar_reparameterization() -> None:
    """A scalar ``NegativeBinomial(r, p)`` emits the same mean /
    dispersion conversion."""
    out = _numpyro(
        "object O : FinSet 8\n"
        "program g : O -> O\n"
        "    sample t <- NegativeBinomial(10.0, 0.3)\n"
        "    return t\n"
        "export g"
    )
    assert "NegativeBinomial2(mean=(10*0.3)/(1-0.3),concentration=10)" in out


# ---------------------------------------------------------------------------
# MatrixNormal -> scale_tril_row / scale_tril_column Cholesky factors
# ---------------------------------------------------------------------------


def test_matrix_normal_emits_cholesky_scale_tril_factors() -> None:
    """NumPyro's ``MatrixNormal`` takes Cholesky factors, not the raw
    row / column covariances; the covariance args are wrapped in
    ``jnp.linalg.cholesky`` and renamed to ``scale_tril_row`` /
    ``scale_tril_column``."""
    out = _numpyro_file("tests/transpile/fixtures/families/matrixnormal.qvr")
    assert (
        "MatrixNormal(loc=m_loc,"
        "scale_tril_row=jnp.linalg.cholesky(m_row_covariance),"
        "scale_tril_column=jnp.linalg.cholesky(m_col_covariance))"
    ) in out
    assert "row_covariance=" not in out
    assert "col_covariance=" not in out


# ---------------------------------------------------------------------------
# LKJCholesky / LKJ -> leading matrix-dimension positional
# ---------------------------------------------------------------------------


def test_lkj_cholesky_prepends_matrix_dimension() -> None:
    """NumPyro's ``LKJCholesky`` requires the correlation-matrix
    dimension as a leading positional arg drawn from the sample's event
    axis (``Dim : FinSet 4``)."""
    out = _numpyro_file("docs/examples/source/lkj_cholesky_correlation.qvr")
    assert "LKJCholesky(4,concentration=eta)" in out


def test_lkj_correlation_factor_prepends_matrix_dimension() -> None:
    """The full-matrix ``LKJCorrelationFactor`` (NumPyro ``LKJ``) also
    requires the dimension positional."""
    out = _numpyro(
        "object Dim : FinSet 5\n"
        "program p : Dim -> Dim\n"
        "    sample corr : Dim <- LKJCorrelationFactor(2.0)\n"
        "    return corr\n"
        "export p"
    )
    assert "LKJ(5,concentration=2)" in out


# ---------------------------------------------------------------------------
# Families NumPyro does not ship -> grafted runtime helper + bare call
# ---------------------------------------------------------------------------


def test_half_student_t_grafts_folded_studentt_helper() -> None:
    """NumPyro has no ``HalfStudentT``; it is emitted as a grafted
    ``FoldedDistribution`` subclass and called by bare name."""
    out = _numpyro_file(
        "docs/examples/source/half_student_t_hierarchical.qvr"
    )
    assert (
        "class HalfStudentT(numpyro.distributions.FoldedDistribution):"
        in out
    )
    assert "HalfStudentT(df=3,scale=1)" in out
    assert "numpyro.distributions.HalfStudentT(" not in out


def test_logit_normal_grafts_sigmoid_transform_helper() -> None:
    """NumPyro has no ``LogitNormal``; it is emitted as a grafted
    sigmoid-of-Normal ``TransformedDistribution`` subclass."""
    out = _numpyro_file("tests/transpile/fixtures/families/logitnormal.qvr")
    assert (
        "class LogitNormal(numpyro.distributions.TransformedDistribution):"
        in out
    )
    assert "SigmoidTransform()" in out
    assert "LogitNormal(loc=0,scale=1)" in out
    assert "numpyro.distributions.LogitNormal(" not in out


def test_continuous_bernoulli_grafts_full_distribution_helper() -> None:
    """NumPyro has no ``ContinuousBernoulli``; a full Distribution with
    the parameter-dependent log-normaliser is grafted, and the helper's
    ``jax.scipy.special`` dependency is imported."""
    out = _numpyro_file(
        "tests/transpile/fixtures/families/continuousbernoulli.qvr"
    )
    assert (
        "class ContinuousBernoulli(numpyro.distributions.Distribution):"
        in out
    )
    assert "import jax.scipy.special as jss" in out
    assert "ContinuousBernoulli(probs=0.5)" in out
    assert "numpyro.distributions.ContinuousBernoulli(" not in out


def test_fisher_snedecor_grafts_full_distribution_helper() -> None:
    """NumPyro has no ``FisherSnedecor``; a full Distribution with the
    log-Beta normaliser is grafted."""
    out = _numpyro_file(
        "tests/transpile/fixtures/families/fishersnedecor.qvr"
    )
    assert (
        "class FisherSnedecor(numpyro.distributions.Distribution):" in out
    )
    assert "import jax.scipy.special as jss" in out
    assert "FisherSnedecor(df1=5,df2=5)" in out
    assert "numpyro.distributions.FisherSnedecor(" not in out


def test_logistic_normal_grafts_stick_breaking_helper() -> None:
    """NumPyro has no ``LogisticNormal``; a stick-breaking-of-Normal
    ``TransformedDistribution`` subclass is grafted."""
    out = _numpyro(
        "object K : FinSet 3\n"
        "program p : K -> K\n"
        "    sample x : K <- LogisticNormal(loc, scale)\n"
        "    return x\n"
        "export p"
    )
    assert (
        "class LogisticNormal(numpyro.distributions.TransformedDistribution):"
        in out
    )
    assert "StickBreakingTransform()" in out
    assert "LogisticNormal(loc=loc,scale=scale)" in out
    assert "numpyro.distributions.LogisticNormal(" not in out


def test_one_hot_categorical_grafts_full_distribution_helper() -> None:
    """NumPyro has no ``OneHotCategorical``; a full Distribution is
    grafted and called by bare name."""
    out = _numpyro(
        "object Cat : FinSet 3\n"
        "program p : Cat -> Cat\n"
        "    sample x : Cat <- OneHotCategorical(probs)\n"
        "    return x\n"
        "export p"
    )
    assert (
        "class OneHotCategorical(numpyro.distributions.Distribution):" in out
    )
    assert "OneHotCategorical(probs=probs)" in out
    assert "numpyro.distributions.OneHotCategorical(" not in out


def test_ordered_probit_grafts_categorical_probs_helper() -> None:
    """NumPyro has no ``OrderedProbit``; a probit-link ``CategoricalProbs``
    subclass is grafted."""
    out = _numpyro(
        "object Cat : FinSet 4\n"
        "program p : Cat -> Cat\n"
        "    sample x <- OrderedProbit(eta, cutpoints)\n"
        "    return x\n"
        "export p"
    )
    assert (
        "class OrderedProbit(numpyro.distributions.CategoricalProbs):" in out
    )
    assert "OrderedProbit(eta=eta,cutpoints=cutpoints)" in out
    assert "numpyro.distributions.OrderedProbit(" not in out


# ---------------------------------------------------------------------------
# The grafted helpers are only emitted when used.
# ---------------------------------------------------------------------------


def test_helper_not_grafted_when_unused() -> None:
    """A model using only built-in NumPyro families must not carry any
    grafted helper class or the extra ``jax.scipy.special`` import."""
    out = _numpyro(
        "object O : FinSet 4\n"
        "program p : O -> O\n"
        "    sample x <- Normal(0.0, 1.0)\n"
        "    return x\n"
        "export p"
    )
    assert "class " not in out
    assert "import jax.scipy.special" not in out


def test_grafted_helper_module_is_valid_python() -> None:
    """The graft (imports + helper class + ``def model``) must compose
    into syntactically valid Python; ``compile`` rejects a malformed
    subtree."""
    for fixture in (
        "tests/transpile/fixtures/families/logitnormal.qvr",
        "tests/transpile/fixtures/families/continuousbernoulli.qvr",
        "tests/transpile/fixtures/families/fishersnedecor.qvr",
    ):
        source = _numpyro_file(fixture)
        compile(source, fixture, "exec")
