"""Regression tests for the audit-confirmed PyMC transpilation fixes.

Each test transpiles a model exercising one family to the PyMC
backend and asserts on the emitted text: the corrected distribution
name, keyword, argument order, and value. The emitted program's joint
log-density must match the QVR model's up to an additive constant, so
these assertions pin the exact call shape that carries that density.

The fixes live in
[`renderers/pymc.py`][quivers.transpile.renderers.pymc] and the
grafted [`runtime_pymc.py`][quivers.transpile.runtime_pymc] helper.
"""

from __future__ import annotations

from quivers.dsl.parser import parse
from quivers.transpile import transpile
from quivers.transpile.family_meta import FAMILY_META
from quivers.transpile.renderers.pymc import _merged_aliases


def _emit(source: str) -> str:
    return transpile(parse(source), target="pymc").decode("utf-8")


def test_poisson_rate_renamed_to_mu() -> None:
    """PyMC `Poisson` takes `mu`, not the torch `rate`."""
    src = (
        "object O : Real 4\n"
        "program g : O -> O\n"
        "    sample t <- Poisson(3.0)\n"
        "    return t\n"
        "export g"
    )
    out = _emit(src)
    assert "pymc.Poisson(" in out
    assert "mu=3" in out
    assert "rate=" not in out


def test_chi2_df_renamed_to_nu() -> None:
    """PyMC `ChiSquared` takes `nu`, not the torch `df`."""
    src = (
        "object Obs : Real 4\n"
        "morphism k : Obs -> Obs [role=kernel] ~ Chi2(3.0)\n"
        "program g : Obs -> Obs\n"
        "    sample x <- k\n"
        "    return x\n"
        "export g"
    )
    out = _emit(src)
    assert "pymc.ChiSquared(" in out
    assert "nu=3" in out
    assert "df=" not in out


def test_kumaraswamy_concentrations_renamed_to_a_b() -> None:
    """PyMC `Kumaraswamy` takes `a` / `b`, not `concentration1` /
    `concentration0`."""
    src = (
        "object Obs : Real 4\n"
        "morphism k : Obs -> Obs [role=kernel] ~ Kumaraswamy(2.0, 3.0)\n"
        "program g : Obs -> Obs\n"
        "    sample x <- k\n"
        "    return x\n"
        "export g"
    )
    out = _emit(src)
    assert "pymc.Kumaraswamy(" in out
    assert "a=2" in out
    assert "b=3" in out
    assert "concentration1" not in out
    assert "concentration0" not in out


def test_negative_binomial_probs_complemented() -> None:
    """PyMC's `NegativeBinomial(n, p)` uses `p == 1 - probs` relative
    to torch's success probability."""
    out = _emit(open("docs/examples/source/negbin_regression.qvr").read())
    assert "pymc.NegativeBinomial(" in out
    assert "p=(1-probs)" in out
    # The count parameter passes through unchanged.
    assert "n=disp" in out


def test_geometric_latent_shifted_off_by_one() -> None:
    """torch `Geometric` counts failures on `{0, 1, ...}`; PyMC counts
    trials on `{1, 2, ...}`. A latent sample keeps the density on the
    PyMC RV and exposes the torch-convention value via a shifted
    `Deterministic`."""
    src = (
        "object O : FinSet 4\n"
        "program g : O -> O\n"
        "    sample t : O <- Geometric(0.3)\n"
        "    return t\n"
        "export g"
    )
    out = _emit(src)
    assert 'pymc.Geometric("t__geom"' in out
    assert 'pymc.Deterministic("t"' in out
    assert "-1" in out
    # The success probability is shared, not complemented.
    assert "p=0.3" in out


def test_geometric_observed_shifted_off_by_one() -> None:
    """An observed `Geometric` feeds `observed + 1` so torch-convention
    data land in PyMC's `{1, 2, ...}` support."""
    src = (
        "object O : FinSet 4\n"
        "program g : O -> O\n"
        "    observe t : O <- Geometric(0.3)\n"
        "    return t\n"
        "export g"
    )
    out = _emit(src)
    assert 'pymc.Geometric("t"' in out
    assert "observed=(t+1)" in out


def test_continuous_bernoulli_grafts_custom_dist_helper() -> None:
    """PyMC ships no `ContinuousBernoulli`; the renderer grafts a
    `CustomDist`-backed helper and calls it by bare name."""
    out = _emit(open("tests/transpile/fixtures/families/continuousbernoulli.qvr").read())
    assert "def ContinuousBernoulli(" in out
    assert "pymc.CustomDist(" in out
    assert 'ContinuousBernoulli("x"' in out
    # The nonexistent `pymc.ContinuousBernoulli` name is never emitted.
    assert "pymc.ContinuousBernoulli(" not in out


def test_wishart_alias_maps_covariance_to_capital_v() -> None:
    """PyMC `Wishart(nu, V)`: the scale matrix is `V`, not the
    torch `covariance_matrix` nor PyMC's `scale_matrix`."""
    aliases = _merged_aliases(FAMILY_META["Wishart"])
    assert aliases["covariance_matrix"] == "V"
    assert aliases["df"] == "nu"
    assert "scale_matrix" not in aliases.values()
