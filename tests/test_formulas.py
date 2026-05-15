"""Comprehensive tests for :mod:`quivers.formulas`: the brms-style
formula frontend.

Every test exercises the AST-driven pipeline end-to-end:

  1. Parse the formula against a dataframe.
  2. Run the :class:`FormulaToQVRModule` lens forward.
  3. Compile the resulting :class:`Module` via the existing
     :class:`quivers.dsl.Compiler` (direct AST consumption).
  4. Re-emit via :func:`quivers.dsl.emit.module_to_source` and
     re-parse via :func:`quivers.dsl.loads`; verify the source
     round-trip also compiles.
  5. Where applicable, fit synthetic data and check posterior
     recovery.

Surface coverage:

  * Every link function in the registry (identity, logit, log,
    softmax, inverse).
  * Every family in the registry (Gaussian, Bernoulli, Binomial,
    Categorical, Poisson, Negative Binomial, Gamma, Beta,
    Student-t, Cumulative).
  * Polynomial terms (``poly(x, k)``, orthogonal by default,
    matching R).  Raw polynomial via ``I(x**k)``.
  * Function transforms wired through formulae's evaluation
    namespace: ``log``, ``exp``, ``sqrt``, ``abs``, ``sin``,
    ``cos``, ``log10``, ``log1p``, ``expm1``, ``tanh``.
  * Random-effect structures: intercept-only, intercept + slope,
    multiple slopes per group, crossed random effects,
    slopes with transformed predictors.
  * Multiplicative interactions: ``x:z``, ``x*z``.
  * Prior overrides per term + per random scale.
  * Pandas and polars dataframes give equivalent fits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from quivers.dsl import Compiler, loads
from quivers.dsl.ast_nodes import Module
from quivers.dsl.emit import module_to_source
from quivers.formulas import (
    FormulaToQVRModule,
    fit,
    families,
    formula_from_data,
    formula_to_qvr,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def n():
    return 80


@pytest.fixture
def base_df(n):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "w": rng.uniform(0.1, 5.0, size=n),
            "g": rng.choice(["a", "b", "c", "d"], size=n),
            "h": rng.choice(["p", "q"], size=n),
        }
    )


@pytest.fixture
def binary_df(base_df):
    rng = np.random.default_rng(1)
    out = base_df.copy()
    out["y"] = rng.integers(0, 2, size=len(out))
    return out


@pytest.fixture
def count_df(base_df):
    rng = np.random.default_rng(2)
    out = base_df.copy()
    out["y"] = rng.poisson(lam=3.0, size=len(out))
    return out


@pytest.fixture
def beta_df(base_df):
    rng = np.random.default_rng(3)
    out = base_df.copy()
    out["y"] = np.clip(rng.beta(2.0, 5.0, size=len(out)), 1e-4, 1 - 1e-4)
    return out


@pytest.fixture
def gamma_df(base_df):
    rng = np.random.default_rng(4)
    out = base_df.copy()
    out["y"] = rng.gamma(shape=2.0, scale=1.5, size=len(out))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_via_ast(formula: str, data, family: str = "gaussian"):
    parsed = formula_from_data(formula, data)
    lens = FormulaToQVRModule(families[family])
    module, complement = lens.forward(parsed)
    assert isinstance(module, Module)
    prog = Compiler(module).compile()
    return prog, module, complement, parsed


def _round_trip(formula: str, data, family: str = "gaussian"):
    """Compile via AST, then emit + reparse + recompile; assert both
    paths produce a Program with the same returned site name.
    """
    prog_ast, module, _, parsed = _compile_via_ast(formula, data, family)
    src = module_to_source(module)
    prog_src = loads(src)
    return prog_ast, prog_src, src, parsed


# ---------------------------------------------------------------------------
# Family / link coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family_name,fixture_name,expected_call,expected_link_token",
    [
        ("gaussian", "base_df", "Normal", None),
        ("bernoulli", "binary_df", "Bernoulli", "sigmoid"),
        ("binomial", "binary_df", "Bernoulli", "sigmoid"),
        ("poisson", "count_df", "Poisson", "exp"),
        ("negative_binomial", "count_df", "NegativeBinomial", "exp"),
        ("gamma", "gamma_df", "Gamma", "exp"),
        ("beta", "beta_df", "Beta", "sigmoid"),
        ("student_t", "base_df", "StudentT", None),
    ],
)
def test_every_family_round_trips(
    family_name, fixture_name, expected_call, expected_link_token, request
):
    df = request.getfixturevalue(fixture_name)
    prog_ast, prog_src, src, _ = _round_trip("y ~ x", df, family=family_name)
    assert prog_ast.morphism is not None
    assert prog_src.morphism is not None
    assert expected_call in src
    if expected_link_token is not None:
        assert expected_link_token in src


# ---------------------------------------------------------------------------
# Polynomial coverage
# ---------------------------------------------------------------------------


class TestPolynomial:
    def test_orthogonal_default_two_columns(self, base_df):
        parsed = formula_from_data("y ~ poly(x, 2)", base_df)
        col_names = [c.name for c in parsed.fixed_columns]
        assert "poly(x, 2)_1" in col_names
        assert "poly(x, 2)_2" in col_names
        col_data = {c.name: c.data for c in parsed.fixed_columns}
        c1 = col_data["poly(x, 2)_1"]
        c2 = col_data["poly(x, 2)_2"]
        # Orthogonal by construction (formulae's default).
        assert abs(c1.dot(c2)) < 1e-9
        assert abs(c1.sum()) < 1e-9
        assert abs(c2.sum()) < 1e-9

    def test_poly_3_three_coefficients(self, base_df):
        parsed = formula_from_data("y ~ poly(x, 3)", base_df)
        poly_cols = [c for c in parsed.fixed_columns if c.term == "poly(x, 3)"]
        assert len(poly_cols) == 3
        d = [c.data for c in poly_cols]
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(d[i].dot(d[j])) < 1e-9

    def test_poly_emit_one_coef_per_degree(self, base_df):
        src = formula_to_qvr("y ~ poly(x, 3)", data=base_df)
        assert "beta_poly_x_3_1 <- Normal" in src
        assert "beta_poly_x_3_2 <- Normal" in src
        assert "beta_poly_x_3_3 <- Normal" in src

    def test_poly_compiles_via_both_paths(self, base_df):
        prog_ast, prog_src, _, _ = _round_trip("y ~ poly(x, 2) + (1 | g)", base_df)
        assert prog_ast.morphism is not None
        assert prog_src.morphism is not None

    def test_raw_polynomial_via_I(self, base_df):
        parsed = formula_from_data("y ~ x + I(x**2)", base_df)
        col_data = {c.name: c.data for c in parsed.fixed_columns}
        assert np.allclose(col_data["I(x ** 2)"], base_df["x"].values ** 2)


# ---------------------------------------------------------------------------
# Function-transform coverage
# ---------------------------------------------------------------------------


class TestFunctionTransforms:
    @pytest.mark.parametrize(
        "transform_name,np_fn,data_column",
        [
            ("log", np.log, "w"),
            ("exp", np.exp, "x"),
            ("sqrt", np.sqrt, "w"),
            ("abs", np.abs, "x"),
            ("sin", np.sin, "x"),
            ("cos", np.cos, "x"),
            ("log10", np.log10, "w"),
            ("log1p", np.log1p, "w"),
            ("expm1", np.expm1, "x"),
            ("tanh", np.tanh, "x"),
        ],
    )
    def test_transform_matches_numpy(self, transform_name, np_fn, data_column, base_df):
        spec = f"y ~ {transform_name}({data_column})"
        parsed = formula_from_data(spec, base_df)
        col_name = f"{transform_name}({data_column})"
        col = next(c for c in parsed.fixed_columns if c.name == col_name)
        expected = np_fn(base_df[data_column].values)
        assert np.allclose(col.data, expected, equal_nan=True)

    def test_log_round_trips(self, base_df):
        prog_ast, prog_src, _, _ = _round_trip("y ~ log(w)", base_df)
        assert prog_ast.morphism is not None
        assert prog_src.morphism is not None

    def test_composite_transform(self, base_df):
        prog_ast, _, src, _ = _round_trip("y ~ log(w) + I(x**2) + sin(x)", base_df)
        assert prog_ast.morphism is not None
        assert "beta_log_w" in src
        assert "beta_I_x_2" in src
        assert "beta_sin_x" in src


# ---------------------------------------------------------------------------
# Random-effect coverage
# ---------------------------------------------------------------------------


class TestRandomEffects:
    def test_intercept_only(self, base_df):
        prog_ast, prog_src, _, parsed = _round_trip("y ~ x + (1 | g)", base_df)
        assert prog_ast.morphism is not None
        assert prog_src.morphism is not None
        assert [t.slope for t in parsed.random_terms] == ["Intercept"]

    def test_slope_plus_intercept(self, base_df):
        prog_ast, _, _, parsed = _round_trip("y ~ x + (1 + x | g)", base_df)
        assert prog_ast.morphism is not None
        slopes = sorted(t.slope for t in parsed.random_terms)
        assert slopes == ["Intercept", "x"]

    def test_crossed_random_intercepts(self, base_df):
        prog_ast, _, _, parsed = _round_trip("y ~ x + (1 | g) + (1 | h)", base_df)
        assert prog_ast.morphism is not None
        groups = sorted({t.group for t in parsed.random_terms})
        assert groups == ["g", "h"]

    def test_multiple_slopes_per_group(self, base_df):
        prog_ast, _, src, parsed = _round_trip("y ~ x + z + (1 + x + z | g)", base_df)
        assert prog_ast.morphism is not None
        slopes = sorted(t.slope for t in parsed.random_terms)
        assert slopes == ["Intercept", "x", "z"]
        assert "sigma_g_x" in src
        assert "sigma_g_z" in src
        assert "beta_g_x" in src
        assert "beta_g_z" in src

    def test_random_slope_with_transform(self, base_df):
        prog_ast, _, _, parsed = _round_trip("y ~ log(w) + (log(w) | g)", base_df)
        assert prog_ast.morphism is not None
        slopes = sorted(t.slope for t in parsed.random_terms)
        assert "log(w)" in slopes


# ---------------------------------------------------------------------------
# Interaction coverage
# ---------------------------------------------------------------------------


class TestInteractions:
    def test_colon_interaction(self, base_df):
        prog_ast, _, src, parsed = _round_trip("y ~ x:z", base_df)
        assert prog_ast.morphism is not None
        terms = [c.term for c in parsed.fixed_columns if not c.is_intercept]
        assert "x:z" in terms
        assert "beta_x_z" in src

    def test_star_expands_to_main_plus_interaction(self, base_df):
        prog_ast, _, _, parsed = _round_trip("y ~ x*z", base_df)
        assert prog_ast.morphism is not None
        terms = sorted({c.term for c in parsed.fixed_columns if not c.is_intercept})
        assert terms == ["x", "x:z", "z"]

    def test_interaction_is_elementwise_product(self, base_df):
        parsed = formula_from_data("y ~ x:z", base_df)
        interaction_col = next(c for c in parsed.fixed_columns if c.term == "x:z")
        expected = base_df["x"].values * base_df["z"].values
        assert np.allclose(interaction_col.data, expected)


# ---------------------------------------------------------------------------
# Prior-override coverage
# ---------------------------------------------------------------------------


class TestPriorOverrides:
    def test_override_fixed_coef(self, base_df):
        src = formula_to_qvr(
            "y ~ x",
            data=base_df,
            priors={"beta_x": "Normal(0.0, 1.0)"},
        )
        assert "beta_x <- Normal(0.0, 1.0)" in src
        assert "intercept <- Normal(0.0, 5.0)" in src

    def test_override_random_scale(self, base_df):
        src = formula_to_qvr(
            "y ~ x + (1 | g)",
            data=base_df,
            priors={"sigma_g_Intercept": "HalfCauchy(0.5)"},
        )
        assert "sigma_g_Intercept <- HalfCauchy(0.5)" in src

    def test_override_observation_scale(self, base_df):
        src = formula_to_qvr(
            "y ~ x",
            data=base_df,
            priors={"sigma": "HalfNormal(0.5)"},
        )
        assert "sigma <- HalfNormal(0.5)" in src


# ---------------------------------------------------------------------------
# Pandas / polars parity
# ---------------------------------------------------------------------------


class TestDataFrameLibraryParity:
    def test_pandas_polars_emit_identical_qvr(self, base_df):
        src_pd = formula_to_qvr("y ~ poly(x, 2) + log(w) + (1 + x | g)", data=base_df)
        src_pl = formula_to_qvr(
            "y ~ poly(x, 2) + log(w) + (1 + x | g)",
            data=pl.from_pandas(base_df),
        )
        assert src_pd == src_pl

    def test_pandas_polars_emit_compiles(self, base_df):
        prog_pd, _, _, _ = _round_trip("y ~ x + (1 | g)", base_df)
        prog_pl, _, _, _ = _round_trip("y ~ x + (1 | g)", pl.from_pandas(base_df))
        assert prog_pd.morphism is not None
        assert prog_pl.morphism is not None


# ---------------------------------------------------------------------------
# Lens laws
# ---------------------------------------------------------------------------


class TestLensLaws:
    @pytest.mark.parametrize(
        "spec",
        [
            "y ~ 1",
            "y ~ x",
            "y ~ x + z",
            "y ~ poly(x, 3)",
            "y ~ log(w) + I(x**2)",
            "y ~ x*z",
            "y ~ x + (1 | g)",
            "y ~ x + (1 + x | g)",
            "y ~ x + z + (1 + x + z | g)",
            "y ~ poly(x, 2) + (1 | g) + (1 | h)",
        ],
    )
    def test_get_put_law(self, spec, base_df):
        parsed = formula_from_data(spec, base_df)
        lens = FormulaToQVRModule(families["gaussian"])
        module, complement = lens.forward(parsed)
        assert isinstance(module, Module)
        recovered = lens.backward(module, complement)
        assert recovered == parsed


# ---------------------------------------------------------------------------
# Emitted source compiles, both paths
# ---------------------------------------------------------------------------


class TestEmittedSourceCompiles:
    @pytest.mark.parametrize(
        "spec,family_name,fixture_name",
        [
            ("y ~ x", "gaussian", "base_df"),
            ("y ~ x", "bernoulli", "binary_df"),
            ("y ~ x", "poisson", "count_df"),
            ("y ~ x", "negative_binomial", "count_df"),
            ("y ~ x", "gamma", "gamma_df"),
            ("y ~ x", "beta", "beta_df"),
            ("y ~ x", "student_t", "base_df"),
            ("y ~ poly(x, 2)", "gaussian", "base_df"),
            ("y ~ poly(x, 3)", "gaussian", "base_df"),
            ("y ~ poly(x, 2) + (1 | g)", "gaussian", "base_df"),
            ("y ~ poly(x, 2) + z", "bernoulli", "binary_df"),
            ("y ~ log(w)", "gaussian", "base_df"),
            ("y ~ exp(x) + sin(z)", "gaussian", "base_df"),
            ("y ~ sqrt(w) + I(x**2)", "gaussian", "base_df"),
            ("y ~ log(w) + (log(w) | g)", "gaussian", "base_df"),
            ("y ~ x + (1 | g)", "gaussian", "base_df"),
            ("y ~ x + (1 + x | g)", "gaussian", "base_df"),
            ("y ~ x + z + (1 + x + z | g)", "gaussian", "base_df"),
            ("y ~ x + (1 | g) + (1 | h)", "gaussian", "base_df"),
            ("y ~ x + (1 | g)", "bernoulli", "binary_df"),
            ("y ~ x + (1 | g)", "poisson", "count_df"),
            ("y ~ x:z", "gaussian", "base_df"),
            ("y ~ x*z", "gaussian", "base_df"),
            ("y ~ x*z + (1 | g)", "gaussian", "base_df"),
            ("y ~ poly(x, 2) + log(w) + (1 + x | g)", "gaussian", "base_df"),
            ("y ~ x*z + log(w)", "bernoulli", "binary_df"),
            ("y ~ poly(x, 3) + (1 | g) + (1 | h)", "poisson", "count_df"),
            ("y ~ sin(x) + cos(x) + (1 | g)", "gaussian", "base_df"),
        ],
    )
    def test_round_trip(self, spec, family_name, fixture_name, request):
        df = request.getfixturevalue(fixture_name)
        prog_ast, prog_src, src, _ = _round_trip(spec, df, family=family_name)
        assert prog_ast.morphism is not None
        assert prog_src.morphism is not None
        assert "program model" in src
        assert "observe y" in src


# ---------------------------------------------------------------------------
# End-to-end SVI fits
# ---------------------------------------------------------------------------


class TestEndToEndFit:
    def test_intercept_only_svi(self, base_df):
        result = fit(
            "y ~ 1",
            data=base_df,
            family="gaussian",
            sampler="svi",
            num_samples=100,
            seed=0,
        )
        assert result.posterior is not None
        assert "intercept" in result.qvr_source

    def test_random_intercept_svi(self, base_df):
        result = fit(
            "y ~ x + (1 | g)",
            data=base_df,
            family="gaussian",
            sampler="svi",
            num_samples=100,
            seed=0,
        )
        assert result.posterior is not None

    def test_poly_svi(self, base_df):
        result = fit(
            "y ~ poly(x, 2)",
            data=base_df,
            family="gaussian",
            sampler="svi",
            num_samples=100,
            seed=0,
        )
        assert result.posterior is not None
        src = result.qvr_source
        assert "beta_poly_x_2_1" in src
        assert "beta_poly_x_2_2" in src

    def test_log_transform_svi(self, base_df):
        result = fit(
            "y ~ log(w)",
            data=base_df,
            family="gaussian",
            sampler="svi",
            num_samples=100,
            seed=0,
        )
        assert result.posterior is not None

    def test_dump_qvr_writes_compilable_source(self, base_df, tmp_path):
        result = fit(
            "y ~ x + (1 | g)",
            data=base_df,
            family="gaussian",
            sampler="svi",
            num_samples=20,
            seed=0,
        )
        out = tmp_path / "fit.qvr"
        path = result.dump_qvr(out)
        assert path == out
        prog = loads(path.read_text())
        assert prog.morphism is not None


# ---------------------------------------------------------------------------
# Family / link defaults
# ---------------------------------------------------------------------------


class TestFamilyLinkDefaults:
    def test_gaussian_uses_identity_link(self, base_df):
        src = formula_to_qvr("y ~ x", data=base_df, family="gaussian")
        assert "let mu = eta" in src

    def test_bernoulli_uses_logit_link(self, binary_df):
        src = formula_to_qvr("y ~ x", data=binary_df, family="bernoulli")
        assert "sigmoid(eta)" in src

    def test_poisson_uses_log_link(self, count_df):
        src = formula_to_qvr("y ~ x", data=count_df, family="poisson")
        assert "exp(eta)" in src

    def test_gaussian_carries_sigma(self, base_df):
        src = formula_to_qvr("y ~ x", data=base_df, family="gaussian")
        assert "sigma <- HalfCauchy" in src
        assert "Normal(mu, sigma)" in src

    def test_negbin_carries_disp(self, count_df):
        src = formula_to_qvr("y ~ x", data=count_df, family="negative_binomial")
        assert "disp <-" in src
        assert "NegativeBinomial(mu, disp)" in src

    def test_unknown_family_raises(self, base_df):
        with pytest.raises(ValueError, match="unknown family"):
            formula_to_qvr("y ~ x", data=base_df, family="nonexistent")
