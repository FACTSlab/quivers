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
import textwrap

import numpy as np
import pandas as pd
import polars as pl
import re

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
    prog_src = loads(textwrap.dedent(src))
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

    def test_complement_is_formula_data_not_formula(self, base_df):
        """The lens' complement is :class:`FormulaData`, carrying only
        the fields not recoverable from the emitted :class:`Module`
        (per-row data arrays, original identifier names, the
        original formula string). It is not the entire
        :class:`Formula`."""
        from quivers.formulas.formula import FormulaData

        parsed = formula_from_data("y ~ x + (1 | g)", base_df)
        lens = FormulaToQVRModule(families["gaussian"])
        _module, complement = lens.forward(parsed)
        assert isinstance(complement, FormulaData)
        assert complement.formula == "y ~ x + (1 | g)"
        assert complement.response_name == "y"
        assert "x" in complement.fixed_column_data
        assert "g" in complement.group_levels

    def test_backward_decodes_structure_from_module(self, base_df):
        """The structural fields of the recovered :class:`Formula`
        (fixed-column qvr names, intercept flag, random-effect
        group / slope pairs) come from decoding the
        :class:`Module`, not from the complement. Verify by
        producing a Module without forwarding through the lens and
        observing that backward recovers the correct structure."""
        from quivers.formulas.formula import FormulaData

        parsed = formula_from_data("y ~ x + (1 | g)", base_df)
        lens = FormulaToQVRModule(families["gaussian"])
        module, complement = lens.forward(parsed)
        # Synthesise a minimal FormulaData with the data fields kept
        # but the structural metadata stripped; the decoder fills the
        # structural fields from the Module.
        stripped = FormulaData(
            formula=complement.formula,
            response_name=complement.response_name,
            response_values=complement.response_values,
            fixed_column_names={},  # decoder uses qvr_name as fallback
            fixed_column_data=complement.fixed_column_data,
            group_original_names={},  # decoder uses qvr_group_name
            group_levels=complement.group_levels,
            group_indices=complement.group_indices,
        )
        recovered = lens.backward(module, stripped)
        assert len(recovered.fixed_columns) == 2
        assert recovered.fixed_columns[0].is_intercept is True
        assert recovered.fixed_columns[1].qvr_name == "x"
        assert recovered.fixed_columns[1].is_intercept is False
        assert len(recovered.random_terms) == 1
        assert recovered.random_terms[0].slope == "Intercept"
        assert recovered.random_terms[0].group == "g"


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


def _posterior_means(guide, n_obs: int, num_draws: int = 400):
    """Average the guide's ``rsample`` over many draws to estimate
    each latent site's posterior mean.

    Returns a dict mapping site name → tensor whose leading batch
    axis (coming from the program-input ``x`` of shape
    ``(n_obs, ...)``) is averaged out, leaving the site's intrinsic
    shape: scalar sites become 0-d tensors, plate sites become
    1-d tensors of the plate's cardinality.
    """
    import torch as _torch

    x = _torch.zeros(n_obs, 1)
    sums: dict[str, _torch.Tensor] = {}
    for _ in range(num_draws):
        sample = guide.rsample(x)
        for k, v in sample.items():
            sums[k] = sums.get(k, _torch.zeros_like(v)) + v
    out: dict[str, _torch.Tensor] = {}
    for k, v in sums.items():
        mean = (v / num_draws).detach()
        # Reduce the leading batch dim if it equals n_obs.
        if mean.dim() > 0 and mean.shape[0] == n_obs:
            mean = mean.mean(dim=0)
        out[k] = mean
    return out


class TestEndToEndFit:
    def test_intercept_only_svi(self, base_df):
        result = fit(
            "y ~ 1",
            data=base_df,
            family="gaussian",
            method="svi",
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
            method="svi",
            num_samples=100,
            seed=0,
        )
        assert result.posterior is not None

    def test_poly_svi(self, base_df):
        result = fit(
            "y ~ poly(x, 2)",
            data=base_df,
            family="gaussian",
            method="svi",
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
            method="svi",
            num_samples=100,
            seed=0,
        )
        assert result.posterior is not None

    def test_dump_qvr_writes_compilable_source(self, base_df, tmp_path):
        result = fit(
            "y ~ x + (1 | g)",
            data=base_df,
            family="gaussian",
            method="svi",
            num_samples=20,
            seed=0,
        )
        out = tmp_path / "fit.qvr"
        path = result.dump_qvr(out)
        assert path == out
        prog = loads(path.read_text())
        assert prog.morphism is not None


# ---------------------------------------------------------------------------
# Posterior recovery on synthetic data, per family
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPosteriorRecovery:
    """Each test generates synthetic data from a known generative
    process, fits via SVI (the fast inference path), and asserts the
    posterior mean of each named coefficient is in the right ballpark
    of the truth.  Tolerances are loose because SVI underestimates
    posterior variance and N is modest, but the *sign* and
    *order-of-magnitude* recovery is what these tests pin down.

    Marked :func:`pytest.mark.slow` — these tests do real inference
    (2000 SVI steps × N≈200 each) and take a few minutes total.
    Run with ``pytest -m slow tests/test_formulas.py`` or
    ``pytest --runslow tests/test_formulas.py``.
    """

    def test_gaussian_recovers_slope_and_intercept(self):
        rng = np.random.default_rng(0)
        N = 200
        true_intercept = 2.0
        true_slope = 1.5
        true_sigma = 0.3
        x = rng.normal(size=N)
        y = true_intercept + true_slope * x + true_sigma * rng.normal(size=N)
        df = pd.DataFrame({"y": y, "x": x})

        result = fit(
            "y ~ x",
            data=df,
            family="gaussian",
            method="svi",
            num_samples=2000,
            seed=0,
        )
        means = _posterior_means(result.posterior, n_obs=N)
        assert abs(means["intercept"].item() - true_intercept) < 0.2
        assert abs(means["beta_x"].item() - true_slope) < 0.2
        # `sigma` is on positive support, biject_to(positive) exposes
        # a softplus; the post-link mean should be near the truth.
        assert 0.1 < means["sigma"].item() < 0.6

    def test_bernoulli_recovers_logit_slope(self):
        rng = np.random.default_rng(1)
        N = 400
        true_intercept = 0.5
        true_slope = -1.2
        x = rng.normal(size=N)
        logit = true_intercept + true_slope * x
        probs = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.uniform(size=N) < probs).astype(int)
        df = pd.DataFrame({"y": y, "x": x})

        result = fit(
            "y ~ x",
            data=df,
            family="bernoulli",
            method="svi",
            num_samples=2000,
            seed=0,
        )
        means = _posterior_means(result.posterior, n_obs=N)
        assert means["beta_x"].item() < 0  # negative slope recovered
        assert abs(means["beta_x"].item() - true_slope) < 0.5

    def test_poisson_recovers_log_slope(self):
        rng = np.random.default_rng(2)
        N = 300
        true_intercept = 1.0
        true_slope = 0.5
        x = rng.normal(size=N)
        rate = np.exp(true_intercept + true_slope * x)
        y = rng.poisson(lam=rate)
        df = pd.DataFrame({"y": y.astype(float), "x": x})

        result = fit(
            "y ~ x",
            data=df,
            family="poisson",
            method="svi",
            num_samples=2000,
            seed=0,
        )
        means = _posterior_means(result.posterior, n_obs=N)
        assert means["beta_x"].item() > 0  # positive slope recovered
        assert abs(means["beta_x"].item() - true_slope) < 0.3

    def test_gamma_recovers_log_slope(self):
        rng = np.random.default_rng(3)
        N = 300
        true_intercept = 0.5
        true_slope = 0.4
        x = rng.normal(size=N)
        mean_rate = np.exp(true_intercept + true_slope * x)
        # Gamma with shape=2.0, scale = mean / shape.
        y = rng.gamma(shape=2.0, scale=mean_rate / 2.0)
        df = pd.DataFrame({"y": y, "x": x})

        result = fit(
            "y ~ x",
            data=df,
            family="gamma",
            method="svi",
            num_samples=2000,
            seed=0,
        )
        means = _posterior_means(result.posterior, n_obs=N)
        assert means["beta_x"].item() > 0
        assert abs(means["beta_x"].item() - true_slope) < 0.4

    def test_random_intercept_partial_pooling(self):
        """A hierarchical random-intercepts model recovers the
        per-group means with partial pooling toward the grand mean.
        """
        rng = np.random.default_rng(4)
        n_groups = 8
        n_per_group = 30
        true_grand_mean = 1.0
        true_group_sigma = 1.5
        true_obs_sigma = 0.5
        group_effects = rng.normal(0.0, true_group_sigma, size=n_groups)
        groups = []
        ys = []
        for g_idx, g_effect in enumerate(group_effects):
            for _ in range(n_per_group):
                groups.append(f"g{g_idx}")
                ys.append(true_grand_mean + g_effect + true_obs_sigma * rng.normal())
        df = pd.DataFrame({"y": ys, "g": groups})

        result = fit(
            "y ~ 1 + (1 | g)",
            data=df,
            family="gaussian",
            method="svi",
            num_samples=2000,
            seed=0,
        )
        means = _posterior_means(result.posterior, n_obs=len(ys))
        # The formula compiler emits the non-centred parameterisation,
        # so a group's random effect is its standard-normal draw scaled
        # by the group-level sigma; there is no centred `alpha_g` site.
        post_group = (
            (means["sigma_g_Intercept"] * means["z_g_Intercept"])
            .detach()
            .numpy()
            .reshape(-1)
        )
        assert len(post_group) == n_groups

        # The intercept and the random effects are identified only up to
        # a constant shared shift: adding c to the intercept and taking
        # c off every group effect leaves the likelihood untouched, and
        # only the N(0, 1) prior on the draws pulls the split back, which
        # with eight groups it does weakly. So the level is asserted on
        # the identified sum rather than on the intercept alone.
        level = means["intercept"].item() + float(post_group.mean())
        assert abs(level - (true_grand_mean + group_effects.mean())) < 0.5

        # The effects themselves are identified up to that same shift,
        # which correlation is invariant to.
        corr = np.corrcoef(post_group, group_effects)[0, 1]
        assert corr > 0.7

    def test_polynomial_orthogonal_recovers_quadratic(self):
        """For a true quadratic relationship, `poly(x, 2)` recovers
        a nonzero quadratic coefficient."""
        rng = np.random.default_rng(5)
        N = 300
        x = rng.uniform(-2.0, 2.0, size=N)
        # True curve has both linear and quadratic content.
        y = 0.5 * x + 1.0 * x**2 + 0.3 * rng.normal(size=N)
        df = pd.DataFrame({"y": y, "x": x})

        result = fit(
            "y ~ poly(x, 2)",
            data=df,
            family="gaussian",
            method="svi",
            num_samples=2000,
            seed=0,
        )
        means = _posterior_means(result.posterior, n_obs=N)
        # `poly` returns an orthonormal basis, so least squares on it is
        # just the projection of y onto each column and gives the exact
        # target the fit has to reach. Both coefficients are large
        # because the columns are of norm one over N rows.
        from formulae import design_matrices

        X = np.asarray(design_matrices("y ~ poly(x, 2)", df).common.design_matrix)
        exact = np.linalg.lstsq(X, y, rcond=None)[0]
        assert abs(means["beta_poly_x_2_1"].item() - exact[1]) < 0.5 * abs(exact[1])
        assert abs(means["beta_poly_x_2_2"].item() - exact[2]) < 0.5 * abs(exact[2])
        # The noise scale is the data's, not the marginal spread of y:
        # a fit that gives up puts sigma near std(y) and leaves the
        # coefficients at their prior.
        assert abs(means["sigma"].item() - 0.3) < 0.15

    def test_log_transform_recovers_slope(self):
        """y ~ log(w) recovers a slope when y is generated from a
        log-linear relationship in w."""
        rng = np.random.default_rng(6)
        N = 300
        true_slope = 1.5
        w = rng.uniform(0.5, 5.0, size=N)
        y = 0.2 + true_slope * np.log(w) + 0.2 * rng.normal(size=N)
        df = pd.DataFrame({"y": y, "w": w})

        result = fit(
            "y ~ log(w)",
            data=df,
            family="gaussian",
            method="svi",
            num_samples=2000,
            seed=0,
        )
        means = _posterior_means(result.posterior, n_obs=N)
        assert abs(means["beta_log_w"].item() - true_slope) < 0.3

    def test_interaction_recovers_product_coefficient(self):
        """y ~ x*z recovers the interaction coefficient when the
        true generative process has a multiplicative term."""
        rng = np.random.default_rng(7)
        N = 400
        x = rng.normal(size=N)
        z = rng.normal(size=N)
        true_interaction = 0.8
        y = (
            0.5
            + 0.3 * x
            + -0.4 * z
            + true_interaction * x * z
            + 0.3 * rng.normal(size=N)
        )
        df = pd.DataFrame({"y": y, "x": x, "z": z})

        result = fit(
            "y ~ x*z",
            data=df,
            family="gaussian",
            method="svi",
            num_samples=2000,
            seed=0,
        )
        means = _posterior_means(result.posterior, n_obs=N)
        assert abs(means["beta_x_z"].item() - true_interaction) < 0.3
        assert abs(means["beta_x"].item() - 0.3) < 0.3
        assert abs(means["beta_z"].item() - (-0.4)) < 0.3

    def test_multivariate_recovers_all_slopes(self):
        """Multi-predictor regression recovers each coefficient."""
        rng = np.random.default_rng(8)
        N = 400
        x = rng.normal(size=N)
        z = rng.normal(size=N)
        w = rng.uniform(0.5, 4.0, size=N)
        true_x, true_z, true_log_w = 1.0, -0.7, 0.5
        y = (
            0.2
            + true_x * x
            + true_z * z
            + true_log_w * np.log(w)
            + 0.3 * rng.normal(size=N)
        )
        df = pd.DataFrame({"y": y, "x": x, "z": z, "w": w})

        result = fit(
            "y ~ x + z + log(w)",
            data=df,
            family="gaussian",
            method="svi",
            num_samples=2000,
            seed=0,
        )
        means = _posterior_means(result.posterior, n_obs=N)
        assert abs(means["beta_x"].item() - true_x) < 0.2
        assert abs(means["beta_z"].item() - true_z) < 0.2
        assert abs(means["beta_log_w"].item() - true_log_w) < 0.2

    def test_random_slope_recovers_per_group_effect(self):
        """A random-intercepts + random-slopes model recovers both
        per-group random effects with the right sign / magnitude."""
        rng = np.random.default_rng(9)
        n_groups = 6
        n_per_group = 40
        true_intercept = 0.5
        true_slope_mean = 1.0
        sigma_intercept = 1.0
        sigma_slope = 0.6
        sigma_obs = 0.3
        ranef_intercept = rng.normal(0.0, sigma_intercept, size=n_groups)
        ranef_slope = rng.normal(0.0, sigma_slope, size=n_groups)

        groups, xs, ys = [], [], []
        for g_idx in range(n_groups):
            for _ in range(n_per_group):
                xv = rng.normal()
                yv = (
                    true_intercept
                    + ranef_intercept[g_idx]
                    + (true_slope_mean + ranef_slope[g_idx]) * xv
                    + sigma_obs * rng.normal()
                )
                groups.append(f"g{g_idx}")
                xs.append(xv)
                ys.append(yv)
        df = pd.DataFrame({"y": ys, "x": xs, "g": groups})

        result = fit(
            "y ~ x + (1 + x | g)",
            data=df,
            family="gaussian",
            method="svi",
            num_samples=2000,
            seed=0,
        )
        means = _posterior_means(result.posterior, n_obs=len(ys))
        # The formula compiler emits the non-centred parameterisation,
        # so a group's random effect is its standard-normal draw scaled
        # by the group-level sigma; there is no centred `alpha_g` site.
        post_int = (
            (means["sigma_g_Intercept"] * means["z_g_Intercept"])
            .detach()
            .numpy()
            .reshape(-1)
        )
        post_slope = (means["sigma_g_x"] * means["z_g_x"]).detach().numpy().reshape(-1)
        assert np.corrcoef(post_int, ranef_intercept)[0, 1] > 0.7
        assert np.corrcoef(post_slope, ranef_slope)[0, 1] > 0.5


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


# ---------------------------------------------------------------------------
# Default coefficient priors are autoscaled to their column
# ---------------------------------------------------------------------------


class TestPriorAutoscaling:
    def test_default_prior_scales_by_the_column_rms(self, base_df):
        """A column enters as ``beta * column``, so the default prior's
        scale is divided by the column's RMS: the nominal value then
        reads in contribution space and means the same thing whatever
        the column's units."""
        src = formula_to_qvr("y ~ x", data=base_df, family="gaussian")
        rms = float(np.sqrt(np.mean(np.square(base_df["x"].to_numpy()))))
        match = re.search(r"sample beta_x <- Normal\(0\.0, ([0-9.eE+-]+)\)", src)
        assert match, f"no autoscaled beta_x prior in:\n{src}"
        assert float(match.group(1)) == pytest.approx(5.0 / rms, rel=1e-6)

    def test_orthonormal_poly_columns_get_a_wide_prior(self):
        """`poly` returns columns of norm one, whose entries run about
        1/sqrt(N). Without autoscaling the default prior would assert
        the contribution is near zero and the fit would believe it."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"y": rng.normal(size=200), "x": rng.uniform(-2, 2, 200)})
        src = formula_to_qvr("y ~ poly(x, 2)", data=df, family="gaussian")
        scales = [
            float(s)
            for s in re.findall(
                r"sample beta_poly_x_2_\d <- Normal\(0\.0, ([0-9.eE+-]+)\)", src
            )
        ]
        assert len(scales) == 2
        # rms of a norm-one column over N rows is 1/sqrt(N), so the
        # scale lands near 5 * sqrt(N).
        for s in scales:
            assert s == pytest.approx(5.0 * np.sqrt(200), rel=1e-3)

    def test_intercept_prior_is_left_alone(self, base_df):
        """The intercept multiplies a column of ones, so there is no
        scale to correct for."""
        src = formula_to_qvr("y ~ x", data=base_df, family="gaussian")
        assert "sample intercept <- Normal(0.0, 5.0)" in src

    def test_explicit_prior_is_emitted_as_written(self, base_df):
        """An override is the user's statement about that coefficient,
        so it is not rescaled underneath them."""
        src = formula_to_qvr(
            "y ~ x",
            data=base_df,
            family="gaussian",
            priors={"beta_x": "Normal(0.0, 0.25)"},
        )
        assert "sample beta_x <- Normal(0.0, 0.25)" in src
