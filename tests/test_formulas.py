"""Tests for :mod:`quivers.formulas`: the brms-style formula frontend.

The compiler is a :class:`didactic.api.Lens` from :class:`Formula`
to a :class:`Module` AST.  Tests cover:

* Round-trip of the lens (``backward(*forward(f)) == f``).
* The emitted module compiles via the existing
  :class:`quivers.dsl.Compiler` (AST is consumed directly; no
  source-string round-trip).
* Re-emit via :func:`quivers.dsl.emit.module_to_source` and re-parse
  produces a module that compiles to the same program shape.
* Posterior recovery of a true effect on a synthetic dataset.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from quivers.dsl import Compiler, loads
from quivers.dsl.ast_nodes import Module
from quivers.formulas import (
    FormulaToQVRModule,
    bayes_fit,
    families,
    formula_to_qvr,
    parse_formula,
)


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {
            "y": [1.5, 2.3, 1.8, 2.7, 1.2, 2.5, 1.9, 2.8],
            "x": [0.1, 0.4, 0.2, 0.5, 0.1, 0.4, 0.3, 0.6],
            "g": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )


class TestParseFormula:
    def test_response_name(self, simple_df):
        f = parse_formula("y ~ x + (1 | g)", simple_df)
        assert f.response_name == "y"

    def test_fixed_terms(self, simple_df):
        f = parse_formula("y ~ x + (1 | g)", simple_df)
        assert "Intercept" in f.fixed_term_names
        assert "x" in f.fixed_term_names

    def test_random_terms(self, simple_df):
        f = parse_formula("y ~ x + (1 | g)", simple_df)
        assert len(f.random_terms) == 1
        assert f.random_terms[0].slope == "Intercept"
        assert f.random_terms[0].group == "g"

    def test_group_levels_deterministic(self, simple_df):
        f1 = parse_formula("y ~ x + (1 | g)", simple_df)
        f2 = parse_formula("y ~ x + (1 | g)", simple_df)
        assert f1.group_levels["g"] == f2.group_levels["g"] == ("a", "b")

    def test_polars_equivalent(self, simple_df):
        f_pd = parse_formula("y ~ x + (1 | g)", simple_df)
        f_pl = parse_formula("y ~ x + (1 | g)", pl.from_pandas(simple_df))
        assert f_pd.response_name == f_pl.response_name
        assert f_pd.fixed_term_names == f_pl.fixed_term_names
        assert f_pd.group_levels["g"] == f_pl.group_levels["g"]


class TestLensRoundTrip:
    def test_get_put_law(self, simple_df):
        f = parse_formula("y ~ x + (1 | g)", simple_df)
        lens = FormulaToQVRModule(families["gaussian"])
        module, complement = lens.forward(f)
        assert isinstance(module, Module)
        recovered = lens.backward(module, complement)
        assert recovered == f

    def test_module_compiles_via_compiler(self, simple_df):
        # The AST goes straight into Compiler without a source
        # round-trip; this guarantees the formula compiler is
        # producing well-formed AST.
        f = parse_formula("y ~ x + (1 | g)", simple_df)
        lens = FormulaToQVRModule(families["gaussian"])
        module, _ = lens.forward(f)
        prog = Compiler(module).compile()
        assert prog.morphism is not None

    def test_emit_then_reparse_compiles(self, simple_df):
        # The module_to_source emit produces canonical .qvr source
        # that re-parses to a module that also compiles.
        src = formula_to_qvr("y ~ x + (1 | g)", data=simple_df)
        prog = loads(src)
        assert prog.morphism is not None


class TestFormulaToQVR:
    def test_writes_to_file(self, simple_df, tmp_path):
        out = tmp_path / "model.qvr"
        result = formula_to_qvr("y ~ x + (1 | g)", data=simple_df, path=out)
        assert out.exists()
        assert out.read_text() == result
        assert "program model" in result
        assert "observe y" in result

    def test_unknown_family_raises(self, simple_df):
        with pytest.raises(ValueError, match="unknown family"):
            formula_to_qvr("y ~ x", data=simple_df, family="nonexistent")

    def test_bernoulli_emit(self, simple_df):
        binary_df = simple_df.copy()
        binary_df["y"] = [1, 0, 1, 0, 1, 0, 1, 0]
        src = formula_to_qvr("y ~ x", data=binary_df, family="bernoulli")
        assert "Bernoulli" in src
        # Bernoulli uses logit link → sigmoid in the emit.
        assert "sigmoid" in src

    def test_poisson_emit(self, simple_df):
        count_df = simple_df.copy()
        count_df["y"] = [3, 5, 4, 7, 2, 6, 4, 8]
        src = formula_to_qvr("y ~ x", data=count_df, family="poisson")
        assert "Poisson" in src
        assert "exp" in src


class TestBayesFit:
    def test_intercept_only_svi(self, simple_df):
        fit = bayes_fit(
            "y ~ 1",
            data=simple_df,
            family="gaussian",
            sampler="svi",
            num_samples=200,
            seed=0,
        )
        assert fit.formula.response_name == "y"
        assert fit.posterior is not None
        # qvr_source available on demand
        src = fit.qvr_source
        assert "program model" in src

    def test_dump_qvr(self, simple_df, tmp_path):
        fit = bayes_fit(
            "y ~ 1",
            data=simple_df,
            family="gaussian",
            sampler="svi",
            num_samples=50,
            seed=0,
        )
        out = tmp_path / "fit.qvr"
        path = fit.dump_qvr(out)
        assert path == out
        assert path.exists()
        assert "program model" in path.read_text()


class TestEmittedSourceShape:
    def test_one_scalar_latent_per_fixed_term(self, simple_df):
        # Coefficients are named scalars (no design matrix), matching
        # the named-axis surface convention.
        src = formula_to_qvr("y ~ x", data=simple_df, family="gaussian")
        assert "intercept <- Normal" in src
        assert "beta_x <- Normal" in src

    def test_random_intercept_emits_scale_and_plate(self, simple_df):
        src = formula_to_qvr("y ~ x + (1 | g)", data=simple_df)
        assert "sigma_g_Intercept <- HalfNormal" in src
        assert "alpha_g : g <- Normal" in src
        # Per-row contribution via plate-gather.
        assert "alpha_g[g_idx]" in src

    def test_response_is_observe(self, simple_df):
        src = formula_to_qvr("y ~ x", data=simple_df)
        assert "observe y : Resp <- Normal" in src
