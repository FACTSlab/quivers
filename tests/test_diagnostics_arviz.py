"""Tests for :mod:`quivers.diagnostics`: the ArviZ adapter.

The adapter is glue between :class:`~quivers.inference.MCMCResult`
records and ArviZ's :class:`xarray.DataTree`-based data model.  The
tests exercise the conversion round-trip plus the user-facing
:func:`compare` and :func:`posterior_predictive_check` wrappers on
synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr

from quivers.diagnostics import (
    compare,
    posterior_predictive_check,
    to_datatree,
)
from quivers.inference.mcmc.driver import MCMCResult


def _make_result(samples: dict[str, torch.Tensor]) -> MCMCResult:
    """Construct an MCMCResult with consistent dimensions for the
    supplied per-site samples."""
    first = next(iter(samples.values()))
    n_chains, n_draws = first.shape[0], first.shape[1]
    return MCMCResult(
        samples=samples,
        log_densities=torch.randn(n_chains, n_draws),
        acceptance_rates=torch.tensor([0.85] * n_chains),
        divergence_counts=torch.tensor([0] * n_chains),
        r_hat={k: torch.tensor(1.003) for k in samples},
        ess={k: torch.tensor(950.0) for k in samples},
        num_warmup=500,
        num_samples=n_draws,
    )


class TestToDataTree:
    def test_returns_datatree(self):
        samples = {"theta": torch.distributions.Beta(80.0, 20.0).sample((4, 1000))}
        dt = to_datatree(_make_result(samples))
        assert isinstance(dt, xr.DataTree)
        assert "posterior" in dt.children
        assert "sample_stats" in dt.children

    def test_posterior_shapes(self):
        samples = {
            "theta": torch.randn(4, 1000),
            "beta": torch.randn(4, 1000, 3),  # per-coordinate site
        }
        dt = to_datatree(_make_result(samples))
        assert dt["posterior"]["theta"].shape == (4, 1000)
        assert dt["posterior"]["beta"].shape == (4, 1000, 3)

    def test_sample_stats_populated(self):
        samples = {"x": torch.randn(4, 100)}
        dt = to_datatree(_make_result(samples))
        assert "lp" in dt["sample_stats"].data_vars
        assert dt["sample_stats"]["lp"].shape == (4, 100)

    def test_observed_data_group(self):
        samples = {"theta": torch.randn(4, 100)}
        observed = {"y": torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)}
        dt = to_datatree(_make_result(samples), observed_data=observed)
        assert "observed_data" in dt.children
        assert dt["observed_data"]["y"].shape == (5,)

    def test_log_likelihood_group(self):
        samples = {"theta": torch.randn(4, 100)}
        loglik = {"y": torch.randn(4, 100, 50)}
        dt = to_datatree(_make_result(samples), log_likelihood=loglik)
        assert "log_likelihood" in dt.children
        assert dt["log_likelihood"]["y"].shape == (4, 100, 50)

    def test_coords_dims_propagate(self):
        samples = {"beta": torch.randn(4, 100, 3)}
        dt = to_datatree(
            _make_result(samples),
            coords={"Verb": ["eat", "drink", "run"]},
            dims={"beta": ["Verb"]},
        )
        assert "Verb" in dt["posterior"]["beta"].coords
        assert list(dt["posterior"]["beta"]["Verb"].values) == [
            "eat",
            "drink",
            "run",
        ]


class TestCompare:
    def test_ranks_known_better_model_first(self):
        torch.manual_seed(0)
        n_chains, n_draws, n_obs = 4, 500, 100
        y = torch.randn(n_obs) * 0.5 + 1.0  # true: N(1, 0.5)

        # Model A: high-likelihood; sample a posterior over the
        # location whose draws hover near the truth so the per-draw
        # log-likelihood has nontrivial spread (PSIS-LOO requires
        # variation across draws to fit the Pareto tail).
        loc_a = torch.randn(n_chains, n_draws) * 0.1 + 1.0
        ll_a = -0.5 * ((y[None, None, :] - loc_a[..., None]) / 0.5) ** 2 - 0.5 * np.log(
            2 * np.pi * 0.5**2
        )
        # Model B: low-likelihood; draws around a wrong centre.
        loc_b = torch.randn(n_chains, n_draws) * 0.2 + 5.0
        ll_b = -0.5 * ((y[None, None, :] - loc_b[..., None]) / 3.0) ** 2 - 0.5 * np.log(
            2 * np.pi * 3.0**2
        )

        samples = {"theta": torch.randn(n_chains, n_draws)}
        dt_a = to_datatree(
            _make_result(samples), log_likelihood={"y": ll_a}, observed_data={"y": y}
        )
        dt_b = to_datatree(
            _make_result(samples), log_likelihood={"y": ll_b}, observed_data={"y": y}
        )
        comp = compare({"good": dt_a, "bad": dt_b})
        assert comp.index[0] == "good"
        assert comp.index[1] == "bad"


class TestPosteriorPredictiveCheck:
    def test_well_specified_ppp_central(self):
        # Posterior predictive identical to observed-data distribution
        # → PPP-value near 0.5 for the mean statistic.
        torch.manual_seed(0)
        y = torch.randn(100) * 2.0 + 3.0
        n_chains, n_draws = 4, 500
        pp = torch.randn(n_chains, n_draws, 100) * 2.0 + 3.0
        samples = {"theta": torch.randn(n_chains, n_draws)}
        dt = to_datatree(
            _make_result(samples),
            observed_data={"y": y},
            posterior_predictive={"y": pp},
        )
        result = posterior_predictive_check(dt, observed_name="y", statistic="mean")
        assert 0.2 < float(result["ppp"]) < 0.8

    def test_mis_specified_ppp_extreme(self):
        # Posterior predictive far from observed → PPP-value near 0 or 1.
        torch.manual_seed(0)
        y = torch.randn(100) * 0.5 + 10.0
        n_chains, n_draws = 4, 500
        pp = torch.randn(n_chains, n_draws, 100) * 0.5
        samples = {"theta": torch.randn(n_chains, n_draws)}
        dt = to_datatree(
            _make_result(samples),
            observed_data={"y": y},
            posterior_predictive={"y": pp},
        )
        result = posterior_predictive_check(dt, observed_name="y", statistic="mean")
        ppp = float(result["ppp"])
        assert ppp < 0.05 or ppp > 0.95

    def test_user_statistic(self):
        torch.manual_seed(0)
        y = torch.randn(100)
        n_chains, n_draws = 2, 100
        pp = torch.randn(n_chains, n_draws, 100)
        samples = {"theta": torch.randn(n_chains, n_draws)}
        dt = to_datatree(
            _make_result(samples),
            observed_data={"y": y},
            posterior_predictive={"y": pp},
        )

        def kurtosis(arr):
            arr = np.asarray(arr)
            m = arr.mean()
            return float(((arr - m) ** 4).mean() / (((arr - m) ** 2).mean()) ** 2)

        result = posterior_predictive_check(dt, observed_name="y", statistic=kurtosis)
        assert result["statistic"] == "kurtosis"
        assert "ppp" in result

    def test_unknown_statistic_raises(self):
        torch.manual_seed(0)
        y = torch.randn(10)
        pp = torch.randn(2, 5, 10)
        samples = {"theta": torch.randn(2, 5)}
        dt = to_datatree(
            _make_result(samples),
            observed_data={"y": y},
            posterior_predictive={"y": pp},
        )
        with pytest.raises(ValueError, match="unknown statistic"):
            posterior_predictive_check(dt, observed_name="y", statistic="nonexistent")
