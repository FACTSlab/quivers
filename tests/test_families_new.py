"""Tests for the Gaussian-process and horseshoe conditional families."""

from __future__ import annotations

import math
import pytest
import torch

from quivers.core.objects import FinSet
from quivers.continuous.spaces import Euclidean
from quivers.continuous.families import (
    ConditionalGaussianProcess,
    ConditionalHorseshoe,
)


class TestConditionalGaussianProcess:
    def test_rsample_shape_single(self):
        D, N = 2, 5
        gp = ConditionalGaussianProcess(
            Euclidean(name="X", dim=D),
            Euclidean(name="Y", dim=N),
            kernel="rbf",
        )
        x = torch.randn(N, D)
        s = gp.rsample(x)
        assert s.shape == (N,)

    def test_rsample_shape_batched(self):
        D, N, B = 3, 5, 4
        gp = ConditionalGaussianProcess(
            Euclidean(name="X", dim=D),
            Euclidean(name="Y", dim=N),
            kernel="rbf",
        )
        x = torch.randn(B, N, D)
        s = gp.rsample(x)
        assert s.shape == (B, N)

    def test_log_prob_round_trip(self):
        D, N = 2, 5
        torch.manual_seed(0)
        gp = ConditionalGaussianProcess(
            Euclidean(name="X", dim=D),
            Euclidean(name="Y", dim=N),
            kernel="rbf",
        )
        x = torch.randn(N, D)
        y = gp.rsample(x)
        lp = gp.log_prob(x, y)
        assert lp.shape == ()
        assert torch.isfinite(lp).all()

    def test_log_prob_batched(self):
        D, N, B = 2, 4, 3
        gp = ConditionalGaussianProcess(
            Euclidean(name="X", dim=D),
            Euclidean(name="Y", dim=N),
            kernel="matern52",
        )
        x = torch.randn(B, N, D)
        y = gp.rsample(x)
        lp = gp.log_prob(x, y)
        assert lp.shape == (B,)
        assert torch.isfinite(lp).all()

    @pytest.mark.parametrize("kernel", ["rbf", "matern52", "linear"])
    def test_kernel_choices_run(self, kernel):
        D, N = 2, 4
        gp = ConditionalGaussianProcess(
            Euclidean(name="X", dim=D),
            Euclidean(name="Y", dim=N),
            kernel=kernel,
        )
        x = torch.randn(N, D)
        y = gp.rsample(x)
        lp = gp.log_prob(x, y)
        assert torch.isfinite(lp).all()

    def test_gradient_flow_kernel_params(self):
        D, N = 2, 5
        gp = ConditionalGaussianProcess(
            Euclidean(name="X", dim=D),
            Euclidean(name="Y", dim=N),
            kernel="rbf",
        )
        x = torch.randn(N, D)
        y = gp.rsample(x)
        loss = -gp.log_prob(x, y).sum()
        loss.backward()
        assert gp._raw_length_scale.grad is not None
        assert gp._raw_amplitude.grad is not None
        assert gp._raw_length_scale.grad.abs().item() > 0
        assert gp._raw_amplitude.grad.abs().item() > 0

    def test_invalid_kernel_rejected(self):
        with pytest.raises(ValueError):
            ConditionalGaussianProcess(
                Euclidean(name="X", dim=2),
                Euclidean(name="Y", dim=3),
                kernel="bogus",
            )

    def test_input_shape_mismatch_rejected(self):
        gp = ConditionalGaussianProcess(
            Euclidean(name="X", dim=2),
            Euclidean(name="Y", dim=5),
        )
        x = torch.randn(3, 2)  # N=3 not 5
        with pytest.raises(ValueError):
            gp.rsample(x)


class TestConditionalHorseshoe:
    def test_rsample_shape_matches_codomain(self):
        Dcoord = 4
        hs = ConditionalHorseshoe(
            FinSet(name="A", cardinality=3),
            Euclidean(name="Y", dim=Dcoord),
            scale=1.0,
        )
        x = torch.tensor([0, 1, 2])
        s = hs.rsample(x)
        assert s.shape == (3, Dcoord)

    def test_rsample_shape_continuous_domain(self):
        Dcoord = 5
        hs = ConditionalHorseshoe(
            Euclidean(name="X", dim=2),
            Euclidean(name="Y", dim=Dcoord),
        )
        x = torch.randn(4, 2)
        s = hs.rsample(x)
        assert s.shape == (4, Dcoord)

    def test_log_prob_finite_at_unit_beta(self):
        Dcoord = 3
        hs = ConditionalHorseshoe(
            FinSet(name="A", cardinality=1),
            Euclidean(name="Y", dim=Dcoord),
            scale=1.0,
        )
        x = torch.tensor([0])
        y = torch.ones(1, Dcoord)
        lp = hs.log_prob(x, y)
        assert lp.shape == (1,)
        assert torch.isfinite(lp).all()
        assert not torch.isnan(lp).any()

    def test_log_prob_finite_across_magnitudes(self):
        hs = ConditionalHorseshoe(
            FinSet(name="A", cardinality=1),
            Euclidean(name="Y", dim=1),
            scale=1.0,
        )
        x = torch.tensor([0])
        for v in (1e-3, 0.1, 1.0, 3.0, 10.0):
            y = torch.tensor([[v]])
            lp = hs.log_prob(x, y)
            assert torch.isfinite(lp).all(), f"non-finite log_prob at beta={v}"

    def test_gradient_flow_scale(self):
        hs = ConditionalHorseshoe(
            FinSet(name="A", cardinality=2),
            Euclidean(name="Y", dim=3),
            scale=0.5,
        )
        x = torch.tensor([0, 1])
        y = torch.tensor([[0.1, -0.2, 0.3], [1.0, 0.5, -0.5]])
        lp = hs.log_prob(x, y).sum()
        lp.backward()
        assert hs._raw_scale.grad is not None
        assert hs._raw_scale.grad.abs().item() > 0

    def test_quadrature_normalisation_sanity(self):
        # The marginal density should integrate (roughly) to 1 over
        # beta. Use a coarse Simpson rule over [-20, 20] and confirm
        # the integral lies within a tolerant band.
        hs = ConditionalHorseshoe(
            FinSet(name="A", cardinality=1),
            Euclidean(name="Y", dim=1),
            scale=1.0,
        )
        x = torch.tensor([0])
        grid = torch.linspace(-20.0, 20.0, 4001).unsqueeze(-1)
        # broadcast x against the grid: treat each grid point as a
        # separate "batch" by repeating x.
        x_rep = torch.zeros(grid.shape[0], dtype=torch.long)
        lps = hs.log_prob(x_rep, grid)
        densities = lps.exp()
        dx = (40.0) / (4001 - 1)
        integral = (densities.sum() * dx).item()
        # The horseshoe marginal has an infinite spike at 0 that the
        # finite Simpson rule under-samples; a band of [0.7, 1.3] is
        # generous but still catches gross normalisation errors.
        assert 0.7 < integral < 1.3, f"integral = {integral}"
