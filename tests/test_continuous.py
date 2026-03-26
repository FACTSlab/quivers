"""Comprehensive tests for the continuous morphism layer."""

from __future__ import annotations


import pytest
import torch

from quivers.core.objects import FinSet
from quivers.continuous.spaces import (
    Euclidean,
    UnitInterval,
    Simplex,
    PositiveReals,
    ProductSpace,
)
from quivers.continuous.morphisms import (
    SampledComposition,
    DiscreteAsContinuous,
)
from quivers.continuous.families import (
    ConditionalNormal,
    ConditionalLogitNormal,
    ConditionalBeta,
    ConditionalTruncatedNormal,
    ConditionalDirichlet,
    # loc-scale
    ConditionalCauchy,
    ConditionalLaplace,
    ConditionalGumbel,
    ConditionalLogNormal,
    ConditionalStudentT,
    # positive-valued
    ConditionalExponential,
    ConditionalGamma,
    ConditionalChi2,
    ConditionalHalfCauchy,
    ConditionalHalfNormal,
    ConditionalInverseGamma,
    ConditionalWeibull,
    ConditionalPareto,
    # (0,1)-valued
    ConditionalKumaraswamy,
    ConditionalContinuousBernoulli,
    # two-df
    ConditionalFisherSnedecor,
    # special
    ConditionalUniform,
    # multivariate
    ConditionalMultivariateNormal,
    ConditionalLowRankMVN,
    # relaxed
    ConditionalRelaxedBernoulli,
    ConditionalRelaxedOneHotCategorical,
    # matrix-valued
    ConditionalWishart,
)
from quivers.continuous.boundaries import Discretize, Embed
from quivers.continuous.flows import AffineCouplingLayer, ConditionalFlow
from quivers.stochastic import StochasticMorphism


# ============================================================================
# spaces
# ============================================================================


class TestEuclidean:
    def test_basic_properties(self):
        R3 = Euclidean("pos", 3)
        assert R3.name == "pos"
        assert R3.dim == 3
        assert R3.event_shape == (3,)
        assert not R3.is_bounded

    def test_bounded(self):
        E = Euclidean("unit", 2, low=0.0, high=1.0)
        assert E.is_bounded
        assert E.low == 0.0
        assert E.high == 1.0

    def test_contains_unbounded(self):
        R2 = Euclidean("R2", 2)
        x = torch.randn(5, 2)
        assert R2.contains(x).all()

    def test_contains_bounded(self):
        E = Euclidean("unit", 2, low=0.0, high=1.0)
        inside = torch.tensor([[0.5, 0.5]])
        outside = torch.tensor([[1.5, 0.5]])
        assert E.contains(inside).item()
        assert not E.contains(outside).item()

    def test_sample_uniform(self):
        E = Euclidean("unit", 3, low=0.0, high=1.0)
        samples = E.sample_uniform(100)
        assert samples.shape == (100, 3)
        assert (samples >= 0.0).all()
        assert (samples <= 1.0).all()

    def test_sample_uniform_unbounded_raises(self):
        R = Euclidean("R", 1)
        with pytest.raises(ValueError, match="unbounded"):
            R.sample_uniform(10)


class TestUnitInterval:
    def test_creates_bounded_euclidean(self):
        U = UnitInterval("prob")
        assert isinstance(U, Euclidean)
        assert U.dim == 1
        assert U.low == 0.0
        assert U.high == 1.0


class TestSimplex:
    def test_basic(self):
        S = Simplex("probs", 3)
        assert S.dim == 3

    def test_contains(self):
        S = Simplex("probs", 3)
        valid = torch.tensor([[0.3, 0.3, 0.4]])
        invalid = torch.tensor([[0.5, 0.5, 0.5]])
        assert S.contains(valid).item()
        assert not S.contains(invalid).item()

    def test_sample_uniform(self):
        S = Simplex("S", 4)
        samples = S.sample_uniform(50)
        assert samples.shape == (50, 4)
        assert torch.allclose(samples.sum(dim=-1), torch.ones(50), atol=1e-5)


class TestPositiveReals:
    def test_basic(self):
        P = PositiveReals("sigma", 2)
        assert P.dim == 2

    def test_contains(self):
        P = PositiveReals("sigma", 2)
        valid = torch.tensor([[0.1, 1.0]])
        invalid = torch.tensor([[-0.1, 1.0]])
        assert P.contains(valid).item()
        assert not P.contains(invalid).item()


class TestProductSpace:
    def test_basic(self):
        A = Euclidean("x", 2)
        B = PositiveReals("sigma", 1)
        P = A * B
        assert isinstance(P, ProductSpace)
        assert P.dim == 3
        assert len(P.components) == 2

    def test_flattens_nested(self):
        A = Euclidean("a", 1)
        B = Euclidean("b", 1)
        C = Euclidean("c", 1)
        P = ProductSpace(A, ProductSpace(B, C))
        assert len(P.components) == 3
        assert P.dim == 3

    def test_contains(self):
        A = Euclidean("x", 1, low=0.0, high=1.0)
        B = PositiveReals("y", 1)
        P = A * B

        valid = torch.tensor([[0.5, 1.0]])
        invalid = torch.tensor([[0.5, -1.0]])
        assert P.contains(valid).item()
        assert not P.contains(invalid).item()


# ============================================================================
# conditional families — discrete domain
# ============================================================================


class TestConditionalNormalDiscrete:
    def test_basic_shapes(self):
        A = FinSet("A", 5)
        Y = Euclidean("Y", 3)
        f = ConditionalNormal(A, Y)

        x = torch.tensor([0, 1, 2])
        samples = f.rsample(x)
        assert samples.shape == (3, 3)

    def test_log_prob_shape(self):
        A = FinSet("A", 4)
        Y = Euclidean("Y", 2)
        f = ConditionalNormal(A, Y)

        x = torch.tensor([0, 1, 2, 3])
        y = torch.randn(4, 2)
        lp = f.log_prob(x, y)
        assert lp.shape == (4,)

    def test_log_prob_finite(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 1)
        f = ConditionalNormal(A, Y)

        x = torch.tensor([0, 1, 2])
        y = torch.randn(3, 1)
        lp = f.log_prob(x, y)
        assert torch.isfinite(lp).all()

    def test_gradient_flow(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalNormal(A, Y)

        x = torch.tensor([0, 1])
        samples = f.rsample(x)
        loss = samples.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in f.parameters()
        )
        assert has_grad

    def test_sample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalNormal(A, Y)

        x = torch.tensor([0, 1])
        samples = f.rsample(x, torch.Size([5]))
        assert samples.shape == (5, 2, 2)

    def test_different_domain_elements_differ(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalNormal(A, Y)

        # draw many samples to check means differ
        torch.manual_seed(42)
        x0 = torch.zeros(100, dtype=torch.long)
        x1 = torch.ones(100, dtype=torch.long)
        s0 = f.rsample(x0).mean(dim=0)
        s1 = f.rsample(x1).mean(dim=0)

        # different domain elements should generally produce different means
        # (they have different parameter entries)
        # this isn't guaranteed but is overwhelmingly likely
        assert not torch.allclose(s0, s1, atol=0.01)


class TestConditionalNormalContinuous:
    def test_continuous_domain(self):
        X = Euclidean("X", 2)
        Y = Euclidean("Y", 3)
        f = ConditionalNormal(X, Y)

        x = torch.randn(4, 2)
        samples = f.rsample(x)
        assert samples.shape == (4, 3)

    def test_log_prob_continuous(self):
        X = Euclidean("X", 1)
        Y = Euclidean("Y", 1)
        f = ConditionalNormal(X, Y)

        x = torch.randn(5, 1)
        y = torch.randn(5, 1)
        lp = f.log_prob(x, y)
        assert lp.shape == (5,)
        assert torch.isfinite(lp).all()

    def test_gradient_through_input(self):
        X = Euclidean("X", 2)
        Y = Euclidean("Y", 2)
        f = ConditionalNormal(X, Y)

        x = torch.randn(3, 2, requires_grad=True)
        samples = f.rsample(x)
        loss = samples.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


class TestConditionalLogitNormal:
    def test_samples_in_unit_interval(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2, low=0.0, high=1.0)
        f = ConditionalLogitNormal(A, Y)

        x = torch.tensor([0, 1, 2])
        samples = f.rsample(x)
        assert (samples > 0.0).all()
        assert (samples < 1.0).all()

    def test_log_prob_finite(self):
        A = FinSet("A", 2)
        Y = Euclidean("Y", 1, low=0.0, high=1.0)
        f = ConditionalLogitNormal(A, Y)

        x = torch.tensor([0, 1])
        y = torch.tensor([[0.3], [0.7]])
        lp = f.log_prob(x, y)
        assert torch.isfinite(lp).all()


class TestConditionalBeta:
    def test_samples_in_unit_interval(self):
        A = FinSet("A", 4)
        Y = Euclidean("Y", 2, low=0.0, high=1.0)
        f = ConditionalBeta(A, Y)

        x = torch.tensor([0, 1, 2, 3])
        samples = f.rsample(x)
        assert samples.shape == (4, 2)
        assert (samples > 0.0).all()
        assert (samples < 1.0).all()

    def test_log_prob(self):
        A = FinSet("A", 2)
        Y = Euclidean("Y", 1, low=0.0, high=1.0)
        f = ConditionalBeta(A, Y)

        x = torch.tensor([0, 1])
        y = torch.tensor([[0.5], [0.3]])
        lp = f.log_prob(x, y)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()

    def test_gradient_flow(self):
        A = FinSet("A", 2)
        Y = Euclidean("Y", 2, low=0.0, high=1.0)
        f = ConditionalBeta(A, Y)

        x = torch.tensor([0, 1])
        samples = f.rsample(x)
        loss = samples.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in f.parameters()
        )
        assert has_grad


class TestConditionalTruncatedNormal:
    def test_samples_in_bounds(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2, low=-1.0, high=1.0)
        f = ConditionalTruncatedNormal(A, Y)

        x = torch.tensor([0, 1, 2])
        samples = f.rsample(x)
        assert samples.shape == (3, 2)
        assert (samples >= -1.0).all()
        assert (samples <= 1.0).all()

    def test_requires_bounded_codomain(self):
        A = FinSet("A", 2)
        Y = Euclidean("Y", 1)
        with pytest.raises(ValueError, match="bounded"):
            ConditionalTruncatedNormal(A, Y)

    def test_log_prob(self):
        A = FinSet("A", 2)
        Y = Euclidean("Y", 1, low=0.0, high=1.0)
        f = ConditionalTruncatedNormal(A, Y)

        x = torch.tensor([0, 1])
        y = torch.tensor([[0.5], [0.3]])
        lp = f.log_prob(x, y)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()


class TestConditionalDirichlet:
    def test_samples_on_simplex(self):
        A = FinSet("A", 3)
        Y = Simplex("probs", 4)
        f = ConditionalDirichlet(A, Y)

        x = torch.tensor([0, 1, 2])
        samples = f.rsample(x)
        assert samples.shape == (3, 4)
        assert (samples > 0.0).all()
        assert torch.allclose(
            samples.sum(dim=-1),
            torch.ones(3),
            atol=1e-5,
        )

    def test_log_prob(self):
        A = FinSet("A", 2)
        Y = Simplex("probs", 3)
        f = ConditionalDirichlet(A, Y)

        x = torch.tensor([0, 1])
        y = torch.tensor([[0.3, 0.3, 0.4], [0.5, 0.2, 0.3]])
        lp = f.log_prob(x, y)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()


# ============================================================================
# composition
# ============================================================================


class TestSampledComposition:
    def test_continuous_continuous(self):
        """Two continuous morphisms compose via sampling."""
        X = Euclidean("X", 2)
        Y = Euclidean("Y", 3)
        Z = Euclidean("Z", 1)

        f = ConditionalNormal(X, Y)
        g = ConditionalNormal(Y, Z)

        h = f >> g
        assert isinstance(h, SampledComposition)

        x = torch.randn(4, 2)
        samples = h.rsample(x)
        assert samples.shape == (4, 1)

    def test_discrete_to_continuous(self):
        """Discrete >> continuous composition."""
        A = FinSet("A", 5)
        Y = Euclidean("Y", 2)

        f = StochasticMorphism(A, FinSet("mid", 3))
        g = ConditionalNormal(FinSet("mid", 3), Y)

        h = f >> g
        assert isinstance(h, SampledComposition)

        x = torch.tensor([0, 1, 2])
        samples = h.rsample(x)
        assert samples.shape == (3, 2)

    def test_log_prob_exact_discrete_intermediate(self):
        """When intermediate is discrete, log_prob is exact."""
        A = FinSet("A", 3)
        B = FinSet("B", 4)
        Y = Euclidean("Y", 2)

        f = StochasticMorphism(A, B)
        g = ConditionalNormal(B, Y)
        h = f >> g

        x = torch.tensor([0, 1, 2])
        y = torch.randn(3, 2)
        lp = h.log_prob(x, y)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()

    def test_log_prob_mc_continuous_intermediate(self):
        """When intermediate is continuous, log_prob uses MC."""
        X = Euclidean("X", 2)
        Y = Euclidean("Y", 2)
        Z = Euclidean("Z", 1)

        f = ConditionalNormal(X, Y)
        g = ConditionalNormal(Y, Z)
        h = SampledComposition(f, g, n_intermediate=200)

        x = torch.randn(3, 2)
        z = torch.randn(3, 1)
        lp = h.log_prob(x, z)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()


class TestDiscreteAsContinuous:
    def test_wrap_stochastic_morphism(self):
        A = FinSet("A", 3)
        B = FinSet("B", 4)
        f = StochasticMorphism(A, B)
        fc = DiscreteAsContinuous(f)

        x = torch.tensor([0, 1, 2])
        samples = fc.rsample(x)
        assert samples.shape == (3,)
        assert (samples >= 0).all()
        assert (samples < 4).all()

    def test_log_prob(self):
        A = FinSet("A", 2)
        B = FinSet("B", 3)
        f = StochasticMorphism(A, B)
        fc = DiscreteAsContinuous(f)

        x = torch.tensor([0, 1])
        y = torch.tensor([0, 2])
        lp = fc.log_prob(x, y)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()
        # probabilities from softmax should give reasonable log-probs
        assert (lp < 0).all()

    def test_rshift_discrete_continuous(self):
        """discrete_morphism >> continuous_morphism via __rrshift__."""
        A = FinSet("A", 3)
        B = FinSet("B", 4)
        Y = Euclidean("Y", 2)

        f = StochasticMorphism(A, B)
        g = ConditionalNormal(B, Y)

        h = f >> g
        assert isinstance(h, SampledComposition)

        x = torch.tensor([0, 1])
        samples = h.rsample(x)
        assert samples.shape == (2, 2)


# ============================================================================
# boundaries
# ============================================================================


class TestDiscretize:
    def test_basic(self):
        Y = Euclidean("Y", 1, low=0.0, high=1.0)
        d = Discretize(Y, n_bins=10)
        assert d.domain == Y
        assert d.codomain.size == 10

    def test_rsample(self):
        Y = Euclidean("Y", 1, low=0.0, high=1.0)
        d = Discretize(Y, n_bins=5)

        x = torch.tensor([[0.1], [0.5], [0.9]])
        bins = d.rsample(x)
        assert bins.shape == (3,)
        assert (bins >= 0).all()
        assert (bins < 5).all()

    def test_log_prob(self):
        Y = Euclidean("Y", 1, low=0.0, high=1.0)
        d = Discretize(Y, n_bins=5)

        x = torch.tensor([[0.1], [0.9]])
        y = torch.tensor([0, 4])
        lp = d.log_prob(x, y)
        assert lp.shape == (2,)
        assert torch.isfinite(lp).all()

    def test_requires_bounded(self):
        Y = Euclidean("Y", 1)
        with pytest.raises(ValueError, match="bounded"):
            Discretize(Y, n_bins=5)

    def test_requires_1d(self):
        Y = Euclidean("Y", 2, low=0.0, high=1.0)
        with pytest.raises(ValueError, match="1-d"):
            Discretize(Y, n_bins=5)


class TestEmbed:
    def test_basic(self):
        A = FinSet("A", 5)
        Y = Euclidean("Y", 2, low=0.0, high=1.0)
        e = Embed(A, Y)

        x = torch.tensor([0, 1, 2, 3, 4])
        samples = e.rsample(x)
        assert samples.shape == (5, 2)

    def test_log_prob(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2, low=0.0, high=1.0)
        e = Embed(A, Y)

        x = torch.tensor([0, 1, 2])
        y = torch.randn(3, 2)
        lp = e.log_prob(x, y)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()

    def test_gradient_flow(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 1)
        e = Embed(A, Y)

        x = torch.tensor([0, 1, 2])
        samples = e.rsample(x)
        loss = samples.sum()
        loss.backward()

        assert e.centers.grad is not None
        assert e.centers.grad.abs().sum() > 0

    def test_sample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        e = Embed(A, Y)

        x = torch.tensor([0, 1])
        samples = e.rsample(x, torch.Size([5]))
        assert samples.shape == (5, 2, 2)


# ============================================================================
# normalizing flows
# ============================================================================


class TestAffineCouplingLayer:
    def test_forward_inverse_roundtrip(self):
        A = FinSet("A", 3)
        layer = AffineCouplingLayer(A, dim=4, mask_even=True)

        x = torch.tensor([0, 1, 2])
        z = torch.randn(3, 4)

        z_out, log_det_fwd = layer.forward(x, z)
        z_back, log_det_inv = layer.inverse(x, z_out)

        assert torch.allclose(z, z_back, atol=1e-5)
        assert torch.allclose(
            log_det_fwd + log_det_inv,
            torch.zeros(3),
            atol=1e-5,
        )


class TestConditionalFlow:
    def test_basic_discrete_domain(self):
        A = FinSet("A", 5)
        Y = Euclidean("Y", 4)
        flow = ConditionalFlow(A, Y, n_layers=4)

        x = torch.tensor([0, 1, 2])
        samples = flow.rsample(x)
        assert samples.shape == (3, 4)

    def test_basic_continuous_domain(self):
        X = Euclidean("X", 2)
        Y = Euclidean("Y", 4)
        flow = ConditionalFlow(X, Y, n_layers=4)

        x = torch.randn(3, 2)
        samples = flow.rsample(x)
        assert samples.shape == (3, 4)

    def test_log_prob(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 4)
        flow = ConditionalFlow(A, Y, n_layers=4)

        x = torch.tensor([0, 1, 2])
        y = torch.randn(3, 4)
        lp = flow.log_prob(x, y)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()

    def test_gradient_flow(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 4)
        flow = ConditionalFlow(A, Y, n_layers=4)

        x = torch.tensor([0, 1, 2])
        samples = flow.rsample(x)
        loss = samples.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in flow.parameters()
        )
        assert has_grad

    def test_requires_dim_ge_2(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 1)
        with pytest.raises(ValueError, match="dim >= 2"):
            ConditionalFlow(A, Y)

    def test_sample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 4)
        flow = ConditionalFlow(A, Y, n_layers=2)

        x = torch.tensor([0, 1, 2])
        samples = flow.rsample(x, torch.Size([5]))
        assert samples.shape == (5, 3, 4)

    def test_log_prob_consistency(self):
        """Log-probs should be higher near the mode of the distribution."""
        A = FinSet("A", 1)
        Y = Euclidean("Y", 2)
        flow = ConditionalFlow(A, Y, n_layers=4, hidden_dim=32)

        x = torch.tensor([0])

        # draw many samples to estimate the mode region
        torch.manual_seed(123)
        samples = flow.rsample(x.expand(500))
        mean = samples.mean(dim=0, keepdim=True)

        # log-prob at mean should be higher than at a distant point
        lp_near = flow.log_prob(x, mean)
        lp_far = flow.log_prob(x, mean + 10.0)
        assert lp_near.item() > lp_far.item()


# ============================================================================
# integration: end-to-end pipelines
# ============================================================================


class TestIntegration:
    def test_discrete_continuous_pipeline(self):
        """FinSet -> FinSet -> Euclidean pipeline (PDS-like)."""
        A = FinSet("context", 5)
        B = FinSet("state", 3)
        Y = Euclidean("response", 2)

        f = StochasticMorphism(A, B)
        g = ConditionalNormal(B, Y)

        h = f >> g

        x = torch.tensor([0, 1, 2, 3, 4])
        samples = h.rsample(x)
        assert samples.shape == (5, 2)

    def test_embed_then_continuous(self):
        """FinSet -Embed-> Euclidean -ConditionalNormal-> Euclidean."""
        A = FinSet("A", 4)
        mid = Euclidean("mid", 2, low=0.0, high=1.0)
        Y = Euclidean("Y", 1)

        f = Embed(A, mid)
        g = ConditionalNormal(mid, Y)

        h = f >> g

        x = torch.tensor([0, 1, 2, 3])
        samples = h.rsample(x)
        assert samples.shape == (4, 1)

    def test_continuous_then_discretize(self):
        """Euclidean -ConditionalNormal-> Euclidean -Discretize-> FinSet."""
        X = Euclidean("X", 1, low=0.0, high=1.0)
        Y = Euclidean("Y", 1, low=0.0, high=1.0)

        f = ConditionalNormal(X, Y)
        d = Discretize(Y, n_bins=10)

        h = f >> d

        x = torch.rand(5, 1)
        bins = h.rsample(x)
        assert bins.shape == (5,)
        assert (bins >= 0).all()
        assert (bins < 10).all()

    def test_flow_composition(self):
        """ConditionalNormal >> ConditionalFlow."""
        X = Euclidean("X", 2)
        mid = Euclidean("mid", 3)
        Y = Euclidean("Y", 4)

        f = ConditionalNormal(X, mid)
        g = ConditionalFlow(mid, Y, n_layers=2, hidden_dim=16)

        h = f >> g

        x = torch.randn(3, 2)
        samples = h.rsample(x)
        assert samples.shape == (3, 4)

    def test_training_loop(self):
        """A minimal training loop with continuous morphisms."""
        torch.manual_seed(42)

        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalNormal(A, Y)

        # generate "target" data
        target_means = torch.tensor([[1.0, 2.0], [-1.0, 0.0], [0.0, -1.0]])

        optimizer = torch.optim.Adam(f.parameters(), lr=0.01)

        initial_loss = None

        for step in range(50):
            optimizer.zero_grad()

            # sample data
            x = torch.arange(3)
            y = target_means[x] + torch.randn(3, 2) * 0.1

            loss = -f.log_prob(x, y).mean()

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # training should decrease the loss
        assert final_loss < initial_loss

    def test_parameter_count(self):
        """Continuous morphisms should have learnable parameters."""
        A = FinSet("A", 5)
        Y = Euclidean("Y", 3)
        f = ConditionalNormal(A, Y)

        n_params = sum(p.numel() for p in f.parameters())
        # lookup table: 5 * (3 + 3) = 30 params
        assert n_params == 30

    def test_mixed_operators(self):
        """The >> operator works between discrete and continuous morphisms."""
        A = FinSet("A", 3)
        B = FinSet("B", 4)
        Y = Euclidean("Y", 2)

        discrete = StochasticMorphism(A, B)
        continuous = ConditionalNormal(B, Y)

        # discrete >> continuous (tests __rrshift__)
        composed = discrete >> continuous
        assert isinstance(composed, SampledComposition)

        x = torch.tensor([0, 1, 2])
        result = composed.rsample(x)
        assert result.shape == (3, 2)


# ============================================================================
# factory-generated families: loc-scale
# ============================================================================


class TestLocScaleFamilies:
    """Tests for Cauchy, Laplace, Gumbel, LogNormal, StudentT."""

    @pytest.fixture(
        params=[
            ConditionalCauchy,
            ConditionalLaplace,
            ConditionalGumbel,
            ConditionalLogNormal,
            ConditionalStudentT,
        ]
    )
    def family_cls(self, request):
        return request.param

    def test_rsample_shape_discrete(self, family_cls):
        A = FinSet("A", 5)
        Y = Euclidean("Y", 3)
        f = family_cls(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert s.shape == (3, 3)

    def test_log_prob_shape_discrete(self, family_cls):
        A = FinSet("A", 4)
        Y = Euclidean("Y", 2)
        f = family_cls(A, Y)

        x = torch.tensor([0, 1, 2, 3])
        y = f.rsample(x).detach()
        lp = f.log_prob(x, y)
        assert lp.shape == (4,)
        assert torch.isfinite(lp).all()

    def test_gradient_flow(self, family_cls):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = family_cls(A, Y)

        x = torch.tensor([0, 1])
        s = f.rsample(x)
        s.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in f.parameters()
        )
        assert has_grad

    def test_continuous_domain(self, family_cls):
        X = Euclidean("X", 2)
        Y = Euclidean("Y", 3)
        f = family_cls(X, Y)

        x = torch.randn(4, 2)
        s = f.rsample(x)
        assert s.shape == (4, 3)

    def test_lognormal_positive(self):
        """LogNormal samples should be positive."""
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalLogNormal(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert (s > 0).all()


# ============================================================================
# factory-generated families: positive-valued
# ============================================================================


class TestPositiveFamilies:
    """Tests for Exponential, Gamma, Chi2, HalfCauchy, HalfNormal,
    InverseGamma, Weibull, Pareto."""

    @pytest.fixture(
        params=[
            ConditionalExponential,
            ConditionalGamma,
            ConditionalChi2,
            ConditionalHalfCauchy,
            ConditionalHalfNormal,
            ConditionalInverseGamma,
            ConditionalWeibull,
            ConditionalPareto,
        ]
    )
    def family_cls(self, request):
        return request.param

    def test_rsample_shape(self, family_cls):
        A = FinSet("A", 4)
        Y = Euclidean("Y", 2)
        f = family_cls(A, Y)

        x = torch.tensor([0, 1, 2, 3])
        s = f.rsample(x)
        assert s.shape == (4, 2)

    def test_samples_positive(self, family_cls):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = family_cls(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert (s > 0).all()

    def test_log_prob_finite(self, family_cls):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = family_cls(A, Y)

        x = torch.tensor([0, 1, 2])
        # use samples from the distribution itself to ensure valid support
        s = f.rsample(x).detach()
        lp = f.log_prob(x, s)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()

    def test_gradient_flow(self, family_cls):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = family_cls(A, Y)

        x = torch.tensor([0, 1])
        s = f.rsample(x)
        s.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in f.parameters()
        )
        assert has_grad


# ============================================================================
# (0,1)-valued families
# ============================================================================


class TestUnitIntervalFamilies:
    """Tests for Kumaraswamy, ContinuousBernoulli."""

    @pytest.fixture(
        params=[
            ConditionalKumaraswamy,
            ConditionalContinuousBernoulli,
        ]
    )
    def family_cls(self, request):
        return request.param

    def test_rsample_shape(self, family_cls):
        A = FinSet("A", 4)
        Y = Euclidean("Y", 2)
        f = family_cls(A, Y)

        x = torch.tensor([0, 1, 2, 3])
        s = f.rsample(x)
        assert s.shape == (4, 2)

    def test_samples_in_unit_interval(self, family_cls):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = family_cls(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert (s > 0).all()
        assert (s < 1).all()

    def test_log_prob_finite(self, family_cls):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = family_cls(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x).detach()
        lp = f.log_prob(x, s)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()


# ============================================================================
# two-df (FisherSnedecor)
# ============================================================================


class TestFisherSnedecor:
    def test_rsample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalFisherSnedecor(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert s.shape == (3, 2)
        assert (s > 0).all()

    def test_log_prob_finite(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalFisherSnedecor(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x).detach()
        lp = f.log_prob(x, s)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()


# ============================================================================
# special: Uniform
# ============================================================================


class TestConditionalUniform:
    def test_rsample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalUniform(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert s.shape == (3, 2)

    def test_log_prob_finite(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalUniform(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x).detach()
        lp = f.log_prob(x, s)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()

    def test_continuous_domain(self):
        X = Euclidean("X", 2)
        Y = Euclidean("Y", 3)
        f = ConditionalUniform(X, Y)

        x = torch.randn(4, 2)
        s = f.rsample(x)
        assert s.shape == (4, 3)


# ============================================================================
# multivariate distributions
# ============================================================================


class TestConditionalMultivariateNormal:
    def test_rsample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 4)
        f = ConditionalMultivariateNormal(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert s.shape == (3, 4)

    def test_log_prob_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 4)
        f = ConditionalMultivariateNormal(A, Y)

        x = torch.tensor([0, 1, 2])
        y = torch.randn(3, 4)
        lp = f.log_prob(x, y)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()

    def test_gradient_flow(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 3)
        f = ConditionalMultivariateNormal(A, Y)

        x = torch.tensor([0, 1])
        s = f.rsample(x)
        s.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in f.parameters()
        )
        assert has_grad

    def test_continuous_domain(self):
        X = Euclidean("X", 2)
        Y = Euclidean("Y", 3)
        f = ConditionalMultivariateNormal(X, Y)

        x = torch.randn(4, 2)
        s = f.rsample(x)
        assert s.shape == (4, 3)


class TestConditionalLowRankMVN:
    def test_rsample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 4)
        f = ConditionalLowRankMVN(A, Y, rank=2)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert s.shape == (3, 4)

    def test_log_prob_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 4)
        f = ConditionalLowRankMVN(A, Y, rank=2)

        x = torch.tensor([0, 1, 2])
        y = torch.randn(3, 4)
        lp = f.log_prob(x, y)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()

    def test_default_rank(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 4)
        f = ConditionalLowRankMVN(A, Y)

        # default rank is 2
        assert f._rank == 2


# ============================================================================
# relaxed discrete distributions
# ============================================================================


class TestConditionalRelaxedBernoulli:
    def test_rsample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 4)
        f = ConditionalRelaxedBernoulli(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert s.shape == (3, 4)

    def test_samples_in_unit_interval(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 4)
        f = ConditionalRelaxedBernoulli(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert (s > 0).all()
        assert (s < 1).all()

    def test_log_prob_finite(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalRelaxedBernoulli(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x).detach()
        lp = f.log_prob(x, s)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()

    def test_custom_temperature(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalRelaxedBernoulli(A, Y, temperature=0.1)

        assert f._temperature == 0.1


class TestConditionalRelaxedOneHotCategorical:
    def test_rsample_shape(self):
        A = FinSet("A", 3)
        Y = Simplex("Y", 5)
        f = ConditionalRelaxedOneHotCategorical(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert s.shape == (3, 5)

    def test_samples_on_simplex(self):
        A = FinSet("A", 3)
        Y = Simplex("Y", 4)
        f = ConditionalRelaxedOneHotCategorical(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        assert (s > 0).all()
        assert torch.allclose(s.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_log_prob_finite(self):
        A = FinSet("A", 3)
        Y = Simplex("Y", 4)
        f = ConditionalRelaxedOneHotCategorical(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x).detach()
        lp = f.log_prob(x, s)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()


# ============================================================================
# matrix-valued: Wishart
# ============================================================================


class TestConditionalWishart:
    def test_rsample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)  # 2x2 matrices, flattened to 4
        f = ConditionalWishart(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x)
        # output is d^2 = 4
        assert s.shape == (3, 4)

    def test_log_prob_finite(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalWishart(A, Y)

        x = torch.tensor([0, 1, 2])
        s = f.rsample(x).detach()
        lp = f.log_prob(x, s)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()


# ============================================================================
# generic _IndependentConditional base: sample_shape
# ============================================================================


class TestFactoryFamilySampleShape:
    """Verify sample_shape works for factory-generated families."""

    def test_cauchy_sample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalCauchy(A, Y)

        x = torch.tensor([0, 1])
        s = f.rsample(x, torch.Size([5]))
        assert s.shape == (5, 2, 2)

    def test_gamma_sample_shape(self):
        A = FinSet("A", 3)
        Y = Euclidean("Y", 2)
        f = ConditionalGamma(A, Y)

        x = torch.tensor([0, 1])
        s = f.rsample(x, torch.Size([5]))
        assert s.shape == (5, 2, 2)
