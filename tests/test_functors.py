"""Tests for Functor ABC and FreeMonoidFunctor."""

import torch
import pytest

from quivers.core.objects import FinSet, FreeMonoid
from quivers.core.morphisms import morphism, observed, FunctorMorphism
from quivers.categorical.functors import FreeMonoidFunctor
from quivers.core.quantales import BOOLEAN


class TestFreeMonoidFunctorObject:
    def test_map_object(self):
        fm = FreeMonoidFunctor(max_length=2)
        a = FinSet("A", 3)
        result = fm.map_object(a)

        assert isinstance(result, FreeMonoid)
        assert result.generators == a
        assert result.max_length == 2
        # 1 + 3 + 9 = 13
        assert result.size == 13

    def test_map_object_accepts_non_finset(self):
        """FreeMonoidFunctor accepts any SetObject via proxy FinSet."""
        fm = FreeMonoidFunctor(max_length=1)
        inner = FreeMonoid(FinSet("A", 2), max_length=1)  # size 3

        result = fm.map_object(inner)

        assert isinstance(result, FreeMonoid)
        # proxy FinSet has cardinality 3, so FreeMonoid size = 1 + 3 = 4
        assert result.size == 4

    def test_max_length_zero(self):
        fm = FreeMonoidFunctor(max_length=0)
        a = FinSet("A", 5)
        result = fm.map_object(a)
        # only empty string
        assert result.size == 1


class TestFreeMonoidFunctorMorphism:
    def test_map_morphism_returns_functor_morphism(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        f = morphism(a, b)
        result = fm.map_morphism(f)

        assert isinstance(result, FunctorMorphism)

    def test_map_morphism_domain_codomain(self):
        fm = FreeMonoidFunctor(max_length=2)
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        f = morphism(a, b)
        result = fm.map_morphism(f)

        assert isinstance(result.domain, FreeMonoid)
        assert isinstance(result.codomain, FreeMonoid)
        assert result.domain.generators == a
        assert result.codomain.generators == b

    def test_map_morphism_tensor_shape(self):
        fm = FreeMonoidFunctor(max_length=2)
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        f = morphism(a, b)
        result = fm.map_morphism(f)

        # domain: 1 + 2 + 4 = 7, codomain: 1 + 3 + 9 = 13
        assert result.tensor.shape == (7, 13)

    def test_block_diagonal_structure(self):
        """The functor morphism should have block-diagonal structure."""
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        f_data = torch.tensor([[0.5, 0.3, 0.1], [0.2, 0.8, 0.4]])
        f = observed(a, b, f_data)
        result = fm.map_morphism(f)

        t = result.tensor  # shape (3, 4): domain=1+2=3, codomain=1+3=4

        # block 0 (empty string -> empty string): 1x1, should be unit
        assert t[0, 0].item() == pytest.approx(1.0, abs=1e-6)

        # block 0 off-diagonal: empty string -> non-empty should be 0
        assert t[0, 1].item() == pytest.approx(0.0, abs=1e-6)
        assert t[0, 2].item() == pytest.approx(0.0, abs=1e-6)
        assert t[0, 3].item() == pytest.approx(0.0, abs=1e-6)

        # block 1 (length 1 -> length 1): should be f_data
        torch.testing.assert_close(t[1:3, 1:4], f_data, atol=1e-6, rtol=1e-6)

        # off-diagonal blocks should be zero
        assert t[1, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert t[2, 0].item() == pytest.approx(0.0, abs=1e-6)


class TestFunctorPreservesComposition:
    """F(f >> g) ≈ F(f) >> F(g) on boolean inputs."""

    def test_preserves_composition_boolean(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet("A", 2)
        b = FinSet("B", 2)
        c = FinSet("C", 2)

        f_data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        g_data = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        f = observed(a, b, f_data)
        g = observed(b, c, g_data)

        # F(f >> g)
        fg = f >> g
        fm_fg = fm.map_morphism(fg)

        # F(f) >> F(g)
        fm_f = fm.map_morphism(f)
        fm_g = fm.map_morphism(g)
        composed = fm_f >> fm_g

        torch.testing.assert_close(fm_fg.tensor, composed.tensor, atol=1e-4, rtol=1e-4)


class TestGradientFlowThroughFunctor:
    def test_gradient_flows(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        f = morphism(a, b)
        result = fm.map_morphism(f)

        loss = result.tensor.sum()
        loss.backward()

        assert f.raw.grad is not None
        assert torch.isfinite(f.raw.grad).all()

    def test_module_delegates_to_inner(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        f = morphism(a, b)
        result = fm.map_morphism(f)

        # parameters should be the same as the inner morphism
        inner_params = list(f.module().parameters())
        functor_params = list(result.module().parameters())

        assert len(inner_params) == len(functor_params)

        for ip, fp in zip(inner_params, functor_params):
            assert ip.data_ptr() == fp.data_ptr()


class TestQuantaleRespected:
    """FreeMonoidFunctor should use the morphism's quantale."""

    def test_boolean_functor(self):
        fm = FreeMonoidFunctor(max_length=1)
        a = FinSet("A", 2)
        b = FinSet("B", 2)

        f_data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        f = observed(a, b, f_data, quantale=BOOLEAN)
        result = fm.map_morphism(f)

        assert result.quantale is BOOLEAN

        # block structure should still hold
        t = result.tensor  # (3, 3)
        assert t[0, 0].item() == pytest.approx(1.0, abs=1e-6)

        torch.testing.assert_close(t[1:3, 1:3], f_data, atol=1e-6, rtol=1e-6)
