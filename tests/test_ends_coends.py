"""Tests for ends and coends."""

import torch
import pytest

from quivers.enriched.ends_coends import coend, end
from quivers.core.quantales import PRODUCT_FUZZY, BOOLEAN


class TestCoend:
    def test_1d_coend_matches_compose(self):
        """Coend over shared dim should match quantale.compose for 1D."""
        torch.manual_seed(42)
        q = PRODUCT_FUZZY
        m = torch.rand(3, 4)
        n = torch.rand(4, 5)

        # manually build the outer product tensor
        # shape: (3, 4, 4, 5)
        outer = q.tensor_op(
            m.unsqueeze(2).unsqueeze(3),
            n.unsqueeze(0).unsqueeze(1),
        )

        # coend over dims 1 (contra) and 2 (co) — both size 4
        result = coend(outer, contra_dims=(1,), co_dims=(2,), quantale=q)
        expected = q.compose(m, n, n_contract=1)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_coend_empty_dims(self):
        """Coend with no dims should return tensor unchanged."""
        t = torch.rand(3, 4)
        result = coend(t, contra_dims=(), co_dims=())
        torch.testing.assert_close(result, t)

    def test_coend_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            coend(torch.rand(3, 4), contra_dims=(0,), co_dims=(0, 1))

    def test_coend_size_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension pair"):
            coend(
                torch.rand(3, 4),
                contra_dims=(0,),
                co_dims=(1,),
                quantale=PRODUCT_FUZZY,
            )

    def test_coend_square_trace(self):
        """Coend of a square matrix = join over diagonal."""
        q = PRODUCT_FUZZY
        t = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        result = coend(t, contra_dims=(0,), co_dims=(1,), quantale=q)

        # diagonal is [0.1, 0.4], noisy-OR = 1 - (1-0.1)(1-0.4) = 1 - 0.54 = 0.46
        expected = q.join(torch.tensor([0.1, 0.4]), dim=0)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_boolean_coend(self):
        """Boolean coend = OR over diagonal."""
        q = BOOLEAN
        t = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        result = coend(t, contra_dims=(0,), co_dims=(1,), quantale=q)

        # diagonal: [1, 0], OR = 1
        assert result.item() == pytest.approx(1.0)


class TestEnd:
    def test_1d_end_is_meet_over_diagonal(self):
        """End should be meet over diagonal entries."""
        q = PRODUCT_FUZZY
        t = torch.tensor([[0.8, 0.2], [0.3, 0.9]])
        result = end(t, contra_dims=(0,), co_dims=(1,), quantale=q)

        # diagonal: [0.8, 0.9], meet (product) = 0.72
        expected = q.meet(torch.tensor([0.8, 0.9]), dim=0)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_boolean_end(self):
        """Boolean end = AND over diagonal."""
        q = BOOLEAN
        t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = end(t, contra_dims=(0,), co_dims=(1,), quantale=q)

        # diagonal: [1, 1], AND = 1
        assert result.item() == pytest.approx(1.0)

    def test_boolean_end_fails(self):
        q = BOOLEAN
        t = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        result = end(t, contra_dims=(0,), co_dims=(1,), quantale=q)

        # diagonal: [1, 0], AND = 0
        assert result.item() == pytest.approx(0.0)

    def test_end_empty_dims(self):
        t = torch.rand(3, 4)
        result = end(t, contra_dims=(), co_dims=())
        torch.testing.assert_close(result, t)
