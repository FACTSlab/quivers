"""Tests for quantale operations."""

import torch
import pytest

from quivers.core.quantales import (
    ProductFuzzy,
    BooleanQuantale,
    PRODUCT_FUZZY,
    BOOLEAN,
)


class TestProductFuzzy:
    def test_tensor_op_is_product(self):
        q = ProductFuzzy()
        a = torch.tensor([0.3, 0.5, 0.8])
        b = torch.tensor([0.4, 0.6, 0.2])
        result = q.tensor_op(a, b)
        expected = a * b
        torch.testing.assert_close(result, expected)

    def test_join_is_noisy_or(self):
        q = ProductFuzzy()
        t = torch.tensor([[0.5, 0.3], [0.8, 0.1]])
        result = q.join(t, dim=1)

        # 1 - (1-0.5)(1-0.3) = 0.65
        # 1 - (1-0.8)(1-0.1) = 0.82
        expected = torch.tensor([0.65, 0.82])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_meet_is_product(self):
        q = ProductFuzzy()
        t = torch.tensor([0.5, 0.3])
        result = q.meet(t, dim=0)
        expected = torch.tensor(0.15)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_negate_is_complement(self):
        q = ProductFuzzy()
        t = torch.tensor([0.3, 0.7])
        result = q.negate(t)
        expected = torch.tensor([0.7, 0.3])
        torch.testing.assert_close(result, expected)

    def test_unit_and_zero(self):
        q = ProductFuzzy()
        assert q.unit == 1.0
        assert q.zero == 0.0

    def test_compose_matches_noisy_or_contract(self):
        """ProductFuzzy.compose should match noisy_or_contract exactly."""
        from quivers.core.tensor_ops import noisy_or_contract

        torch.manual_seed(42)
        m = torch.rand(3, 4)
        n = torch.rand(4, 5)

        q = ProductFuzzy()
        result_q = q.compose(m, n, n_contract=1)
        result_legacy = noisy_or_contract(m, n, n_contract=1)

        torch.testing.assert_close(result_q, result_legacy, atol=1e-6, rtol=1e-6)

    def test_compose_higher_order(self):
        """Compose over multi-dimensional shared set."""
        q = ProductFuzzy()
        m = torch.rand(3, 4, 5)
        n = torch.rand(4, 5, 2)
        result = q.compose(m, n, n_contract=2)
        assert result.shape == (3, 2)

    def test_compose_shape_mismatch_raises(self):
        q = ProductFuzzy()

        with pytest.raises(ValueError, match="shared dimensions"):
            q.compose(torch.rand(3, 4), torch.rand(5, 2), n_contract=1)

    def test_name(self):
        assert ProductFuzzy().name == "ProductFuzzy"

    def test_singleton(self):
        assert isinstance(PRODUCT_FUZZY, ProductFuzzy)


class TestBooleanQuantale:
    def test_tensor_op_is_and(self):
        q = BooleanQuantale()
        a = torch.tensor([1.0, 0.0, 1.0, 0.0])
        b = torch.tensor([1.0, 1.0, 0.0, 0.0])
        result = q.tensor_op(a, b)
        expected = torch.tensor([1.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_join_is_or(self):
        q = BooleanQuantale()
        t = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = q.join(t, dim=1)
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_meet_is_and(self):
        q = BooleanQuantale()
        t = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        result = q.meet(t, dim=1)
        expected = torch.tensor([1.0, 0.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_negate_is_not(self):
        q = BooleanQuantale()
        t = torch.tensor([0.0, 1.0])
        result = q.negate(t)
        expected = torch.tensor([1.0, 0.0])
        torch.testing.assert_close(result, expected)

    def test_compose_boolean_matmul(self):
        """Boolean composition should match boolean matrix multiply."""
        q = BooleanQuantale()
        m = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        n = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        result = q.compose(m, n, n_contract=1)

        expected = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_name(self):
        assert BooleanQuantale().name == "Boolean"

    def test_singleton(self):
        assert isinstance(BOOLEAN, BooleanQuantale)


class TestIdentityTensor:
    def test_1d_identity(self):
        q = ProductFuzzy()
        identity = q.identity_tensor((3,))
        assert identity.shape == (3, 3)
        torch.testing.assert_close(identity, torch.eye(3))

    def test_multi_dim_identity(self):
        q = ProductFuzzy()
        identity = q.identity_tensor((2, 3))
        assert identity.shape == (2, 3, 2, 3)

        # diagonal entries should be unit (1.0)
        for i in range(2):
            for j in range(3):
                assert identity[i, j, i, j].item() == 1.0

        # off-diagonal should be zero
        for i1 in range(2):
            for j1 in range(3):
                for i2 in range(2):
                    for j2 in range(3):
                        if (i1, j1) != (i2, j2):
                            assert identity[i1, j1, i2, j2].item() == 0.0

    def test_boolean_identity(self):
        q = BooleanQuantale()
        identity = q.identity_tensor((4,))
        torch.testing.assert_close(identity, torch.eye(4))


class TestCompatibility:
    def test_same_type_compatible(self):
        assert PRODUCT_FUZZY.is_compatible(ProductFuzzy())
        assert BOOLEAN.is_compatible(BooleanQuantale())

    def test_different_type_incompatible(self):
        assert not PRODUCT_FUZZY.is_compatible(BOOLEAN)
        assert not BOOLEAN.is_compatible(PRODUCT_FUZZY)


class TestDeMorgan:
    """Verify quantale-level De Morgan duality."""

    def test_product_fuzzy_de_morgan(self):
        """¬(⋁ x_i) = ⋀ (¬x_i) in ProductFuzzy."""
        q = ProductFuzzy()
        torch.manual_seed(42)
        x = torch.rand(8)

        lhs = q.negate(q.join(x, dim=0))
        rhs = q.meet(q.negate(x), dim=0)

        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)

    def test_boolean_de_morgan(self):
        """¬(⋁ x_i) = ⋀ (¬x_i) in BooleanQuantale."""
        q = BooleanQuantale()
        x = torch.tensor([1.0, 0.0, 1.0, 0.0])

        lhs = q.negate(q.join(x, dim=0))
        rhs = q.meet(q.negate(x), dim=0)

        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)
