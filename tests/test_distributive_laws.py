"""Tests for distributive laws."""

import torch
import pytest
from quivers.core.objects import FinSet, FreeMonoid
from quivers.core.morphisms import identity
from quivers.monadic.distributive_laws import FreeMonoidPowersetLaw
from quivers.core.quantales import PRODUCT_FUZZY


class TestFreeMonoidPowersetLaw:
    def test_distribute_is_identity(self):
        """For identity on A, the distributive law is the identity on A*."""
        law = FreeMonoidPowersetLaw(max_length=2)
        a = FinSet(name="A", cardinality=2)
        result = law.distribute(a)
        fm = FreeMonoid(generators=a, max_length=2)
        expected = identity(fm).tensor
        torch.testing.assert_close(result.tensor, expected)

    def test_distribute_shape(self):
        law = FreeMonoidPowersetLaw(max_length=1)
        a = FinSet(name="A", cardinality=3)
        result = law.distribute(a)
        assert result.tensor.shape == (4, 4)

    def test_distribute_block_diagonal(self):
        """The distributive law tensor should be block-diagonal."""
        law = FreeMonoidPowersetLaw(max_length=1)
        a = FinSet(name="A", cardinality=2)
        result = law.distribute(a)
        t = result.tensor
        assert t[0, 0].item() == pytest.approx(1.0)
        assert t[0, 1].item() == pytest.approx(0.0)
        torch.testing.assert_close(t[1:3, 1:3], torch.eye(2), atol=1e-06, rtol=1e-06)

    def test_requires_finset(self):
        law = FreeMonoidPowersetLaw(max_length=1)
        with pytest.raises(TypeError, match="FinSet"):
            law.distribute(
                FreeMonoid(generators=FinSet(name="A", cardinality=2), max_length=1)
            )

    def test_outer_inner_monads(self):
        law = FreeMonoidPowersetLaw(max_length=2)
        assert law.outer_monad.max_length == 2
        assert law.inner_monad.quantale is PRODUCT_FUZZY
