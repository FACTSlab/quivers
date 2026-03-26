"""Tests for Day convolution."""

import torch
import pytest

from quivers.core.objects import FinSet, ProductSet
from quivers.enriched.profunctors import Profunctor
from quivers.categorical.monoidal import CartesianMonoidal
from quivers.enriched.day_convolution import (
    day_unit,
    day_convolution_profunctors,
)


class TestDayUnit:
    def test_unit_presheaf(self):
        """Unit presheaf has I at the unit index, ⊥ elsewhere."""
        objects = [FinSet("A", 2), FinSet("B", 3), FinSet("I", 1)]
        result = day_unit(objects, unit_index=2)

        assert result.shape == (3,)
        assert result[0].item() == 0.0
        assert result[1].item() == 0.0
        assert result[2].item() == 1.0


class TestDayConvolutionProfunctors:
    def test_external_tensor_product(self):
        """(P ⊛ Q)(ac, bd) = P(a, b) ⊗ Q(c, d)."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        c = FinSet("C", 2)
        d = FinSet("D", 2)

        p_data = torch.rand(2, 3)
        q_data = torch.rand(2, 2)
        p = Profunctor(contra=a, co=b, tensor=p_data)
        q = Profunctor(contra=c, co=d, tensor=q_data)

        monoidal = CartesianMonoidal()
        result = day_convolution_profunctors(p, q, monoidal)

        # result should be (A×C) ↛ (B×D), shape (2, 2, 3, 2)
        assert result.tensor.shape == (2, 2, 3, 2)

        # check a specific entry
        # result[a, c, b, d] = p[a, b] * q[c, d]
        for ai in range(2):
            for ci in range(2):
                for bi in range(3):
                    for di in range(2):
                        expected = p_data[ai, bi] * q_data[ci, di]
                        actual = result.tensor[ai, ci, bi, di]
                        assert actual.item() == pytest.approx(expected.item(), abs=1e-6)

    def test_profunctor_objects(self):
        """Day convolution produces correct contra/co objects."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        c = FinSet("C", 4)
        d = FinSet("D", 5)

        p = Profunctor(contra=a, co=b, tensor=torch.rand(2, 3))
        q = Profunctor(contra=c, co=d, tensor=torch.rand(4, 5))

        monoidal = CartesianMonoidal()
        result = day_convolution_profunctors(p, q, monoidal)

        assert isinstance(result.contra, ProductSet)
        assert isinstance(result.co, ProductSet)
        assert result.contra.shape == (2, 4)
        assert result.co.shape == (3, 5)
