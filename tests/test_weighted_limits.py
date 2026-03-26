"""Tests for weighted (co)limits."""

import torch
import pytest

from quivers.core.objects import FinSet
from quivers.core.morphisms import observed, identity
from quivers.core.quantales import PRODUCT_FUZZY, BOOLEAN
from quivers.enriched.weighted_limits import (
    Weight,
    Diagram,
    weighted_limit,
    weighted_colimit,
    weighted_limit_morphisms,
    weighted_colimit_morphisms,
    representable_weight,
    terminal_weight,
)


class TestWeight:
    def test_creation(self):
        w = Weight(values=torch.tensor([0.5, 1.0, 0.0]))
        assert w.size == 3

    def test_requires_1d(self):
        with pytest.raises(ValueError, match="1D"):
            Weight(values=torch.tensor([[0.5, 1.0]]))


class TestDiagram:
    def test_creation(self):
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        d = Diagram(objects=[a, b])
        assert d.size == 2


class TestRepresentableWeight:
    def test_representable(self):
        j = FinSet("J", 3)
        w = representable_weight(j, represented_at=1)

        assert w.values[0].item() == 0.0
        assert w.values[1].item() == 1.0
        assert w.values[2].item() == 0.0


class TestTerminalWeight:
    def test_terminal(self):
        j = FinSet("J", 3)
        w = terminal_weight(j)

        for i in range(3):
            assert w.values[i].item() == 1.0


class TestWeightedColimitMorphisms:
    def test_uniform_weight_colimit(self):
        """With uniform weights, colimit is the join over morphisms."""
        a = FinSet("A", 2)
        b = FinSet("B", 2)

        f1 = observed(a, b, torch.tensor([[0.8, 0.1], [0.2, 0.7]]))
        f2 = observed(a, b, torch.tensor([[0.3, 0.6], [0.5, 0.2]]))

        w = Weight(values=torch.tensor([1.0, 1.0]))
        result = weighted_colimit_morphisms(w, [f1, f2])

        # should be elementwise product of each with weight,
        # then join (noisy-OR) over j
        expected_00 = 1 - (1 - 0.8) * (1 - 0.3)
        assert result[0, 0].item() == pytest.approx(expected_00, abs=1e-5)

    def test_zero_weight_contributes_nothing(self):
        """A zero-weighted morphism should not contribute."""
        a = FinSet("A", 2)
        b = FinSet("B", 2)

        f1 = observed(a, b, torch.tensor([[0.8, 0.1], [0.2, 0.7]]))
        f2 = observed(a, b, torch.tensor([[0.3, 0.6], [0.5, 0.2]]))

        w = Weight(values=torch.tensor([1.0, 0.0]))
        result = weighted_colimit_morphisms(w, [f1, f2])

        # only f1 contributes (scaled by 1.0), f2 scaled by 0.0
        expected = f1.tensor
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestWeightedLimitMorphisms:
    def test_uniform_weight_limit(self):
        """With uniform unit weights, limit is the meet over morphisms."""
        a = FinSet("A", 2)
        b = FinSet("B", 2)

        f1 = observed(a, b, torch.tensor([[0.8, 0.1], [0.2, 0.7]]))
        f2 = observed(a, b, torch.tensor([[0.3, 0.6], [0.5, 0.2]]))

        w = Weight(values=torch.tensor([1.0, 1.0]))
        result = weighted_limit_morphisms(w, [f1, f2])

        # meet (product) over the two morphisms
        # [1.0, f_j] = f_j, so meet = elementwise product
        expected = f1.tensor * f2.tensor
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
