"""Tests for change of enrichment."""

import torch
import pytest

from quivers.core.objects import FinSet
from quivers.core.morphisms import observed
from quivers.categorical.base_change import BoolToFuzzy, FuzzyToBool
from quivers.core.quantales import PRODUCT_FUZZY, BOOLEAN


class TestBoolToFuzzy:
    def test_preserves_boolean_tensor(self):
        bc = BoolToFuzzy()
        t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        result = bc.apply_to_values(t)
        torch.testing.assert_close(result, t)

    def test_source_target(self):
        bc = BoolToFuzzy()
        assert bc.source is BOOLEAN
        assert bc.target is PRODUCT_FUZZY

    def test_apply_to_morphism(self):
        bc = BoolToFuzzy()
        a = FinSet("A", 2)
        b = FinSet("B", 2)
        data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        f = observed(a, b, data, quantale=BOOLEAN)

        result = bc.apply_to_morphism(f)
        assert result.quantale is PRODUCT_FUZZY
        torch.testing.assert_close(result.tensor, data)


class TestFuzzyToBool:
    def test_threshold_default(self):
        bc = FuzzyToBool()
        t = torch.tensor([0.3, 0.5, 0.7, 0.9])
        result = bc.apply_to_values(t)
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_threshold_custom(self):
        bc = FuzzyToBool(threshold=0.8)
        t = torch.tensor([0.3, 0.5, 0.7, 0.9])
        result = bc.apply_to_values(t)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0])
        torch.testing.assert_close(result, expected)

    def test_source_target(self):
        bc = FuzzyToBool()
        assert bc.source is PRODUCT_FUZZY
        assert bc.target is BOOLEAN

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            FuzzyToBool(threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            FuzzyToBool(threshold=1.0)


class TestRoundTrip:
    def test_bool_fuzzy_bool(self):
        """Bool → Fuzzy → Bool should be identity on {0,1} tensors."""
        bc_to_fuzzy = BoolToFuzzy()
        bc_to_bool = FuzzyToBool(threshold=0.5)

        t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        fuzzy = bc_to_fuzzy.apply_to_values(t)
        back = bc_to_bool.apply_to_values(fuzzy)

        torch.testing.assert_close(back, t)

    def test_apply_to_morphism_roundtrip(self):
        a = FinSet("A", 2)
        b = FinSet("B", 2)
        data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        f = observed(a, b, data, quantale=BOOLEAN)

        fuzzy_f = BoolToFuzzy().apply_to_morphism(f)
        bool_f = FuzzyToBool().apply_to_morphism(fuzzy_f)

        torch.testing.assert_close(bool_f.tensor, data)
