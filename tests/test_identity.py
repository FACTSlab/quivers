"""Tests for identity morphisms and categorical identity laws."""

import torch
import pytest

from quivers.core.objects import FinSet, ProductSet
from quivers.core.morphisms import morphism, observed, identity
from quivers.core.quantales import PRODUCT_FUZZY, BOOLEAN


class TestIdentityMorphism:
    def test_basic_properties(self):
        x = FinSet("X", 3)
        idx = identity(x)

        assert idx.domain == x
        assert idx.codomain == x
        assert idx.tensor.shape == (3, 3)

    def test_is_diagonal(self):
        x = FinSet("X", 4)
        idx = identity(x)
        torch.testing.assert_close(idx.tensor, torch.eye(4))

    def test_boolean_identity(self):
        x = FinSet("X", 3)
        idx = identity(x, quantale=BOOLEAN)

        assert idx.quantale is BOOLEAN
        torch.testing.assert_close(idx.tensor, torch.eye(3))


class TestIdentityLaws:
    """f >> id ≈ f and id >> f ≈ f."""

    def test_right_identity(self):
        """f >> identity(Y) ≈ f."""
        torch.manual_seed(42)
        x = FinSet("X", 3)
        y = FinSet("Y", 4)

        f_data = torch.rand(3, 4)
        f = observed(x, y, f_data)
        idx_y = identity(y)

        result = f >> idx_y
        torch.testing.assert_close(
            result.tensor, f_data, atol=1e-5, rtol=1e-5
        )

    def test_left_identity(self):
        """identity(X) >> f ≈ f."""
        torch.manual_seed(42)
        x = FinSet("X", 3)
        y = FinSet("Y", 4)

        f_data = torch.rand(3, 4)
        f = observed(x, y, f_data)
        idx_x = identity(x)

        result = idx_x >> f
        torch.testing.assert_close(
            result.tensor, f_data, atol=1e-5, rtol=1e-5
        )

    def test_identity_laws_with_latent(self):
        """Identity laws hold for latent (sigmoid-valued) morphisms."""
        torch.manual_seed(42)
        x = FinSet("X", 3)
        y = FinSet("Y", 4)

        f = morphism(x, y)
        idx_x = identity(x)
        idx_y = identity(y)

        left_id = idx_x >> f
        right_id = f >> idx_y

        torch.testing.assert_close(
            left_id.tensor, f.tensor, atol=1e-5, rtol=1e-5
        )

        torch.testing.assert_close(
            right_id.tensor, f.tensor, atol=1e-5, rtol=1e-5
        )

    def test_boolean_identity_laws(self):
        """Identity laws hold in the Boolean quantale."""
        x = FinSet("X", 3)
        y = FinSet("Y", 2)

        f_data = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        f = observed(x, y, f_data, quantale=BOOLEAN)
        idx_x = identity(x, quantale=BOOLEAN)
        idx_y = identity(y, quantale=BOOLEAN)

        left_id = idx_x >> f
        right_id = f >> idx_y

        torch.testing.assert_close(
            left_id.tensor, f_data, atol=1e-5, rtol=1e-5
        )

        torch.testing.assert_close(
            right_id.tensor, f_data, atol=1e-5, rtol=1e-5
        )

    def test_identity_compose_identity(self):
        """id >> id = id."""
        x = FinSet("X", 3)
        idx = identity(x)
        result = idx >> idx

        torch.testing.assert_close(
            result.tensor, torch.eye(3), atol=1e-5, rtol=1e-5
        )
