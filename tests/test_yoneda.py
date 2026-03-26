"""Tests for Yoneda embedding and representable profunctors."""

import torch
import pytest

from quivers.core.objects import FinSet
from quivers.core.morphisms import observed, identity
from quivers.core.quantales import PRODUCT_FUZZY
from quivers.enriched.profunctors import Profunctor
from quivers.enriched.yoneda import (
    representable_profunctor,
    corepresentable_profunctor,
    yoneda_embedding,
    yoneda_density,
    verify_yoneda_fully_faithful,
)


class TestRepresentableProfunctor:
    def test_representable_is_identity(self):
        """y(A) should be the identity profunctor."""
        a = FinSet("A", 3)
        p = representable_profunctor(a)

        expected = torch.eye(3)
        torch.testing.assert_close(p.tensor, expected)

    def test_contra_and_co(self):
        a = FinSet("A", 3)
        p = representable_profunctor(a)

        assert p.contra == a
        assert p.co == a


class TestYonedaEmbedding:
    def test_embedding_equals_profunctor_view(self):
        """y(f) should equal Profunctor.from_morphism(f)."""
        torch.manual_seed(42)
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        f_data = torch.rand(3, 4)
        f = observed(a, b, f_data)

        y_f = yoneda_embedding(f)
        p_f = Profunctor.from_morphism(f)

        torch.testing.assert_close(y_f.tensor, p_f.tensor)

    def test_preserves_composition(self):
        """y(f >> g) ≈ y(f) ; y(g)."""
        torch.manual_seed(42)
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        c = FinSet("C", 2)

        f = observed(a, b, torch.rand(3, 4))
        g = observed(b, c, torch.rand(4, 2))

        assert verify_yoneda_fully_faithful(f, g)


class TestYonedaDensity:
    def test_density_recovers_morphism(self):
        """Yoneda density: id_A >> f = f."""
        torch.manual_seed(42)
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        f_data = torch.rand(3, 4)
        f = observed(a, b, f_data)

        result = yoneda_density(f)

        torch.testing.assert_close(
            result, f.tensor, atol=1e-4, rtol=1e-4
        )
