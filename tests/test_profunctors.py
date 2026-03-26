"""Tests for profunctors."""

import torch
import pytest

from quivers.core.objects import FinSet
from quivers.core.morphisms import observed
from quivers.enriched.profunctors import Profunctor
from quivers.core.quantales import PRODUCT_FUZZY


class TestProfunctor:
    def test_from_morphism(self):
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        f_data = torch.rand(3, 4)
        f = observed(a, b, f_data)

        p = Profunctor.from_morphism(f)
        assert p.contra == a
        assert p.co == b
        torch.testing.assert_close(p.tensor, f_data)

    def test_compose_matches_rshift(self):
        """Profunctor composition should match morphism >>."""
        torch.manual_seed(42)
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        c = FinSet("C", 2)

        f_data = torch.rand(3, 4)
        g_data = torch.rand(4, 2)
        f = observed(a, b, f_data)
        g = observed(b, c, g_data)

        pf = Profunctor.from_morphism(f)
        pg = Profunctor.from_morphism(g)
        composed = pf.compose(pg)

        expected = (f >> g).tensor
        torch.testing.assert_close(
            composed.tensor, expected, atol=1e-5, rtol=1e-5
        )

    def test_to_morphism(self):
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        data = torch.rand(3, 4)
        p = Profunctor(a, b, data)
        m = p.to_morphism()

        assert m.domain == a
        assert m.codomain == b
        torch.testing.assert_close(m.tensor, data)

    def test_shape_validation(self):
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        with pytest.raises(ValueError, match="tensor shape"):
            Profunctor(a, b, torch.rand(3, 5))

    def test_compose_mismatch_raises(self):
        a = FinSet("A", 3)
        b = FinSet("B", 4)
        c = FinSet("C", 2)
        d = FinSet("D", 5)

        p1 = Profunctor(a, b, torch.rand(3, 4))
        p2 = Profunctor(c, d, torch.rand(2, 5))

        with pytest.raises(TypeError, match="cannot compose"):
            p1.compose(p2)
