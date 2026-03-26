"""Tests for categorical objects."""

import pytest
from quivers.core.objects import FinSet, ProductSet, CoproductSet, FreeMonoid, Unit


class TestFinSet:
    def test_basic(self):
        x = FinSet("X", 5)
        assert x.size == 5
        assert x.shape == (5,)
        assert x.ndim == 1
        assert x.name == "X"

    def test_equality(self):
        a = FinSet("X", 5)
        b = FinSet("X", 5)
        assert a == b

    def test_invalid_cardinality(self):
        with pytest.raises(ValueError):
            FinSet("X", 0)

    def test_repr(self):
        x = FinSet("X", 3)
        assert "X" in repr(x)
        assert "3" in repr(x)


class TestUnit:
    def test_unit(self):
        assert Unit.size == 1
        assert Unit.shape == (1,)


class TestProductSet:
    def test_two_sets(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        p = ProductSet(x, y)
        assert p.shape == (3, 4)
        assert p.size == 12
        assert p.ndim == 2

    def test_flattens_nested(self):
        x = FinSet("X", 2)
        y = FinSet("Y", 3)
        z = FinSet("Z", 4)
        p = ProductSet(ProductSet(x, y), z)
        assert p.shape == (2, 3, 4)
        assert len(p.components) == 3

    def test_mul_operator(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        p = x * y
        assert isinstance(p, ProductSet)
        assert p.shape == (3, 4)

    def test_equality(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        assert ProductSet(x, y) == ProductSet(x, y)
        assert ProductSet(x, y) != ProductSet(y, x)


class TestCoproductSet:
    def test_two_sets(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        c = CoproductSet(x, y)
        assert c.size == 7
        assert c.shape == (7,)

    def test_offsets(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        c = CoproductSet(x, y)
        assert c.offset(0) == 0
        assert c.offset(1) == 3
        assert c.component_range(0) == (0, 3)
        assert c.component_range(1) == (3, 7)

    def test_flattens_nested(self):
        x = FinSet("X", 2)
        y = FinSet("Y", 3)
        z = FinSet("Z", 4)
        c = CoproductSet(CoproductSet(x, y), z)
        assert c.size == 9
        assert len(c.components) == 3

    def test_add_operator(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        c = x + y
        assert isinstance(c, CoproductSet)
        assert c.size == 7


class TestFreeMonoid:
    def test_cardinality(self):
        g = FinSet("abc", 3)
        fm = FreeMonoid(g, max_length=2)
        # 1 + 3 + 9 = 13
        assert fm.size == 13

    def test_cardinality_binary(self):
        g = FinSet("bin", 2)
        fm = FreeMonoid(g, max_length=3)
        # 1 + 2 + 4 + 8 = 15
        assert fm.size == 15

    def test_encode_empty(self):
        g = FinSet("abc", 3)
        fm = FreeMonoid(g, max_length=2)
        assert fm.encode(()) == 0

    def test_encode_single(self):
        g = FinSet("abc", 3)
        fm = FreeMonoid(g, max_length=2)
        # length-1 strings start at offset 1
        assert fm.encode((0,)) == 1
        assert fm.encode((1,)) == 2
        assert fm.encode((2,)) == 3

    def test_encode_pair(self):
        g = FinSet("abc", 3)
        fm = FreeMonoid(g, max_length=2)
        # length-2 strings start at offset 4
        assert fm.encode((0, 0)) == 4
        assert fm.encode((0, 1)) == 5
        assert fm.encode((0, 2)) == 6
        assert fm.encode((1, 0)) == 7
        assert fm.encode((2, 2)) == 12

    def test_decode_roundtrip(self):
        g = FinSet("abc", 3)
        fm = FreeMonoid(g, max_length=3)

        for idx in range(fm.size):
            word = fm.decode(idx)
            assert fm.encode(word) == idx

    def test_encode_too_long(self):
        g = FinSet("abc", 3)
        fm = FreeMonoid(g, max_length=2)

        with pytest.raises(ValueError):
            fm.encode((0, 0, 0))

    def test_encode_invalid_generator(self):
        g = FinSet("abc", 3)
        fm = FreeMonoid(g, max_length=2)

        with pytest.raises(ValueError):
            fm.encode((5,))
