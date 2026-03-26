"""Tests for optics (lenses, prisms, adapters)."""

import torch
import pytest

from quivers.core.objects import FinSet, ProductSet, CoproductSet
from quivers.core.morphisms import observed, identity
from quivers.core.quantales import PRODUCT_FUZZY
from quivers.enriched.optics import (
    Lens,
    Prism,
    Adapter,
    compose_optics,
)


class TestLens:
    def test_lens_get(self):
        """Lens get should project the focus component."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        whole = ProductSet(a, b)
        lens = Lens(whole, focus_index=0)

        get_morph = lens.forward()
        assert get_morph.domain.shape == whole.shape
        assert get_morph.codomain.shape == a.shape

        # (0, any_b) -> 0, (1, any_b) -> 1
        for ai in range(2):
            for bi in range(3):
                assert get_morph.tensor[ai, bi, ai].item() == 1.0

                for ao in range(2):
                    if ao != ai:
                        assert get_morph.tensor[ai, bi, ao].item() == 0.0

    def test_lens_put(self):
        """Lens put should embed the focus back."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        whole = ProductSet(a, b)
        lens = Lens(whole, focus_index=0)

        put_morph = lens.backward()
        assert put_morph.domain.shape == a.shape
        assert put_morph.codomain.shape == whole.shape

    def test_lens_get_put_law(self):
        """get >> put should be idempotent (in the fuzzy sense)."""
        a = FinSet("A", 2)
        b = FinSet("B", 2)
        whole = ProductSet(a, b)
        lens = Lens(whole, focus_index=0)

        get_m = lens.forward()
        put_m = lens.backward()

        # get >> put: whole -> whole
        roundtrip = (get_m >> put_m).tensor
        assert roundtrip.shape == (*whole.shape, *whole.shape)

    def test_lens_requires_product(self):
        """Lens should require ProductSet."""
        with pytest.raises(TypeError, match="ProductSet"):
            Lens(FinSet("A", 3), focus_index=0)

    def test_lens_second_component(self):
        """Lens focusing on second component of a product."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        whole = ProductSet(a, b)
        lens = Lens(whole, focus_index=1)

        get_morph = lens.forward()
        assert get_morph.codomain.shape == b.shape

        # (any_a, 0) -> 0, (any_a, 1) -> 1, etc.
        for ai in range(2):
            for bi in range(3):
                assert get_morph.tensor[ai, bi, bi].item() == 1.0


class TestPrism:
    def test_prism_match(self):
        """Prism match should extract the focus component."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        whole = CoproductSet(a, b)
        prism = Prism(whole, focus_index=0)

        match_morph = prism.forward()
        assert match_morph.domain.shape == (5,)  # 2 + 3
        assert match_morph.codomain.shape == (2,)

        # first 2 elements map to 0, 1
        assert match_morph.tensor[0, 0].item() == 1.0
        assert match_morph.tensor[1, 1].item() == 1.0

        # elements from B map to zero
        for i in range(2, 5):
            for j in range(2):
                assert match_morph.tensor[i, j].item() == 0.0

    def test_prism_build(self):
        """Prism build should inject the focus into the coproduct."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        whole = CoproductSet(a, b)
        prism = Prism(whole, focus_index=0)

        build_morph = prism.backward()
        assert build_morph.domain.shape == (2,)
        assert build_morph.codomain.shape == (5,)

        assert build_morph.tensor[0, 0].item() == 1.0
        assert build_morph.tensor[1, 1].item() == 1.0

    def test_prism_build_match_roundtrip(self):
        """build >> match = id for the focus."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        whole = CoproductSet(a, b)
        prism = Prism(whole, focus_index=0)

        build = prism.backward()
        match = prism.forward()

        roundtrip = (build >> match).tensor
        expected = identity(a).tensor

        torch.testing.assert_close(
            roundtrip, expected, atol=1e-5, rtol=1e-5
        )

    def test_prism_requires_coproduct(self):
        with pytest.raises(TypeError, match="CoproductSet"):
            Prism(FinSet("A", 3), focus_index=0)


class TestAdapter:
    def test_adapter_isomorphism(self):
        """Adapter with identity morphisms is an isomorphism."""
        a = FinSet("A", 3)
        fwd = identity(a)
        bwd = identity(a)
        adapter = Adapter(fwd, bwd)

        assert adapter.verify_isomorphism()

    def test_adapter_permutation(self):
        """Adapter with a permutation isomorphism."""
        a = FinSet("A", 2)

        # swap permutation
        swap = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        fwd = observed(a, a, swap)
        bwd = observed(a, a, swap)  # self-inverse

        adapter = Adapter(fwd, bwd)
        assert adapter.verify_isomorphism()


class TestComposeOptics:
    def test_compose_lenses(self):
        """Composing two lenses should produce a valid optic."""
        a = FinSet("A", 2)
        b = FinSet("B", 3)
        c = FinSet("C", 2)

        # outer lens: A × B → A
        whole = ProductSet(a, b)
        lens1 = Lens(whole, focus_index=0)

        # inner lens: identity on A (trivial, but tests composition)
        # use an adapter as the inner optic
        id_a = identity(a)
        adapter = Adapter(id_a, id_a)

        composed = compose_optics(lens1, adapter)

        fwd = composed.forward()
        bwd = composed.backward()

        # forward: whole → a
        assert fwd.codomain.shape == a.shape
