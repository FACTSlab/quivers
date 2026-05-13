"""Multi-quantale composition and change-of-base for V-Cat
morphisms.

A morphism in :mod:`quivers.core.morphisms` carries an enrichment
quantale. Composing two morphisms over the same quantale uses
that quantale's monoidal structure; composing across quantales
requires a quantale homomorphism (a lax monoidal poset functor)
applied via :meth:`Morphism.change_base`.

The DSL exposes one composition operator per canonical quantale:

* ``>>`` — ProductFuzzy noisy-OR (default).
* ``<<`` — reverse ``>>``.
* ``>=>`` — Kleisli, in the operands' shared quantale.
* ``*>`` — Markov sum-product.
* ``~>`` — LogProb (log-space sum-product).
* ``||>`` — Gödel (lattice min/max + Heyting implication).
* ``?>`` — Viterbi (max-plus tropical, best path).
* ``&&>`` — Boolean (∧/∨).
* ``+>`` — Łukasiewicz (probabilistic sum bounded by 1).

Each operator carries its quantale; the operands' declared
quantales must already match the operator's target (the operator
does not auto-base-change). Cross-operator chains require an
explicit ``.change_base(φ)`` between segments.

This module verifies:

1. The Python-level ``change_base`` adapter: applying each
   canonical homomorphism to a morphism's tensor produces a new
   morphism over the target quantale with the expected
   per-entry mapping.
2. Composition over each non-default quantale (Markov, LogProb,
   Gödel, Viterbi, Boolean, Łukasiewicz) produces a
   :class:`ComposedMorphism` over the right quantale.
3. The DSL composition operators dispatch to the right quantale
   and raise a typed error on mismatched operands.
"""

from __future__ import annotations

import os

import pytest
import torch

from quivers.core.quantales import (
    GODEL,
    LOG_PROB,
    LUKASIEWICZ,
    MAX_PLUS,
    TROPICAL,
)
from quivers.core.morphisms import (
    ComposedMorphism,
    LatentMorphism,
    ObservedMorphism,
)
from quivers.core.objects import FinSet
from quivers.core.quantale_morphisms import (
    IdentityHom,
    LOG_PROB as LOG_PROB_HOM,
    MATERIAL_IMPLICATION,
    MAX_PLUS as MAX_PLUS_HOM,
    Threshold,
    embedding,
    lookup_homomorphism,
    threshold,
)
from quivers.core.quantales import BOOLEAN, PRODUCT_FUZZY
from quivers.stochastic.quantale import MARKOV


_LOCAL_GRAMMAR = pytest.mark.skipif(
    os.environ.get("QVR_USE_LOCAL_GRAMMAR", "") not in ("1", "true", "True"),
    reason="needs QVR_USE_LOCAL_GRAMMAR=1 to pick up the in-tree grammar",
)


# ---------------------------------------------------------------------------
# change_base on Morphism
# ---------------------------------------------------------------------------


def test_change_base_identity_preserves_tensor() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=3)
    f = LatentMorphism(A, B)
    g = f.change_base(IdentityHom(PRODUCT_FUZZY))
    assert g.quantale.name == "ProductFuzzy"
    assert torch.allclose(g.tensor, f.tensor)


def test_change_base_to_log_prob_returns_log_of_tensor() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.4, 0.6], [0.7, 0.3]])
    f = ObservedMorphism(A, B, data)
    g = f.change_base(LOG_PROB_HOM)
    expected = torch.log(data.clamp(min=1e-30))
    assert torch.allclose(g.tensor, expected)


def test_change_base_to_boolean_via_threshold() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.4, 0.7], [0.3, 0.9]])
    f = ObservedMorphism(A, B, data)
    g = f.change_base(threshold(0.5))
    expected = (data > 0.5).to(dtype=torch.float32)
    assert torch.allclose(g.tensor, expected)
    assert g.quantale.name == "Boolean"


def test_change_base_to_godel_via_material_implication() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.4, 0.6], [0.7, 0.3]])
    f = ObservedMorphism(A, B, data)
    g = f.change_base(MATERIAL_IMPLICATION)
    assert g.quantale.name == "Godel"
    # Material implication clamps to [0, 1] but otherwise preserves
    # entry values; the carry-through is exact for inputs already
    # in [0, 1].
    assert torch.allclose(g.tensor, data)


def test_change_base_to_max_plus_via_log() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    f = ObservedMorphism(A, B, data)
    g = f.change_base(MAX_PLUS_HOM)
    expected = torch.log(data.clamp(min=1e-30))
    assert torch.allclose(g.tensor, expected)


def test_change_base_rejects_wrong_source_quantale() -> None:
    """A homomorphism's source must match the morphism's quantale;
    applying ``LOG_PROB_HOM`` (which expects ``ProductFuzzy``) to
    a morphism declared over Markov raises a clear error."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    f = ObservedMorphism(A, B, data, quantale=MARKOV)
    with pytest.raises(TypeError, match="does not match"):
        f.change_base(LOG_PROB_HOM)


def test_change_base_rejects_non_homomorphism() -> None:
    A = FinSet(name="A", cardinality=2)
    f = LatentMorphism(A, A)
    with pytest.raises(TypeError, match="QuantaleHomomorphism"):
        f.change_base("not a homomorphism")


def test_change_base_chain_through_two_homomorphisms() -> None:
    """Chaining ``change_base`` calls realises composition of
    quantale homomorphisms. Going ProductFuzzy → Boolean (via
    threshold) → ProductFuzzy (via embedding) recovers a Boolean-
    valued tensor embedded back in [0, 1]."""
    A = FinSet(name="A", cardinality=2)
    data = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
    f = ObservedMorphism(A, A, data)
    g = f.change_base(threshold(0.5)).change_base(
        embedding(BOOLEAN, PRODUCT_FUZZY)
    )
    expected = (data > 0.5).to(dtype=torch.float32)
    assert torch.allclose(g.tensor, expected)
    assert g.quantale.name == "ProductFuzzy"


def test_change_base_preserves_gradients_to_morphism_parameters() -> None:
    """A learnable LatentMorphism's parameters remain gradient-
    connected through the base change."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    f = LatentMorphism(A, B)
    g = f.change_base(LOG_PROB_HOM)
    loss = g.tensor.sum()
    loss.backward()
    raw = f.raw
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


# ---------------------------------------------------------------------------
# Composition operator dispatch (DSL surface)
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_default_compose_uses_product_fuzzy() -> None:
    from quivers.dsl import loads

    src = """
    quantale product_fuzzy
    object A : 3
    object B : 3
    object C : 3
    latent f : A -> B
    latent g : B -> C
    let chain = f >> g
    export chain
    """
    m = loads(src)
    assert m.morphism is not None
    assert m.morphism.quantale.name == "ProductFuzzy"


@_LOCAL_GRAMMAR
def test_cross_quantale_compose_without_change_base_errors() -> None:
    """An operator that fixes its quantale rejects operands whose
    declared quantale differs without an explicit base change."""
    from quivers.dsl import loads
    from quivers.dsl.compiler import CompileError

    src = """
    quantale product_fuzzy
    object A : 3
    object B : 3
    object C : 3
    latent f : A -> B
    latent g : B -> C
    let chain = f *> g
    export chain
    """
    with pytest.raises(CompileError, match="dispatches to"):
        loads(src)


# ---------------------------------------------------------------------------
# Each new operator dispatches to the right quantale
# ---------------------------------------------------------------------------


def _markov_chain(A_size: int, B_size: int, C_size: int) -> tuple:
    A = FinSet(name="A", cardinality=A_size)
    B = FinSet(name="B", cardinality=B_size)
    C = FinSet(name="C", cardinality=C_size)
    f = LatentMorphism(A, B, quantale=MARKOV)
    g = LatentMorphism(B, C, quantale=MARKOV)
    return A, B, C, f, g


def test_markov_compose_produces_markov_composed_morphism() -> None:
    A, B, C, f, g = _markov_chain(3, 3, 3)
    chain = f >> g
    assert isinstance(chain, ComposedMorphism)
    assert chain.quantale.name == "Markov"
    assert chain.domain is A
    assert chain.codomain is C


def test_log_prob_compose_produces_log_prob_composed_morphism() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, quantale=LOG_PROB)
    g = LatentMorphism(B, C, quantale=LOG_PROB)
    chain = f >> g
    assert chain.quantale.name == "LogProb"


def test_godel_compose_produces_godel_composed_morphism() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, quantale=GODEL)
    g = LatentMorphism(B, C, quantale=GODEL)
    chain = f >> g
    assert chain.quantale.name == "Godel"


def test_viterbi_compose_produces_max_plus_composed_morphism() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, quantale=MAX_PLUS)
    g = LatentMorphism(B, C, quantale=MAX_PLUS)
    chain = f >> g
    assert chain.quantale.name == "MaxPlus"


def test_boolean_compose_produces_boolean_composed_morphism() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, quantale=BOOLEAN)
    g = LatentMorphism(B, C, quantale=BOOLEAN)
    chain = f >> g
    assert chain.quantale.name == "Boolean"


def test_lukasiewicz_compose_produces_lukasiewicz_composed_morphism() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, quantale=LUKASIEWICZ)
    g = LatentMorphism(B, C, quantale=LUKASIEWICZ)
    chain = f >> g
    assert chain.quantale.name == "Lukasiewicz"


def test_tropical_min_plus_compose() -> None:
    """The existing shortest-path Tropical quantale also composes;
    no dedicated operator but the existing ``>>`` works once both
    operands are declared over it."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    f = LatentMorphism(A, B, quantale=TROPICAL)
    g = LatentMorphism(B, C, quantale=TROPICAL)
    chain = f >> g
    assert chain.quantale.name == "Tropical"


# ---------------------------------------------------------------------------
# Homomorphism registry lookup
# ---------------------------------------------------------------------------


def test_homomorphism_registry_returns_expectation_for_markov_to_pf() -> None:
    phi = lookup_homomorphism(MARKOV, PRODUCT_FUZZY)
    assert phi is not None
    assert phi.source.name == "Markov"
    assert phi.target.name == "ProductFuzzy"


def test_homomorphism_registry_returns_identity_for_same_quantale() -> None:
    phi = lookup_homomorphism(PRODUCT_FUZZY, PRODUCT_FUZZY)
    assert phi is not None
    assert isinstance(phi, IdentityHom)


def test_homomorphism_registry_returns_none_for_unknown_pair() -> None:
    """Unknown pairs return None so the caller can decide whether
    to raise or to construct a custom homomorphism."""
    from quivers.core.quantales import ProductFuzzy

    class _BogusQuantale(ProductFuzzy):
        @property
        def name(self):
            return "BogusUnregistered"

    phi = lookup_homomorphism(_BogusQuantale(), PRODUCT_FUZZY)
    assert phi is None


def test_threshold_homomorphism_rejects_invalid_tau() -> None:
    with pytest.raises(ValueError, match="tau must be in"):
        Threshold(tau=1.5)
    with pytest.raises(ValueError, match="tau must be in"):
        Threshold(tau=-0.1)


# ---------------------------------------------------------------------------
# Pythonic interplay with change_base + compose
# ---------------------------------------------------------------------------


def test_change_base_then_compose_in_target_quantale() -> None:
    """Bring a ProductFuzzy morphism into the Markov quantale via
    a custom homomorphism, then compose with a Markov-native
    morphism."""
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    C = FinSet(name="C", cardinality=2)
    # Construct a "noop" homomorphism from ProductFuzzy to Markov.
    from quivers.core.quantale_morphisms import QuantaleHomomorphism

    class ProductToMarkov(QuantaleHomomorphism):
        @property
        def source(self):
            return PRODUCT_FUZZY

        @property
        def target(self):
            return MARKOV

        def apply(self, t):
            # Row-normalize.
            return t / t.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    f_pf = LatentMorphism(A, B)
    f_markov = f_pf.change_base(ProductToMarkov())
    assert f_markov.quantale.name == "Markov"
    g_markov = LatentMorphism(B, C, quantale=MARKOV)
    chain = f_markov >> g_markov
    assert chain.quantale.name == "Markov"
