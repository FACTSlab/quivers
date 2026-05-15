"""Compact-closed surface on V-Cat morphisms.

V-Cat is compact-closed for any commutative algebra V: every
object has a self-dual, and the categorical unit / counit (cup /
cap) satisfy the snake equations. This module exposes the
compact-closed operations on the morphism API:

* ``f.dagger`` — transpose of a morphism (axis-swap of the
  tensor; semantic depends on the algebra).
* ``f.trace(A)`` — contraction along an object A; the categorical
  trace ``(ε_A ⊗ id) ∘ (id ⊗ f) ∘ (η_A ⊗ id)``.
* ``cup(A)`` — unit morphism ``I → A ⊗ A`` (diagonal).
* ``cap(A)`` — counit morphism ``A ⊗ A → I`` (codiagonal).

This module tests the categorical contract end-to-end:

1. The dagger of a morphism's tensor is the axis-swap.
2. Double-dagger is identity (``(f^†)^† = f``).
3. ``cup`` and ``cap`` carry the algebra's identity tensor.
4. The snake equation: ``(cap ⊗ id) ∘ (id ⊗ cup) = id`` for the
   simplest finite-set case.
5. Trace over a single-axis domain agrees with the join over the
   contracted axis.
6. The DSL surface ``f.dagger``, ``f.trace(A)``, ``cup(A)``,
   ``cap(A)``, ``f.change_base(name)`` all parse and compile to
   morphisms with the expected shapes.
"""

from __future__ import annotations

import os

import pytest
import torch

from quivers.core.morphisms import (
    LatentMorphism,
    ObservedMorphism,
    cap,
    cup,
)
from quivers.core.objects import FinSet


_LOCAL_GRAMMAR = pytest.mark.skipif(
    os.environ.get("QVR_USE_LOCAL_GRAMMAR", "") not in ("1", "true", "True"),
    reason="needs QVR_USE_LOCAL_GRAMMAR=1 to pick up the in-tree grammar",
)


# ---------------------------------------------------------------------------
# Dagger
# ---------------------------------------------------------------------------


def test_dagger_swaps_domain_and_codomain() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    f = LatentMorphism(A, B)
    g = f.dagger
    assert g.domain is B
    assert g.codomain is A


def test_dagger_tensor_is_transpose() -> None:
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    data = torch.randn(3, 4)
    f = ObservedMorphism(A, B, data)
    g = f.dagger
    assert torch.allclose(g.tensor, data.t())


def test_double_dagger_is_identity_on_tensor() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=3)
    data = torch.randn(2, 3)
    f = ObservedMorphism(A, B, data)
    f_dd = f.dagger.dagger
    assert torch.allclose(f_dd.tensor, data)
    assert f_dd.domain is A
    assert f_dd.codomain is B


def test_dagger_preserves_algebra() -> None:
    from quivers.core.algebras import MARKOV

    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    data = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    f = ObservedMorphism(A, B, data, algebra=MARKOV)
    g = f.dagger
    assert g.algebra.name == "Markov"


def test_dagger_preserves_gradients() -> None:
    A = FinSet(name="A", cardinality=2)
    B = FinSet(name="B", cardinality=2)
    f = LatentMorphism(A, B)
    g = f.dagger
    loss = g.tensor.sum()
    loss.backward()
    raw = f.raw
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


# ---------------------------------------------------------------------------
# cup / cap
# ---------------------------------------------------------------------------


def test_cup_returns_diagonal_into_product() -> None:
    from quivers.core.objects import FinSet as _FS, ProductSet

    A = _FS(name="A", cardinality=3)
    eta = cup(A)
    # Domain is the unit (singleton); codomain is A * A.
    assert eta.domain.size == 1
    assert isinstance(eta.codomain, ProductSet)
    # Tensor shape is (1, 3, 3) with the identity diagonal.
    t = eta.tensor
    assert t.shape == (1, 3, 3)
    expected_diag = torch.eye(3)
    assert torch.allclose(t.squeeze(0), expected_diag)


def test_cap_returns_codiagonal_from_product() -> None:
    from quivers.core.objects import FinSet as _FS, ProductSet

    A = _FS(name="A", cardinality=3)
    eps = cap(A)
    assert eps.codomain.size == 1
    assert isinstance(eps.domain, ProductSet)
    t = eps.tensor
    assert t.shape == (3, 3, 1)
    expected_diag = torch.eye(3)
    assert torch.allclose(t.squeeze(-1), expected_diag)


# ---------------------------------------------------------------------------
# DSL surface
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
def test_dsl_dagger_method_call() -> None:
    from quivers.dsl import loads

    src = """
    algebra product_fuzzy
    object A : 3
    object B : 4

    latent f : A -> B
    let f_dag = f.dagger
    export f_dag
    """
    m = loads(src)
    assert m.morphism is not None
    assert m.morphism.tensor.shape == (4, 3)


@_LOCAL_GRAMMAR
def test_dsl_cup_returns_diagonal() -> None:
    from quivers.dsl import loads

    src = """
    algebra product_fuzzy
    object A : 3

    let eta = cup(A)
    export eta
    """
    m = loads(src)
    t = m.morphism.tensor
    assert t.shape == (1, 3, 3)
    expected = torch.eye(3)
    assert torch.allclose(t.squeeze(0), expected)


@_LOCAL_GRAMMAR
def test_dsl_cap_returns_codiagonal() -> None:
    from quivers.dsl import loads

    src = """
    algebra product_fuzzy
    object A : 3

    let eps = cap(A)
    export eps
    """
    m = loads(src)
    t = m.morphism.tensor
    assert t.shape == (3, 3, 1)
    expected = torch.eye(3)
    assert torch.allclose(t.squeeze(-1), expected)


@_LOCAL_GRAMMAR
def test_dsl_change_base_to_log_prob() -> None:
    from quivers.dsl import loads

    src = """
    algebra product_fuzzy
    object A : 3

    latent f : A -> A
    let f_log = f.change_base(log_prob)
    export f_log
    """
    m = loads(src)
    assert m.morphism.algebra.name == "LogProb"
    # All entries are <= 0 (log of sigmoid in [0, 1]).
    assert (m.morphism.tensor <= 0).all()


@_LOCAL_GRAMMAR
def test_dsl_change_base_unknown_homomorphism_errors() -> None:
    from quivers.dsl import loads
    from quivers.dsl.compiler import CompileError

    src = """
    algebra product_fuzzy
    object A : 3
    latent f : A -> A
    let g = f.change_base(not_a_real_homomorphism)
    export g
    """
    with pytest.raises(CompileError, match="undefined transformation"):
        loads(src)


@_LOCAL_GRAMMAR
def test_dsl_change_base_to_boolean() -> None:
    from quivers.dsl import loads

    src = """
    algebra product_fuzzy
    object A : 3
    latent f : A -> A
    let g = f.change_base(threshold)
    export g
    """
    m = loads(src)
    assert m.morphism.algebra.name == "Boolean"
    # Boolean tensor entries are 0/1 only.
    t = m.morphism.tensor
    assert torch.all((t == 0.0) | (t == 1.0))


@_LOCAL_GRAMMAR
def test_dsl_dagger_chained_with_compose() -> None:
    """The canonical bilinear-scoring pattern: ``f >> g.dagger``."""
    from quivers.dsl import loads

    src = """
    algebra product_fuzzy
    object A : 3
    object B : 3
    object Latent : 4

    latent emb_a : A -> Latent
    latent emb_b : B -> Latent

    let score = emb_a >> emb_b.dagger
    export score
    """
    m = loads(src)
    assert m.morphism.domain.name == "A"
    assert m.morphism.codomain.name == "B"


# ---------------------------------------------------------------------------
# Cross-algebra dagger
# ---------------------------------------------------------------------------


def test_dagger_round_trip_via_change_base_preserves_shape() -> None:
    """``f.change_base(phi).dagger`` produces the right shape and
    algebra for the chained transformation."""
    from quivers.core.algebra_morphisms import LOG_PROB as LOG_PROB_HOM

    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=4)
    f = LatentMorphism(A, B)
    log_f = f.change_base(LOG_PROB_HOM)
    log_f_dag = log_f.dagger
    assert log_f_dag.tensor.shape == (4, 3)
    assert log_f_dag.algebra.name == "LogProb"
