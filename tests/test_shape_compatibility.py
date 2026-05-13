"""Tests for storage-level shape compatibility on morphism inits
and ``Morphism.refactor`` for object-iso reshapes.

Covers:

* ``from_data`` / expression-derived inits whose flat tensor numel
  matches a declared product / factored codomain are accepted
  and reshaped automatically.
* ``Morphism.refactor`` switches between flat and product views
  while preserving the underlying tensor data.
* Domain / codomain numel mismatches still raise a clear error.
"""

from __future__ import annotations

import pytest
import torch

from quivers.core.morphisms import observed
from quivers.core.objects import FinSet, ProductSet


def test_refactor_flat_to_product() -> None:
    A = FinSet(name="A", cardinality=4)
    flat = FinSet(name="Flat", cardinality=12)
    B = FinSet(name="B", cardinality=3)
    C = FinSet(name="C", cardinality=4)
    prod = ProductSet(components=(B, C))
    tensor = torch.randn(4, 12)
    m = observed(A, flat, tensor)
    m2 = m.refactor(codomain=prod)
    assert m2.domain is A
    assert m2.codomain is prod
    assert tuple(m2.tensor.shape) == (4, 3, 4)
    # The data is the same — storage-level no-op.
    assert torch.allclose(m2.tensor.reshape(4, 12), tensor)


def test_refactor_product_to_flat() -> None:
    A = FinSet(name="A", cardinality=4)
    B = FinSet(name="B", cardinality=3)
    C = FinSet(name="C", cardinality=4)
    prod = ProductSet(components=(B, C))
    flat = FinSet(name="Flat", cardinality=12)
    tensor = torch.randn(4, 3, 4)
    m = observed(A, prod, tensor)
    m2 = m.refactor(codomain=flat)
    assert m2.codomain is flat
    assert tuple(m2.tensor.shape) == (4, 12)


def test_refactor_domain_and_codomain_together() -> None:
    A = FinSet(name="A", cardinality=6)
    B = FinSet(name="B", cardinality=8)
    A_prod = ProductSet(components=(FinSet(name="A1", cardinality=2), FinSet(name="A2", cardinality=3)))
    B_prod = ProductSet(components=(FinSet(name="B1", cardinality=2), FinSet(name="B2", cardinality=4)))
    tensor = torch.randn(6, 8)
    m = observed(A, B, tensor)
    m2 = m.refactor(domain=A_prod, codomain=B_prod)
    assert m2.domain is A_prod
    assert m2.codomain is B_prod
    assert tuple(m2.tensor.shape) == (2, 3, 2, 4)


def test_refactor_rejects_numel_mismatch() -> None:
    A = FinSet(name="A", cardinality=4)
    B = FinSet(name="B", cardinality=12)
    wrong = FinSet(name="Wrong", cardinality=10)
    m = observed(A, B, torch.randn(4, 12))
    with pytest.raises(ValueError, match="numel"):
        m.refactor(codomain=wrong)


def test_refactor_no_args_returns_equivalent() -> None:
    """No-op refactor is identity (different ObservedMorphism instance
    but identical content)."""
    A = FinSet(name="A", cardinality=3)
    B = FinSet(name="B", cardinality=5)
    tensor = torch.randn(3, 5)
    m = observed(A, B, tensor)
    m2 = m.refactor()
    assert m2.domain is A
    assert m2.codomain is B
    assert torch.allclose(m2.tensor, tensor)
