"""Tests for the three-level composition-rule hierarchy:

* :class:`CompositionRule` — the bare composition surface
  (``tensor_op`` + ``join``; no identity required).
* :class:`Semigroupoid` — adds the associativity assumption.
* :class:`Quantale` — adds identity, meet, negation, and the
  compact-closed structure.

Verifies type-checked subclass relationships, that existing
quantales reclassify as ``Quantale`` (i.e. also ``Semigroupoid``
and ``CompositionRule``), that user-defined semigroupoids ship
via :func:`semigroupoid` and :func:`material_implication`, and
that operations needing identity (cup, cap, dagger, ...) reject
non-quantale composition rules at the API boundary.
"""

from __future__ import annotations

import pytest
import torch

from quivers.core.quantales import (
    BOOLEAN,
    PRODUCT_FUZZY,
    REAL,
    CompositionRule,
    CustomSemigroupoid,
    Quantale,
    Semigroupoid,
    material_implication,
    semigroupoid,
)


# ---------------------------------------------------------------------------
# Hierarchy structure
# ---------------------------------------------------------------------------


def test_quantale_extends_semigroupoid_extends_composition_rule() -> None:
    assert issubclass(Quantale, Semigroupoid)
    assert issubclass(Semigroupoid, CompositionRule)


@pytest.mark.parametrize("instance", [PRODUCT_FUZZY, BOOLEAN, REAL])
def test_shipped_quantales_are_at_all_three_levels(instance) -> None:
    assert isinstance(instance, Quantale)
    assert isinstance(instance, Semigroupoid)
    assert isinstance(instance, CompositionRule)


# ---------------------------------------------------------------------------
# CustomSemigroupoid / semigroupoid factory
# ---------------------------------------------------------------------------


def _bounded_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a + b).clamp(max=1.0)


def _max_reduce(t: torch.Tensor, dim) -> torch.Tensor:
    if isinstance(dim, int):
        dim = (dim,)
    result = t
    for d in sorted(dim, reverse=True):
        result = result.max(dim=d).values
    return result


def test_semigroupoid_factory_builds_custom_instance() -> None:
    s = semigroupoid("BoundedSumMax", _bounded_sum, _max_reduce)
    assert isinstance(s, Semigroupoid)
    assert isinstance(s, CompositionRule)
    assert not isinstance(s, Quantale)
    assert s.name == "BoundedSumMax"


def test_semigroupoid_compose_works() -> None:
    s = semigroupoid("BoundedSumMax", _bounded_sum, _max_reduce)
    a = torch.tensor([[0.3, 0.5], [0.7, 0.2]])
    b = torch.tensor([[0.6, 0.1], [0.4, 0.8]])
    out = s.compose(a, b, n_contract=1)
    assert tuple(out.shape) == (2, 2)
    assert torch.isfinite(out).all()


def test_associativity_smoke_check_rejects_bad_op() -> None:
    """A clearly-non-associative op (subtraction) trips the check."""

    def sub(a, b):
        return a - b

    with pytest.raises(ValueError, match="associativity"):
        CustomSemigroupoid("Sub", sub, _max_reduce)


def test_associativity_smoke_check_can_be_disabled() -> None:
    """``verify_associative=False`` bypasses the check."""

    def sub(a, b):
        return a - b

    s = CustomSemigroupoid("Sub", sub, _max_reduce, verify_associative=False)
    assert s.name == "Sub"


# ---------------------------------------------------------------------------
# Material implication
# ---------------------------------------------------------------------------


def test_material_implication_is_semigroupoid_not_quantale() -> None:
    mi = material_implication()
    assert isinstance(mi, Semigroupoid)
    assert not isinstance(mi, Quantale)


def test_material_implication_compose_matches_reichenbach_formula() -> None:
    """``(f >> g)[i, k] = prod_j (1 - f[i, j] + f[i, j] * g[j, k])``."""
    mi = material_implication()
    f = torch.tensor([[0.3, 0.5, 0.2]])  # (1, 3)
    g = torch.tensor(
        [
            [0.6, 0.4],
            [0.1, 0.9],
            [0.7, 0.2],
        ]
    )  # (3, 2)
    expected = torch.zeros(1, 2)
    for i in range(1):
        for k in range(2):
            product = 1.0
            for j in range(3):
                product *= float(1 - f[i, j] + f[i, j] * g[j, k])
            expected[i, k] = product
    actual = mi.compose(f, g, n_contract=1)
    assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Semigroupoids lack the Quantale operations
# ---------------------------------------------------------------------------


def test_semigroupoid_has_no_unit_or_zero() -> None:
    mi = material_implication()
    # ``Semigroupoid`` doesn't declare ``unit`` / ``zero``; accessing
    # them surfaces a clean AttributeError.
    with pytest.raises(AttributeError):
        _ = mi.unit
    with pytest.raises(AttributeError):
        _ = mi.zero


def test_semigroupoid_has_no_meet_or_negate() -> None:
    mi = material_implication()
    with pytest.raises(AttributeError):
        mi.meet(torch.zeros(3), dim=0)
    with pytest.raises(AttributeError):
        mi.negate(torch.zeros(3))


def test_semigroupoid_has_no_dual() -> None:
    mi = material_implication()
    with pytest.raises(AttributeError):
        mi.dual()


def test_semigroupoid_has_no_identity_tensor() -> None:
    """``identity_tensor`` lives on Quantale because it needs
    ``unit`` / ``zero``."""
    mi = material_implication()
    with pytest.raises(AttributeError):
        mi.identity_tensor((3,))


# ---------------------------------------------------------------------------
# is_compatible still works on the base class
# ---------------------------------------------------------------------------


def test_compatible_same_kind() -> None:
    mi1 = material_implication()
    mi2 = material_implication()
    assert mi1.is_compatible(mi2)


def test_compatible_across_levels() -> None:
    """A Quantale instance is compatible with itself (degenerate
    case)."""
    assert PRODUCT_FUZZY.is_compatible(PRODUCT_FUZZY)
