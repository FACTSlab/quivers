"""Tests for ``Quantale.dual()`` and ``CustomQuantale``.

Verifies:

* Every shipped quantale exposes a ``.dual()`` returning a usable
  quantale instance.
* The dual swaps the roles of ``tensor_op`` and ``join`` under the
  de-Morgan involution; ``dual.dual()`` recovers the original
  semantics.
* Named singletons ``REICHENBACH``, ``BOOLEAN_DUAL``,
  ``DUAL_LUKASIEWICZ``, ``DUAL_GODEL`` match the algebraic
  expectations.
* :class:`CustomQuantale` accepts user-supplied operations and
  catches obviously-wrong axiom violations at construction time.
"""

from __future__ import annotations

import pytest
import torch

from quivers.core.quantales import (
    BOOLEAN,
    BOOLEAN_DUAL,
    DUAL_GODEL,
    DUAL_LUKASIEWICZ,
    GODEL,
    LUKASIEWICZ,
    PRODUCT_FUZZY,
    REICHENBACH,
    CustomQuantale,
    DualQuantale,
)


_SAMPLES_3 = torch.tensor([0.3, 0.5, 0.8])
_SAMPLES_3_B = torch.tensor([0.4, 0.6, 0.9])


# ---------------------------------------------------------------------------
# Reichenbach (ProductFuzzy.dual)
# ---------------------------------------------------------------------------


def test_reichenbach_is_dual_of_product_fuzzy() -> None:
    """``REICHENBACH.tensor_op`` is the noisy-OR (base join)."""
    a, b = _SAMPLES_3, _SAMPLES_3_B
    actual = REICHENBACH.tensor_op(a, b)
    # Noisy-OR: a + b - a*b.
    expected = a + b - a * b
    assert torch.allclose(actual, expected, atol=1e-6)


def test_reichenbach_join_is_product_reduction() -> None:
    """``REICHENBACH.join`` reduces with the base tensor_op (product)."""
    t = torch.stack([_SAMPLES_3, _SAMPLES_3_B], dim=-1)
    actual = REICHENBACH.join(t, dim=-1)
    expected = _SAMPLES_3 * _SAMPLES_3_B
    assert torch.allclose(actual, expected, atol=1e-6)


def test_reichenbach_unit_and_zero_swap() -> None:
    """For ProductFuzzy ``(unit=1, zero=0)``; the dual swaps them."""
    assert REICHENBACH.unit == 0.0
    assert REICHENBACH.zero == 1.0


def test_dual_of_dual_recovers_base() -> None:
    """Double-dualizing returns the original base quantale."""
    back = REICHENBACH.dual()
    assert back is PRODUCT_FUZZY


# ---------------------------------------------------------------------------
# Boolean.dual = (OR, AND)
# ---------------------------------------------------------------------------


def test_boolean_dual_tensor_op_is_or() -> None:
    """Dual of Boolean has ``tensor_op = max = OR``."""
    a = torch.tensor([0.0, 0.0, 1.0, 1.0])
    b = torch.tensor([0.0, 1.0, 0.0, 1.0])
    actual = BOOLEAN_DUAL.tensor_op(a, b)
    expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
    assert torch.allclose(actual, expected)


def test_boolean_dual_join_is_and() -> None:
    """Dual Boolean ``join`` reduces with the base tensor_op (AND)."""
    t = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    actual = BOOLEAN_DUAL.join(t, dim=-1)
    expected = torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# Łukasiewicz / Gödel duals
# ---------------------------------------------------------------------------


def test_dual_lukasiewicz_tensor_op_is_bounded_sum() -> None:
    """Łukasiewicz t-conorm: ``a + b`` clamped to 1."""
    a, b = _SAMPLES_3, _SAMPLES_3_B
    actual = DUAL_LUKASIEWICZ.tensor_op(a, b)
    expected = torch.minimum(a + b, torch.ones_like(a))
    assert torch.allclose(actual, expected, atol=1e-6)


def test_dual_godel_tensor_op_is_max() -> None:
    """Gödel t-conorm: ``max(a, b)``."""
    a, b = _SAMPLES_3, _SAMPLES_3_B
    actual = DUAL_GODEL.tensor_op(a, b)
    expected = torch.maximum(a, b)
    assert torch.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# DualQuantale generic behavior
# ---------------------------------------------------------------------------


def test_dual_is_dual_quantale_instance() -> None:
    """Calling ``base.dual()`` returns a ``DualQuantale`` instance."""
    for base in (PRODUCT_FUZZY, BOOLEAN, LUKASIEWICZ, GODEL):
        d = base.dual()
        assert isinstance(d, DualQuantale)
        assert d.base is base


def test_dual_name_records_base() -> None:
    """``Dual(Foo)`` carries the base name through the ``name`` property."""
    d = PRODUCT_FUZZY.dual()
    assert d.name == "Dual(ProductFuzzy)"


def test_dual_is_compatible_with_itself() -> None:
    """Two dual instances over the same base compose."""
    d1 = PRODUCT_FUZZY.dual()
    d2 = PRODUCT_FUZZY.dual()
    assert d1.is_compatible(d2)


# ---------------------------------------------------------------------------
# CustomQuantale
# ---------------------------------------------------------------------------


def _bounded_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a + b, torch.ones_like(a))


def _bounded_sum_reduce(t: torch.Tensor, dim) -> torch.Tensor:
    if isinstance(dim, int):
        dim = (dim,)
    s = t.sum(dim=dim)
    return torch.minimum(s, torch.ones_like(s))


def test_custom_quantale_round_trip() -> None:
    """``CustomQuantale`` constructs with user-supplied callables and
    proxies them through ``tensor_op`` / ``join``."""
    q = CustomQuantale(
        name="bounded_sum",
        tensor_op=_bounded_sum,
        join=_bounded_sum_reduce,
        unit=0.0,
        zero=1.0,
        negate=lambda t: 1.0 - t,
    )
    a, b = _SAMPLES_3, _SAMPLES_3_B
    assert torch.allclose(q.tensor_op(a, b), _bounded_sum(a, b))


def test_custom_quantale_meet_raises_without_spec() -> None:
    """``meet`` is None unless the user supplies it."""
    q = CustomQuantale(
        name="bs",
        tensor_op=_bounded_sum,
        join=_bounded_sum_reduce,
        unit=0.0,
        zero=1.0,
    )
    with pytest.raises(NotImplementedError):
        q.meet(_SAMPLES_3, dim=0)


def test_custom_quantale_negate_raises_without_spec() -> None:
    """``negate`` is None unless the user supplies it."""
    q = CustomQuantale(
        name="bs",
        tensor_op=_bounded_sum,
        join=_bounded_sum_reduce,
        unit=0.0,
        zero=1.0,
    )
    with pytest.raises(NotImplementedError):
        q.negate(_SAMPLES_3)


def test_custom_quantale_rejects_bad_unit() -> None:
    """A user who passes a wrong ``unit`` value triggers the sanity
    check at construction time."""

    def bad_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b  # product → unit should be 1.0

    with pytest.raises(ValueError, match="left-identity"):
        CustomQuantale(
            name="bad",
            tensor_op=bad_op,
            join=lambda t, dim: t.sum(dim=dim),
            unit=2.0,  # wrong
            zero=0.0,
        )


def test_custom_quantale_verify_off_skips_check() -> None:
    """Setting ``verify=False`` bypasses the construction-time
    axiom check."""
    q = CustomQuantale(
        name="unchecked",
        tensor_op=lambda a, b: a * b,
        join=lambda t, dim: t.sum(dim=dim),
        unit=42.0,
        zero=0.0,
        verify=False,
    )
    assert q.name == "unchecked"


def test_custom_quantale_blank_name_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CustomQuantale(
            name="",
            tensor_op=lambda a, b: a * b,
            join=lambda t, dim: t.sum(dim=dim),
            unit=1.0,
            zero=0.0,
        )


# ---------------------------------------------------------------------------
# DSL surface for named duals
# ---------------------------------------------------------------------------


def test_dsl_quantale_registry_includes_reichenbach() -> None:
    """``quantale reichenbach`` resolves to the
    ``ProductFuzzy.dual`` singleton via the compiler registry."""
    from quivers.dsl.compiler import _QUANTALE_REGISTRY, _register_extra_quantales

    _register_extra_quantales()
    assert "reichenbach" in _QUANTALE_REGISTRY
    assert _QUANTALE_REGISTRY["reichenbach"] is REICHENBACH


def test_dsl_quantale_registry_includes_named_duals() -> None:
    from quivers.dsl.compiler import _QUANTALE_REGISTRY, _register_extra_quantales

    _register_extra_quantales()
    for key in ("reichenbach", "boolean_dual", "dual_lukasiewicz", "dual_godel"):
        assert key in _QUANTALE_REGISTRY, f"DSL quantale registry missing {key!r}"
