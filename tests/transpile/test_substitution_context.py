"""Context-aware substitution tests for
[`_substitute_let_expr`][quivers.transpile.renderers._stan_helpers._substitute_let_expr].

The helper distinguishes two replacement slots: the `index_value`
substituted into the `indices` tuple of every
[`LetExprIndex`][quivers.dsl.ast_nodes.LetExprIndex] (the
``[ ]``-bracketed positions), and the `scalar_value` substituted
everywhere else. Stan needs the two values to differ by one
(Stan arrays are 1-based, QVR arithmetic on factor binders is
0-based); Python / NumPyro / Pyro / PyMC / Edward2 / JavaScript /
WebPPL pass the same value for both slots because their host
arrays are already 0-based.

These tests exercise the helper directly at the AST level rather
than through a full renderer; the substituted tree is structurally
equal to the hand-built expected tree.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    LetExprBinOp,
    LetExprCall,
    LetExprIndex,
    LetExprLiteral,
    LetExprNode,
    LetExprUnaryOp,
    LetExprVar,
)
from quivers.transpile.renderers._stan_helpers import _substitute_let_expr


def _stan_substitute(expr: LetExprNode, name: str, value: int) -> LetExprNode:
    """Stan call-site convention: 1-indexed in index slots, 0-indexed
    in scalar slots (matches `_build_nested_array` in `_stan_helpers`)."""
    return _substitute_let_expr(
        expr,
        name,
        index_value=LetExprLiteral(value=value + 1),
        scalar_value=LetExprLiteral(value=value),
    )


def _python_substitute(
    expr: LetExprNode, name: str, value: int
) -> LetExprNode:
    """Python / NumPyro / Pyro / PyMC / Edward2 call-site convention:
    same 0-indexed value in both slots (matches `_build_nested_python`
    in `_python_helpers` and `_build_nested_array` in
    `_javascript_helpers`)."""
    literal = LetExprLiteral(value=value)
    return _substitute_let_expr(
        expr,
        name,
        index_value=literal,
        scalar_value=literal,
    )


def test_index_slot_only_stan_picks_one_based() -> None:
    """`delta[v]` with `v=0` substitutes to `delta[1]` in Stan (the
    binder appears in an index slot only, so the Stan call site's
    1-indexed `index_value` wins)."""
    expr = LetExprIndex(
        array=LetExprVar(name="delta"),
        indices=(LetExprVar(name="v"),),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprIndex)
    assert isinstance(substituted.array, LetExprVar)
    assert substituted.array.name == "delta"
    assert len(substituted.indices) == 1
    idx = substituted.indices[0]
    assert isinstance(idx, LetExprLiteral)
    assert idx.value == 1


def test_index_slot_only_python_picks_zero_based() -> None:
    """`delta[v]` with `v=0` substitutes to `delta[0]` in Python /
    NumPyro / Pyro / PyMC / Edward2 (the binder appears in an index
    slot but the call site supplies the 0-indexed value for both
    slots, since the host arrays are already 0-based)."""
    expr = LetExprIndex(
        array=LetExprVar(name="delta"),
        indices=(LetExprVar(name="v"),),
    )
    substituted = _python_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprIndex)
    assert len(substituted.indices) == 1
    idx = substituted.indices[0]
    assert isinstance(idx, LetExprLiteral)
    assert idx.value == 0


def test_scalar_slot_only_stan_picks_zero_based() -> None:
    """`2 * v + 1` with `v=0` substitutes to `2 * 0 + 1` in Stan
    (every binder occurrence is in scalar position, so the
    0-indexed `scalar_value` wins and the expression evaluates to
    `1` rather than the `3` a context-blind substitution would
    produce)."""
    expr = LetExprBinOp(
        op="+",
        left=LetExprBinOp(
            op="*",
            left=LetExprLiteral(value=2),
            right=LetExprVar(name="v"),
        ),
        right=LetExprLiteral(value=1),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprBinOp)
    assert substituted.op == "+"
    left = substituted.left
    assert isinstance(left, LetExprBinOp)
    assert left.op == "*"
    assert isinstance(left.left, LetExprLiteral)
    assert left.left.value == 2
    assert isinstance(left.right, LetExprLiteral)
    assert left.right.value == 0
    assert isinstance(substituted.right, LetExprLiteral)
    assert substituted.right.value == 1


def test_mixed_slots_in_one_expression_stan() -> None:
    """`delta[v] * v` with `v=0` substitutes to `delta[1] * 0` in
    Stan: the binder occurrence under `LetExprIndex.indices` uses
    the 1-indexed `index_value`, the bare occurrence on the right
    of the multiply uses the 0-indexed `scalar_value`."""
    expr = LetExprBinOp(
        op="*",
        left=LetExprIndex(
            array=LetExprVar(name="delta"),
            indices=(LetExprVar(name="v"),),
        ),
        right=LetExprVar(name="v"),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprBinOp)
    assert substituted.op == "*"
    left = substituted.left
    assert isinstance(left, LetExprIndex)
    assert isinstance(left.array, LetExprVar)
    assert left.array.name == "delta"
    idx = left.indices[0]
    assert isinstance(idx, LetExprLiteral)
    assert idx.value == 1
    right = substituted.right
    assert isinstance(right, LetExprLiteral)
    assert right.value == 0


def test_mixed_slots_in_one_expression_python() -> None:
    """`delta[v] * v` with `v=0` substitutes to `delta[0] * 0` in
    Python: both occurrences resolve to 0 because the call site
    passes the same `LetExprLiteral(value=0)` for both slots."""
    expr = LetExprBinOp(
        op="*",
        left=LetExprIndex(
            array=LetExprVar(name="delta"),
            indices=(LetExprVar(name="v"),),
        ),
        right=LetExprVar(name="v"),
    )
    substituted = _python_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprBinOp)
    left = substituted.left
    assert isinstance(left, LetExprIndex)
    idx = left.indices[0]
    assert isinstance(idx, LetExprLiteral)
    assert idx.value == 0
    right = substituted.right
    assert isinstance(right, LetExprLiteral)
    assert right.value == 0


def test_array_child_of_indexed_is_scalar_slot_stan() -> None:
    """The `array` child of a `LetExprIndex` is a scalar slot, not
    an index slot. A pathological body `v[v]` with `v=0` substitutes
    to `0[1]` in Stan: the outer `v` is the array being indexed
    (scalar), the inner `v` is the index (1-based)."""
    expr = LetExprIndex(
        array=LetExprVar(name="v"),
        indices=(LetExprVar(name="v"),),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprIndex)
    array = substituted.array
    assert isinstance(array, LetExprLiteral)
    assert array.value == 0
    idx = substituted.indices[0]
    assert isinstance(idx, LetExprLiteral)
    assert idx.value == 1


def test_nested_indices_inner_index_is_index_slot_stan() -> None:
    """`arr[subj[v]]` with `v=0` substitutes to `arr[subj[1]]` in
    Stan. The inner `LetExprIndex` lives inside the outer node's
    `indices` tuple, but the helper resets `in_index_slot` on
    descent into the inner's `array` child and re-sets it on
    descent into the inner's `indices` tuple. The bare `v` ends
    up in `subj[...]`'s index slot and therefore picks the
    1-indexed value."""
    expr = LetExprIndex(
        array=LetExprVar(name="arr"),
        indices=(
            LetExprIndex(
                array=LetExprVar(name="subj"),
                indices=(LetExprVar(name="v"),),
            ),
        ),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprIndex)
    inner = substituted.indices[0]
    assert isinstance(inner, LetExprIndex)
    assert isinstance(inner.array, LetExprVar)
    assert inner.array.name == "subj"
    inner_idx = inner.indices[0]
    assert isinstance(inner_idx, LetExprLiteral)
    assert inner_idx.value == 1


def test_unrelated_var_unchanged() -> None:
    """A `LetExprVar` whose name differs from `name` passes through
    untouched regardless of slot context."""
    expr = LetExprIndex(
        array=LetExprVar(name="arr"),
        indices=(LetExprVar(name="k"),),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprIndex)
    assert isinstance(substituted.array, LetExprVar)
    assert substituted.array.name == "arr"
    idx = substituted.indices[0]
    assert isinstance(idx, LetExprVar)
    assert idx.name == "k"


def test_unary_operand_is_scalar_slot() -> None:
    """The operand of a `LetExprUnaryOp` is a scalar slot; `-v`
    with `v=0` becomes `-0` under the Stan call-site convention."""
    expr = LetExprUnaryOp(operand=LetExprVar(name="v"))
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprUnaryOp)
    operand = substituted.operand
    assert isinstance(operand, LetExprLiteral)
    assert operand.value == 0


def test_call_args_are_scalar_slots() -> None:
    """Arguments to a `LetExprCall` are scalar slots; `log(v + 1)`
    with `v=0` becomes `log(0 + 1)` under the Stan call-site
    convention."""
    expr = LetExprCall(
        func="log",
        args=(
            LetExprBinOp(
                op="+",
                left=LetExprVar(name="v"),
                right=LetExprLiteral(value=1),
            ),
        ),
    )
    substituted = _stan_substitute(expr, "v", 0)
    assert isinstance(substituted, LetExprCall)
    assert substituted.func == "log"
    inner = substituted.args[0]
    assert isinstance(inner, LetExprBinOp)
    assert isinstance(inner.left, LetExprLiteral)
    assert inner.left.value == 0
