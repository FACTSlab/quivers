"""Shared in-container reshape helper for probe scripts.

The host harness sends each (param, data) value as a flat
row-major Python list under `points.json`. The model the probe
loads, however, typically declares the same name with a multi-
dimensional declared shape (e.g. Stan `array[16] int obs` or PyMC
`obs.shape == (16, 4)`). Each probe script imports
:func:`reshape_point` and rebuilds the multi-dimensional native
container the underlying runtime expects.

The reshape table arrives in `/io/shapes.json`:

    {"obs": [16], "rule_weights": [16], "logits": [3, 4]}

The dtype table (optional) arrives in `/io/dtypes.json`:

    {"obs": "int", "logits": "float"}

A name absent from either table falls through to its raw
list value. The float / int cast applies even for scalars so
backends that distinguish `int` and `real` (Stan, JAGS) get the
right type.
"""
import json
import pathlib
from collections.abc import Sequence
from typing import cast

# The probe payload is strictly numeric: scalars (Python int or
# float) and arbitrarily nested lists of the same. PEP 695 `type`
# statements keep the recursion declarative; `Any` and `object` are
# intentionally avoided.
type Number = int | float
type NestedNumber = Number | list[NestedNumber]
type PointSection = dict[str, NestedNumber]
type Point = dict[str, PointSection]


def load_tables(io: pathlib.Path) -> tuple[
    dict[str, list[int]], dict[str, str],
]:
    """Read `/io/shapes.json` and `/io/dtypes.json`. Each absent
    file resolves to an empty dict so the helper is a no-op in
    legacy probe runs that don't ship the tables."""
    shapes_path = io / "shapes.json"
    dtypes_path = io / "dtypes.json"
    shapes = (
        json.loads(shapes_path.read_text())
        if shapes_path.exists() else {}
    )
    dtypes = (
        json.loads(dtypes_path.read_text())
        if dtypes_path.exists() else {}
    )
    return shapes, dtypes


def _flat_to_nested(
    flat: Sequence[Number], shape: list[int],
) -> NestedNumber:
    """Reshape a flat row-major list into a nested list with the
    given shape. Returns the scalar element when `shape == []`."""
    if not shape:
        if len(flat) != 1:
            raise ValueError(
                f"scalar shape but len(flat)={len(flat)}"
            )
        return flat[0]
    expected = 1
    for d in shape:
        expected *= d
    if len(flat) != expected:
        raise ValueError(
            f"flat length {len(flat)} does not match shape "
            f"{shape} (expected {expected})"
        )
    if len(shape) == 1:
        return list(flat)
    stride = expected // shape[0]
    return [
        _flat_to_nested(flat[i * stride:(i + 1) * stride], shape[1:])
        for i in range(shape[0])
    ]


def reshape_value(
    name: str,
    value: NestedNumber,
    shapes: dict[str, list[int]],
    dtypes: dict[str, str],
) -> NestedNumber:
    """Reshape one (name, value) pair using the loaded tables.

    - When `name` has a shape entry, treat `value` as flat row-major
      and rebuild a nested list of the declared shape (scalar shape
      `[]` collapses a length-1 list to its element).
    - When `name` has a dtype entry, recursively cast every leaf to
      `int` or `float`.
    - When neither table mentions `name`, return the value unchanged
      (preserves the existing 7-fixture probe contract).
    """
    if name in shapes:
        if isinstance(value, (int, float)):
            value = _flat_to_nested([value], shapes[name])
        elif isinstance(value, list):
            value = _flat_to_nested(
                cast(Sequence[Number], value), shapes[name],
            )
    if name in dtypes:
        value = _cast_leaves(value, dtypes[name])
    return value


def _cast_leaves(value: NestedNumber, dtype: str) -> NestedNumber:
    """Recursively cast every leaf of a nested list to the target
    dtype (`"int"` -> Python int, anything else -> float)."""
    if isinstance(value, list):
        return [_cast_leaves(v, dtype) for v in value]
    if dtype == "int":
        return int(value)
    return float(value)


def reshape_point(
    point: Point,
    shapes: dict[str, list[int]],
    dtypes: dict[str, str],
) -> Point:
    """Reshape every entry under `params` / `data` in one point."""
    out: Point = {}
    for section in ("params", "data"):
        out[section] = {
            name: reshape_value(name, value, shapes, dtypes)
            for name, value in point.get(section, {}).items()
        }
    return out


__all__ = [
    "load_tables",
    "reshape_point",
    "reshape_value",
    "NestedNumber",
    "Number",
    "Point",
    "PointSection",
]
