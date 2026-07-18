"""Reading adapter over the DSL `DrawArg` AST for the transpile.

The DSL models a draw-step argument with a recursive `DrawArg`
tagged union: an atom is a `DrawArgScalar` (number), a `DrawArgName`
(identifier), or a `DrawArgIndex` (bracket-indexed reference); a
vector literal ``[a, b]`` is a `DrawArgList` whose ``items`` are
atoms; a matrix literal ``[[a, b], [c, d]]`` is a `DrawArgList`
whose ``items`` are themselves `DrawArgList`s.

The transpile pipeline reads those arguments as flat forms: atoms
(numbers, identifiers, bracket-encoded references), vectors of
atoms, and matrices of atoms. This module is the single boundary
that translates the recursive AST shape into those flat forms, so
the lowering pass, the resolver, and the backend renderers never
match `DrawArg` variants or reach into ``items`` themselves. They
call `atom_value`, `list_items`, `list_atoms`, `is_matrix`, and
`matrix_rows` instead.

A bracket-indexed reference re-serialises to the repeated-bracket
surface form ``theta[i]`` / ``w[i][j]`` that the backends' own
index regex consumes, via `encode_index`.
"""

from __future__ import annotations

from quivers.dsl.ast_nodes import (
    DrawArg,
    DrawArgIndex,
    DrawArgList,
    DrawArgName,
    DrawArgScalar,
    atom_to_draw_arg as _lift_atom,
)

Atom = str | float


def encode_index(arg: DrawArgIndex) -> str:
    """Serialise a bracket-indexed reference to its repeated-bracket
    surface form: ``theta[i]`` for a single index, ``w[i][j]`` for
    several. This is the wire form the backends' index parser reads.
    """
    return arg.name + "".join(f"[{index}]" for index in arg.indices)


def is_atom(arg: DrawArg) -> bool:
    """True when `arg` is an atomic variant: a scalar, a bare name,
    or a bracket-indexed reference."""
    return isinstance(arg, (DrawArgScalar, DrawArgName, DrawArgIndex))


def atom_value(arg: DrawArg) -> Atom:
    """Return the flat wire value of an atomic `DrawArg`.

    A `DrawArgScalar` yields its float, a `DrawArgName` its
    identifier text, and a `DrawArgIndex` its bracket-encoded
    surface string. Raises for compound variants (`DrawArgList`,
    `DrawArgDist`), which carry no single atomic value.
    """
    if isinstance(arg, DrawArgScalar):
        return arg.value
    if isinstance(arg, DrawArgName):
        return arg.text
    if isinstance(arg, DrawArgIndex):
        return encode_index(arg)
    raise TypeError(
        f"atom_value: not an atomic DrawArg: {type(arg).__name__}"
    )


def list_items(arg: DrawArgList) -> tuple[DrawArg, ...]:
    """Return the recursive child `DrawArg`s of a list literal.

    Use this when the children may themselves be compound (a matrix
    row, a nested distribution); use `list_atoms` when every child
    is known to be atomic.
    """
    return arg.items


def list_atoms(arg: DrawArgList) -> tuple[Atom, ...]:
    """Return the atomic wire values of a vector-literal
    `DrawArgList`. Raises when any child is compound (guard matrix
    inputs with `is_matrix`)."""
    return tuple(atom_value(item) for item in arg.items)


def is_matrix(arg: DrawArg) -> bool:
    """True when `arg` is a matrix literal: a non-empty `DrawArgList`
    whose every item is itself a `DrawArgList`."""
    return (
        isinstance(arg, DrawArgList)
        and len(arg.items) > 0
        and all(isinstance(item, DrawArgList) for item in arg.items)
    )


def matrix_rows(arg: DrawArgList) -> tuple[tuple[Atom, ...], ...]:
    """Return the row-major atom values of a matrix-literal
    `DrawArgList` (each item is a `DrawArgList` of atoms).

    Callers should gate this with `is_matrix`.
    """
    rows: list[tuple[Atom, ...]] = []
    for item in arg.items:
        if not isinstance(item, DrawArgList):
            raise TypeError(
                f"matrix_rows: row is not a DrawArgList: "
                f"{type(item).__name__}"
            )
        rows.append(list_atoms(item))
    return tuple(rows)


def atom_to_draw_arg(value: Atom) -> DrawArg:
    """Lift a flat wire value into the atomic `DrawArg` it denotes: a
    string becomes a `DrawArgName`, a number a `DrawArgScalar`."""
    return _lift_atom(value)


__all__ = [
    "Atom",
    "atom_to_draw_arg",
    "atom_value",
    "encode_index",
    "is_atom",
    "is_matrix",
    "list_atoms",
    "list_items",
    "matrix_rows",
]
