"""AST nodes for draw-step arguments.

The DSL's program-step grammar (`SampleStep`, `ObserveStep`,
`MarginalizeStep`) and the family-init clause
(`MorphismInitFamily`) accept argument positions that are scalars
(identifiers, numbers), bracket-indexed references (``mu[cls]``),
or compound literals:

* `DrawArgList` -- a vector literal ``[a, b, c]`` whose elements
  are atomic draw args (identifiers, numbers, or bracket-indexed
  references).
* `DrawArgMatrix` -- a row-major matrix literal
  ``[[a, b], [c, d]]`` whose rows are `DrawArgList` instances.

The four surface forms are gathered under the
`DrawArg` tagged union so step / init `args` slots type-check as
``tuple[DrawArg, ...]`` and serialise through didactic's encode /
decode round trip. Atomic positions wrap as either `DrawArgScalar`
(numeric literal) or `DrawArgName` (identifier or bracket-encoded
reference); their wire-form ``str | float`` is held verbatim in
the wrapper. The compound positions hold structural element
sequences directly.

`atom_to_draw_arg` lifts a wire-form ``str | float`` into the
appropriate atomic variant; `draw_arg_atom_value` extracts the
wire-form from an atomic variant (raising on a compound variant).
`draw_args_to_wire` is the bulk helper used by emit / lower call
sites that only handle atomic arg lists.
"""

from __future__ import annotations

from typing import Literal

import didactic.api as dx


DrawArgAtom = str | float


class DrawArg(dx.TaggedUnion, discriminator="kind"):
    """One argument position in a draw step or family initialiser.

    Variants:

    * `DrawArgScalar` -- a numeric literal.
    * `DrawArgName` -- an identifier or encoded bracket-indexed
      reference (``"theta[N]"`` round-trips as a `DrawArgName`
      whose ``text`` carries the bracket form).
    * `DrawArgList` -- a vector literal ``[a, b, c]``.
    * `DrawArgMatrix` -- a matrix literal ``[[a, b], [c, d]]``.
    """


class DrawArgScalar(DrawArg):
    """A numeric literal argument."""

    value: float
    line: int = 0
    col: int = 0
    kind: Literal["draw_arg_scalar"] = "draw_arg_scalar"


class DrawArgName(DrawArg):
    """An identifier or encoded bracket-indexed reference argument.

    The text carries the surface form verbatim: a bare identifier
    (``"phi"``) or the encoded bracket form (``"phi[cls]"``).
    """

    text: str
    line: int = 0
    col: int = 0
    kind: Literal["draw_arg_name"] = "draw_arg_name"


class DrawArgList(DrawArg):
    """Vector literal ``[a, b, c]`` in a draw-arg position.

    ``elements`` are atomic draw args as their wire-form
    ``str | float`` representations. A `DrawArgList` of length
    ``n`` is the structural surface form of a length-``n`` vector
    parameter (e.g. a Dirichlet concentration or a Categorical
    probability vector).
    """

    elements: tuple[DrawArgAtom, ...]
    line: int = 0
    col: int = 0
    kind: Literal["draw_arg_list"] = "draw_arg_list"


class DrawArgMatrix(DrawArg):
    """Matrix literal ``[[a, b], [c, d]]`` in a draw-arg position.

    ``rows`` is a tuple of `DrawArgList` instances of equal
    length. The number of rows is the matrix's first axis size;
    the row length is the second axis size.
    """

    rows: tuple[DrawArgList, ...]
    line: int = 0
    col: int = 0
    kind: Literal["draw_arg_matrix"] = "draw_arg_matrix"


def atom_to_draw_arg(value: DrawArgAtom, *, line: int = 0, col: int = 0) -> DrawArg:
    """Wrap an atomic wire-form value into the appropriate
    `DrawArg` variant.

    Numeric values become `DrawArgScalar`; strings (identifiers
    and encoded bracket forms) become `DrawArgName`.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"atom_to_draw_arg expected str | float, got bool {value!r}"
        )
    if isinstance(value, (int, float)):
        return DrawArgScalar(value=float(value), line=line, col=col)
    if isinstance(value, str):
        return DrawArgName(text=value, line=line, col=col)
    raise TypeError(
        f"atom_to_draw_arg expected str | float, got {type(value).__name__}"
    )


def draw_arg_atom_value(arg: DrawArg) -> DrawArgAtom:
    """Return the atomic wire-form value carried by an atomic
    `DrawArg` variant, raising for compound variants.
    """
    if isinstance(arg, DrawArgScalar):
        return arg.value
    if isinstance(arg, DrawArgName):
        return arg.text
    raise TypeError(
        f"draw_arg_atom_value: not an atomic variant: "
        f"{type(arg).__name__}"
    )


def draw_args_to_wire(
    args: tuple[DrawArg, ...] | None,
) -> tuple[DrawArgAtom, ...] | None:
    """Bulk-unwrap a tuple of `DrawArg` into a tuple of wire-form
    ``str | float`` atomic values.

    Raises [`TypeError`][TypeError] when any position carries a
    compound variant (`DrawArgList` or `DrawArgMatrix`). Used by
    callers that only support atomic positional args.
    """
    if args is None:
        return None
    return tuple(draw_arg_atom_value(a) for a in args)


def wire_args_to_draw_args(
    args: tuple[DrawArgAtom, ...] | None,
) -> tuple[DrawArg, ...] | None:
    """Bulk-wrap a tuple of wire-form ``str | float`` atomic values
    into `DrawArg` variants."""
    if args is None:
        return None
    return tuple(atom_to_draw_arg(a) for a in args)


__all__ = [
    "DrawArg",
    "DrawArgAtom",
    "DrawArgList",
    "DrawArgMatrix",
    "DrawArgName",
    "DrawArgScalar",
    "atom_to_draw_arg",
    "draw_arg_atom_value",
    "draw_args_to_wire",
    "wire_args_to_draw_args",
]
