"""Low-level helpers shared by the per-statement walkers."""

from __future__ import annotations

from quivers.dsl.ast_nodes.draw_args import (
    DrawArg,
    DrawArgAtom,
    DrawArgList,
    DrawArgMatrix,
    DrawArgName,
    DrawArgScalar,
    atom_to_draw_arg,
)
from quivers.dsl.parser._registry import ParseError, _Tree


def _required_text(
    t: _Tree, child_vid: str | None, parent_vid: str, field_name: str
) -> str:
    """Return the text of a required-by-grammar field, raising if missing.

    Tree-sitter guarantees the field exists on a successful parse, so
    a ``None`` here means the parse was corrupted (an ``ERROR`` node
    leaked through, or the grammar was edited without updating the
    walker).
    """
    if child_vid is None:
        raise ParseError(
            f"missing required {field_name!r} field at {parent_vid} (malformed parse)"
        )
    return t.text(child_vid)


def _walk_draw_arg_atom(t: _Tree, vid: str) -> DrawArgAtom:
    """Walk an atomic draw arg into its wire-form ``str | float``
    representation.

    Identifiers and numeric literals walk to their natural Python
    values. A ``bracket_index_arg`` (e.g. ``theta[N]``) is encoded
    as the string ``"theta[N]"``; downstream consumers detect the
    bracket and unpack the section's name and index set when
    resolving the argument at draw / observe time.
    """
    k = t.kind(vid)
    if k == "identifier":
        return t.text(vid)
    if k == "signed_number":
        return float(t.text(vid))
    if k in ("integer", "float"):
        return float(t.text(vid))
    if k == "bracket_index_arg":
        nv = t.field(vid, "name")
        iv = t.field(vid, "index")
        if nv is None or iv is None:
            raise ParseError(f"bracket_index_arg malformed at {vid}")
        return f"{t.text(nv)}[{t.text(iv)}]"
    raise ParseError(f"unexpected draw arg atom kind: {k}")


def _walk_draw_arg(t: _Tree, vid: str) -> DrawArg:
    """Walk a family-argument into its `DrawArg` tagged-union variant.

    Atomic positions (identifier, signed_number, bracket_index_arg)
    wrap into `DrawArgScalar` / `DrawArgName`; compound forms
    ``draw_arg_list`` and ``draw_arg_matrix`` walk into the
    `DrawArgList` and `DrawArgMatrix` variants respectively.
    """
    k = t.kind(vid)
    if k == "draw_arg_list":
        elements = tuple(
            _walk_draw_arg_atom(t, child)
            for child in t.positional(vid)
            if t.kind(child)
            in ("identifier", "signed_number", "bracket_index_arg")
        )
        line, col = t.line_col(vid)
        return DrawArgList(elements=elements, line=line, col=col)
    if k == "draw_arg_matrix":
        rows: list[DrawArgList] = []
        for child in t.positional(vid):
            if t.kind(child) != "draw_arg_list":
                continue
            row_elements = tuple(
                _walk_draw_arg_atom(t, atom)
                for atom in t.positional(child)
                if t.kind(atom)
                in ("identifier", "signed_number", "bracket_index_arg")
            )
            row_line, row_col = t.line_col(child)
            rows.append(
                DrawArgList(elements=row_elements, line=row_line, col=row_col)
            )
        line, col = t.line_col(vid)
        return DrawArgMatrix(rows=tuple(rows), line=line, col=col)
    line, col = t.line_col(vid)
    return atom_to_draw_arg(_walk_draw_arg_atom(t, vid), line=line, col=col)


__all__ = [
    "DrawArg",
    "DrawArgList",
    "DrawArgMatrix",
    "DrawArgName",
    "DrawArgScalar",
    "_required_text",
    "_walk_draw_arg",
    "_walk_draw_arg_atom",
]
