"""Low-level helpers shared by the per-statement walkers."""

from __future__ import annotations

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
            f"missing required {field_name!r} field at "
            f"{parent_vid} (malformed parse)"
        )
    return t.text(child_vid)

def _walk_draw_arg(t: _Tree, vid: str) -> str | float:
    """Walk a family-argument into its compiler representation.

    Identifiers and numeric literals walk to their natural Python
    values. A ``bracket_index_arg`` (e.g. ``theta[N]``) is encoded
    as the string ``"theta[N]"``; the compiler detects the bracket
    and unpacks the section's name and index set when resolving the
    argument at draw / observe time.
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
    raise ParseError(f"unexpected draw arg kind: {k}")

__all__ = ["_required_text", "_walk_draw_arg"]
