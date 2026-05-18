"""Low-level helpers shared by the per-statement walkers."""

from __future__ import annotations

from quivers.dsl.parser._registry import ParseError, _Tree


def _required_text(
    t: _Tree, child_vid: str | None, parent_vid: str, field_name: str
) -> str:
    """Return the text of a required-by-grammar field, raising if missing.

    Several Statement variants — algebra, object, morphism, space, etc. —
    declare an identifier ``name`` field. Tree-sitter guarantees the
    field exists on a successful parse, so a ``None`` here means the
    parse was corrupted (an ``ERROR`` node leaked through, or the
    grammar was edited without updating the walker).
    """
    if child_vid is None:
        raise ParseError(
            f"missing required {field_name!r} field at {parent_vid} (malformed parse)"
        )
    return t.text(child_vid)


def _walk_options(t: _Tree, vid: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in t.positional(vid):
        if t.kind(entry) != "option_entry":
            continue
        kvid = t.field(entry, "key")
        vvid = t.field(entry, "value")
        out[_required_text(t, kvid, entry, "key")] = _required_text(
            t, vvid, entry, "value"
        )
    return out


def _walk_return_pattern(
    t: _Tree, vid: str
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    """Walk a return clause into (vars, labels).

    Three forms:

    * ``return x`` — single variable; ``vars=(x,)``, ``labels=None``.
    * ``return (x, y, z)`` — positional tuple; ``vars=(x, y, z)``,
      ``labels=None``.
    * ``return (a: x, b: y)`` — labelled tuple; ``vars=(x, y)``,
      ``labels=(a, b)``. The labels rename the coordinates of the
      output product space at the schema level.
    """
    k = t.kind(vid)
    if k == "identifier":
        return (t.text(vid),), None
    if k == "return_tuple":
        return tuple(t.text(c) for c in t.positional(vid)), None
    if k == "return_labeled_tuple":
        labels: list[str] = []
        vars_l: list[str] = []
        for entry in t.positional(vid):
            if t.kind(entry) != "return_label_entry":
                continue
            lvid = t.field(entry, "label")
            vvid = t.field(entry, "var")
            labels.append(_required_text(t, lvid, entry, "label"))
            vars_l.append(_required_text(t, vvid, entry, "var"))
        return tuple(vars_l), tuple(labels)
    raise ParseError(f"unexpected return pattern kind: {k}")


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def _walk_draw_arg(t: _Tree, vid: str) -> str | float:
    """Walk a family-argument into its compiler representation.

    Identifiers and numeric literals are walked into their natural
    Python values. A ``bracket_index_arg`` (e.g., ``theta[N]``) is
    encoded as the string ``"theta[N]"``; the compiler detects
    the bracket and unpacks the section's name and index set when
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
    raise ParseError(f"unexpected draw arg kind: {k}")
