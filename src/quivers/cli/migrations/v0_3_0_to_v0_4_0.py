"""One-hop migrator: v0.3.0 source to v0.4.0 source.

Walks the v0.3.0 parse tree, dispatches per-declaration converters
where the grammar rule shape changed between v0.3.0 and v0.4.0, clones
everything else structurally, and emits canonical v0.4.0 source via
panproto's grammar-bound ``emit_pretty``.
"""

from __future__ import annotations

from quivers.cli.migrations._common import (
    DeclConverter,
    MigrationError,
    SchemaView,
    migrate_source,
)


_Edit = tuple[int, int, bytes]


def _walk(view: SchemaView, root_vid: str) -> list[str]:
    pending = [root_vid]
    result: list[str] = []
    while pending:
        vid = pending.pop()
        result.append(vid)
        pending.extend(reversed(view.outgoing_vids(vid)))
    return result


def _token_edit(
    view: SchemaView,
    vid: str,
    token: bytes,
    replacement: bytes,
    *,
    consume_trailing_space: bool = False,
) -> _Edit:
    for start, text in view.interstitials(vid):
        encoded = text.encode("utf-8")
        offset = _syntax_token_offset(encoded, token)
        if offset >= 0:
            lo = start + offset
            end = offset + len(token)
            if consume_trailing_space:
                while end < len(encoded) and encoded[end : end + 1] in {b" ", b"\t"}:
                    end += 1
            return lo, start + end, replacement
    raise MigrationError(
        f"v0.3.0 migration could not locate {token!r} in {view.kind(vid)}"
    )


def _syntax_token_offset(text: bytes, token: bytes) -> int:
    """Find a grammar token while ignoring v0.3 line-comment text."""
    in_comment = False
    index = 0
    while index < len(text):
        byte = text[index : index + 1]
        if byte == b"\n":
            in_comment = False
        elif byte == b"#":
            in_comment = True
        elif not in_comment and text.startswith(token, index):
            return index
        index += 1
    return -1


def _apply_decl_edits(view: SchemaView, vid: str, edits: list[_Edit]) -> str:
    lo, hi = view.span(vid)
    cursor = lo
    pieces: list[bytes] = []
    for start, end, replacement in sorted(edits):
        if start < cursor or end > hi:
            raise MigrationError("overlapping or out-of-bounds v0.3.0 migration edit")
        pieces.append(view.source[cursor:start])
        pieces.append(replacement)
        cursor = end
    pieces.append(view.source[cursor:hi])
    result = b"".join(pieces).decode("utf-8")
    return result if result.endswith("\n") else result + "\n"


def _convert_program_decl(view: SchemaView, vid: str) -> str:
    edits: list[_Edit] = []
    for descendant in _walk(view, vid):
        if view.kind(descendant) != "draw_step":
            continue
        edits.append(
            _token_edit(
                view,
                descendant,
                b"draw",
                b"",
                consume_trailing_space=True,
            )
        )
        edits.append(_token_edit(view, descendant, b"~", b"<-"))
    return _apply_decl_edits(view, vid, edits)


def _convert_output_decl(view: SchemaView, vid: str) -> str:
    edit = _token_edit(view, vid, b"output", b"export")
    return _apply_decl_edits(view, vid, [edit])


_DECL_CONVERTERS: dict[str, DeclConverter] = {
    "output_decl": _convert_output_decl,
    "program_decl": _convert_program_decl,
}


def migrate(source: bytes) -> bytes:
    return migrate_source(source, "v0.3.0", "v0.4.0", _DECL_CONVERTERS)


# `program_decl` owns both removed step shapes: `draw_step` is rewritten and
# the already-canonical `arrow_draw_step` is verified by target parsing.
SOURCE_RULE_COVERAGE: frozenset[str] = frozenset(
    {"arrow_draw_step", "draw_step", "output_decl"}
)
