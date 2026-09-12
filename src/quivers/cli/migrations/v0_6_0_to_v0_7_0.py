"""One-hop migrator: v0.6.0 source to v0.7.0 source.

Walks the v0.6.0 parse tree, dispatches per-declaration converters
where the grammar rule shape changed between v0.6.0 and v0.7.0, clones
everything else structurally, and emits canonical v0.7.0 source via
panproto's grammar-bound ``emit_pretty``.
"""

from __future__ import annotations

from quivers.cli.migrations._common import (
    DeclConverter,
    MigrationError,
    SchemaView,
    migrate_source,
)


def _convert_quantale(view: SchemaView, vid: str) -> str:
    lo, hi = view.span(vid)
    source = view.source[lo:hi]
    if source.startswith(b"quantale"):
        source = b"algebra" + source[len(b"quantale") :]
    elif not source.startswith(
        (b"semigroupoid", b"bilinear_form", b"composition_rule")
    ):
        raise MigrationError("unrecognized v0.6.0 quantale declaration form")
    text = source.decode("utf-8")
    return text if text.endswith("\n") else text + "\n"


_DECL_CONVERTERS: dict[str, DeclConverter] = {
    "quantale_decl": _convert_quantale,
}


def migrate(source: bytes) -> bytes:
    return migrate_source(source, "v0.6.0", "v0.7.0", _DECL_CONVERTERS)


SOURCE_RULE_COVERAGE: frozenset[str] = frozenset(_DECL_CONVERTERS)
