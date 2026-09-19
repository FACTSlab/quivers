"""One-hop migrator: v0.4.0 source to v0.5.0 source.

Walks the v0.4.0 parse tree, dispatches per-declaration converters
where the grammar rule shape changed between v0.4.0 and v0.5.0, clones
everything else structurally, and emits canonical v0.5.0 source via
panproto's grammar-bound ``emit_pretty``.
"""

from __future__ import annotations

from quivers.cli.migrations._common import (
    DeclConverter,
    MigrationError,
    SchemaView,
    migrate_source,
)


def _as_kernel(view: SchemaView, vid: str, keyword: str) -> str:
    lo, hi = view.span(vid)
    source = view.source[lo:hi]
    encoded = keyword.encode("ascii")
    if not source.startswith(encoded):
        raise MigrationError(f"v0.4.0 {view.kind(vid)} does not start with {keyword!r}")
    result = b"kernel" + source[len(encoded) :]
    text = result.decode("utf-8")
    return text if text.endswith("\n") else text + "\n"


def _convert_continuous(view: SchemaView, vid: str) -> str:
    return _as_kernel(view, vid, "continuous")


def _convert_stochastic(view: SchemaView, vid: str) -> str:
    return _as_kernel(view, vid, "stochastic")


_DECL_CONVERTERS: dict[str, DeclConverter] = {
    "continuous_decl": _convert_continuous,
    "stochastic_decl": _convert_stochastic,
}


def migrate(source: bytes) -> bytes:
    return migrate_source(source, "v0.4.0", "v0.5.0", _DECL_CONVERTERS)


SOURCE_RULE_COVERAGE: frozenset[str] = frozenset(_DECL_CONVERTERS)
