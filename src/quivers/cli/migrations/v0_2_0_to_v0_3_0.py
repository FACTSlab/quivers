"""One-hop migrator: v0.2.0 source to v0.3.0 source.

Walks the v0.2.0 parse tree, dispatches per-declaration converters
where the grammar rule shape changed between v0.2.0 and v0.3.0,
clones everything else structurally, and emits canonical v0.3.0
source via panproto's grammar-bound ``emit_pretty``.

The per-declaration converters live in this module; rules that
are byte-identical between v0.2.0 and v0.3.0 fall through to
[`quivers.cli.migrations._common.clone_vertex`][quivers.cli.migrations._common.clone_vertex].
"""

from __future__ import annotations

from quivers.cli.migrations._common import DeclConverter, SchemaView, migrate_source


def _preserve_rule_decl(view: SchemaView, vid: str) -> str:
    """Preserve syntax while v0.3 reclassifies category-pattern nodes."""
    text = view.text(vid)
    return text if text.endswith("\n") else text + "\n"


_DECL_CONVERTERS: dict[str, DeclConverter] = {
    "rule_decl": _preserve_rule_decl,
}


def migrate(source: bytes) -> bytes:
    return migrate_source(source, "v0.2.0", "v0.3.0", _DECL_CONVERTERS)


# v0.3 folds the category-pattern grammar into `_type_expr`. The concrete
# syntax and the containing `rule_decl` stay unchanged; the explicit parent
# converter above passes the declaration through and target-validates it.
SOURCE_RULE_COVERAGE: frozenset[str] = frozenset(
    {"_cat_pattern", "cat_atom", "cat_paren", "cat_product", "cat_slash"}
)
