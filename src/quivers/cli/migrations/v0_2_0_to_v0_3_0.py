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

from quivers.cli.migrations._common import DeclConverter, migrate_source


_DECL_CONVERTERS: dict[str, DeclConverter] = {}


def migrate(source: bytes) -> bytes:
    return migrate_source(source, "v0.2.0", "v0.3.0", _DECL_CONVERTERS)

# Identity hop or no converters declared yet. The chain-coverage
# check will flag every removed source rule as uncovered until
# the hop's converters are written.
SOURCE_RULE_COVERAGE: frozenset[str] = frozenset()
