"""One-hop migrator from the released v0.18 surface to QVR v0.19.

QVR v0.19 added the indexed-family and algebraic-effect surface. Every v0.18
production remains accepted without reinterpretation, so migration is
byte-preserving. This is not an identity edge: both revisions are parsed and
validated explicitly, and Panproto reports the added target rules through the
coverage gate.
"""

from __future__ import annotations

from quivers.cli.migrations._common import parse_validated_source


_SOURCE_REV = "v0.18.0"
_TARGET_REV = "v0.19.0"


def migrate(source: bytes) -> bytes:
    """Validate a v0.18 source under both grammars and preserve its bytes."""
    parse_validated_source(_SOURCE_REV, source)
    parse_validated_source(_TARGET_REV, source, role="additive target")
    return source


SOURCE_RULE_COVERAGE: frozenset[str] = frozenset()
