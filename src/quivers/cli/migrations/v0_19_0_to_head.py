"""One-hop migrator from the released v0.19 surface to QVR HEAD.

QVR HEAD accepts hanging-indented argument and list forms while retaining the
v0.19 layouts. Migration is thus byte-preserving, but it remains a validating
hop because the generated parser identity changed.
"""

from __future__ import annotations

from quivers.cli.migrations._common import parse_validated_source


_SOURCE_REV = "v0.19.0"
_TARGET_REV = "HEAD"


def migrate(source: bytes) -> bytes:
    """Validate a v0.19 source under both grammars and preserve its bytes."""
    parse_validated_source(_SOURCE_REV, source)
    parse_validated_source(_TARGET_REV, source, role="additive target")
    return source


SOURCE_RULE_COVERAGE: frozenset[str] = frozenset()
