"""One-hop migrator: v0.9.0 source to v0.10.0 source.

Grammar identity: v0.10.0's ``grammars/qvr/grammar.js`` is byte-
identical to v0.9.0's, so this hop is a no-op pass-through. The
module exists so the migration chain stays uniform (every adjacent
release pair has an entry); composers can walk through it without
special-casing.
"""

from __future__ import annotations


def migrate(source: bytes) -> bytes:
    return source


# Identity hop or no converters declared yet. The chain-coverage
# check will flag every removed source rule as uncovered until
# the hop's converters are written.
SOURCE_RULE_COVERAGE: frozenset[str] = frozenset()
