"""Byte-preserving migration hops for grammar-identical releases."""

from __future__ import annotations

from typing import Callable

from quivers.cli.migrations._common import MigrationError, validate_parsed_schema
from quivers.cli.migrations._grammar import grammar_identity, parse
from quivers.cli.migrations._manifest import SCHEMA_COMMITS
from quivers.cli.migrations._vcs import schemas_identical


SOURCE_RULE_COVERAGE: frozenset[str] = frozenset()


def validate_identity_pair(source_rev: str, target_rev: str) -> None:
    """Prove that a declared byte-preserving edge is still an identity."""
    # Import lazily because the migration registry imports this module while
    # it is constructing `commit_id`.
    from quivers.cli.migrations import commit_id

    for revision in (source_rev, target_rev):
        expected = SCHEMA_COMMITS.get(revision)
        actual = commit_id(revision)
        if expected is None or actual != expected:
            raise MigrationError(
                f"identity hop {source_rev} -> {target_rev} has an unpinned "
                f"schema commit for {revision}: expected {expected!r}, got {actual!r}"
            )
    try:
        source_grammar = grammar_identity(source_rev)
        target_grammar = grammar_identity(target_rev)
        schema_identity = schemas_identical(source_rev, target_rev)
    except (OSError, ValueError, RuntimeError) as error:
        raise MigrationError(str(error)) from error
    if source_grammar != target_grammar:
        raise MigrationError(
            f"identity hop {source_rev} -> {target_rev} has a parser/grammar "
            "delta; replace it with an explicit converter"
        )
    if not schema_identity:
        raise MigrationError(
            f"identity hop {source_rev} -> {target_rev} has a panproto schema "
            "delta; replace it with an explicit converter"
        )


def migrate(source: bytes, source_rev: str, target_rev: str) -> bytes:
    """Validate an identity edge and return target-parseable bytes unchanged.

    Identity hops remain explicit in the chain so release names do not
    collapse into a mutable ``HEAD`` alias. Both the parser-source hash and
    panproto schema identity are checked before the source is parsed using the
    target snapshot.
    """
    validate_identity_pair(source_rev, target_rev)
    try:
        schema = parse(target_rev, source)
    except Exception as error:
        raise MigrationError(
            f"source does not parse under identity target {target_rev}: {error}"
        ) from error
    validate_parsed_schema(schema, f"identity target {target_rev}", role="source")
    return source


def migrator(source_rev: str, target_rev: str) -> Callable[[bytes], bytes]:
    """Bind an identity migrator to one explicit release edge."""

    def migrate_edge(source: bytes) -> bytes:
        return migrate(source, source_rev, target_rev)

    setattr(migrate_edge, "_qvr_identity_pair", (source_rev, target_rev))
    return migrate_edge


__all__ = [
    "SOURCE_RULE_COVERAGE",
    "migrate",
    "migrator",
    "validate_identity_pair",
]
