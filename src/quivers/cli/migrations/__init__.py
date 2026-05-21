"""Registry of structural one-hop ``.qvr`` source migrations.

Each adjacent-pair migrator is a callable ``bytes -> bytes`` built
on the panproto pipeline in `quivers.cli.migrations._common`:
parse with the source revision's tree-sitter parser, build the
target schema by walking the source parse tree and emitting target
vertices/edges/constraints, then emit canonical target bytes via
the target revision's ``emit_pretty``. The third stage is grammar-
bound by panproto: a target schema that does not satisfy the target
grammar's rules cannot emit.

The `CHAIN` tuple lists releases in chronological order with
``"HEAD"`` always last. Each adjacent pair ``(CHAIN[i], CHAIN[i+1])``
must have a registered migrator in `MIGRATORS`.
`compose_migration` walks ``CHAIN`` between any two listed
releases and composes the intermediate hops into a single
``bytes -> bytes`` callable.

To add the next release:

1. Tag the release in git.
2. Run ``python grammars/qvr/vcs/build_schemas.py`` and
   ``python grammars/qvr/vcs/build_parsers.py``.
3. Rename the current ``v<latest>_to_head`` migrator module to
   ``v<latest>_to_v<next>`` (it now migrates between two pinned
   revisions), and write a new ``v<next>_to_head`` covering the
   next batch of grammar changes if any.
4. Append the new release to `CHAIN` and register the
   migrator(s) in `MIGRATORS`.

The ``qvr migrate`` CLI consumes `compose_migration` and is
agnostic to which pairs are registered.
"""

from __future__ import annotations

from typing import Callable

from quivers.cli.migrations import v0_2_0_to_v0_3_0 as _hop_2_3
from quivers.cli.migrations import v0_3_0_to_v0_4_0 as _hop_3_4
from quivers.cli.migrations import v0_4_0_to_v0_5_0 as _hop_4_5
from quivers.cli.migrations import v0_5_0_to_v0_6_0 as _hop_5_6
from quivers.cli.migrations import v0_6_0_to_v0_7_0 as _hop_6_7
from quivers.cli.migrations import v0_7_0_to_v0_9_0 as _hop_7_9
from quivers.cli.migrations import v0_9_0_to_v0_10_0 as _hop_9_10
from quivers.cli.migrations import v0_10_0_to_v0_11_0 as _hop_10_11
from quivers.cli.migrations._vcs import (
    BlameReport,
    DiffCoverageReport,
    blame_kind,
    check_chain_coverage,
    commit_id_for,
    diff_coverage,
)

__all__ = [
    "BlameReport",
    "DiffCoverageReport",
    "CHAIN",
    "MIGRATORS",
    "COVERAGE",
    "MigrationError",
    "blame_kind",
    "check_chain_coverage",
    "commit_id_for",
    "compose_migration",
    "diff_coverage",
    "vcs_coverage_report",
]


_Migrator = Callable[[bytes], bytes]


CHAIN: tuple[str, ...] = (
    "v0.2.0",
    "v0.3.0",
    "v0.4.0",
    "v0.5.0",
    "v0.6.0",
    "v0.7.0",
    "v0.9.0",
    "v0.10.0",
    "v0.11.0",
)


MIGRATORS: dict[tuple[str, str], _Migrator] = {
    ("v0.2.0", "v0.3.0"): _hop_2_3.migrate,
    ("v0.3.0", "v0.4.0"): _hop_3_4.migrate,
    ("v0.4.0", "v0.5.0"): _hop_4_5.migrate,
    ("v0.5.0", "v0.6.0"): _hop_5_6.migrate,
    ("v0.6.0", "v0.7.0"): _hop_6_7.migrate,
    ("v0.7.0", "v0.9.0"): _hop_7_9.migrate,
    ("v0.9.0", "v0.10.0"): _hop_9_10.migrate,
    ("v0.10.0", "v0.11.0"): _hop_10_11.migrate,
}

# Per-hop coverage declarations: source-side rule names each hop's
# converter dispatch table covers. Used by
# `vcs_coverage_report` to validate the migration system
# against the panproto VCS schema diff between adjacent revisions.
COVERAGE: dict[tuple[str, str], frozenset[str]] = {
    ("v0.2.0", "v0.3.0"): _hop_2_3.SOURCE_RULE_COVERAGE,
    ("v0.3.0", "v0.4.0"): _hop_3_4.SOURCE_RULE_COVERAGE,
    ("v0.4.0", "v0.5.0"): _hop_4_5.SOURCE_RULE_COVERAGE,
    ("v0.5.0", "v0.6.0"): _hop_5_6.SOURCE_RULE_COVERAGE,
    ("v0.6.0", "v0.7.0"): _hop_6_7.SOURCE_RULE_COVERAGE,
    ("v0.7.0", "v0.9.0"): _hop_7_9.SOURCE_RULE_COVERAGE,
    ("v0.9.0", "v0.10.0"): _hop_9_10.SOURCE_RULE_COVERAGE,
    ("v0.10.0", "v0.11.0"): _hop_10_11.SOURCE_RULE_COVERAGE,
}

# Commit-id index: maps each `CHAIN` release name to its
# panproto VCS commit id, computed once at module-import time.
# Released tags resolve directly; releases whose grammar is byte-
# identical to a predecessor share that predecessor's commit;
# the last entry (typically the upcoming-but-untagged release)
# resolves to HEAD when the working-tree grammar differs from
# every tagged release.
_COMMIT_IDS: dict[str, str] = {}


def _build_commit_index() -> dict[str, str]:
    """Resolve every `CHAIN` entry to a panproto VCS commit id."""
    out: dict[str, str] = {}
    last_resolved: str = ""
    for i, ref in enumerate(CHAIN):
        is_last = i == len(CHAIN) - 1
        resolved = commit_id_for(ref)
        if resolved:
            out[ref] = resolved
            last_resolved = resolved
            continue
        if is_last:
            # The upcoming/untagged latest revision: HEAD when
            # head differs from the last tagged commit; otherwise
            # share the predecessor's commit.
            from quivers.cli.migrations._vcs import _open_repo

            head_id = _open_repo().head() or ""
            out[ref] = (
                head_id if head_id and head_id != last_resolved else last_resolved
            )
            last_resolved = out[ref]
            continue
        # Mid-chain entry sharing its predecessor's grammar (e.g.
        # v0.10.0 == v0.9.0 byte-for-byte; the VCS skips the
        # commit). Reuse last_resolved.
        out[ref] = last_resolved
    return out


def commit_id(ref: str) -> str:
    """Resolve a release name to its panproto VCS commit id. Cached."""
    if not _COMMIT_IDS:
        _COMMIT_IDS.update(_build_commit_index())
    return _COMMIT_IDS.get(ref, "")


def vcs_coverage_report() -> list[DiffCoverageReport]:
    """Run the panproto-VCS-driven coverage check across every
    adjacent pair in `CHAIN`. Each report carries the schema
    diff and the set of removed source rules not covered by the
    corresponding hop's ``SOURCE_RULE_COVERAGE``.

    Use this from ``qvr migrate --check`` (CLI) or from a CI test
    to catch migrators that drift behind grammar changes."""
    reports: list[DiffCoverageReport] = []
    for i in range(len(CHAIN) - 1):
        from_ref = CHAIN[i]
        to_ref = CHAIN[i + 1]
        decl = COVERAGE.get((from_ref, to_ref), frozenset())
        from_id = commit_id(from_ref)
        to_id = commit_id(to_ref)
        if from_id == to_id:
            reports.append(
                DiffCoverageReport(
                    from_ref=from_ref,
                    to_ref=to_ref,
                    added_rules=(),
                    removed_rules=(),
                    uncovered_removed=(),
                )
            )
            continue
        # Re-implement diff_coverage's body using the resolved
        # commit ids directly to avoid double-resolution.
        from quivers.cli.migrations._vcs import _open_repo
        import panproto

        repo = _open_repo()
        src_schema = repo.schema_at(from_id)
        tgt_schema = repo.schema_at(to_id)
        diff_dict = panproto.diff_schemas(src_schema, tgt_schema).to_dict()
        added = tuple(sorted(diff_dict.get("added_vertices", [])))
        removed = tuple(sorted(diff_dict.get("removed_vertices", [])))
        uncovered = tuple(r for r in removed if r not in decl)
        reports.append(
            DiffCoverageReport(
                from_ref=from_ref,
                to_ref=to_ref,
                added_rules=added,
                removed_rules=removed,
                uncovered_removed=uncovered,
            )
        )
    return reports


class MigrationError(Exception):
    """Raised when the requested migration cannot be composed."""


def _identity(source: bytes) -> bytes:
    return source


def _chain_slice(from_ref: str, to_ref: str) -> list[tuple[str, str]]:
    """Return the ordered list of adjacent ``(src, dst)`` pairs that
    walk `CHAIN` from ``from_ref`` to ``to_ref``."""
    if from_ref not in CHAIN:
        raise MigrationError(
            f"unknown source revision {from_ref!r}; known: {CHAIN}",
        )
    if to_ref not in CHAIN:
        raise MigrationError(
            f"unknown target revision {to_ref!r}; known: {CHAIN}",
        )
    i_from = CHAIN.index(from_ref)
    i_to = CHAIN.index(to_ref)
    if i_to < i_from:
        raise MigrationError(
            f"backward migration ({from_ref} -> {to_ref}) is not yet "
            "implemented; only forward composition through CHAIN is "
            "supported",
        )
    return [(CHAIN[i], CHAIN[i + 1]) for i in range(i_from, i_to)]


def compose_migration(from_ref: str, to_ref: str) -> _Migrator:
    """Return a single ``bytes -> bytes`` callable that composes
    every adjacent-pair migrator between ``from_ref`` and ``to_ref``
    on `CHAIN`."""
    pairs = _chain_slice(from_ref, to_ref)
    if not pairs:
        return _identity
    missing = [pair for pair in pairs if pair not in MIGRATORS]
    if missing:
        raise MigrationError(
            "missing migrator(s) for pair(s): "
            + ", ".join(f"{a} -> {b}" for a, b in missing),
        )
    migrators = [MIGRATORS[pair] for pair in pairs]

    def _composite(source: bytes) -> bytes:
        for step in migrators:
            source = step(source)
        return source

    return _composite


def available_targets(from_ref: str) -> tuple[str, ...]:
    """Return every revision reachable forward from ``from_ref`` on
    `CHAIN` (inclusive of ``from_ref`` itself)."""
    if from_ref not in CHAIN:
        return ()
    i = CHAIN.index(from_ref)
    return CHAIN[i:]


__all__ = [
    "BlameReport",
    "CHAIN",
    "COVERAGE",
    "DiffCoverageReport",
    "MIGRATORS",
    "MigrationError",
    "available_targets",
    "blame_kind",
    "commit_id",
    "compose_migration",
    "diff_coverage",
    "vcs_coverage_report",
]
