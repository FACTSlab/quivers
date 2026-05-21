"""VCS integration helpers for the migration system.

The panproto VCS at ``grammars/qvr/vcs/.panproto/`` carries one commit
per distinct grammar revision. This module wraps the panproto Python
API to drive three integration points used by the rest of the
migrations package:

* `diff_coverage` -- for an adjacent hop, computes the
  panproto schema diff and reports any source rule that the
  hop's declared converters do not cover. Used by
  `check_chain_coverage` and exposed via ``qvr migrate --check``.
* `blame_kind` -- given a tree-sitter rule name (vertex kind),
  reports the commit and tag that first introduced or last carried
  that rule. Used by the migrator's error reporter when an unknown
  source vertex kind is encountered.
* `commit_id_for` -- resolves a release name (``"v0.10.0"``,
  ``"HEAD"``) to its panproto VCS commit id, so converters and the
  CLI can index by content-hash and survive tag renames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import panproto


_VCS_ROOT = Path(__file__).resolve().parents[4] / "grammars" / "qvr" / "vcs"


def _open_repo() -> panproto.Repository:
    """Open the qvr-grammar VCS. Raises a ``FileNotFoundError`` if
    ``.panproto/`` has never been built; users see a clear message
    pointing at ``build_schemas.py``."""
    panproto_dir = _VCS_ROOT / ".panproto"
    if not panproto_dir.is_dir():
        raise FileNotFoundError(
            f"no panproto VCS at {panproto_dir}; run "
            "``python grammars/qvr/vcs/build_schemas.py --reset`` "
            "to build the grammar-evolution chain first",
        )
    return panproto.Repository.open(str(_VCS_ROOT))


# ---------------------------------------------------------------------------
# Tag / commit-id resolution
# ---------------------------------------------------------------------------


def commit_id_for(ref: str) -> str:
    """Resolve a release name (e.g. ``"v0.10.0"``) or ``"HEAD"`` to
    the panproto VCS commit id.

    Falls back to ``""`` if the ref cannot be resolved (e.g. the
    VCS chain doesn't include a separate commit for the requested
    name, as is the case for ``"v0.10.0"`` whose grammar is byte-
    identical to ``"v0.9.0"`` and shares its commit). Callers
    use the empty-string sentinel to indicate "same as predecessor."
    """
    repo = _open_repo()
    if ref == "HEAD":
        head = repo.head()
        return head or ""
    for tag_name, commit_id in repo.list_tags():
        if tag_name == ref:
            return commit_id
    return ""


# ---------------------------------------------------------------------------
# Schema diff + coverage check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffCoverageReport:
    """The schema diff between two revisions, classified against a
    hop's declared converter dispatch table."""

    from_ref: str
    to_ref: str
    added_rules: tuple[str, ...]
    removed_rules: tuple[str, ...]
    uncovered_removed: tuple[str, ...]
    """Rules removed at the target revision that are also absent
    from ``declared_converters``. Each one is a source-side vertex
    kind that the migrator's dispatch will silently pass through as
    a structural clone — and ``emit_pretty`` will then either
    misrender or drop. These are the actionable misses."""

    @property
    def is_complete(self) -> bool:
        return not self.uncovered_removed

    def format(self) -> str:
        """Render the report for CLI display."""
        lines = [f"{self.from_ref} -> {self.to_ref}:"]
        if not self.added_rules and not self.removed_rules:
            lines.append("    (grammar identical; no diff)")
            return "\n".join(lines)
        if self.removed_rules:
            lines.append(f"    removed: {', '.join(self.removed_rules)}")
        if self.added_rules:
            lines.append(f"    added:   {', '.join(self.added_rules)}")
        if self.uncovered_removed:
            lines.append(
                f"    UNCOVERED removed rules (no converter): "
                f"{', '.join(self.uncovered_removed)}",
            )
        elif self.removed_rules:
            lines.append("    all removed rules have converters [OK]")
        return "\n".join(lines)


def diff_coverage(
    from_ref: str,
    to_ref: str,
    declared_converters: frozenset[str],
) -> DiffCoverageReport:
    """Compute the schema diff between ``from_ref`` and ``to_ref``
    in the VCS, classified against the set of source-side rule
    names the migrator declares it can handle.

    A rule appearing in ``from_ref``'s schema but not in
    ``to_ref``'s is a "removed" rule. If a removed rule is a top-
    level declaration kind that the migrator's
    ``_DECL_CONVERTERS`` dict does not list, the migrator will
    silently pass it through (likely producing incorrect output);
    these surface in ``uncovered_removed``.

    Identity hops (where ``from_ref`` and ``to_ref`` share a
    commit) report no diff.
    """
    repo = _open_repo()
    from_id = commit_id_for(from_ref) or _resolve_via_chain(repo, from_ref)
    to_id = commit_id_for(to_ref) or _resolve_via_chain(repo, to_ref)
    if from_id == to_id:
        return DiffCoverageReport(
            from_ref=from_ref,
            to_ref=to_ref,
            added_rules=(),
            removed_rules=(),
            uncovered_removed=(),
        )
    src_schema = repo.schema_at(from_id)
    tgt_schema = repo.schema_at(to_id)
    diff = panproto.diff_schemas(src_schema, tgt_schema)
    diff_dict = diff.to_dict()
    added = tuple(sorted(diff_dict.get("added_vertices", [])))
    removed = tuple(sorted(diff_dict.get("removed_vertices", [])))
    uncovered = tuple(r for r in removed if r not in declared_converters)
    return DiffCoverageReport(
        from_ref=from_ref,
        to_ref=to_ref,
        added_rules=added,
        removed_rules=removed,
        uncovered_removed=uncovered,
    )


def _semver_key(tag: str) -> tuple[int, ...]:
    """Parse ``v0.10.0`` into ``(0, 10, 0)`` for numeric comparison.
    Tags that don't match the ``vX.Y.Z`` pattern sort last."""
    if not tag.startswith("v"):
        return (1 << 30,)
    try:
        return tuple(int(p) for p in tag[1:].split("."))
    except ValueError:
        return (1 << 30,)


def _resolve_via_chain(repo: panproto.Repository, ref: str) -> str:
    """For names that don't directly tag a commit (``v0.10.0`` shares
    its commit with ``v0.9.0`` because their grammars are byte-
    identical, so only one VCS commit holds both), walk the chain
    and return the last commit whose tag is ``<= ref`` by semver
    order. For ``"HEAD"`` returns the VCS head commit (the working-
    tree grammar's commit if it differs from the last tagged
    release; otherwise the same as the latest release)."""
    if ref == "HEAD":
        return repo.head() or ""
    tags = {name: cid for name, cid in repo.list_tags()}
    if ref in tags:
        return tags[ref]
    head_id = repo.head() or ""
    target_key = _semver_key(ref)
    sorted_tags = sorted(tags.items(), key=lambda nc: _semver_key(nc[0]))
    if not sorted_tags:
        return head_id
    latest_tag_name, latest_tag_id = sorted_tags[-1]
    latest_key = _semver_key(latest_tag_name)
    # If ``ref`` is greater than the most recent tag AND head() points
    # past it, the working-tree commit IS this revision (typical for
    # the next-release name that hasn't been tagged yet -- e.g.
    # ``v0.11.0`` when only ``v0.10.0`` and earlier are tagged).
    if target_key > latest_key and head_id and head_id != latest_tag_id:
        return head_id
    # Otherwise: highest tag whose semver <= target_key.
    candidates = [
        (name, cid) for name, cid in tags.items() if _semver_key(name) <= target_key
    ]
    if candidates:
        candidates.sort(key=lambda nc: _semver_key(nc[0]))
        return candidates[-1][1]
    return head_id


def check_chain_coverage(
    chain: tuple[str, ...],
    converters_by_pair: dict[tuple[str, str], frozenset[str]],
) -> list[DiffCoverageReport]:
    """Run `diff_coverage` on every adjacent pair in
    ``chain``. ``converters_by_pair`` maps each ``(from, to)`` pair
    to the set of source-side rule names that hop's
    ``_DECL_CONVERTERS`` declares; pairs not present in the map are
    treated as having an empty converter set (so every removed rule
    will surface as uncovered)."""
    reports: list[DiffCoverageReport] = []
    for i in range(len(chain) - 1):
        from_ref = chain[i]
        to_ref = chain[i + 1]
        decl = converters_by_pair.get((from_ref, to_ref), frozenset())
        reports.append(diff_coverage(from_ref, to_ref, decl))
    return reports


# ---------------------------------------------------------------------------
# Blame: when a migrator encounters an unknown rule, ask the VCS
# when it appeared or disappeared.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlameReport:
    """Where in the grammar's history a rule first appeared (or was
    last seen)."""

    rule: str
    introduced_at_commit: str | None
    introduced_at_tag: str | None
    last_present_at_commit: str | None
    last_present_at_tag: str | None


def blame_kind(rule: str) -> BlameReport:
    """Report when a tree-sitter rule was introduced or removed in
    the grammar's VCS history. Used by the migrator's failure
    path to point the user at the specific release that needs a
    new converter."""
    repo = _open_repo()
    head = repo.head() or ""

    introduced_commit: str | None = None
    try:
        info = repo.blame_vertex(head, rule)
        introduced_commit = str(info.get("commit"))  # type: ignore[union-attr]
    except Exception:
        introduced_commit = None

    introduced_tag = (
        _tag_for_commit(repo, introduced_commit) if introduced_commit else None
    )

    # Walk log oldest-first, find the last commit whose schema
    # contains the rule. If the current HEAD schema contains it,
    # introduced_commit is the answer to both "introduced" and
    # "last present." Otherwise scan history.
    last_commit: str | None = None
    for entry in repo.log():
        cid = str(entry["id"])
        try:
            schema = repo.schema_at(cid)
        except Exception:
            continue
        if schema.has_vertex(rule):
            last_commit = cid
            break  # log() is newest-first; the first hit is the last presence.
    last_tag = _tag_for_commit(repo, last_commit) if last_commit else None

    return BlameReport(
        rule=rule,
        introduced_at_commit=introduced_commit,
        introduced_at_tag=introduced_tag,
        last_present_at_commit=last_commit,
        last_present_at_tag=last_tag,
    )


def _tag_for_commit(
    repo: panproto.Repository,
    commit_id: str | None,
) -> str | None:
    if commit_id is None:
        return None
    for tag_name, cid in repo.list_tags():
        if cid == commit_id:
            return tag_name
    return None
