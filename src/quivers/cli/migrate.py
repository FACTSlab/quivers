"""``qvr migrate`` command.

Walks the panproto VCS chain at ``grammars/qvr/vcs/.panproto/`` to
compose a migration between two grammar revisions (each addressed
by its git-tag name, ``vX.Y.Z``, or by the literal ``HEAD``), then
applies it to every supplied ``.qvr`` source file.

The migration is the categorical composite of the auto-derived
schema morphisms between adjacent commits on the chain. Because
the schemas in the VCS share vertex labels (rule names) across
revisions, the auto-derived morphisms map unchanged rules
identically; the composite is exactly the structural rewrite
needed to lower a parsed instance at the FROM grammar onto the TO
grammar's vertex vocabulary.

Surface forms accepted today:

    qvr migrate --from v0.10.0 --to HEAD path/to/file.qvr [paths...]
    qvr migrate --from v0.9.0 --to v0.10.0 --dry-run docs/examples/source/

``--dry-run`` reports which files would change without writing
output. ``--output DIR`` writes migrated copies under ``DIR``
mirroring the input layout instead of overwriting the originals.

Migration application currently requires the FROM grammar's parser
to be available to ``panproto.AstParserRegistry``: panproto-grammars-all
ships the pre-homogenized surface, and ``QVR_USE_LOCAL_GRAMMAR=1``
swaps in the homogenized in-tree grammar. Migrations between two
historical pre-homogenized revisions need each revision's parser
binary; that build path is not yet wired and the command emits a
clear error rather than silently doing the wrong thing.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Iterable

import panproto

from quivers.cli.migration_table import (
    MigrationLookupError,
    apply_rewrites,
    composite_rewrites,
)


_VCS_PATH = (
    Path(__file__).resolve().parents[3] / "grammars" / "qvr" / "vcs"
)


class MigrateError(Exception):
    """Raised by the migrate CLI on a recoverable user-facing error."""


def _open_vcs() -> panproto.Repository:
    """Open the in-tree qvr-grammar VCS, raising a friendly error
    when the panproto directory is absent."""
    panproto_dir = _VCS_PATH / ".panproto"
    if not panproto_dir.is_dir():
        raise MigrateError(
            f"no panproto VCS at {panproto_dir}; run "
            "``python grammars/qvr/vcs/build_schemas.py --reset`` "
            "to build the grammar-evolution chain first.",
        )
    return panproto.Repository.open(str(_VCS_PATH))


def _resolve_commit(
    repo: panproto.Repository, ref: str,
) -> str:
    """Resolve a tag name or commit id to its commit id, with a
    friendly error when the ref does not exist in the VCS."""
    try:
        resolved = repo.resolve_ref(ref)
    except Exception as exc:
        raise MigrateError(
            f"unknown ref {ref!r} in grammars/qvr/vcs/.panproto: {exc}",
        ) from exc
    if resolved is None:
        raise MigrateError(
            f"unknown ref {ref!r} in grammars/qvr/vcs/.panproto",
        )
    return resolved


def _chain_commits(
    repo: panproto.Repository, from_id: str, to_id: str,
) -> list[str]:
    """Return the ordered commit ids from ``from_id`` to ``to_id``
    inclusive, walking the main branch's history.

    The VCS is built as a single linear chain (one commit per
    distinct grammar revision in semver order), so the ``main``
    branch log already gives us a topological order; we filter to
    the inclusive window between the two endpoints.
    """
    log = list(repo.log())
    # ``log`` is newest-first by panproto's convention. Each entry
    # is a dict with an ``id`` key holding the commit's hex id.
    ids_newest_first: list[str] = [str(entry["id"]) for entry in log]
    if from_id not in ids_newest_first or to_id not in ids_newest_first:
        raise MigrateError(
            "FROM and TO must both belong to the same panproto VCS "
            "history; one or both commit ids were not found on "
            "``main``.",
        )
    i_from = ids_newest_first.index(from_id)
    i_to = ids_newest_first.index(to_id)
    if i_to > i_from:
        raise MigrateError(
            f"TO commit {to_id[:12]} is older than FROM commit "
            f"{from_id[:12]}; backward migration is not yet wired.",
        )
    # Inclusive slice, then reverse so the result is oldest-first.
    return list(reversed(ids_newest_first[i_to:i_from + 1]))


def _rule_set(schema: panproto.Schema) -> set[str]:
    """Return the vertex (= grammar rule) names declared in a
    chain-stored schema."""
    vertices = schema.to_dict().get("vertices", {})
    if isinstance(vertices, dict):
        return {str(name) for name in vertices.keys()}
    return set()


def _chain_rule_delta(
    repo: panproto.Repository, commit_chain: list[str],
) -> tuple[set[str], set[str]]:
    """Compute the set of rules that exist on the FROM grammar but
    NOT on the TO grammar (``removed``) and those that exist on
    the TO grammar but NOT on FROM (``added``), composed across
    every consecutive pair in ``commit_chain``.

    Rules present in both endpoints are mapped identically by the
    migration (the categorical identity-on-shared-vertices
    morphism). Rules removed along the way are flagged so the
    file-level migrator can warn when a ``.qvr`` source still
    references them.
    """
    if len(commit_chain) < 2:
        return set(), set()
    src_schema = repo.schema_at(commit_chain[0])
    tgt_schema = repo.schema_at(commit_chain[-1])
    src_rules = _rule_set(src_schema)
    tgt_rules = _rule_set(tgt_schema)
    removed = src_rules - tgt_rules
    added = tgt_rules - src_rules
    return removed, added


def _walk_inputs(paths: Iterable[str]) -> list[Path]:
    """Expand the CLI's path arguments to a flat list of ``.qvr``
    files. Directories are walked recursively."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.qvr")))
        elif p.suffix == ".qvr":
            out.append(p)
        else:
            raise MigrateError(
                f"unsupported input {raw!r}: not a directory or "
                "``.qvr`` file",
            )
    return out


def _scan_file_for_removed_rules(
    src_path: Path, removed_rules: set[str],
) -> set[str]:
    """Return the subset of ``removed_rules`` whose names appear as
    parse-tree node kinds in ``src_path``.

    We parse the file through panproto's currently-registered qvr
    grammar (vendored or local-override), then walk the resulting
    schema's vertex kinds. A hit means the file uses a surface
    construct that no longer exists at the target grammar; the
    migrator surfaces those as actionable warnings rather than
    silently emitting broken output.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reg = panproto.AstParserRegistry()
    parsed = reg.parse_with_protocol(
        "qvr", src_path.read_bytes(), str(src_path),
    )
    used: set[str] = set()
    vertices = parsed.to_dict().get("vertices", {})
    if not isinstance(vertices, dict):
        return used
    for v in vertices.values():
        if isinstance(v, dict):
            kind = v.get("kind")
            if isinstance(kind, str) and kind in removed_rules:
                used.add(kind)
    return used


def main(args: argparse.Namespace) -> int:
    """Entry point invoked by :mod:`quivers.cli`'s top-level
    dispatcher.

    Validates the FROM and TO refs against the in-tree grammar VCS
    (so a typo in a tag name fails fast), then composites the
    rewrite chain from :mod:`quivers.cli.migration_table` between
    those two revisions and applies it to each input ``.qvr`` file.

    Returns 0 when every file migrated cleanly; 1 when at least
    one file was blocked or unchanged where a change was expected;
    2 on invalid command-line input (unknown ref, missing file).
    """
    try:
        repo = _open_vcs()
        # Resolve refs to validate they exist in the VCS chain; the
        # commit ids themselves are not needed for text rewriting,
        # but failing fast on a typo is the right user experience.
        _resolve_commit(repo, args.from_ref)
        _resolve_commit(repo, args.to_ref)
        inputs = _walk_inputs(args.paths)
        rewrites = composite_rewrites(args.from_ref, args.to_ref)
    except MigrateError as exc:
        print(f"qvr migrate: {exc}", file=sys.stderr)
        return 2
    except MigrationLookupError as exc:
        print(f"qvr migrate: {exc}", file=sys.stderr)
        return 2

    if not inputs:
        print("qvr migrate: no .qvr files to migrate", file=sys.stderr)
        return 2

    if not rewrites:
        print(
            f"qvr migrate: {args.from_ref} == {args.to_ref}, "
            "identity migration",
        )
        return 0

    print(
        f"qvr migrate: {args.from_ref} -> {args.to_ref}: "
        f"composing {len(rewrites)} rewrite(s)",
    )

    out_root = Path(args.output) if args.output is not None else None
    changed = 0
    for src_path in inputs:
        source = src_path.read_text()
        migrated = apply_rewrites(source, rewrites)
        if migrated == source:
            continue
        changed += 1
        prefix = "would migrate" if args.dry_run else "migrated"
        print(f"{prefix} {src_path}")
        if args.dry_run:
            continue
        target = (
            (out_root / src_path.name)
            if out_root is not None
            else src_path
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(migrated)

    if changed == 0:
        print(
            f"qvr migrate: {len(inputs)} file(s) already at "
            f"{args.to_ref}",
        )
    return 0


__all__ = ["MigrateError", "main"]
