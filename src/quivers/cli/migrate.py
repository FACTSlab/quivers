"""``qvr migrate`` command.

Lowers ``.qvr`` source forward along the QVR grammar release chain
declared in [`quivers.cli.migrations`][quivers.cli.migrations].

Surface::

    qvr migrate path/to/file.qvr [paths...]
    qvr migrate --from v0.10.0 --to HEAD --dry-run docs/examples/source/
    qvr migrate --output OUT_DIR --to HEAD path/to/file.qvr

``--from`` defaults to the most recent release on the chain (the
penultimate entry of `quivers.cli.migrations.CHAIN`);
``--to`` defaults to ``HEAD``. Both must be members of
`quivers.cli.migrations.CHAIN`; the CLI composes the
intermediate adjacent-pair migrators automatically, so adding a new
release is purely additive to the migrations package.

``--dry-run`` reports which files would change without writing
output. ``--output DIR`` writes migrated copies under ``DIR``
instead of overwriting the originals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from quivers.cli.migrations import (
    CHAIN,
    MigrationError,
    compose_migration,
    vcs_coverage_report,
)


class MigrateError(Exception):
    """Raised by the migrate CLI on a recoverable user-facing error."""


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
                f"unsupported input {raw!r}: not a directory or ``.qvr`` file",
            )
    return out


def _default_from_ref() -> str:
    """Most recent named release on the chain."""
    if len(CHAIN) < 2:
        raise MigrateError(
            "migration chain is empty; no released grammar to migrate from",
        )
    return CHAIN[-2]


def main(args: argparse.Namespace) -> int:
    """Entry point invoked by the top-level ``qvr`` dispatcher.

    Returns 0 when every file migrated cleanly; 2 on invalid input.
    """
    # ``--check`` runs the panproto-VCS coverage check across the
    # migration chain. For each adjacent (from, to) pair, computes
    # the schema diff and reports any source rule removed at the
    # target whose hop migrator has no converter. Non-zero exit
    # if any pair has uncovered removed rules.
    if getattr(args, "check", False):
        reports = vcs_coverage_report()
        any_uncovered = False
        for r in reports:
            print(r.format())
            if r.uncovered_removed:
                any_uncovered = True
        return 1 if any_uncovered else 0

    from_ref = args.from_ref or _default_from_ref()
    to_ref = args.to_ref or "HEAD"

    try:
        migrate_fn = compose_migration(from_ref, to_ref)
        inputs = _walk_inputs(args.paths)
    except (MigrateError, MigrationError) as exc:
        print(f"qvr migrate: {exc}", file=sys.stderr)
        return 2

    if not inputs:
        print("qvr migrate: no .qvr files to migrate", file=sys.stderr)
        return 2

    out_root = Path(args.output) if args.output is not None else None
    changed = 0
    for src_path in inputs:
        source = src_path.read_bytes()
        migrated = migrate_fn(source)
        if migrated == source:
            continue
        changed += 1
        prefix = "would migrate" if args.dry_run else "migrated"
        print(f"{prefix} {src_path}")
        if args.dry_run:
            continue
        target = (out_root / src_path.name) if out_root is not None else src_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(migrated)

    if changed == 0:
        print(
            f"qvr migrate: {len(inputs)} file(s) already at {to_ref}",
        )
    return 0


__all__ = ["MigrateError", "main"]
