"""``qvr check`` — parse + compile .qvr files and report diagnostics.

Implementation
--------------

For each input file:

1. ``parse(source)`` is invoked; ``ParseError`` produces a
   ``code="parse"`` diagnostic.
2. ``Compiler(module).compile()`` is invoked; ``CompileError``
   produces a ``code="compile"`` diagnostic.
3. The constraint solver in [`quivers.dsl.constraints`][quivers.dsl.constraints] walks the
   parsed AST for residuated-universe and effect-typed-application
   well-formedness violations; each emits a ``code="residuated_constraint"``
   or ``code="effect_constraint"`` diagnostic.

Successful files emit no diagnostics; on success the human-readable
mode prints ``"OK file.qvr"``.

Exit codes:

- ``0`` — every file compiled without diagnostics,
- ``1`` — at least one file produced an ``error`` diagnostic,
- ``2`` — usage / IO error.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from quivers.dsl import Compiler, CompileError, ParseError, parse
from quivers.dsl.constraints import check_constraints


type Severity = Literal["error", "warning", "note"]


@dataclass(frozen=True)
class Diagnostic:
    """One structured diagnostic message."""

    file: str
    line: int
    col: int
    severity: Severity
    code: str
    message: str


def _check_one(path: Path) -> list[Diagnostic]:
    """Run the parse + constraint + compile pipeline on a single file."""
    try:
        source = path.read_bytes()
    except OSError as e:
        return [
            Diagnostic(
                file=str(path),
                line=0,
                col=0,
                severity="error",
                code="io",
                message=f"could not read file: {e}",
            )
        ]

    diags: list[Diagnostic] = []

    try:
        module = parse(source, file_path=str(path))
    except ParseError as e:
        diags.append(
            Diagnostic(
                file=str(path),
                line=0,
                col=0,
                severity="error",
                code="parse",
                message=str(e),
            )
        )
        return diags

    # Constraint solver runs before compile so users see structural
    # diagnostics even when compilation would also fail.
    diags.extend(
        Diagnostic(
            file=str(path),
            line=v.line,
            col=v.col,
            severity="error",
            code=v.code,
            message=v.message,
        )
        for v in check_constraints(module)
    )

    try:
        Compiler(module).compile()
    except CompileError as e:
        diags.append(
            Diagnostic(
                file=str(path),
                line=getattr(e, "line", 0),
                col=getattr(e, "col", 0),
                severity="error",
                code="compile",
                message=str(e),
            )
        )

    return diags


def main(files: list[str], *, json_output: bool = False) -> int:
    """Run ``qvr check`` on a list of paths.

    Parameters
    ----------
    files : list of str
        Paths to ``.qvr`` files.
    json_output : bool
        When True, emit a single JSON document on stdout containing
        the full diagnostic list. When False (default), emit
        human-readable lines.

    Returns
    -------
    int
        Exit code. 0 on full success; 1 on any error diagnostic.
    """
    paths = [Path(f) for f in files]
    all_diags: list[Diagnostic] = []
    for p in paths:
        all_diags.extend(_check_one(p))

    has_error = any(d.severity == "error" for d in all_diags)

    if json_output:
        payload = {
            "files": [str(p) for p in paths],
            "diagnostics": [asdict(d) for d in all_diags],
            "ok": not has_error,
        }
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
    else:
        files_with_diags: set[str] = set()
        for d in all_diags:
            files_with_diags.add(d.file)
            loc = f"{d.file}:{d.line}:{d.col}" if d.line else d.file
            sys.stderr.write(f"{loc}: {d.severity}[{d.code}]: {d.message}\n")
        for p in paths:
            if str(p) not in files_with_diags:
                sys.stdout.write(f"OK {p}\n")
        if has_error:
            sys.stderr.write(
                f"\n{sum(1 for d in all_diags if d.severity == 'error')} "
                f"error(s) across {len(paths)} file(s)\n"
            )

    return 1 if has_error else 0


__all__ = ["Diagnostic", "main"]
