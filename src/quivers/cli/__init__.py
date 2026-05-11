"""Command-line entry points for quivers.

The :func:`main` function is registered as the ``qvr`` console script
in ``pyproject.toml``.

Subcommands:

- ``qvr check FILES...`` — parse + compile every supplied ``.qvr``
  file, emitting structured diagnostics. Exits 0 on full success,
  non-zero when any file produces an error.

Output format: human-readable by default, structured JSON when
``--json`` is supplied. Each diagnostic carries:

- ``file``: source path,
- ``line``, ``col``: 1-indexed source location,
- ``severity``: ``"error"``, ``"warning"``, or ``"note"``,
- ``code``: stable diagnostic code (``parse``, ``compile``,
  ``effect_constraint``, ``residuated_constraint``),
- ``message``: human-readable description.
"""

from quivers.cli.check import main as check_main


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="qvr", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    check = sub.add_parser(
        "check",
        help="Parse + compile .qvr files and report diagnostics.",
    )
    check.add_argument("files", nargs="+", help="Paths to .qvr files.")
    check.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON diagnostics on stdout.",
    )

    args = parser.parse_args()
    if args.cmd == "check":
        return check_main(args.files, json_output=args.json)
    parser.print_help(sys.stderr)
    return 2


__all__ = ["main"]
