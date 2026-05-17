"""``qvr repl`` entry point.

Chooses the Textual TUI when stdin/stdout are a TTY and Textual is
importable, otherwise falls back to the prompt_toolkit single-line
front end. ``--plain`` forces the fallback even when the TTY exists.
"""

from __future__ import annotations

import argparse
import sys

from quivers.cli.repl_session import ReplSession


def main(args: argparse.Namespace) -> int:
    session = ReplSession()
    if args.file is not None:
        response = session.load_file(args.file)
        for diag in response.diagnostics:
            sys.stderr.write(_format_diag(diag) + "\n")
        if response.body and not args.plain:
            # The TUI will re-render once it starts; in plain mode we
            # echo so the user sees the load result before the prompt.
            pass

    if args.plain or not sys.stdin.isatty():
        from quivers.cli.repl_prompt import run_plain

        return run_plain(session)

    try:
        from quivers.cli.repl_tui import run_tui
    except ImportError as e:
        sys.stderr.write(
            f"textual not installed ({e}); falling back to plain mode. "
            "Install with `pip install 'quivers[repl]'`.\n"
        )
        from quivers.cli.repl_prompt import run_plain

        return run_plain(session)

    return run_tui(session)


def _format_diag(d) -> str:
    loc = f":{d.line}:{d.col}" if d.line else ""
    return f"{d.severity}[{d.code}]{loc}: {d.message}"


__all__ = ["main"]
