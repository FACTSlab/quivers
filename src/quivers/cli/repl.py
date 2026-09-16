"""``qvr repl`` entry point.

Chooses the Textual TUI when stdin/stdout are a TTY and Textual is
importable, otherwise falls back to the prompt_toolkit single-line
front end. ``--plain`` forces the fallback even when the TTY exists.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import sys

from quivers.cli.repl_prompt import run_plain
from quivers.cli.repl_session import ReplSession

try:
    from quivers.cli.repl_tui import run_tui
except ImportError as error:
    _TUI_UNAVAILABLE: ImportError | None = error
    run_tui: Callable[[ReplSession], int] | None = None
else:
    _TUI_UNAVAILABLE = None


def main(args: argparse.Namespace) -> int:
    """Start a session, loading a file first when one is given.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed arguments: an optional file, ``--plain``, and
        ``--target``.

    Returns
    -------
    int
        The exit status of the front end.
    """
    session = ReplSession()
    target = getattr(args, "target", None)
    if target:
        response = session.set_option(f"target={target}")
        if not response.ok:
            for diag in response.diagnostics:
                sys.stderr.write(_format_diag(diag) + "\n")
            return 1
    if args.file is not None:
        response = session.load_file(args.file)
        for diag in response.diagnostics:
            sys.stderr.write(_format_diag(diag) + "\n")

    if args.plain or not sys.stdin.isatty():
        return run_plain(session)

    if run_tui is None:
        sys.stderr.write(
            f"textual not installed ({_TUI_UNAVAILABLE}); falling back to plain "
            "mode. Install with `pip install 'quivers[repl]'`.\n"
        )
        return run_plain(session)

    return run_tui(session)


def _format_diag(d) -> str:
    loc = f":{d.line}:{d.col}" if d.line else ""
    return f"{d.severity}[{d.code}]{loc}: {d.message}"


__all__ = ["main"]
