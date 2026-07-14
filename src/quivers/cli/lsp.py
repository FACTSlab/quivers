"""``qvr lsp`` subcommand: start the QVR language server.

By default runs over stdio (which is what every editor expects).
Pass ``--tcp PORT`` to bind a TCP listener for debugging or for
clients that prefer socket transport.
"""

from __future__ import annotations

import argparse
import sys


def main(args: argparse.Namespace) -> int:
    try:
        from quivers.lsp import build_server
    except ImportError as e:
        sys.stderr.write(
            f"pygls not installed ({e}); install with `pip install 'quivers[lsp]'`.\n"
        )
        return 2
    server = build_server()
    if args.tcp is not None:
        server.start_tcp("127.0.0.1", args.tcp)
    else:
        server.start_io()
    return 0


def _entry() -> int:
    """Console-script entry point for ``qvr-lsp``."""
    parser = argparse.ArgumentParser(prog="qvr-lsp")
    parser.add_argument("--tcp", type=int, default=None, metavar="PORT")
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="run over stdio (the default; accepted for compatibility).",
    )
    args = parser.parse_args()
    return main(args)


__all__ = ["main", "_entry"]
