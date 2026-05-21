"""QVR Language Server.

Public entry: `build_server` returns a configured pygls
``LanguageServer`` that speaks LSP 3.17 over stdio. The
[`quivers.cli.lsp.main`][quivers.cli.lsp.main] CLI subcommand wraps this with stdio /
TCP plumbing.

The server reuses every analytic component the REPL uses:
[`quivers.cli.repl_session.ReplSession`][quivers.cli.repl_session.ReplSession]-style state per document,
[`quivers.cli.repl_highlight`][quivers.cli.repl_highlight] for semantic tokens,
[`quivers.cli.repl_complete`][quivers.cli.repl_complete] for completion. There is no duplicate
parser, type-checker, or token vocabulary.
"""

from quivers.lsp.server import build_server

__all__ = ["build_server"]
