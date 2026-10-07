"""QVR Language Server.

Public entry: `build_server` returns a configured pygls
``LanguageServer`` that speaks LSP 3.17 over stdio. The ``qvr lsp``
subcommand wraps it with stdio and TCP plumbing.

The server reuses every analytic component the REPL uses: per-document
state in the style of [`ReplSession`][quivers.cli.ReplSession],
[`tokenize`][quivers.cli.tokenize] for semantic tokens, and
[`all_completions`][quivers.cli.all_completions] for completion. There
is no duplicate parser, type-checker, or token vocabulary.
"""

from quivers.lsp.server import SERVER_NAME, SERVER_VERSION, build_server

__all__ = ["SERVER_NAME", "SERVER_VERSION", "build_server"]
