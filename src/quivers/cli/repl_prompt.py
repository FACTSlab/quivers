"""prompt_toolkit single-line / multi-line REPL frontend.

Used when stdin/stdout is not a TTY or when the user passes
``--plain``. Honours the same :class:`ReplSession` as the Textual TUI
and the Jupyter kernel; the only difference is how input is gathered
and output is rendered.

Two render paths:

- If ``rich`` is available we render :class:`ReplResponse` bodies with
  syntax highlighting via :func:`quivers.cli.repl_highlight.to_rich_text`.
- Otherwise we print the body verbatim. Diagnostics always print to
  stderr.

History persists at ``~/.config/quivers/history``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quivers.cli.repl_session import ReplResponse, ReplSession


def run_plain(session: "ReplSession") -> int:
    """Drive ``session`` with prompt_toolkit (or bare stdin if missing)."""
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.lexers import PygmentsLexer

        from quivers.cli.repl_complete import all_completions
        from quivers.dsl.pygments_lexer import QvrLexer

        history_path = _history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)

        class _Completer(Completer):
            def get_completions(self, document, complete_event):  # noqa: D401
                del complete_event
                word = document.get_word_before_cursor(WORD=True)
                for c in all_completions(session, word):
                    yield Completion(
                        text=c.text,
                        start_position=-len(word),
                        display_meta=c.detail,
                    )

        ps: PromptSession = PromptSession(
            message="qvr> ",
            history=FileHistory(str(history_path)),
            completer=_Completer(),
            lexer=PygmentsLexer(QvrLexer),
            multiline=False,
        )
        getline = lambda: ps.prompt()
    except ImportError:
        getline = _bare_input

    while True:
        try:
            line = getline()
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n")
            return 0
        if line is None:
            return 0
        # Allow `:reload` to auto-fire if file changed on disk.
        auto = session.autoreload_if_stale()
        if auto is not None:
            _render(auto)
        response = session.dispatch(line)
        if response.body == "__quit__":
            return 0
        _render(response)


def _bare_input() -> str | None:
    try:
        return input("qvr> ")
    except EOFError:
        return None


def _render(response: "ReplResponse") -> None:
    # Try rich for nicer output, fall back to plain print.
    try:
        from rich.console import Console
        from rich.syntax import Syntax

        console = Console()
        if response.body:
            if response.body_kind == "qvr":
                console.print(Syntax(response.body, "qvr", theme="ansi_dark"))
            elif response.body_kind == "json":
                console.print(Syntax(response.body, "json", theme="ansi_dark"))
            elif response.body_kind == "markdown":
                from rich.markdown import Markdown

                console.print(Markdown(response.body))
            else:
                from quivers.cli.repl_highlight import to_rich_text

                # Plain text bodies aren't QVR, so render as-is.
                console.print(response.body)
                del to_rich_text
        for d in response.diagnostics:
            console.print(_format_diag(d), style="red" if d.severity == "error" else "yellow")
    except ImportError:
        if response.body:
            sys.stdout.write(response.body + "\n")
        for d in response.diagnostics:
            sys.stderr.write(_format_diag(d) + "\n")


def _format_diag(d) -> str:
    loc = f":{d.line}:{d.col}" if d.line else ""
    return f"{d.severity}[{d.code}]{loc}: {d.message}"


def _history_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "quivers" / "history"


__all__ = ["run_plain"]
