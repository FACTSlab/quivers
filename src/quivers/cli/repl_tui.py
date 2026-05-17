"""Textual TUI for ``qvr repl``.

Layout
------

::

    +-------------------------------+-------------------+
    | Input (multiline TextArea)    | Environment       |
    |                               | (objects/spaces/  |
    |                               |  morphisms/rules) |
    +-------------------------------+                   |
    | Output log (rich text)        |                   |
    |                               |                   |
    +-------------------------------+-------------------+
    | Diagnostics                                       |
    +---------------------------------------------------+

Key bindings:

- ``ctrl+enter`` evaluate
- ``ctrl+l`` clear output
- ``ctrl+r`` reload
- ``ctrl+d`` / ``ctrl+q`` quit
- ``f1`` help

The widgets all draw from a single :class:`ReplSession` instance.
Highlighting is driven by :mod:`quivers.cli.repl_highlight`, so the
TUI tracks the live grammar.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quivers.cli.repl_session import ReplSession


def run_tui(session: "ReplSession") -> int:
    """Run the Textual REPL App on ``session``."""
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, RichLog, Static, TextArea, Tree

    from quivers.cli.repl_highlight import to_rich_text

    class QvrRepl(App):
        CSS = """
        Screen { layout: vertical; }
        #top { height: 1fr; }
        #input-pane { width: 2fr; }
        #env-pane { width: 1fr; border-left: tall $primary; }
        #input { height: 1fr; }
        #output { height: 1fr; border-top: tall $primary; }
        #diagnostics { height: 6; border-top: tall $error; }
        """

        BINDINGS = [
            Binding("ctrl+enter", "submit", "Eval", show=True),
            Binding("ctrl+j", "submit", "Eval", show=False),
            Binding("ctrl+l", "clear", "Clear", show=True),
            Binding("ctrl+r", "reload", "Reload", show=True),
            Binding("ctrl+d", "quit", "Quit", show=True),
            Binding("ctrl+q", "quit", "Quit", show=False),
            Binding("f1", "help", "Help", show=True),
        ]

        prompt: reactive[str] = reactive("qvr> ")

        def __init__(self, session) -> None:  # type: ignore[no-untyped-def]
            super().__init__()
            self.session = session

        def compose(self) -> ComposeResult:
            yield Header(name="quivers REPL")
            with Horizontal(id="top"):
                with Vertical(id="input-pane"):
                    yield TextArea(
                        id="input",
                        language=None,
                        show_line_numbers=True,
                    )
                    yield RichLog(
                        id="output",
                        wrap=True,
                        markup=False,
                        highlight=False,
                    )
                with Vertical(id="env-pane"):
                    yield Tree("environment", id="env")
            yield Static("", id="diagnostics")
            yield Footer()

        def on_mount(self) -> None:
            input_widget: TextArea = self.query_one("#input", TextArea)
            input_widget.focus()
            self._refresh_env()
            if self.session.loaded_path is not None:
                self._log(f"loaded {self.session.loaded_path}")
                self._refresh_env()

        # --- actions ----------------------------------------------------

        def action_submit(self) -> None:
            input_widget: TextArea = self.query_one("#input", TextArea)
            text = input_widget.text
            if not text.strip():
                return
            self._log(f"> {text}")
            response = self.session.dispatch(text)
            if response.body == "__quit__":
                self.exit(0)
                return
            self._render(response)
            self._refresh_env()
            input_widget.clear()

        def action_clear(self) -> None:
            self.query_one("#output", RichLog).clear()

        def action_reload(self) -> None:
            self._render(self.session.reload())
            self._refresh_env()

        def action_help(self) -> None:
            self._render(self.session.help(""))

        def on_tree_node_selected(self, event) -> None:  # type: ignore[no-untyped-def]
            """Clicking a leaf in the env browser runs `:info NAME`.

            Branch nodes (`objects`, `spaces`, `morphisms`, `rules` and
            the file-path root) have no associated name in the env, so
            ignore them.
            """
            node = event.node
            if not node.is_root and not node.allow_expand:
                name = str(node.label)
                self._log(f"> :info {name}")
                self._render(self.session.info(name))

        # --- helpers ----------------------------------------------------

        def _log(self, message: str) -> None:
            self.query_one("#output", RichLog).write(message)

        def _render(self, response) -> None:  # type: ignore[no-untyped-def]
            log: RichLog = self.query_one("#output", RichLog)
            if response.body:
                if response.body_kind == "qvr":
                    # Highlight QVR bodies line-by-line; comment-only
                    # lines (starting with `--`, the REPL's own
                    # annotation marker, or `##` / `#` doc / line
                    # comments from the source) pass through dimmed.
                    for line in response.body.splitlines() or [""]:
                        stripped = line.lstrip()
                        if stripped.startswith("--") and not stripped.startswith(
                            "->"
                        ):
                            from rich.text import Text

                            log.write(Text(line, style="italic dim"))
                        else:
                            log.write(to_rich_text(line))
                else:
                    log.write(response.body)
            diag_panel: Static = self.query_one("#diagnostics", Static)
            if response.diagnostics:
                lines = []
                for d in response.diagnostics:
                    loc = f":{d.line}:{d.col}" if d.line else ""
                    lines.append(f"[{d.severity}] {d.code}{loc}: {d.message}")
                diag_panel.update("\n".join(lines))
            else:
                diag_panel.update("")

        def _refresh_env(self) -> None:
            tree: Tree = self.query_one("#env", Tree)
            tree.clear()
            compiler = self.session._compiler  # noqa: SLF001
            if compiler is None:
                tree.root.label = "(no module loaded)"
                return
            tree.root.label = (
                str(self.session.loaded_path)
                if self.session.loaded_path is not None
                else "<module>"
            )
            tree.root.expand()
            objects = tree.root.add("objects", expand=True)
            for name in sorted(compiler.objects):
                objects.add_leaf(name)
            spaces = tree.root.add("spaces", expand=True)
            for name in sorted(compiler.spaces):
                spaces.add_leaf(name)
            morphisms = tree.root.add("morphisms", expand=True)
            for name in sorted(compiler.morphisms):
                morphisms.add_leaf(name)
            rules = tree.root.add("rules", expand=False)
            for name in sorted(compiler.rules):
                rules.add_leaf(name)

    app = QvrRepl(session)
    app.run()
    return 0


__all__ = ["run_tui"]
