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

Key bindings (chosen to work uniformly on macOS, Linux, and Windows):

- ``ctrl+g``  evaluate the current buffer (semantic: "go"; not bound by
  Textual's TextArea, not reserved by any default OS shortcut, not
  blocked by terminal XON/XOFF)
- ``ctrl+o``  evaluate (alternate; same reasoning)
- ``f8``      evaluate (for users who enabled "F1, F2 as standard
  function keys" in System Settings; F5 is reserved by macOS Dictation,
  F3/F4 by Mission Control/Launchpad, F11 by Show Desktop)
- ``ctrl+l``  clear the eval log
- ``ctrl+r``  reload the loaded file
- ``ctrl+q``  quit
- ``f1``      help

Ctrl+Enter / Alt+Enter / Cmd+Enter are deliberately NOT used because
macOS Terminal.app and the default iTerm2 profile drop them at the
emulator layer before the TTY ever receives them. They can be enabled
per-emulator (iTerm2: send ``\\x1b[13;5u``; Windows Terminal: same),
in which case the hidden ctrl+enter / ctrl+j fallbacks below will fire.

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
            # Eval. priority=True makes the app-level binding fire
            # even when the TextArea would otherwise consume the key
            # for its own editor action. ctrl+g and ctrl+o are
            # uniquely free across:
            #   * Textual's TextArea (verified against its BINDINGS
            #     table -- a/b/c/d/e/f/k/u/v/w/x/y/z are TextArea ops)
            #   * macOS system shortcuts (no global capture)
            #   * Linux/Windows terminals
            #   * Terminal driver flow control (XON/XOFF claims s/q;
            #     g/o are unaffected)
            Binding("ctrl+g", "submit", "Eval", show=True, priority=True),
            Binding("ctrl+o", "submit", "Eval", show=True, priority=True),
            # F8 reaches us only when the user has set "Use F1, F2 as
            # standard function keys" in System Settings (or is on a
            # keyboard without the Fn row). F5 is reserved by macOS
            # Dictation; F3/F4/F11 by Mission Control/Launchpad/Show
            # Desktop. F8 is unclaimed on every OS we target.
            Binding("f8", "submit", "Eval", show=True, priority=True),
            # Hidden fallbacks for terminals configured to forward
            # Ctrl+Enter (modifyOtherKeys / CSI-u): iTerm2 with a
            # custom keymap, Windows Terminal, Kitty, Wezterm, etc.
            Binding("ctrl+enter", "submit", "Eval", show=False, priority=True),
            Binding("ctrl+j", "submit", "Eval", show=False, priority=True),
            Binding("ctrl+l", "clear", "Clear", show=True),
            Binding("ctrl+r", "reload", "Reload", show=True),
            Binding("ctrl+q", "quit", "Quit", show=True),
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
                    env_kinds = self.session.env_kinds()
                    for line in response.body.splitlines() or [""]:
                        stripped = line.lstrip()
                        if stripped.startswith("--") and not stripped.startswith(
                            "->"
                        ):
                            from rich.text import Text

                            log.write(Text(line, style="italic dim"))
                        else:
                            log.write(to_rich_text(line, env_kinds=env_kinds))
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
