"""Textual TUI for ``qvr repl``.

Layout
------

::

    +--- Status bar -----------------------------------------+
    | Input (TextArea, multi-line)    | Env filter           |
    |                                 |----------------------|
    |                                 | Env tree             |
    +---------------------------------|                      |
    | Output log (rich text)          |                      |
    |                                 |                      |
    +---------------------------------+----------------------+
    | Watches (hidden if empty)                              |
    +--------------------------------------------------------+
    | Diagnostics (hidden if empty)                          |
    +--------------------------------------------------------+

Key bindings (chosen to work uniformly on macOS, Linux, and Windows):

- ``ctrl+g``      evaluate the current buffer (semantic: "go")
- ``ctrl+o``      evaluate (alternate)
- ``f8``          evaluate (Fn-row alternate)
- ``ctrl+up``     previous input from history
- ``ctrl+down``   next input from history
- ``tab``         complete the identifier under the cursor
- ``ctrl+p``      command palette
- ``ctrl+l``      clear the eval log
- ``ctrl+r``      reload the loaded file
- ``ctrl+q``      quit
- ``f1``          help

The widgets all draw from a single `ReplSession` instance.
Highlighting is driven by [`quivers.cli.repl_highlight`][quivers.cli.repl_highlight], so the
TUI tracks the live grammar and the live env.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quivers.cli.repl_session import ReplSession


_HISTORY_PATH = (
    Path(os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config"))
    / "quivers"
    / "history"
)


def run_tui(session: "ReplSession") -> int:
    """Run the Textual REPL App on ``session``."""
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.command import Hit, Hits, Provider
    from textual.containers import Horizontal, Vertical
    from textual.widgets import (
        Footer,
        Input,
        RichLog,
        Static,
        TextArea,
        Tree,
    )

    from quivers.cli.repl_complete import all_completions
    from quivers.cli.repl_highlight import to_rich_text

    META_COMMANDS = (
        "load",
        "reload",
        "type",
        "kind",
        "info",
        "doc",
        "browse",
        "dump",
        "edit",
        "trace",
        "save",
        "watch",
        "unwatch",
        "set",
        "help",
        "quit",
    )

    class _MetaCommandProvider(Provider):
        """Surface every meta-command in the Ctrl-P palette."""

        async def search(self, query: str) -> Hits:
            matcher = self.matcher(query)
            for name in META_COMMANDS:
                label = f":{name}"
                score = matcher.match(label)
                if score > 0:
                    yield Hit(
                        score,
                        matcher.highlight(label),
                        partial_run := _make_runner(self.app, label),
                        help=f"meta-command :{name}",
                    )
                    del partial_run

    def _make_runner(app, command: str):  # type: ignore[no-untyped-def]
        def _run() -> None:
            app.run_meta_command(command)

        return _run

    class _QvrTextArea(TextArea):
        """TextArea with brace-balancing and auto-indent on Enter.

        These are pure key handlers; no language hook required, so we
        avoid Textual's tree-sitter shipping list (which lacks QVR).
        """

        BRACE_PAIRS = {"(": ")", "[": "]", "{": "}"}

        async def _on_key(self, event) -> None:  # type: ignore[no-untyped-def]
            char = event.character
            if char in self.BRACE_PAIRS:
                self.insert(char + self.BRACE_PAIRS[char])
                row, col = self.cursor_location
                self.move_cursor((row, col - 1))
                event.prevent_default()
                event.stop()
                return
            if event.key == "enter":
                row, col = self.cursor_location
                current_line = self.document.get_line(row)
                indent = ""
                for c in current_line[:col]:
                    if c in (" ", "\t"):
                        indent += c
                    else:
                        break
                # Indent one level after a line ending in an open
                # brace / colon / arrow / `=` (declaration body).
                trailing = current_line[:col].rstrip()
                extra = ""
                if trailing.endswith(("(", "[", "{", ":", "->", "=", "<-")):
                    extra = "    "
                self.insert("\n" + indent + extra)
                event.prevent_default()
                event.stop()
                return
            await super()._on_key(event)

    class QvrRepl(App):
        CSS = """
        Screen { layout: vertical; }
        #status { dock: top; height: 1; background: $boost; padding: 0 1; }
        #top { height: 1fr; }
        #input-pane { width: 2fr; }
        #env-pane { width: 1fr; border-left: tall $primary; }
        #env-filter { height: 1; border: none; padding: 0 1; }
        #env { height: 1fr; }
        #input { height: 1fr; }
        #output { height: 1fr; border-top: tall $primary; }
        #watches { height: auto; max-height: 8; border-top: tall $accent;
                   padding: 0 1; display: none; }
        #watches.has-content { display: block; }
        #diagnostics { height: auto; max-height: 8; border-top: tall $error;
                       padding: 0 1; display: none; }
        #diagnostics.has-content { display: block; }
        """

        COMMANDS = {_MetaCommandProvider}

        BINDINGS = [
            Binding("ctrl+g", "submit", "Eval", show=True, priority=True),
            Binding("ctrl+o", "submit", "Eval", show=True, priority=True),
            Binding("f8", "submit", "Eval", show=True, priority=True),
            Binding("ctrl+enter", "submit", "Eval", show=False, priority=True),
            Binding("ctrl+j", "submit", "Eval", show=False, priority=True),
            Binding("ctrl+up", "history_prev", "Prev", show=True, priority=True),
            Binding("ctrl+down", "history_next", "Next", show=False, priority=True),
            Binding("tab", "complete", "Complete", show=True, priority=True),
            Binding("ctrl+l", "clear", "Clear", show=True),
            Binding("ctrl+r", "reload", "Reload", show=True),
            Binding("ctrl+q", "quit", "Quit", show=True),
            Binding("f1", "help", "Help", show=True),
        ]

        def __init__(self, session) -> None:  # type: ignore[no-untyped-def]
            super().__init__()
            self.session = session
            self._history: list[str] = _load_history()
            self._history_idx: int | None = None
            self._completion_cycle: list[str] = []
            self._completion_anchor: tuple[int, int] | None = None
            self._completion_prefix_len: int = 0
            self._completion_idx: int = 0

        def compose(self) -> ComposeResult:
            yield Static("", id="status")
            with Horizontal(id="top"):
                with Vertical(id="input-pane"):
                    yield _QvrTextArea(
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
                    yield Input(placeholder="filter env...", id="env-filter")
                    yield Tree("environment", id="env")
            yield Static("", id="watches")
            yield Static("", id="diagnostics")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#input", TextArea).focus()
            self._refresh_status()
            self._refresh_env()
            if self.session.loaded_path is not None:
                self._log(Text(f"loaded {self.session.loaded_path}", style="dim"))
            # File watcher: poll the loaded file every 1s and fire :reload
            # automatically when its mtime advances.
            self.set_interval(1.0, self._poll_for_file_change)

        # --- actions ----------------------------------------------------

        def action_submit(self) -> None:
            input_widget: TextArea = self.query_one("#input", TextArea)
            text = input_widget.text
            if not text.strip():
                return
            self._log(Text("> ", style="bold cyan") + Text(text))
            self._push_history(text)
            response = self.session.dispatch(text)
            if response.body == "__quit__":
                self.exit(0)
                return
            self._render(response)
            self._refresh_env()
            self._refresh_status()
            self._refresh_watches()
            input_widget.clear()
            self._reset_completion()

        def action_clear(self) -> None:
            self.query_one("#output", RichLog).clear()

        def action_reload(self) -> None:
            self._render(self.session.reload())
            self._refresh_env()
            self._refresh_status()
            self._refresh_watches()

        def action_help(self) -> None:
            self._render(self.session.help(""))

        def action_info(self, name: str) -> None:
            """Click handler for identifiers rendered with @click=info."""
            self._log(Text(f"> :info {name}", style="bold cyan"))
            self._render(self.session.info(name))

        def action_open_at(self, path: str, line: str) -> None:
            """Click handler for `path:line:col` footer entries.

            Opens the file at the given line in ``$EDITOR``. Defaults
            to ``vi`` when the env var is unset.
            """
            editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
            try:
                subprocess.Popen(
                    [editor, f"+{line}", path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError as e:
                self._log(Text(f"could not launch editor: {e}", style="red"))

        def action_history_prev(self) -> None:
            if not self._history:
                return
            idx = (
                len(self._history) - 1
                if self._history_idx is None
                else max(0, self._history_idx - 1)
            )
            self._history_idx = idx
            self._set_input_text(self._history[idx])

        def action_history_next(self) -> None:
            if self._history_idx is None:
                return
            idx = self._history_idx + 1
            if idx >= len(self._history):
                self._history_idx = None
                self._set_input_text("")
                return
            self._history_idx = idx
            self._set_input_text(self._history[idx])

        def action_complete(self) -> None:
            input_widget: TextArea = self.query_one("#input", TextArea)
            cursor_row, cursor_col = input_widget.cursor_location
            line = input_widget.document.get_line(cursor_row)
            prefix = _word_prefix(line, cursor_col)
            anchor = (cursor_row, cursor_col)
            if self._completion_anchor == anchor and self._completion_cycle:
                # Cycle: rotate through the existing candidate list.
                self._completion_idx = (self._completion_idx + 1) % len(
                    self._completion_cycle
                )
                self._apply_completion(input_widget, prefix)
                return
            # Build a fresh candidate list at this cursor position.
            completions = all_completions(self.session, prefix)
            if not completions:
                return
            self._completion_cycle = [c.text for c in completions]
            self._completion_idx = 0
            self._completion_prefix_len = len(prefix)
            self._apply_completion(input_widget, prefix)
            self._completion_anchor = (
                cursor_row,
                cursor_col - len(prefix) + len(self._completion_cycle[0]),
            )

        def run_meta_command(self, command: str) -> None:
            """Entry point for command-palette execution."""
            self._set_input_text(command + " ")
            self.query_one("#input", TextArea).focus()

        # --- env / status / watches ------------------------------------

        def on_tree_node_selected(self, event) -> None:  # type: ignore[no-untyped-def]
            node = event.node
            if not node.is_root and not node.allow_expand:
                name = str(node.label)
                self._log(Text(f"> :info {name}", style="bold cyan"))
                self._render(self.session.info(name))

        def on_input_changed(self, event) -> None:  # type: ignore[no-untyped-def]
            if event.input.id == "env-filter":
                self._refresh_env(filter_text=event.value.strip())

        # --- file watcher ----------------------------------------------

        def _poll_for_file_change(self) -> None:
            response = self.session.autoreload_if_stale()
            if response is None:
                return
            self._log(Text("auto-reload", style="italic dim cyan"))
            self._render(response)
            self._refresh_env()
            self._refresh_status()
            self._refresh_watches()

        # --- helpers ----------------------------------------------------

        def _log(self, message) -> None:  # type: ignore[no-untyped-def]
            self.query_one("#output", RichLog).write(message)

        def _set_input_text(self, text: str) -> None:
            input_widget: TextArea = self.query_one("#input", TextArea)
            input_widget.text = text
            input_widget.move_cursor(input_widget.document.end)

        def _apply_completion(self, input_widget, prefix: str) -> None:  # type: ignore[no-untyped-def]
            candidate = self._completion_cycle[self._completion_idx]
            cursor_row, cursor_col = input_widget.cursor_location
            start = (cursor_row, cursor_col - self._completion_prefix_len)
            end = (cursor_row, cursor_col)
            input_widget.replace(candidate, start, end)
            # Reanchor at the new cursor position so subsequent Tab
            # presses cycle relative to the same prefix span.
            new_col = cursor_col - self._completion_prefix_len + len(candidate)
            self._completion_anchor = (cursor_row, new_col)
            self._completion_prefix_len = len(candidate)

        def _reset_completion(self) -> None:
            self._completion_cycle = []
            self._completion_anchor = None
            self._completion_prefix_len = 0
            self._completion_idx = 0

        def _push_history(self, line: str) -> None:
            line = line.strip()
            if not line:
                return
            if self._history and self._history[-1] == line:
                self._history_idx = None
                return
            self._history.append(line)
            self._history_idx = None
            _append_history(line)

        def _render(self, response) -> None:  # type: ignore[no-untyped-def]
            log: RichLog = self.query_one("#output", RichLog)
            if response.body:
                if response.body_kind == "qvr":
                    env_kinds = self.session.env_kinds()
                    for line in response.body.splitlines() or [""]:
                        stripped = line.lstrip()
                        if stripped.startswith("--") and not stripped.startswith("->"):
                            log.write(_decorate_comment_line(line))
                        else:
                            log.write(
                                to_rich_text(
                                    line,
                                    env_kinds=env_kinds,
                                    link_action="info",
                                )
                            )
                else:
                    log.write(response.body)
            self._update_diagnostics(response.diagnostics)

        def _update_diagnostics(self, diagnostics) -> None:  # type: ignore[no-untyped-def]
            panel: Static = self.query_one("#diagnostics", Static)
            if diagnostics:
                rendered = Text()
                first = True
                for d in diagnostics:
                    if not first:
                        rendered.append("\n")
                    first = False
                    loc = f":{d.line}:{d.col}" if d.line else ""
                    style = "bold red" if d.severity == "error" else "yellow"
                    rendered.append(f"[{d.severity}] ", style=style)
                    rendered.append(f"{d.code}{loc}: ", style="dim")
                    rendered.append(d.message)
                panel.update(rendered)
                panel.add_class("has-content")
                # Move the input cursor to the first diagnostic with a
                # known location so the user sees where to look.
                first_loc = next(
                    (d for d in diagnostics if d.line > 0 and d.severity == "error"),
                    None,
                )
                if first_loc is not None:
                    input_widget = self.query_one("#input", TextArea)
                    target_row = max(0, first_loc.line - 1)
                    target_col = max(0, first_loc.col)
                    try:
                        line_text = input_widget.document.get_line(target_row)
                        end_col = min(len(line_text), target_col + 1)
                        input_widget.selection = (  # type: ignore[assignment]
                            (target_row, target_col),
                            (target_row, end_col),
                        )
                    except Exception:
                        pass
            else:
                panel.update("")
                panel.remove_class("has-content")

        def _refresh_watches(self) -> None:
            panel: Static = self.query_one("#watches", Static)
            watches = self.session.watch_results()
            if not watches:
                panel.update("")
                panel.remove_class("has-content")
                return
            env_kinds = self.session.env_kinds()
            text = Text()
            first = True
            for expr, line in watches.items():
                if not first:
                    text.append("\n")
                first = False
                text.append("watch ", style="bold magenta")
                text.append(expr, style="bold")
                text.append(" => ", style="dim")
                text.append(to_rich_text(line, env_kinds=env_kinds))
            panel.update(text)
            panel.add_class("has-content")

        def _refresh_status(self) -> None:
            bar: Static = self.query_one("#status", Static)
            bar.update(self._status_text())

        def _status_text(self):  # type: ignore[no-untyped-def]
            compiler = self.session._compiler  # noqa: SLF001
            path = (
                str(self.session.loaded_path)
                if self.session.loaded_path is not None
                else "<no file>"
            )
            counts = "no env"
            algebra = ""
            if compiler is not None:
                # Show every populated bucket; suppress empty ones so a
                # tiny module's status line stays readable.
                parts: list[str] = []
                for label, mapping in (
                    ("obj", compiler.objects),
                    ("space", compiler.spaces),
                    ("morph", compiler.morphisms),
                    ("rule", compiler.rules),
                    ("prog", compiler.programs),
                    ("ded", compiler.deductions),
                    ("sig", compiler.signatures),
                    ("enc", compiler.encoders),
                    ("dec", compiler.decoders),
                    ("loss", compiler.losses),
                    ("bundle", compiler.bundles),
                    ("contr", compiler.contractions),
                ):
                    n = len(mapping)
                    if n:
                        parts.append(f"{n} {label}")
                counts = " · ".join(parts) if parts else "empty env"
                algebra = type(compiler.algebra).__name__
            text = Text()
            text.append("● ", style="bold green")
            text.append(path, style="bold")
            if algebra:
                text.append("  algebra:", style="dim")
                text.append(f" {algebra}", style="bold cyan")
            text.append("  ", style="dim")
            text.append(counts, style="dim")
            return text

        def _refresh_env(self, *, filter_text: str = "") -> None:
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
            needle = filter_text.lower()

            def keep(name: str) -> bool:
                return not needle or needle in name.lower()

            for ns_name, mapping, builder in (
                ("objects", compiler.objects, _children_for_object),
                ("spaces", compiler.spaces, _children_for_space),
                ("morphisms", compiler.morphisms, _children_for_morphism),
                ("rules", compiler.rules, _children_for_rule),
                ("programs", compiler.programs, _children_for_program),
                ("deductions", compiler.deductions, _children_for_deduction),
                ("signatures", compiler.signatures, _children_for_signature),
                ("encoders", compiler.encoders, _children_for_encoder),
                ("decoders", compiler.decoders, _children_for_decoder),
                ("losses", compiler.losses, _children_for_loss),
                ("bundles", compiler.bundles, _children_for_bundle),
                ("contractions", compiler.contractions, _children_for_contraction),
            ):
                names = [n for n in sorted(mapping) if keep(n)]
                if not names:
                    continue
                ns_node = tree.root.add(ns_name, expand=True)
                for name in names:
                    head, children = builder(name, mapping[name])
                    if children:
                        sub = ns_node.add(head, expand=False)
                        _populate_children(sub, children)
                    else:
                        ns_node.add_leaf(head)

    app = QvrRepl(session)
    app.run()
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_LOCATION_RE = __import__("re").compile(r"(\S+\.qvr):(\d+):(\d+)")


def _decorate_comment_line(line: str):  # type: ignore[no-untyped-def]
    """Render a ``-- ...`` comment line, linking any path:line:col span."""
    from rich.style import Style
    from rich.text import Text

    base = Style.parse("italic dim")
    out = Text(style=base)
    cursor = 0
    for m in _LOCATION_RE.finditer(line):
        if m.start() > cursor:
            out.append(line[cursor : m.start()])
        path, lineno, _col = m.group(1), m.group(2), m.group(3)
        link_style = base + Style(
            underline=True,
            meta={"@click": f"open_at('{path}', '{lineno}')"},
        )
        out.append(m.group(0), style=link_style)
        cursor = m.end()
    if cursor < len(line):
        out.append(line[cursor:])
    return out


def _word_prefix(line: str, col: int) -> str:
    """Return the identifier-ish substring ending at ``col``.

    A leading ``:`` is preserved so meta-command completion works.
    """
    i = col
    while i > 0 and (line[i - 1].isalnum() or line[i - 1] in "_:"):
        i -= 1
    return line[i:col]


def _load_history() -> list[str]:
    if not _HISTORY_PATH.exists():
        return []
    try:
        return [
            line.rstrip("\n")
            for line in _HISTORY_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError:
        return []


def _append_history(line: str) -> None:
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Env-tree child builders
# ---------------------------------------------------------------------------
#
# Each ``_children_for_*`` returns ``(head_label, children)`` where
# ``children`` is a possibly nested list of either ``str`` leaves or
# ``(label, sub_children)`` tuples. ``_populate_children`` walks that
# structure onto a Textual ``Tree`` node. Builders consult only public
# accessors on the runtime objects so they never look into compiler
# internals.


Children = list  # list[str | tuple[str, "Children"]]


def _populate_children(node, items):  # type: ignore[no-untyped-def]
    for item in items:
        if isinstance(item, str):
            node.add_leaf(item)
            continue
        label, sub = item
        if sub:
            sub_node = node.add(label, expand=False)
            _populate_children(sub_node, sub)
        else:
            node.add_leaf(label)


def _pretty(obj):  # type: ignore[no-untyped-def]
    name = getattr(obj, "name", None)
    if isinstance(name, str) and name:
        return name
    return repr(obj)


def _children_for_object(name, obj):  # type: ignore[no-untyped-def]
    card = getattr(obj, "cardinality", None)
    if card is not None:
        return f"{name} : FinSet {card}", []
    return f"{name} : {_pretty(obj)}", []


def _children_for_space(name, sp):  # type: ignore[no-untyped-def]
    dim = getattr(sp, "dim", None)
    if dim is not None:
        return f"{name} : Real {dim}", []
    return f"{name} : {_pretty(sp)}", []


def _children_for_morphism(name, morph):  # type: ignore[no-untyped-def]
    dom = _pretty(getattr(morph, "domain", None))
    cod = _pretty(getattr(morph, "codomain", None))
    return f"{name} : {dom} -> {cod}", []


def _children_for_rule(name, rule):  # type: ignore[no-untyped-def]
    return name, []


def _children_for_program(name, tmpl):  # type: ignore[no-untyped-def]
    param_strs = []
    for n in getattr(tmpl, "params", None) or ():
        param_strs.append(str(n))
    for p in getattr(tmpl, "type_params", None) or ():
        pname = getattr(p, "name", "?")
        kind = type(p).__name__
        if kind == "ScalarParam":
            param_strs.append(f"{pname} : {getattr(p, 'scalar_kind', '?')}")
        elif kind == "ObjectParam":
            param_strs.append(f"{pname} : {getattr(p, 'universe', '?')}")
        elif kind == "MorphismParam":
            dom = _pretty(getattr(p, "domain", None))
            cod = _pretty(getattr(p, "codomain", None))
            param_strs.append(f"{pname} : Mor[{dom}, {cod}]")
        else:
            param_strs.append(str(pname))
    dom = _pretty(getattr(tmpl, "domain", None))
    cod = _pretty(getattr(tmpl, "codomain", None))
    head = f"{name}"
    if param_strs:
        head += f"({', '.join(param_strs)})"
    head += f" : {dom} -> {cod}"
    steps = getattr(tmpl, "draws", ()) or ()
    return head, [_step_node(step) for step in steps]


def _step_node(step):  # type: ignore[no-untyped-def]
    cls = type(step).__name__
    if cls == "SampleStep":
        vars_ = getattr(step, "vars", ()) or ()
        var = vars_[0] if vars_ else "?"
        idx = _index_suffix(getattr(step, "index", None))
        return f"sample {var}{idx} <- {_call_str(step)}", []
    if cls == "ObserveStep":
        var = getattr(step, "var", "?")
        idx = _index_suffix(getattr(step, "index", None))
        return f"observe {var}{idx} <- {_call_str(step)}", []
    if cls == "LetStep":
        return f"let {getattr(step, 'name', '?')} = ...", []
    if cls == "ScoreStep":
        return f"score {getattr(step, 'name', '?')} = ...", []
    if cls == "MarginalizeStep":
        var = getattr(step, "var", "?")
        idx = _index_suffix(getattr(step, "index", None))
        head = f"marginalize {var}{idx} <- {_call_str(step)}"
        body = [_step_node(s) for s in getattr(step, "scope", ()) or ()]
        return head, body
    if cls == "ReturnStep":
        vars_ = getattr(step, "vars", ()) or ()
        return f"return {', '.join(vars_)}", []
    return cls, []


def _index_suffix(idx):  # type: ignore[no-untyped-def]
    if idx is None:
        return ""
    return f" : {_pretty(idx)}"


def _call_str(step):  # type: ignore[no-untyped-def]
    head = getattr(step, "morphism", "?") or "?"
    args = getattr(step, "args", None)
    if not args:
        return str(head)
    return f"{head}({', '.join(str(a) for a in args)})"


def _children_for_deduction(name, system):  # type: ignore[no-untyped-def]
    head = name
    semiring = type(getattr(system, "semiring", system)).__name__
    children = []
    rules = getattr(system, "rules", ()) or ()
    if rules:
        rule_kids = [
            (f"{getattr(r, 'name', '?')} : {_rule_line(r)}", []) for r in rules
        ]
        children.append(("rules", rule_kids))
    children.append((f"semiring: {semiring}", []))
    tol = getattr(system, "tolerance", None)
    if tol is not None and tol != 0:
        children.append((f"tolerance: {tol}", []))
    return head, children


def _rule_line(rule):  # type: ignore[no-untyped-def]
    premises = getattr(rule, "premises", ()) or ()
    conclusion = getattr(rule, "conclusion", None)
    prem_str = ", ".join(_pat_str(p) for p in premises)
    return f"{prem_str} |- {_pat_str(conclusion)}"


def _pat_str(pat):  # type: ignore[no-untyped-def]
    if pat is None:
        return "?"
    if isinstance(pat, tuple):
        return "(" + ", ".join(_pat_str(p) for p in pat) + ")"
    return str(pat)


def _children_for_signature(name, sig):  # type: ignore[no-untyped-def]
    children = []
    sorts = getattr(sig, "sorts_t", ()) or ()
    if sorts:
        children.append(
            (
                "sorts",
                [
                    (
                        f"{s.name} : {getattr(s, 'kind', '?')}"
                        + (f" [dim={s.dim}]" if getattr(s, "dim", None) else ""),
                        [],
                    )
                    for s in sorts
                ],
            )
        )
    ctors = getattr(sig, "constructors_t", ()) or ()
    if ctors:
        children.append(
            (
                "constructors",
                [(f"{c.name} : {_ctor_line(c)}", []) for c in ctors],
            )
        )
    binders = getattr(sig, "binders_t", ()) or ()
    if binders:
        children.append(("binders", [(b.name, []) for b in binders]))
    vkinds = getattr(sig, "vertex_kinds_t", ()) or ()
    if vkinds:
        children.append(("vertex_kinds", [(v.name, []) for v in vkinds]))
    ekinds = getattr(sig, "edge_kinds_t", ()) or ()
    if ekinds:
        children.append(("edge_kinds", [(e.name, []) for e in ekinds]))
    return name, children


def _ctor_line(ctor):  # type: ignore[no-untyped-def]
    args = getattr(ctor, "args", ()) or ()
    ret = getattr(ctor, "return_sort", None) or getattr(ctor, "result", "?")
    return f"{', '.join(str(a) for a in args)} -> {ret}"


def _children_for_encoder(name, enc):  # type: ignore[no-untyped-def]
    sig_name = getattr(enc, "signature_name", None) or getattr(enc, "signature", "?")
    return f"{name} : {sig_name}", []


def _children_for_decoder(name, dec):  # type: ignore[no-untyped-def]
    sig_name = getattr(dec, "signature_name", None) or getattr(dec, "signature", "?")
    return f"{name} : {sig_name}", []


def _children_for_loss(name, entry):  # type: ignore[no-untyped-def]
    kind = getattr(entry, "attachment_kind", "global")
    target = getattr(entry, "target", None)
    if target:
        return f"{name} [on={kind}({target})]", []
    return f"{name} [on={kind}]", []


def _children_for_bundle(name, members):  # type: ignore[no-untyped-def]
    return f"{name}", [(m, []) for m in members]


def _children_for_contraction(name, contr):  # type: ignore[no-untyped-def]
    return name, []


__all__ = ["run_tui"]
