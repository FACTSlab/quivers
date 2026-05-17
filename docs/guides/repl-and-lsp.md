# Interactive surface: REPL, kernel, language server

quivers ships an interactive type-explorer and a matching language
server. Both share the same parser
([panproto](https://github.com/panproto/panproto) tree-sitter), the
same elaborator
([`quivers.dsl.Compiler`](../api/dsl/compiler.md)), and the same highlight
table — so the colour you see in the TUI matches the colour your
editor shows over LSP, which matches what gets emitted as Jupyter
output. One source of truth, four surfaces.

This page is the comprehensive reference. For a one-paragraph
overview see [Quickstart](../getting-started/quickstart.md).

## Installation

```bash
pip install 'quivers[repl,lsp]'
```

The `repl` extra brings in [Textual](https://textual.textualize.io/),
[prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/),
[rich](https://rich.readthedocs.io/), and an
[ipykernel](https://ipykernel.readthedocs.io/)-based Jupyter kernel.
The `lsp` extra brings in
[pygls](https://github.com/openlawlibrary/pygls) and `lsprotocol`.

Both are optional: `pip install quivers` alone gives you the library
and `qvr check`, nothing else.

After install:

| Command | What it does |
| --- | --- |
| `qvr repl` | Textual TUI (or prompt_toolkit fallback) |
| `qvr lsp` / `qvr-lsp` | Language Server over stdio |
| `qvr kernel install --user` / `qvr-kernel install --user` | Register a Jupyter kernelspec |
| `qvr check FILE...` | Batch parse + compile (the default subcommand) |

## The REPL

### Starting it

```bash
qvr repl                                              # blank session
qvr repl docs/examples/source/seq2seq.qvr             # load on startup
qvr repl --plain                                      # single-line prompt_toolkit mode
qvr repl --plain < session.qvrrepl                    # non-TTY scripted mode
```

When stdin is a TTY and Textual is importable, the four-pane TUI
launches. `--plain` forces the prompt_toolkit single-line frontend.

### Layout (TUI)

```
+---- Status bar -----------------------------------------+
|                                                         |
|  Input editor                  |  Env filter            |
|  (multi-line TextArea          +-----------------------+
|   with bracket pairing,        |                       |
|   auto-indent, Tab completion) |  Env tree             |
|                                |  (objects/spaces/     |
+--------------------------------+   morphisms/rules,    |
|  Output log                    |   click any leaf to   |
|  (history of evaluated         |   :info it)           |
|   commands and rendered        |                       |
|   responses)                   |                       |
+--------------------------------+-----------------------+
|  Watches  (hidden when empty; pinned :watch results)    |
+---------------------------------------------------------+
|  Diagnostics (hidden when empty; errors with locations) |
+---------------------------------------------------------+
|  Footer: visible key bindings                           |
```

The watches and diagnostics strips collapse when there is nothing to
show, so unused panels never cost screen real estate.

### Status bar

The top bar shows, separated by `·`:

- the loaded file path (or `<no file>`),
- the active algebra (`ProductFuzzyAlgebra`, `LogProbAlgebra`, …),
- counts: `N obj · M space · K morph · J rule`.

It refreshes after every evaluation, reload, or watch update.

### Environment browser

The right-hand `Tree` widget groups bindings by namespace:

- `objects` — every `object` and `alias` declaration
- `spaces` — every `space` and `type` declaration
- `morphisms` — `latent`, `observed`, `kernel`, `embed`, `program`, `let`
- `rules` — `rule` declarations

The root node auto-expands on every refresh; each namespace expands
its leaves. Click a leaf (or arrow-key to it and press Enter) to fire
`:info NAME` in the output log.

Above the tree is a one-line **filter input** — type any substring
and the tree fuzzy-collapses to the bindings whose names contain it.
Clear the filter to bring everything back.

### Output log

Rendered responses appear here in source order. Bodies tagged as QVR
(every `:type`, `:info`, `:doc`, `:dump`, `:watch`, `:browse` result)
are passed through the shared
[`tokenize`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_highlight.py)
pipeline and rendered with the
[`STYLE_TABLE`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_highlight.py)
colours. Identifiers known to the env are upgraded to their semantic
colour even when the surrounding line wasn't a parseable QVR
declaration; the same identifier always reads the same way.

Identifiers rendered as type / function / namespace are also
**clickable links** — clicking any name fires `:info NAME`. The
`-- declared at PATH:LINE:COL` footer underneath each `:info` body
is also clickable; it opens that file at that line in `$EDITOR`
(falling back to `$VISUAL`, then `vi`).

### Input editor

The input pane is a multi-line `TextArea` with three quality-of-life
behaviours layered on:

1. **Bracket pairing.** Typing `(`, `[`, or `{` inserts the matching
   closer and parks the cursor between them.
2. **Auto-indent on Enter.** Lines that end with `(`, `[`, `{`, `:`,
   `->`, `=`, or `<-` get one extra indent on the next line so
   `program` / `deduction` blocks stay aligned without effort.
3. **Tab completion** (described below).

Press the eval key (next section) to evaluate the buffer; the input
is cleared on success.

### Meta-commands

GHCi-shaped, leading `:`. Any prefix that uniquely identifies a
command works, plus the explicit short aliases listed.

| Command | Short | What it does |
| --- | --- | --- |
| `:load FILE` | `:l` | Parse + elaborate, rebind the session env |
| `:reload` | `:r` | Re-`:load` the last file, print added/removed/changed names |
| `:type EXPR` | `:t` | Print EXPR's resolved type as canonical QVR (`latent f : A -> B`, `object X : 3`, `space Z : Euclidean(64)`) |
| `:kind T` | `:k` | Print T's AST variant and enumerate the sibling TypeExpr variants |
| `:info NAME` | `:i` | Show NAME's declaration verbatim from the source, plus its location and doc comment. Pass `--python` for the didactic AST `repr()` instead |
| `:doc NAME` | — | Render only the doc comment(s) for NAME |
| `:browse [NS]` | `:b` | List every binding, optionally filtered by namespace (`objects`/`spaces`/`morphisms`/`rules`) |
| `:dump NAME [--json]` | — | Pretty-print NAME's AST node (`--json` for didactic's `model_dump_json`) |
| `:edit NAME` | — | Open `$EDITOR` on NAME's source, splice the edited text back into the module, recompile |
| `:trace EXPR` | — | Step through elaboration of a morphism expression, surfacing each intermediate domain/codomain |
| `:save [FILE]` | `:s` | Write the live module to FILE (or back to the loaded path) via [`module_to_source`](../api/dsl/emit.md) |
| `:watch EXPR` | `:w` | Pin EXPR for re-eval on every recompile; result appears in the Watches strip |
| `:unwatch [EXPR]` | — | Remove one watch, or clear all when no argument is given |
| `:set k=v` | — | Toggle session options (`highlight`, `unicode`, `paranoid`, `autoload_on_save`, `theme`) |
| `:help [CMD]` | `:h` | Without arg: full command list. With one: detailed help for CMD |
| `:quit` | `:q` | Exit the REPL |

A bare line (no leading `:`) is evaluated as **either** appended
statements (parsed, compiled into the live module, env updated)
**or**, if parsing as statements fails, treated as an expression and
piped through `:type`.

### Key bindings

Selected for cross-platform reliability — every binding here reaches
the application on macOS, Linux, and Windows without per-terminal
configuration.

| Key | Action |
| --- | --- |
| `Ctrl-G` | Evaluate the buffer |
| `Ctrl-O` | Evaluate the buffer (alternate) |
| `F8` | Evaluate the buffer (Fn-row alternate) |
| `Ctrl-Up` | Recall the previous input from history |
| `Ctrl-Down` | Advance to the next input |
| `Tab` | Cycle completion candidates at the cursor |
| `Ctrl-P` | Open the command palette (fuzzy meta-command picker) |
| `Ctrl-L` | Clear the eval log |
| `Ctrl-R` | Reload the loaded file |
| `Ctrl-Q` | Quit |
| `F1` | Show the meta-command help in the eval log |
| `Ctrl-Enter`, `Ctrl-J` | Evaluate (hidden fallback; only fires in terminals that forward Ctrl-Enter, like Wezterm/Kitty/Windows Terminal/iTerm2 with a CSI-u keymap) |

Inside the TextArea, all of Textual's emacs-style editing bindings
remain available: `Ctrl-A`/`Ctrl-E` line nav, `Ctrl-W` delete word
back, `Ctrl-K` kill to end of line, `Ctrl-X`/`Ctrl-C`/`Ctrl-V`
cut/copy/paste, `Ctrl-Z`/`Ctrl-Y` undo/redo. The eval bindings
above use `priority=True` so they fire even though `Ctrl-G` is not
otherwise a TextArea action.

> **macOS Fn-row note.** `F5` is reserved by macOS Dictation, `F3`
> and `F4` by Mission Control and Launchpad, `F11` by Show Desktop.
> `F8` is the only function key macOS doesn't already claim by
> default. Enable "Use F1, F2, etc. keys as standard function keys"
> in System Settings → Keyboard if you want to use F-keys without
> holding Fn.

> **Ctrl-Enter, Alt-Enter, Cmd-Enter.** These never reach the
> application on macOS Terminal.app or the default iTerm2 profile.
> The terminal emulator drops them at the wire layer. To make them
> work you have to configure your emulator to send them; for iTerm2
> map them to `\x1b[13;5u` (the CSI-u "modifyOtherKeys" encoding),
> for Windows Terminal use the same string. Wezterm, Kitty, and
> Alacritty forward them out of the box.

### Tab completion

`Tab` at the cursor:

1. Builds a candidate list from four sources:
   - Meta-command names (`:load`, `:type`, …) when the prefix starts with `:`.
   - Env names: every object, space, morphism, rule in the live env, tagged with its namespace.
   - QVR grammar keywords (`latent`, `observed`, `program`, `over`, `iid`, `via`, …) and builtins (`Normal`, `Euclidean`, `softmax`, …).
   - File-system paths after `:load`.
2. Inserts the first candidate, replacing the prefix under the cursor.
3. Subsequent `Tab` presses cycle through the remaining candidates without rebuilding the list.

The completer is the exact one the LSP and the Jupyter kernel call;
the surfaces never disagree about what's available.

### Command palette

`Ctrl-P` opens Textual's command palette pre-populated with every
meta-command. Type any substring to fuzzy-filter; Enter selects.
Selecting a meta-command inserts it into the input pane with a
trailing space, ready for the argument. This is the discoverable
substitute for memorising shortcuts.

### Input history

Every evaluated line is appended to `~/.config/quivers/history`
(`$XDG_CONFIG_HOME/quivers/history` if set). The file is plain
newline-separated entries; the same history is loaded the next time
you start `qvr repl`.

- `Ctrl-Up` walks backwards from the most recent entry.
- `Ctrl-Down` walks forward; at the end you return to an empty
  buffer.
- The current input is replaced wholesale; the TextArea is not
  modal, so just keep typing to continue editing.

The prompt_toolkit fallback (`qvr repl --plain`) uses the same file
through `prompt_toolkit.history.FileHistory`.

### File watcher / auto-reload

When a file is loaded, the TUI polls its mtime once per second. If
the mtime advances (your editor saved it), the session re-runs
`:reload` automatically and logs `auto-reload` in the eval log. Any
diagnostics from the new parse show up immediately in the
Diagnostics strip; any pinned `:watch` expressions re-evaluate
against the new env.

The option is on by default and controlled by
`:set autoload_on_save=true|false`.

### Watches

`:watch EXPR` pins EXPR for re-evaluation after every recompile —
every `:load`, `:reload`, bare-line statement, or autoreload. The
result appears in a dedicated **Watches** strip (auto-hidden when
empty) above the diagnostics strip. Format: `watch EXPR => result`,
syntax-highlighted.

```
qvr> :watch f
watch f => latent f : Alpha -> Beta
qvr> object Gamma : 7
qvr> latent g : Alpha -> Gamma
qvr> :watch g
# both f and g are now pinned; modifying Alpha or Beta updates both
```

`:unwatch EXPR` removes one; `:unwatch` alone clears every watch.

### Inline diagnostic markers

Every error diagnostic with a known location (parse errors, compile
errors, constraint violations) selects the offending span in the
input pane so you see where the failure originated. The Diagnostics
strip shows the same message in human-readable form:

```
[error] compile:5:12: undefined object 'X'
```

The strip is hidden when there are no diagnostics.

### Click handlers

Two surfaces are click-aware:

- **Identifiers in rendered output.** Clicking a type-, function-,
  or namespace-coloured identifier fires `:info NAME` in the eval
  log. Works on any text the renderer painted with one of those
  semantic colours: env-known names, grammar-classified types,
  built-in functions, algebras.
- **`PATH:LINE:COL` footers.** The `-- declared at ...` line under
  every `:info` body is a clickable link that launches
  `$EDITOR +LINE PATH` (e.g. `vi +5 model.qvr`,
  `code --goto model.qvr:5`).

### Themes

`:set theme=NAME` controls the Rich syntax theme used when bodies
are rendered through `rich.syntax.Syntax`. Useful values:

- `ansi_dark` (default), `ansi_light`
- `monokai`, `dracula`, `solarized-dark`, `solarized-light`
- `nord`, `gruvbox-dark`, `github-dark`
- any other [Pygments style](https://pygments.org/styles/) shipped with your install

The Textual app itself follows your terminal's colour scheme; the
theme option only affects fenced QVR / JSON blocks in `:info` and
`:dump` output.

### Bracket pairing and auto-indent

In the input TextArea:

- `(`, `[`, `{` → insert the matching closer, leave cursor between.
- Enter after `(`, `[`, `{`, `:`, `->`, `=`, `<-` → add one indent
  level (four spaces by default) on the new line.
- Indentation is otherwise preserved verbatim.

These rules are heuristic; they cover the common shape of QVR
`program` and `deduction` blocks without needing a full grammar
configuration.

### Plain mode

`qvr repl --plain` (or any non-TTY invocation) uses
[prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/)
instead of Textual. The same `ReplSession` drives the prompt, so
every meta-command works identically; the only differences are:

- Single-line input, with prompt_toolkit's built-in history
  navigation (Up/Down) instead of `Ctrl-Up`/`Ctrl-Down`.
- Output is rendered through [Rich](https://rich.readthedocs.io/) if
  available, plain text otherwise.
- No env tree, no watches panel, no command palette, no click
  handlers.

This is the mode CI uses for scripted runs: pipe `.qvrrepl` files
(plain lists of meta-commands and bare statements) on stdin.

## The Jupyter kernel

`qvr-kernel install --user` registers a `quivers` kernelspec. After
that, every Jupyter frontend (Notebook, JupyterLab, VS Code
notebooks, `jupyter console`) sees QVR as a first-class language.

```bash
qvr-kernel install --user
jupyter console --kernel quivers
```

```
In [1]: :load model.qvr
loaded model.qvr: 17 binding(s)

In [2]: :type backbone
latent backbone : Source * Target -> Combined

In [3]: object Extra : 8
        latent g : Extra -> Combined
installed module: 19 binding(s)
```

Cell semantics:

- Leading-`:` lines are dispatched as meta-commands.
- Other lines are appended to the live module and compiled.
- Blank lines separate independent chunks within a cell, so you can
  mix meta-commands and declarations freely.

Notebook-side features:

- **Tab completion** routes through the same completer as the TUI
  and LSP.
- **Inspect (`Shift-Tab`)** routes through `:info`, showing the
  declaration in the inspector panel.
- The kernelspec advertises `pygments_lexer: qvr` so cell content
  is highlighted by the bundled Pygments lexer.

## The Language Server

`qvr-lsp` speaks LSP 3.17 over stdio. Pointed at any LSP-aware
editor it provides hover, go-to-definition, references, document
symbols, semantic highlighting, completion, formatting, and live
diagnostics.

### VS Code / Cursor

The [`vscode-qvr`](https://github.com/FACTSlab/quivers/tree/main/editors/vscode-qvr)
extension in the repository ships:

- the TextMate grammar (for the initial render before the LSP
  attaches),
- a [`vscode-languageclient`](https://github.com/microsoft/vscode-languageclient-node)
  bridge that auto-discovers `qvr-lsp` in:
  1. an explicit `qvr.lsp.path` setting (supports `${workspaceFolder}` expansion),
  2. `<workspace>/.venv/bin/qvr-lsp` (uv / venv convention),
  3. `<workspace>/.venv/Scripts/qvr-lsp.exe` (Windows venv),
  4. `$VIRTUAL_ENV/bin/qvr-lsp`,
  5. plain `qvr-lsp` on `$PATH`.

Settings:

| Key | Default | Purpose |
| --- | --- | --- |
| `qvr.lsp.enabled` | `true` | Master toggle |
| `qvr.lsp.path` | `qvr-lsp` | Override the executable path |
| `qvr.lsp.args` | `[]` | Extra CLI arguments |

Install the packaged `.vsix` directly:

```bash
cd editors/vscode-qvr
npm install
npx tsc -p .
npx @vscode/vsce package --allow-missing-repository --no-yarn
code --install-extension vscode-qvr-*.vsix         # or cursor --install-extension
```

### Zed

The [`zed-extension-qvr`](https://github.com/FACTSlab/quivers/tree/main/editors/zed-extension-qvr)
extension exports `qvr-lsp` as a `language_server`. Symlink the
extension into Zed's extensions directory:

```bash
mkdir -p ~/.config/zed/extensions
ln -s "$PWD/editors/zed-extension-qvr" ~/.config/zed/extensions/qvr
```

Reload extensions from the command palette. Zed spawns `qvr-lsp` on
`$PATH` automatically when you open a `.qvr` file.

### Neovim

```lua
require("lspconfig").configs.qvr = {
  default_config = {
    cmd = { "qvr-lsp" },
    filetypes = { "qvr" },
    root_dir = require("lspconfig.util").root_pattern(
      "pyproject.toml", ".git"
    ),
  },
}
require("lspconfig").qvr.setup({})
```

### Capabilities

Every capability advertised by `qvr-lsp`:

| LSP method | What it returns |
| --- | --- |
| `textDocument/publishDiagnostics` | Parser, constraint-solver, and compiler diagnostics with source ranges |
| `textDocument/hover` | The declaration as a fenced `qvr` block, with the didactic AST `repr()` stacked beneath under a collapsed `<details>` so both forms are always available |
| `textDocument/definition` | Jump to the originating declaration |
| `textDocument/references` | Every textual occurrence of the name |
| `textDocument/documentSymbol` | All top-level declarations grouped by symbol kind (`Class` for objects/spaces, `Function` for morphisms, `Variable` otherwise) |
| `textDocument/completion` | Env names + grammar keywords + builtins + paths, same source as the REPL completer |
| `textDocument/semanticTokens/full` | Env-aware semantic token stream driven by the shared [`STYLE_TABLE`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_highlight.py) |
| `textDocument/formatting` | Canonical re-emission via [`module_to_source`](../api/dsl/emit.md) (no-op when the module contains a statement variant the canonical emitter doesn't yet cover) |
| `textDocument/didOpen` / `didChange` / `didSave` / `didClose` | Incremental sync, full re-analysis per change |

### Hover format

A hover always shows two visually-separated panes:

````markdown
Optional doc comment lines if the decl had `## doc` blocks above it.

**QVR source**

```qvr
latent f : Alpha -> Beta
```

---

**AST (didactic)**

<details><summary><i>click to expand</i></summary>

```python
MorphismDecl(morphism_kind='latent', name='f', domain=TypeName(...), ...)
```

</details>
````

- The QVR block is sliced from the original source so user formatting
  and comments survive verbatim.
- A horizontal rule draws the divider between the panes.
- The Python AST is the didactic `repr()`, hidden by default in the
  collapsed `<details>` element so it only takes vertical space when
  you click "click to expand".

## Architecture

All four surfaces fan out from one class:

```
                    ┌─────────────────────┐
                    │   ReplSession       │  pure Python state +
                    │  (state + dispatch) │  meta-command dispatch
                    └────────┬────────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        │                    │                     │
   ┌────▼─────┐         ┌────▼─────┐          ┌────▼─────┐
   │ Textual  │         │ prompt_  │          │ Jupyter  │
   │ TUI      │         │ toolkit  │          │ kernel   │
   └──────────┘         └──────────┘          └──────────┘

                       ┌────────────────────┐
                       │  qvr-lsp           │  builds its own
                       │  (pygls)           │  per-document state
                       └────────────────────┘  but shares the
                                               highlighter +
                                               completer
```

Components shared across all surfaces:

- [`quivers.cli.repl_session`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_session.py) — `ReplSession`, the meta-command dispatcher, the live env, the watch list.
- [`quivers.cli.repl_complete`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_complete.py) — `all_completions(session, prefix)`; fans out to env, grammar, paths.
- [`quivers.cli.repl_highlight`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_highlight.py) — `tokenize`, `STYLE_TABLE`, `to_rich_text`, `to_semantic_token_data`; one classifier feeds every renderer.

Each frontend is a thin adapter:

- [`quivers.cli.repl_tui`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_tui.py) — Textual app.
- [`quivers.cli.repl_prompt`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/repl_prompt.py) — prompt_toolkit single-line frontend.
- [`quivers.kernel.quivers_kernel`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/kernel/quivers_kernel.py) — ipykernel adapter.
- [`quivers.lsp.server`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/lsp/server.py) — pygls server.

This is the seam that keeps the four surfaces in sync: anywhere a
user sees `Source`, it's classified by the same call and rendered
with the same colour, regardless of which frontend is asking.
