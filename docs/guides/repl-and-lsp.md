# Interactive type exploration: `qvr repl` and `qvr-lsp`

quivers ships a GHCi-style interactive surface and a matching
Language Server. Both are powered by the same parser
([`panproto`](https://github.com/panproto/panproto)) and the same
elaborator ([`quivers.dsl.Compiler`](../api/dsl.md)), so what you see
in the REPL is what you see in your editor.

## Install

```bash
pip install 'quivers[repl,lsp]'
```

`repl` brings in [Textual](https://textual.textualize.io/),
[prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/),
[rich](https://rich.readthedocs.io/), and an
[ipykernel](https://ipykernel.readthedocs.io/)-based Jupyter
kernel. `lsp` brings in [pygls](https://github.com/openlawlibrary/pygls)
and `lsprotocol`.

## REPL

```bash
qvr repl                      # Textual TUI when stdin/stdout is a TTY
qvr repl path/to/model.qvr    # load a file on startup
qvr repl --plain              # prompt_toolkit single-line mode
```

The TUI splits four panes: input editor, evaluation log, environment
browser (objects / spaces / morphisms / rules), and a diagnostics
strip. Live highlighting is driven directly off the tree-sitter QVR
grammar, so adding a keyword to `grammars/qvr/grammar.js` paints it
automatically.

### Meta-commands

| Command | Effect |
| --- | --- |
| `:load FILE` / `:l` | parse + elaborate; rebind the environment |
| `:reload` / `:r` | re-parse the last file and diff the env |
| `:type EXPR` / `:t` | print the resolved type or `dom -> cod` signature |
| `:kind T` / `:k` | print the AST variant of a type expression |
| `:info NAME` / `:i` | declaration source, location, and doc comment |
| `:doc NAME` | only the doc comment |
| `:browse [NS]` / `:b` | list every binding, optionally filtered by namespace |
| `:dump NAME [--json]` | AST node `repr` or `model_dump_json` |
| `:edit NAME` | open `$EDITOR`, splice the result back, recompile |
| `:trace EXPR` | step through elaboration of a morphism expression |
| `:set k=v` | toggle session options |
| `:help [CMD]` | overview, or per-command help |
| `:quit` / `:q` | exit |

A bare line is first tried as a sequence of statements (appended to the
current module and recompiled); if parsing fails, it is treated as an
expression and passed through `:type`.

## Jupyter kernel

```bash
qvr-kernel install --user           # register the kernelspec
jupyter console --kernel quivers    # cell-driven REPL
```

The kernel reuses the same `ReplSession` that drives `qvr repl`, so
notebook cells, the Textual TUI, and the plain prompt all behave
identically. Hover (`Shift-Tab` in Jupyter) routes through `:info`,
and tab completion routes through the same completer the LSP uses.

## Language Server

`qvr-lsp` speaks LSP 3.17 over stdio. It is what you point your
editor at.

### VS Code

The `vscode-qvr` extension in `editors/vscode-qvr` ships a TextMate
grammar plus a [`vscode-languageclient`](https://github.com/microsoft/vscode-languageclient-node)
bridge to `qvr-lsp`. Settings:

- `qvr.lsp.enabled` (`true`): turn the LSP off if needed.
- `qvr.lsp.path` (`qvr-lsp`): absolute path or PATH name of the server.
- `qvr.lsp.args` (`[]`): extra CLI arguments.

### Zed

The `editors/zed-extension-qvr` extension exports `qvr-lsp` as a
`language_server` entry; Zed will spawn it automatically for `.qvr`
files.

### Neovim and others

```lua
require("lspconfig").qvr.setup({
  cmd = { "qvr-lsp" },
  filetypes = { "qvr" },
  root_dir = require("lspconfig.util").root_pattern("pyproject.toml", ".git"),
})
```

### Capabilities

- `textDocument/publishDiagnostics` — parser, constraint solver, and
  compiler errors with source ranges.
- `textDocument/semanticTokens/full` — full grammar-driven highlight
  legend; shared with the TUI's `STYLE_TABLE`.
- `textDocument/hover` — declaration source plus doc comment.
- `textDocument/definition` — jump to the originating declaration.
- `textDocument/references` — name-scan over the document.
- `textDocument/documentSymbol` — every declaration, grouped by
  symbol kind.
- `textDocument/completion` — environment names, grammar keywords,
  builtin types and functions, algebra names, file paths.
- `textDocument/formatting` — canonical re-emission via
  [`quivers.dsl.emit.module_to_source`](../api/dsl.md).
