# Editor support and syntax highlighting

`.qvr` files are highlighted by two complementary systems, depending on what reads the file:

| Consumer | Engine | Comes from |
|---|---|---|
| Python tools: mkdocs, Sphinx, Jupyter, IPython, `pygmentize` | [Pygments](https://pygments.org/) | bundled with the `quivers` PyPI package |
| Editors: Neovim, Helix, Emacs, Zed, VS Code (via tree-sitter extension) | [tree-sitter](https://tree-sitter.github.io/) | bundled with `panproto-grammars-all` and the in-tree `grammars/qvr/` source |
| GitHub.com source view, GitHub gists | [Linguist](https://github.com/github-linguist/linguist) | language registration (not yet upstream) |

This page documents how to wire each one up.

## Pygments (Python doc and notebook tools)

Quivers ships a Pygments lexer registered as an entry point. After

```bash
pip install quivers
```

every Pygments-based tool finds `.qvr` files automatically. No further configuration is needed for mkdocs, Sphinx, `pygmentize`, IPython / Jupyter cell magic (`%%highlight qvr`), GitHub Gists rendered via Pygments locally, or any other consumer that asks Pygments to choose a lexer by filename or by `name="qvr"`.

To confirm the lexer is visible:

```bash
python -c "from pygments.lexers import get_lexer_by_name; print(get_lexer_by_name('qvr').name)"
# QVR
```

To highlight a file to a terminal:

```bash
pygmentize -l qvr -O style=monokai docs/examples/source/bayesian_regression.qvr
```

The lexer source lives at [`src/quivers/dsl/pygments_lexer.py`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/dsl/pygments_lexer.py); the highlight rules and keyword list are kept in sync with the [tree-sitter grammar](https://github.com/FACTSlab/quivers/tree/main/grammars/qvr) by `tests/test_dsl_extensions.py`.

## Tree-sitter (editors)

The tree-sitter grammar lives at [`grammars/qvr/`](https://github.com/FACTSlab/quivers/tree/main/grammars/qvr) in the quivers repository and is vendored by the [`panproto-grammars-all`](https://pypi.org/project/panproto-grammars-all/) distribution. Most editor integrations consume the grammar through one of two paths:

1. The vendored `parser.c` inside `panproto-grammars-all`.
2. A direct clone of the quivers repo with a per-editor build step against `grammars/qvr/`.

The Neovim, Helix, Emacs, and Zed instructions below cover (2). The grammar follows standard tree-sitter conventions; if your editor has a `:TSInstall qvr` or equivalent command and the grammar isn't on the upstream list yet, the manual paths below also work.

### Neovim

With [nvim-treesitter](https://github.com/nvim-treesitter/nvim-treesitter):

```lua
-- init.lua  (or any config file loaded after nvim-treesitter)
local parsers = require("nvim-treesitter.parsers").get_parser_configs()
parsers.qvr = {
  install_info = {
    url = "https://github.com/FACTSlab/quivers",
    files = { "grammars/qvr/src/parser.c" },
    location = "grammars/qvr",
    branch = "main",
    generate_requires_npm = false,
    requires_generate_from_grammar = false,
  },
  filetype = "qvr",
}
vim.filetype.add({ extension = { qvr = "qvr" } })
```

Then `:TSInstall qvr` from inside Neovim. The repository ships `queries/highlights.scm` at `grammars/qvr/queries/highlights.scm`; nvim-treesitter picks it up automatically once the parser is registered.

### Helix

Helix has native tree-sitter support. Add to `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "qvr"
scope = "source.qvr"
file-types = ["qvr"]
roots = []
comment-token = "#"
indent = { tab-width = 4, unit = "    " }

[[grammar]]
name = "qvr"
source = { git = "https://github.com/FACTSlab/quivers", subpath = "grammars/qvr" }
```

Then:

```bash
hx --grammar fetch
hx --grammar build
```

Place the highlight queries at `~/.config/helix/runtime/queries/qvr/highlights.scm`; copy the file from `grammars/qvr/queries/highlights.scm` in this repository, or symlink to a checkout.

### Emacs (tree-sitter-langs)

With [tree-sitter-langs](https://github.com/emacs-tree-sitter/tree-sitter-langs) and built-in `treesit` (Emacs 29+):

```elisp
(add-to-list 'treesit-language-source-alist
             '(qvr . ("https://github.com/FACTSlab/quivers"
                      "main"  ; branch
                      "grammars/qvr/src")))
(define-derived-mode qvr-mode prog-mode "QVR"
  "Major mode for editing .qvr files."
  (when (treesit-ready-p 'qvr)
    (treesit-parser-create 'qvr)
    (setq-local treesit-font-lock-settings
                (treesit-font-lock-rules
                 :language 'qvr
                 :feature 'comment
                 '((comment) @font-lock-comment-face)
                 :language 'qvr
                 :feature 'keyword
                 '(["morphism" "object" "program"
                    "composition" "category" "contraction" "bundle"
                    "deduction" "atoms" "start" "depth"
                    "lexicon" "let" "in" "sample" "observe"
                    "marginalize" "return" "factor" "score"
                    "export" "as" "over"
                    "algebra" "semigroupoid" "bilinear_form"
                    "rule"]
                   @font-lock-keyword-face)))
    (treesit-major-mode-setup)))
(add-to-list 'auto-mode-alist '("\\.qvr\\'" . qvr-mode))
```

Run `M-x treesit-install-language-grammar RET qvr RET` once after Emacs starts to fetch the parser.

### Zed

The quivers repository ships a Zed extension at
[`editors/zed-extension-qvr/`](https://github.com/FACTSlab/quivers/tree/main/editors/zed-extension-qvr).
Until the extension is published to Zed's public registry, install it
locally from a checkout:

```bash
git clone https://github.com/FACTSlab/quivers
mkdir -p ~/.config/zed/extensions
ln -s "$(pwd)/quivers/editors/zed-extension-qvr" ~/.config/zed/extensions/qvr
```

Reload extensions (`zed: reload extensions` from the command palette).
The extension's manifest pins the grammar against `grammars/qvr/` in the
same repository, so a single checkout drives both the grammar source
and the editor packaging.

### VS Code / Cursor

The repository ships a first-party VS Code extension at
[`editors/vscode-qvr/`](https://github.com/FACTSlab/quivers/tree/main/editors/vscode-qvr).
It provides:

- a TextMate grammar for the initial render,
- a [`vscode-languageclient`](https://github.com/microsoft/vscode-languageclient-node)
  bridge to [`qvr-lsp`](../guides/repl-and-lsp.md) for hover,
  go-to-definition, semantic tokens, document symbols, completion,
  formatting, and live diagnostics.

Install the packaged extension into VS Code (or Cursor):

```bash
cd editors/vscode-qvr
npm install
npx tsc -p .
npx @vscode/vsce package --allow-missing-repository --no-yarn
code --install-extension vscode-qvr-*.vsix
# or: cursor --install-extension vscode-qvr-*.vsix
```

The extension auto-discovers `qvr-lsp` in (1) the `qvr.lsp.path`
setting (with `${workspaceFolder}` expansion), (2)
`<workspace>/.venv/bin/qvr-lsp`, (3) `$VIRTUAL_ENV/bin/qvr-lsp`, or
(4) plain `qvr-lsp` on `$PATH`. Install the LSP extra with
`pip install 'quivers[lsp]'`.

The full configuration surface and the launch resolution order live in
the [Interactive guide](../guides/repl-and-lsp.md#vs-code-cursor).

## GitHub source view (Linguist)

GitHub's syntax highlighting on `.qvr` files in repository source view and gists is driven by [github-linguist](https://github.com/github-linguist/linguist). QVR is not yet upstream; until the language is registered, repositories can override locally with a `.gitattributes` entry pointing at the closest existing language for at least *some* color:

```gitattributes
*.qvr linguist-language=Haskell
```

Haskell is the closest match for `<-`, `let`, `>>=`, the offside-rule program-block bodies, and the `|`-separated rule head / body shape; it is not perfect (the categorial slash, the `~ Family` clause, and the axis-role surface get no special treatment), but is the closest readily-available approximation.

Registering QVR upstream with Linguist is tracked at [panproto/panproto#84](https://github.com/panproto/panproto/issues/84); upvoting helps prioritize it.

## Verifying your setup

A short check that exercises every distinguishing surface feature:

<!-- compile: false -->
```qvr
object D : FinSet 32
object K : FinSet 64
object SD : Real 32
object SK : Real 64

morphism W : SD -> SK [role=latent, over=[dom, cod]] ~ MatrixNormal(0.0, 1.0)
morphism softmax_link : SK -> SK ~ Normal
program regression : SD -> SK [effects = [Sample, Score]]
    let z = W >> softmax_link
    observe y : K <- Normal(z, 0.5)
    return y

export regression
```

When the highlighting is wired up, keywords (`object`, `morphism`, `program`, `let`, `observe`, `return`, `export`), the option-block keys and the family-init sigil (`role=`, `over=`, `effects=`, `~`), the `<-` Kleisli-bind punctuation, the family identifiers (`MatrixNormal`, `Normal`), and the comments (none here, but `#`-prefixed lines anywhere in the file) all carry distinct colors.

If a token is unhighlighted (rendered as default-foreground text), the corresponding rule in the editor's highlight query is the place to look; the canonical reference is [`grammars/qvr/queries/highlights.scm`](https://github.com/FACTSlab/quivers/blob/main/grammars/qvr/queries/highlights.scm).
