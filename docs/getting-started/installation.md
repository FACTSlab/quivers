# Installation

## Prerequisites

- **Python** >= 3.14
- **PyTorch** >= 2.0
- **didactic** >= 0.7.1
- **panproto** >= 0.58.0 (provides the schema/lens machinery)
- **panproto-grammars-all** >= 0.58.0 (ships the QVR tree-sitter parser)
- **Pygments** >= 2.10, **tree-sitter** >= 0.21

The didactic, panproto, panproto-grammars-all, Pygments, and
tree-sitter packages are pulled in automatically by
`pip install quivers`.

## From PyPI

```bash
pip install quivers
```

## From Source

To install directly from source:

```bash
git clone https://github.com/FACTSlab/quivers
cd quivers
pip install -e .
```

## Development Installation

If you intend to run tests and contribute:

```bash
pip install -e ".[dev]"
```

This adds:

- `pytest >= 7.0`: test runner
- `pytest-cov`: coverage reporting
- `ruff`: linter and formatter
- `pyright`: static type checker
- `numpy`, `pandas`, `polars`, `pyarrow`: data-frame fixtures used by the data-encoding tests

The `[docs]` extra (mkdocs, mkdocstrings[python], mkdocs-terminal,
pymdown-extensions, pygments) is needed to build the documentation
site locally; `mkdocs-terminal` is the theme `mkdocs.yml` declares.

## Optional capability extras

Opt-in extras pull in the dependencies for surfaces that quivers can
do without:

```bash
pip install 'quivers[repl]'         # qvr repl + Jupyter kernel
pip install 'quivers[lsp]'          # qvr-lsp language server
pip install 'quivers[data]'         # narwhals + scikit-learn data adapters
pip install 'quivers[diagnostics]'  # ArviZ / xarray / netCDF4 trace I/O
pip install 'quivers[formulas]'     # data + diagnostics + formulae DSL
pip install 'quivers[repl,lsp]'     # any combination is fine
```

`[repl]` pulls in [Textual](https://textual.textualize.io/),
[prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/),
[rich](https://rich.readthedocs.io/),
[ipykernel](https://ipykernel.readthedocs.io/), and
[jupyter_client](https://jupyter-client.readthedocs.io/).

`[lsp]` pulls in [pygls](https://github.com/openlawlibrary/pygls) and
`lsprotocol`.

`[data]` pulls in [narwhals](https://narwhals-dev.github.io/narwhals/)
and [scikit-learn](https://scikit-learn.org/); `[diagnostics]` adds
[ArviZ](https://python.arviz.org/), [xarray](https://docs.xarray.dev/),
and [netCDF4](https://unidata.github.io/netcdf4-python/). `[formulas]`
is a superset that also brings in
[formulae](https://bambinos.github.io/formulae/) for the R-style
formula surface.

After installing the extras you have these new console scripts:

| Command | Provided by | What it does |
| --- | --- | --- |
| `qvr repl` | `[repl]` | Four-pane Textual TUI (or prompt_toolkit fallback) |
| `qvr-lsp` | `[lsp]` | LSP 3.17 server over stdio |
| `qvr lsp` | `[lsp]` | The same, as a `qvr` subcommand |
| `qvr-kernel install` | `[repl]` | Register the Jupyter kernelspec |
| `qvr kernel install` | `[repl]` | Same, as a subcommand |

See [Interactive surface](../guides/repl-and-lsp.md) for the full
guide.

## Verify Installation

Check that the import works:

```python
import quivers
print(quivers.__version__)
```

Run the test suite:

```bash
pytest tests/
```

## Dependencies

Quivers depends on:

- **torch** (>= 2.0): differentiable tensors and automatic differentiation
- **didactic** (>= 0.7.1): typed-data layer that backs every value-type in quivers (`dx.Model`, `dx.TaggedUnion`, `dx.Lens`)
- **panproto** (>= 0.58.0): schema/theory machinery used to extract a `Schema` from each `.qvr` program for diff/migrate workflows
- **panproto-grammars-all** (>= 0.58.0): ships the QVR tree-sitter parser registered with panproto; quivers does not run a hand-written lexer or recursive-descent parser
- **Pygments** (>= 2.10): in-tree `qvr` lexer for documentation and notebooks
- **tree-sitter** (>= 0.21): runtime bindings for the QVR grammar

All core functionality is built as pure Python atop PyTorch; no other
system dependencies are required at runtime. The optional capability
extras (`[repl]`, `[lsp]`, `[data]`, `[diagnostics]`, `[formulas]`)
pull in the interactive surfaces and data/diagnostics integrations
described above.
