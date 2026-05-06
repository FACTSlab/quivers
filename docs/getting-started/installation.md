# Installation

## Prerequisites

- **Python** >= 3.14
- **PyTorch** >= 2.0
- **didactic** >= 0.6.0
- **panproto** >= 0.45.0 (provides the schema/lens machinery)
- **panproto-grammars-all** >= 0.45.0 (ships the QVR tree-sitter parser)

The didactic, panproto, and panproto-grammars-all packages are pulled in automatically by `pip install quivers`.

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

The `[docs]` extra (mkdocs, mkdocstrings, mkdocs-cinder, pymdown-extensions, pygments) is needed to build the documentation site locally.

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
- **didactic** (>= 0.6.0): typed-data layer that backs every value-type in quivers (`dx.Model`, `dx.TaggedUnion`, `dx.Lens`)
- **panproto** (>= 0.45.0): schema/theory machinery used to extract a `Schema` from each `.qvr` program for diff/migrate workflows
- **panproto-grammars-all** (>= 0.45.0): ships the QVR tree-sitter parser registered with panproto; quivers does not run a hand-written lexer or recursive-descent parser

All functionality is built as pure Python atop PyTorch; no other system dependencies are required at runtime.
