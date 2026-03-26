# Installation

## Prerequisites

- **Python** >= 3.12
- **PyTorch** >= 2.0

## From Source

Quivers is not yet available on PyPI. Install directly from source:

```bash
# Clone the repository
git clone https://github.com/aaronstevenwhite/quivers
cd quivers

# Install in development mode
pip install -e .
```

This installs the `quivers` package with its core dependency on PyTorch.

## Development Installation

If you intend to run tests and contribute:

```bash
pip install -e ".[dev]"
```

This adds:
- `pytest >= 7.0` — test runner
- `pytest-cov` — coverage reporting

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
- **torch** (>= 2.0) — differentiable tensors and automatic differentiation

All functionality is built as pure Python atop PyTorch; no other system dependencies are required.
