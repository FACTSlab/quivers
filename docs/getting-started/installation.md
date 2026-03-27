# Installation

## Prerequisites

- **Python** >= 3.12
- **PyTorch** >= 2.0

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

All functionality is built as pure Python atop PyTorch; no other system dependencies are required.
