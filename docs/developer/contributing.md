# Contributing to Quivers

This guide covers setting up a development environment, understanding the project structure, and contributing code to the quivers library.

## Development Environment Setup

### Prerequisites

- Python 3.12 or later
- pip or conda
- git

### Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/aaronstevenwhite/quivers
cd quivers
pip install -e ".[dev]"
```

This installs the package in editable mode along with development dependencies.

### Running Tests

Run the full test suite:

```bash
python -m pytest tests/ -x
```

The `-x` flag stops at the first failure, which is useful for iterative development. For more options:

```bash
python -m pytest tests/ -v                  # verbose
python -m pytest tests/ -k test_name        # run specific tests
python -m pytest tests/path/to/test_file.py # run specific file
```

## Project Structure

```
quivers/
├── docs/                          # Documentation
│   ├── api/                       # API reference
│   ├── developer/                 # Developer documentation
│   ├── getting-started/           # User guides
│   ├── guides/                    # Long-form guides
│   └── tutorials/                 # Tutorials
├── src/quivers/                   # Main package
│   ├── __init__.py
│   ├── categorical/               # Categorical algebra
│   ├── continuous/                # Continuous distributions (30+ families)
│   ├── core/                      # Core types and utilities
│   ├── dsl/                       # QVR DSL (lexer, parser, interpreter)
│   │   ├── tokens.py              # Token definitions
│   │   ├── parser.py              # Recursive descent parser
│   │   ├── ast_nodes.py           # AST node definitions
│   │   ├── interpreter.py         # Execution engine
│   │   ├── examples/              # Example .qvr files
│   │   └── ...
│   ├── enriched/                  # Enriched categories
│   ├── inference/                 # Variational inference
│   ├── monadic/                   # Monadic programs (draw, observe, return)
│   ├── stochastic/                # Stochastic morphisms
│   └── ...
├── tests/                         # Test suite (mirrors src structure)
├── pyproject.toml                 # Package metadata
└── mkdocs.yml                     # Documentation config
```

## Code Style Conventions

### Type Hints

Include type hints in all function signatures. Use modern Python 3.12+ syntax:

- Use `dict[K, V]` not `Dict[K, V]`
- Use `list[T]` not `List[T]`
- Use `X | None` not `Optional[X]`
- Use `tuple[T, ...]` for variable-length tuples

```python
def process_data(values: list[float], multiplier: float = 1.0) -> dict[str, float]:
    """Process numeric data."""
    return {str(i): v * multiplier for i, v in enumerate(values)}
```

Do not use type hints in function bodies or variable assignments unless necessary for clarity in complex code sections.

### Docstrings

Use numpy-style docstrings for all public modules, classes, and functions:

```python
def calculate_entropy(probabilities: list[float]) -> float:
    """Calculate Shannon entropy of a probability distribution.

    Parameters
    ----------
    probabilities : list[float]
        Probabilities that sum to 1.0.

    Returns
    -------
    float
        Shannon entropy in nats.

    Raises
    ------
    ValueError
        If probabilities do not sum to approximately 1.0.

    Examples
    --------
    >>> entropy([0.5, 0.5])
    0.6931471805599453
    """
```

### Comments

Use lowercase inline comments to clarify non-obvious logic:

```python
# compute sufficient statistics for exponential family
sufficient_stats = compute_stats(data)

# handle edge case where prior is uniform
if prior_strength == 0:
    posterior = likelihood
```

Avoid stating the obvious. Comments should explain "why," not "what."

### Python Version and Modern Features

Maintain compatibility with Python 3.12 and later. Use modern features:

- Type union syntax: `X | None` instead of `Union[X, None]`
- Positional-only parameters: `def func(a, /, b)`
- Named tuple literals (Python 3.12+, consider `dataclass` for earlier versions)

## The DSL Pipeline

The QVR DSL processes `.qvr` files through these stages:

### 1. Tokenization (tokens.py)

The lexer breaks source text into tokens. Each token carries type, value, line, and column:

```python
class TokenType(Enum):
    QUANTALE = auto()
    OBJECT = auto()
    PROGRAM = auto()
    DRAW = auto()
    OBSERVE = auto()
    ...
```

Keywords like `program`, `draw`, `observe`, `return` map to specific token types. Operators (`->`, `>>`, `@`, `~`) and punctuation are also tokenized.

### 2. Parsing (parser.py)

The recursive descent parser transforms the token stream into an Abstract Syntax Tree (AST). The grammar is documented in the module docstring:

- **Statements**: quantale, object, morphism, space, continuous, stochastic, discretize, embed, program, let, output declarations
- **Programs**: blocks with draw/observe steps and return statements
- **Expressions**: identity, composition (>>), tensor product (@), marginalization
- **Types**: products (*), coproducts (+)

### 3. AST Nodes (ast_nodes.py)

Each syntax construct maps to a dataclass:

```python
@dataclass
class ProgramDecl(Statement):
    name: str
    params: tuple[str, ...] | None
    domain: TypeExpr
    codomain: TypeExpr
    draws: tuple[DrawStep | LetStep, ...]
    return_vars: tuple[str, ...]
    return_labels: tuple[str, ...] | None
```

### 4. Interpretation (interpreter.py)

The interpreter walks the AST and executes programs:

- Builds up a scope mapping variable names to values
- Executes draw steps by sampling from morphism distributions
- Executes observe steps by conditioning
- Processes let steps to bind computed expressions
- Returns final values according to the return statement

## Adding a New Distribution Family

To add a new continuous distribution family:

### 1. Define the Distribution Class

Create a new class in `src/quivers/continuous/distributions.py` or a new module:

```python
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class MyDistribution:
    """My custom probability distribution.

    Parameters
    ----------
    param1 : float
        First parameter.
    param2 : float
        Second parameter.
    """
    param1: float
    param2: float

    def sample(self, size: int) -> torch.Tensor:
        """Draw samples from this distribution."""
        # implement sampling
        pass

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability density."""
        # implement log-density
        pass
```

### 2. Register in the DSL

Add the distribution to the DSL's family registry in `src/quivers/dsl/interpreter.py`:

```python
DISTRIBUTION_FAMILIES: dict[str, type] = {
    "Normal": ...,
    "Beta": ...,
    "MyDistribution": MyDistribution,
    ...
}
```

Update `src/quivers/dsl/tokens.py` if the distribution name should be syntax-highlighted as a keyword.

### 3. Add Tests

Create test cases in `tests/continuous/test_mydistribution.py`:

```python
def test_mydistribution_sample_shape():
    dist = MyDistribution(param1=1.0, param2=2.0)
    samples = dist.sample(1000)
    assert samples.shape == (1000,)

def test_mydistribution_log_prob():
    dist = MyDistribution(param1=1.0, param2=2.0)
    value = torch.tensor([0.5])
    log_prob = dist.log_prob(value)
    assert log_prob.shape == ()
```

### 4. Update Documentation

Add the distribution to `docs/api/continuous/distributions.md` with usage examples and parameter descriptions.

## Testing Philosophy

- Write tests for all public APIs
- Test both happy paths and edge cases
- Use pytest fixtures for common setup
- Organize tests to mirror the source tree structure
- Aim for clear, descriptive test names: `test_<function>_<condition>_<expected>`

## Git Workflow

1. Create a feature branch: `git checkout -b feature/description`
2. Make focused commits with clear messages
3. Push to your fork and open a pull request
4. Ensure all tests pass before requesting review
5. Respond to feedback and update as needed

## Questions and Issues

Open an issue on the repository for bugs, feature requests, or questions. For development-specific questions, use discussions.
