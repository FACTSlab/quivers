# Contributing to Quivers

This guide covers setting up a development environment, understanding the project structure, and contributing code to the quivers library.

## Development Environment Setup

### Prerequisites

- Python 3.14 or later
- pip or conda
- git

### Installation

Clone the repository and install in development mode:

```bash
git clone https://github.com/FACTSlab/quivers
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
│   ├── continuous/                # Continuous distributions (40+ families)
│   ├── core/                      # Core types and utilities
│   ├── dsl/                       # QVR DSL (grammar walker, compiler, emitter)
│   ├── enriched/                  # Enriched categories
│   ├── inference/                 # Variational inference
│   ├── monadic/                   # Monadic programs (draw, observe, return)
│   └── stochastic/                # Stochastic morphisms
├── tests/                         # Test suite (mirrors src structure)
├── pyproject.toml                 # Package metadata
└── mkdocs.yml                     # Documentation config
```

## Code Style Conventions

### Type Hints

Include type hints in all function signatures. Use modern Python 3.14+ syntax:

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

Maintain compatibility with Python 3.14 and later. Use modern features:

- Type union syntax: `X | None` instead of `Union[X, None]`
- Positional-only parameters: `def func(a, /, b)`

## The DSL Pipeline

The QVR DSL processes `.qvr` files through these stages:

### 1. Grammar and parsing (`grammars/qvr/`, `dsl/parser/`)

There is no hand-written lexer or recursive-descent parser. The
grammar is a [tree-sitter](https://tree-sitter.github.io/) grammar
(`grammars/qvr/grammar.js`), compiled to a parser that ships vendored
in `panproto-grammars-all` and is served through panproto's
`AstParserRegistry`. Parsing a `.qvr` source yields a panproto schema
(the parse tree as vertices, edges, and field constraints), and the
walkers in `src/quivers/dsl/parser/` turn that schema into AST nodes.
The walker rejects any `ERROR` or missing node, so a malformed source
fails loudly with a line and column rather than parsing to a silently
different tree.

Editing the grammar means editing `grammar.js`, regenerating with
`tree-sitter generate`, and re-vendoring through `panproto-grammars-all`;
the `grammars/qvr/vcs/` panproto store and the `qvr migrate` chain
carry `.qvr` sources across grammar releases.

### 2. AST nodes (`dsl/ast_nodes/`)

Each grammar production maps to a [didactic](https://github.com/panproto/didactic)
model, not a dataclass. Statement variants live under a `dx.TaggedUnion`
keyed on `kind`; leaf records are `dx.Model` subclasses:

```python
class ProgramDecl(Statement):
    name: str
    params: tuple[str, ...] | None = None
    type_params: tuple[ProgramParam, ...] | None = None
    domain: ObjectExpr
    codomain: ObjectExpr
    options: tuple[OptionEntry, ...] = ()
    draws: tuple[ProgramStep, ...] = ()
    return_vars: tuple[str, ...] = ()
    return_labels: tuple[str, ...] | None = None
    docs: tuple[str, ...] = ()
    line: int = 0
    col: int = 0
    kind: Literal["program_decl"] = "program_decl"
```

### 3. Compilation (`dsl/compiler/`)

The compiler walks the AST and builds executable programs:

- Resolves object and morphism references against the module's declarations.
- Reads each declaration's option block against a closed key set, reporting an unknown key with a did-you-mean suggestion.
- Expands surface program steps (`sample`, `observe`, `marginalize`, `let`, `score`) into the internal bind IR.
- Emits a [`Program`][quivers.program.Program] whose morphism is a Kleisli arrow in the (discrete or continuous) Giry monad.

## Adding a New Distribution Family

To add a new continuous distribution family:

### 1. Define the Distribution Class

Create a new class in `src/quivers/continuous/families.py` or a new module:

Distribution families subclass `torch.distributions.Distribution`
(often alongside the measure-algebra combinators in
`quivers.continuous.measure`), so they compose with PyTorch's
sampling and scoring machinery:

```python
import torch
from torch import Tensor
from torch.distributions.distribution import Distribution


class MyDistribution(Distribution):
    """My custom probability distribution."""

    def __init__(self, param1: Tensor, param2: Tensor) -> None:
        self.param1 = param1
        self.param2 = param2
        super().__init__(batch_shape=param1.shape)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> Tensor:
        """Draw samples from this distribution."""
        ...

    def log_prob(self, value: Tensor) -> Tensor:
        """Compute the log probability of ``value``."""
        ...
```

### 2. Register in the DSL

Add the distribution to the DSL's family registry so it can be used in `.qvr` files.

### 3. Add Tests

Create test cases in `tests/continuous/`:

```python
def test_mydistribution_sample_shape():
    dist = MyDistribution(param1=torch.tensor(1.0), param2=torch.tensor(2.0))
    samples = dist.sample(1000)
    assert samples.shape == (1000,)

def test_mydistribution_log_prob():
    dist = MyDistribution(param1=torch.tensor(1.0), param2=torch.tensor(2.0))
    value = torch.tensor([0.5])
    log_prob = dist.log_prob(value)
    assert log_prob.shape == ()
```

### 4. Update Documentation

Add the distribution to the API reference with usage examples and parameter descriptions.

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

Open an issue on the [GitHub repository](https://github.com/FACTSlab/quivers) for bugs, feature requests, or questions.
