# Contributing to Quivers

This guide covers setting up a development environment, understanding the project structure, and contributing code to the quivers library.

## Development Environment Setup

### Prerequisites

- Python 3.14 or later
- pip or conda
- git
- A C toolchain (the panproto-grammars-all wheel ships pre-built tree-sitter parsers; building it from source requires a working C compiler)

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
├── grammars/qvr/                  # Tree-sitter grammar for the QVR DSL
│   ├── grammar.js                 # Grammar source of truth
│   ├── grammar.json               # Generated; vendored by panproto
│   ├── src/                       # Generated parser.c + node-types.json
│   ├── test/corpus/               # Tree-sitter fixtures
│   └── queries/                   # Editor highlight queries
├── src/quivers/                   # Main package
│   ├── __init__.py
│   ├── categorical/               # Categorical algebra
│   ├── continuous/                # Continuous distributions (30+ families)
│   ├── core/                      # Core types (didactic Models)
│   ├── dsl/                       # QVR DSL pipeline
│   │   ├── parser.py              # panproto-driven parser walker
│   │   ├── ast_nodes.py           # didactic Model AST nodes
│   │   ├── compiler.py            # AST -> Program lowering
│   │   ├── resolution.py          # dx.Lens family for type/space resolution
│   │   ├── program_theory.py      # QVR_PROGRAM_PROTOCOL + Schema extractor
│   │   ├── pygments_lexer.py      # Pygments lexer for docs highlighting
│   │   └── examples/              # Example .qvr files
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

Target Python 3.14 and later. Use modern features:

- Type union syntax: `X | None` instead of `Union[X, None]`
- Positional-only parameters: `def func(a, /, b)`
- `Literal[...]` discriminators for `dx.TaggedUnion` variants

### Value types are didactic Models

Every record-shaped value (AST nodes, `FinSet`, `ProductSet`, `CoproductSet`, `ContinuousSpace` variants, `Category` variants, `RuleSystem`) subclasses `didactic.api.Model`. Recursive sums are `dx.TaggedUnion` roots discriminated by a `kind: Literal[...]` field. Use `dx.field(..., converter=...)` for normalization (e.g., flattening nested `ProductSet` components), `@dx.derived` for computed fields, and `__axioms__` for cross-field invariants.

Tensor-bearing accumulators (`Presheaf`, `Weight`, `SampleSite`, `Trace`) remain `@dataclass` because they hold mutable `torch.Tensor` fields.

## The DSL Pipeline

The QVR DSL processes `.qvr` files through these stages:

### 1. Parsing

`quivers.dsl.parser.parse(source)` and `parse_file(path)` delegate to panproto's tree-sitter–driven `AstParserRegistry`, which loads the QVR grammar from `panproto-grammars-all`. The parser walker then converts the parse tree into a tree of `dx.Model` AST nodes. Lexical and syntactic errors both raise `ParseError`.

### 2. AST Nodes

Each syntax construct is a `dx.Model`. Recursive sums (`TypeExpr`, `SpaceExpr`, `Expr`, `LetExprNode`, `ProgramStep`, `Statement`) are `dx.TaggedUnion` roots:

```python
import didactic.api as dx
from typing import Literal

class ProgramDecl(dx.Model):
    kind: Literal["program_decl"] = "program_decl"
    name: str
    params: tuple[str, ...] | None
    domain: TypeExpr
    codomain: TypeExpr
    body: tuple[ProgramStep, ...]
    return_vars: tuple[str, ...]
    return_labels: tuple[str, ...] | None
```

### 3. Resolution Lenses

`quivers.dsl.resolution` exposes `TypeExprToSetObject(env)` and `SpaceExprToContinuousSpace(env_spaces, env_objects, name)` as `dx.Lens` instances. The compiler invokes their `forward` direction; the complement preserves the original AST node so `backward` recovers it verbatim.

### 4. Compilation

`quivers.dsl.compiler.Compiler(ast).compile()` walks the AST, calls the resolution lenses, validates domain/codomain compatibility, builds the morphism DAG, and wraps the result in a `quivers.Program`.

### 5. Schema Extraction

`extract_program_schema(compiler)` walks the resolved environment and emits a `panproto.Schema` over `QVR_PROGRAM_PROTOCOL`. Use this to compare two programs with `panproto schema diff` or generate migration lenses between them.

## Adding a New Distribution Family

To add a new continuous distribution family:

### 1. Define the Distribution Class

Create a new class in `src/quivers/continuous/families.py` or a new module:

```python
import didactic.api as dx
import torch

class MyDistribution(dx.Model):
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
        ...

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        ...
```

### 2. Register in the DSL

Add the family to the compiler's family registry in `src/quivers/dsl/compiler.py` so that `~ MyDistribution(...)` clauses resolve to your class.

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
