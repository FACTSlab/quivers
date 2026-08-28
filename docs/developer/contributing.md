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
│   ├── continuous/                # Continuous distributions (more than 40 families)
│   ├── core/                      # Core types (didactic Models)
│   ├── dsl/                       # QVR DSL pipeline
│   │   ├── parser/                # panproto-driven parser walker
│   │   ├── ast_nodes/             # didactic Model AST nodes
│   │   ├── compiler/              # AST -> Program lowering and resolution
│   │   ├── emit.py                # canonical AST -> QVR source printer
│   │   ├── program_theory.py      # QVR_PROGRAM_PROTOCOL + Schema extractor
│   │   ├── pygments_lexer.py      # Pygments lexer for docs highlighting
│   ├── effects/                   # Algebraic effect handlers
│   ├── enriched/                  # Enriched categories
│   ├── inference/                 # Variational inference
│   ├── monadic/                   # Monadic programs (draw, observe, return)
│   ├── stochastic/                # Stochastic morphisms
│   ├── transpile/                 # Cross-language PPL emitters
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

Each syntax construct is a `dx.Model`. Recursive sums (`ObjectExpr`,
`Expr`, `LetExprNode`, `ProgramStep`, `Statement`) are
`dx.TaggedUnion` roots:

<!-- python: skip -->
```python
from typing import Literal
from quivers.dsl.ast_nodes import ObjectExpr, OptionEntry, ProgramParam, ProgramStep, Statement

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

### 3. Resolution

`quivers.dsl.compiler.resolution._ResolutionMixin` resolves an
`ObjectExpr` to either a discrete `SetObject` or a
`ContinuousSpace`. `_resolve_any_space` performs the shared dispatch;
the narrower helpers reject a result in the wrong category.

### 4. Compilation

`quivers.dsl.compiler.Compiler(ast).compile()` walks the AST, calls the resolution lenses, validates domain/codomain compatibility, builds the morphism DAG, and wraps the result in a `quivers.Program`.

### 5. Schema Extraction

`extract_program_schema(compiler)` walks the resolved environment and emits a `panproto.Schema` over `QVR_PROGRAM_PROTOCOL`. Use this to compare two programs with `panproto schema diff` or generate migration lenses between them.

## Adding a New Distribution Family

Most independent families use the single registry in
`src/quivers/continuous/family_spec.py`. If PyTorch already provides
the distribution, add an `_make_family(...)` call in
`src/quivers/continuous/families.py` with the parameter names and
bijectors that map unconstrained parameters onto their supports.
That call creates the `Conditional*` class and registers both inline
DSL paths.

<!-- python: skip -->
```python
ConditionalMyFamily = _make_family(
    "ConditionalMyFamily",
    MyTorchDistribution,
    [("loc", "id"), ("scale", "softplus")],
    "Conditional MyFamily(loc(x), scale(x)).",
)
```

Families with vector or matrix events, custom constructors, or
nonstandard sampling behavior need a hand-written
`ContinuousMorphism` plus a `FamilySpec` whose override fields point
to that implementation. An entirely new underlying distribution may
also require a `torch.distributions.Distribution` implementation.

Add registry, fixed-inline, mixed-inline, conditional sampling, and
`log_prob` coverage. `tests/test_family_registry.py` contains the
shared registry checks; specialized behavior belongs under
`tests/continuous/`. Document the family in
`docs/guides/continuous-families.md` and
`docs/api/continuous/families.md`.

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
