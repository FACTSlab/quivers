"""Fixture discovery for the transpile correctness suite.

Every fixture is a complete `.qvr` file that
[`qvr check`][quivers.cli.check] accepts. The loader exposes them as
``[(name, source, path)]`` tuples so test files can parametrise
directly:

```python
@pytest.mark.parametrize("fixture", load_compositions(),
                         ids=lambda f: f.name)
def test_something(fixture):
    module = parse(fixture.source)
    ...
```

Categories:

- **compositions** — full real-world programs. Reuses the curated
  inference-benchmark corpus at `tests/benchmarks/models/` so every
  fixture has an associated reference posterior (see
  `tests/benchmarks/references/`).
- **families** — programmatically generated, one minimal program per
  entry in
  [`_get_family_registry`][quivers.dsl.compiler._prelude._get_family_registry].
  Used by the family × backend coverage matrix.
- **statements** — one minimal program per `Statement` discriminator
  in [`quivers.dsl.ast_nodes.declarations`][quivers.dsl.ast_nodes].
- **steps** — one minimal program per `ProgramStep` discriminator.
- **let_expressions** — one minimal program per `LetExprNode`
  discriminator.
- **options** — one minimal program per `OptionValue` discriminator.
- **axes** — one minimal program per `AxisSpec` pattern.
"""

from __future__ import annotations

import dataclasses
import pathlib


_THIS_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]
_BENCHMARKS_DIR = _REPO_ROOT / "tests" / "benchmarks" / "models"


@dataclasses.dataclass(frozen=True)
class Fixture:
    """One fixture's `.qvr` source plus its discovered metadata."""

    name: str
    """Stable identifier (filename stem); used as the pytest parametrise id."""

    source: str
    """The full `.qvr` source text."""

    path: pathlib.Path
    """Absolute path to the source file."""

    category: str
    """One of ``compositions`` / ``families`` / ``statements`` / ``steps``
    / ``let_expressions`` / ``options`` / ``axes``."""


def _load_dir(category: str, directory: pathlib.Path) -> list[Fixture]:
    if not directory.is_dir():
        return []
    fixtures: list[Fixture] = []
    for path in sorted(directory.glob("*.qvr")):
        fixtures.append(
            Fixture(
                name=path.stem,
                source=path.read_text(),
                path=path,
                category=category,
            )
        )
    return fixtures


def load_compositions() -> list[Fixture]:
    """Real-world programs from the inference-benchmark corpus."""
    return _load_dir("compositions", _BENCHMARKS_DIR)


def load_families() -> list[Fixture]:
    """One fixture per distribution family in the QVR registry."""
    return _load_dir("families", _THIS_DIR / "families")


def load_statements() -> list[Fixture]:
    """One fixture per `Statement` discriminator."""
    return _load_dir("statements", _THIS_DIR / "statements")


def load_steps() -> list[Fixture]:
    """One fixture per `ProgramStep` discriminator."""
    return _load_dir("steps", _THIS_DIR / "steps")


def load_let_expressions() -> list[Fixture]:
    """One fixture per `LetExprNode` discriminator."""
    return _load_dir("let_expressions", _THIS_DIR / "let_expressions")


def load_options() -> list[Fixture]:
    """One fixture per `OptionValue` discriminator."""
    return _load_dir("options", _THIS_DIR / "options")


def load_axes() -> list[Fixture]:
    """One fixture per `AxisSpec` pattern."""
    return _load_dir("axes", _THIS_DIR / "axes")


def load_all() -> list[Fixture]:
    """Every fixture across every category."""
    return (
        load_compositions()
        + load_families()
        + load_statements()
        + load_steps()
        + load_let_expressions()
        + load_options()
        + load_axes()
    )


__all__ = [
    "Fixture",
    "load_all",
    "load_axes",
    "load_compositions",
    "load_families",
    "load_let_expressions",
    "load_options",
    "load_statements",
    "load_steps",
]
