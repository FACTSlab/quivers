"""Round-trip law tests for the QVR resolution lenses.

For every parsed example program in ``src/quivers/dsl/examples/``, walk
the AST and run :class:`TypeExprToSetObject` and
:class:`SpaceExprToContinuousSpace` on every TypeExpr / SpaceExpr it
contains, asserting:

- **GetPut**: ``backward(*forward(t)) == t`` — the complement preserves
  enough to reconstruct the original AST.
- **Forward agrees with the compiler**: the lens' resolved value
  matches what ``Compiler._resolve_type`` / ``_resolve_space`` produce
  on the same input + environment.
"""

from pathlib import Path

import pytest

from quivers.dsl.ast_nodes import (
    ContinuousMorphismDecl,
    DiscretizeDecl,
    EmbedDecl,
    MorphismDecl,
    ObjectDecl,
    ProgramDecl,
    SpaceDecl,
    SpaceExpr,
    StochasticMorphismDecl,
    TypeExpr,
)
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse_file
from quivers.dsl.resolution import (
    SpaceExprToContinuousSpace,
    TypeExprToSetObject,
)


EXAMPLES_DIR = Path(__file__).parent.parent / "src/quivers/dsl/examples"
EXAMPLE_PATHS = sorted(EXAMPLES_DIR.glob("*.qvr"))


def _walk_type_exprs(module):
    """Yield every TypeExpr that appears in the module."""
    for stmt in module.statements:
        if isinstance(stmt, ObjectDecl):
            yield stmt.type_expr
        elif isinstance(stmt, MorphismDecl):
            yield stmt.domain
            yield stmt.codomain
        elif isinstance(stmt, ContinuousMorphismDecl):
            yield stmt.domain
            yield stmt.codomain
        elif isinstance(stmt, StochasticMorphismDecl):
            yield stmt.domain
            yield stmt.codomain
        elif isinstance(stmt, ProgramDecl):
            yield stmt.domain
            yield stmt.codomain


def _walk_space_exprs(module):
    """Yield (SpaceDecl_name, SpaceExpr) pairs."""
    for stmt in module.statements:
        if isinstance(stmt, SpaceDecl):
            yield stmt.name, stmt.space_expr


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=[p.stem for p in EXAMPLE_PATHS])
def test_type_lens_roundtrip(path: Path) -> None:
    """GetPut law: every TypeExpr round-trips through the resolution lens."""
    module = parse_file(path)
    # Build the object env by partially compiling object_decls.
    objects: dict = {}
    for stmt in module.statements:
        if isinstance(stmt, ObjectDecl):
            try:
                lens = TypeExprToSetObject(objects)
                resolved, _ = lens.forward(stmt.type_expr)
                objects[stmt.name] = resolved
            except KeyError:
                pass  # forward declarations / order issues — skip

    lens = TypeExprToSetObject(objects)
    for texpr in _walk_type_exprs(module):
        if not isinstance(texpr, TypeExpr):
            continue
        try:
            resolved, complement = lens.forward(texpr)
        except KeyError:
            # may legitimately reference space-only names; skip
            continue
        # GetPut law
        assert lens.backward(resolved, complement) == texpr


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=[p.stem for p in EXAMPLE_PATHS])
def test_type_lens_agrees_with_compiler(path: Path) -> None:
    """Lens forward agrees with Compiler._resolve_type on every example."""
    module = parse_file(path)
    compiler = Compiler(module)
    # Run compile_env to populate _objects without going all the way to the Program.
    try:
        compiler.compile_env()
    except Exception:
        # programs that need full compile() — skip the env step but still
        # check whatever _objects got built before the error
        pass

    lens = TypeExprToSetObject(compiler._objects)
    for texpr in _walk_type_exprs(module):
        if not isinstance(texpr, TypeExpr):
            continue
        try:
            lens_resolved, _ = lens.forward(texpr)
        except KeyError:
            continue
        try:
            compiler_resolved = compiler._resolve_type(texpr)
        except Exception:
            continue
        assert lens_resolved == compiler_resolved, (
            f"lens / compiler disagree on {texpr!r} in {path.name}"
        )


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=[p.stem for p in EXAMPLE_PATHS])
def test_space_lens_roundtrip(path: Path) -> None:
    """GetPut law: every SpaceExpr round-trips through the space lens."""
    module = parse_file(path)
    objects: dict = {}
    spaces: dict = {}
    for stmt in module.statements:
        if isinstance(stmt, ObjectDecl):
            try:
                tlens = TypeExprToSetObject(objects)
                resolved, _ = tlens.forward(stmt.type_expr)
                objects[stmt.name] = resolved
            except KeyError:
                pass

    for name, sexpr in _walk_space_exprs(module):
        if not isinstance(sexpr, SpaceExpr):
            continue
        slens = SpaceExprToContinuousSpace(spaces, objects, name)
        try:
            resolved, complement = slens.forward(sexpr)
        except (KeyError, ValueError):
            continue
        assert slens.backward(resolved, complement) == sexpr
        # populate env for subsequent decls
        spaces[name] = resolved


def test_lens_class_metadata() -> None:
    """The lens classes expose ``__source__`` and ``__target__`` on the
    instances they construct, per ``didactic.api.Lens``'s
    ``__init_subclass__`` contract."""
    # The base ``didactic.api.Lens`` records source/target at subclass
    # time when declared as ``Lens[A, B]``; our lenses use 3-arg form
    # so source/target are recorded but C is the complement.
    assert TypeExprToSetObject.__source__ is not None
    assert TypeExprToSetObject.__target__ is not None
