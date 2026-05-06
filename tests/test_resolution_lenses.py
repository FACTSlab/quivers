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
    MorphismDecl,
    ObjectDecl,
    SpaceDecl,
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
    """Yield every TypeExpr that the compiler routes through ``_resolve_type``.

    Continuous and stochastic morphism declarations and program-block
    domains/codomains may reference space-only names; the compiler routes
    those through ``_resolve_any_space`` instead. They're excluded here so
    the lens-vs-compiler comparison stays apples-to-apples.
    """
    for stmt in module.statements:
        if isinstance(stmt, ObjectDecl):
            yield stmt.type_expr
        elif isinstance(stmt, MorphismDecl):
            yield stmt.domain
            yield stmt.codomain


def _walk_space_exprs(module):
    """Yield (SpaceDecl_name, SpaceExpr) pairs."""
    for stmt in module.statements:
        if isinstance(stmt, SpaceDecl):
            yield stmt.name, stmt.space_expr


def _build_object_env(module) -> dict:
    """Build the object environment by walking object_decls in source order."""
    objects: dict = {}
    for stmt in module.statements:
        if isinstance(stmt, ObjectDecl):
            lens = TypeExprToSetObject(objects)
            resolved, _ = lens.forward(stmt.type_expr)
            objects[stmt.name] = resolved
    return objects


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=[p.stem for p in EXAMPLE_PATHS])
def test_type_lens_roundtrip(path: Path) -> None:
    """GetPut law: every TypeExpr round-trips through the resolution lens."""
    module = parse_file(path)
    objects = _build_object_env(module)
    lens = TypeExprToSetObject(objects)
    for texpr in _walk_type_exprs(module):
        resolved, complement = lens.forward(texpr)
        # GetPut law
        assert lens.backward(resolved, complement) == texpr


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=[p.stem for p in EXAMPLE_PATHS])
def test_type_lens_agrees_with_compiler(path: Path) -> None:
    """Lens forward agrees with Compiler._resolve_type on every example."""
    module = parse_file(path)
    compiler = Compiler(module)
    compiler.compile_env()

    lens = TypeExprToSetObject(compiler._objects)
    for texpr in _walk_type_exprs(module):
        lens_resolved, _ = lens.forward(texpr)
        compiler_resolved = compiler._resolve_type(texpr)
        assert lens_resolved == compiler_resolved, (
            f"lens / compiler disagree on {texpr!r} in {path.name}"
        )


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=[p.stem for p in EXAMPLE_PATHS])
def test_space_lens_roundtrip(path: Path) -> None:
    """GetPut law: every SpaceExpr round-trips through the space lens."""
    module = parse_file(path)
    objects = _build_object_env(module)
    spaces: dict = {}

    for name, sexpr in _walk_space_exprs(module):
        slens = SpaceExprToContinuousSpace(spaces, objects, name)
        resolved, complement = slens.forward(sexpr)
        assert slens.backward(resolved, complement) == sexpr
        # populate env for subsequent decls
        spaces[name] = resolved


def test_lens_class_metadata() -> None:
    """The lens classes expose ``__source__`` and ``__target__`` on the
    classes themselves, per ``didactic.api.Lens``'s
    ``__init_subclass__`` contract.

    ``didactic.api.Lens.__init_subclass__`` walks ``__orig_bases__`` for a
    ``Lens[A, B, C]`` parameterisation and records ``A`` as
    ``__source__`` and ``B`` as ``__target__`` only when each is a
    plain class. Union-typed targets (e.g.
    ``ContinuousSpace | SetObject``) record as ``None``; that's the
    case for :class:`SpaceExprToContinuousSpace`.
    """
    from quivers.core.objects import SetObject
    from quivers.dsl.ast_nodes import SpaceExpr, TypeExpr

    assert TypeExprToSetObject.__source__ is TypeExpr
    assert TypeExprToSetObject.__target__ is SetObject
    assert SpaceExprToContinuousSpace.__source__ is SpaceExpr
    # target is a Union[ContinuousSpace, SetObject]; not a single class,
    # so didactic records None.
    assert SpaceExprToContinuousSpace.__target__ is None
