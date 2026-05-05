"""quivers DSL: parse .qvr files into trainable PyTorch models.

The DSL provides a declarative syntax for specifying V-enriched
categorical morphism networks that compile to ``quivers.Program``
instances (``nn.Module`` subclasses). Parsing is delegated to panproto
via the ``qvr`` tree-sitter grammar; the AST is a tree of
:mod:`quivers.dsl.ast_nodes` didactic Models.

Quick start
-----------
::

    from quivers.dsl import load, loads

    program = loads('''
        object X : 3
        object Y : 4
        latent f : X -> Y
        output f
    ''')

    program = load("model.qvr")

    optimizer = torch.optim.Adam(program.parameters())
"""

from pathlib import Path

from quivers.dsl.ast_nodes import Module
from quivers.dsl.compiler import Compiler, CompileError
from quivers.dsl.parser import ParseError, parse, parse_file
from quivers.program import Program


def loads(source: str) -> Program:
    """Compile .qvr source text into a trainable Program."""
    ast = parse(source)
    return Compiler(ast).compile()


def load(path: str | Path) -> Program:
    """Load and compile a .qvr file into a trainable Program."""
    ast = parse_file(path)
    return Compiler(ast).compile()


__all__ = [
    "parse",
    "parse_file",
    "loads",
    "load",
    "ParseError",
    "CompileError",
    "Module",
    "Compiler",
]
