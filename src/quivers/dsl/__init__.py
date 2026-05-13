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
from quivers.dsl.program_theory import (
    QVR_PROGRAM_PROTOCOL,
    extract_program_schema,
)
from quivers.program import Program


def loads(
    source: str,
    *,
    data: dict | None = None,
) -> Program:
    """Compile .qvr source text into a trainable Program.

    Parameters
    ----------
    source : str
        The ``.qvr`` source.
    data : dict, optional
        Maps string keys to tensors (or tensor-like objects) for
        any ``from_data("KEY")`` initialisers in the source. The
        compiler looks each key up at compile time; an unknown key
        raises :class:`CompileError`.
    """
    ast = parse(source)
    compiler = Compiler(ast)
    if data is not None:
        compiler.bind_data(data)
    return compiler.compile()


def load(
    path: str | Path,
    *,
    data: dict | None = None,
) -> Program:
    """Load and compile a .qvr file into a trainable Program."""
    ast = parse_file(path)
    compiler = Compiler(ast)
    if data is not None:
        compiler.bind_data(data)
    return compiler.compile()


__all__ = [
    "parse",
    "parse_file",
    "loads",
    "load",
    "ParseError",
    "CompileError",
    "Module",
    "Compiler",
    "QVR_PROGRAM_PROTOCOL",
    "extract_program_schema",
]
