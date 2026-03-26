"""quivers DSL: parse .qvr files into trainable PyTorch models.

The DSL provides a declarative syntax for specifying V-enriched
categorical morphism networks that compile to ``quivers.Program``
instances (``nn.Module`` subclasses).

Quick start
-----------
::

    from quivers.dsl import load, loads

    # from a string
    program = loads('''
        object X : 3
        object Y : 4
        latent f : X -> Y
        output f
    ''')

    # from a .qvr file
    program = load("model.qvr")

    # train as usual
    optimizer = torch.optim.Adam(program.parameters())

File extension
--------------
The canonical extension is ``.qvr``.

Syntax overview
---------------
::

    # comments start with #

    # quantale selection (optional, defaults to product_fuzzy)
    quantale product_fuzzy

    # object declarations
    object X : 3           # FinSet("X", 3)
    object Y : 4           # FinSet("Y", 4)
    object XY : X * Y      # ProductSet(X, Y)
    object XZ : X + Z      # CoproductSet(X, Z)

    # morphism declarations
    latent f : X -> Y                    # learnable
    latent g : Y -> X [scale=0.3]        # with init scale
    observed h : X -> X = identity(X)    # fixed identity

    # compositions
    let fg = f >> g               # sequential (V-enriched)
    let par = f @ g               # parallel (tensor product)
    let m = par.marginalize(Y)    # marginalize over Y

    # output (compiles to Program)
    output fg

Continuous extensions::

    # continuous space declarations
    space R3 : Euclidean(3)
    space S4 : Simplex(4)
    space P2 : PositiveReals(2)
    space U : UnitInterval
    space B : Euclidean(2, low=0.0, high=1.0)
    space PS : R3 * S4            # product space

    # stochastic morphisms (Markov kernels)
    stochastic s : X -> Y

    # continuous morphisms with distribution family
    continuous g : X -> R3 ~ Normal
    continuous k : R3 -> S4 ~ Dirichlet
    continuous fl : X -> R3 ~ Flow [n_layers=6, hidden_dim=32]

    # boundary morphisms
    discretize d : B -> 10
    embed e : X -> R3

    # composition works across discrete/continuous
    let pipeline = s >> g
    output pipeline
"""

from __future__ import annotations

from pathlib import Path

from quivers.dsl.lexer import Lexer, LexError
from quivers.dsl.parser import Parser, ParseError
from quivers.dsl.compiler import Compiler, CompileError
from quivers.dsl.ast_nodes import Module
from quivers.program import Program


def parse(source: str) -> Module:
    """Parse .qvr source text into an AST.

    Parameters
    ----------
    source : str
        The raw .qvr source code.

    Returns
    -------
    Module
        The parsed AST.

    Raises
    ------
    LexError
        On lexical errors.
    ParseError
        On syntax errors.
    """
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def loads(source: str) -> Program:
    """Compile .qvr source text into a trainable Program.

    Parameters
    ----------
    source : str
        The raw .qvr source code.

    Returns
    -------
    Program
        A trainable ``nn.Module`` wrapping the morphism DAG.

    Raises
    ------
    LexError
        On lexical errors.
    ParseError
        On syntax errors.
    CompileError
        On semantic errors.

    Examples
    --------
    >>> from quivers.dsl import loads
    >>> prog = loads('''
    ...     object X : 3
    ...     object Y : 4
    ...     latent f : X -> Y
    ...     output f
    ... ''')
    >>> prog().shape
    torch.Size([3, 4])
    """
    ast = parse(source)
    return Compiler(ast).compile()


def load(path: str | Path) -> Program:
    """Load and compile a .qvr file into a trainable Program.

    Parameters
    ----------
    path : str or Path
        Path to a .qvr file.

    Returns
    -------
    Program
        A trainable ``nn.Module`` wrapping the morphism DAG.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    LexError
        On lexical errors.
    ParseError
        On syntax errors.
    CompileError
        On semantic errors.
    """
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    return loads(source)


__all__ = [
    "parse",
    "loads",
    "load",
    "LexError",
    "ParseError",
    "CompileError",
    "Module",
    "Compiler",
    "Lexer",
    "Parser",
]
