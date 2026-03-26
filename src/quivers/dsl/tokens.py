"""Token types for the quivers DSL lexer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    """All token types in the .qvr language."""

    # keywords
    QUANTALE = auto()
    CATEGORY = auto()
    OBJECT = auto()
    RULE = auto()
    LATENT = auto()
    OBSERVED = auto()
    LET = auto()
    OUTPUT = auto()
    IDENTITY = auto()
    MARGINALIZE = auto()

    # continuous/stochastic keywords
    SPACE = auto()
    CONTINUOUS = auto()
    STOCHASTIC = auto()
    DISCRETIZE = auto()
    EMBED = auto()

    # monadic program keywords
    PROGRAM = auto()
    DRAW = auto()
    OBSERVE = auto()
    RETURN = auto()

    # combinators
    FAN = auto()
    REPEAT = auto()
    STACK = auto()
    SCAN = auto()
    PARSER = auto()
    CCG = auto()  # backward compat alias for parser with CCG defaults
    LAMBEK = auto()  # backward compat alias for parser with Lambek defaults

    # type alias
    TYPE = auto()

    # literals
    IDENT = auto()
    INT = auto()
    FLOAT = auto()

    # operators and punctuation
    COLON = auto()  # :
    ARROW = auto()  # ->
    DARROW = auto()  # =>
    COMPOSE = auto()  # >>
    COMPOSE_BACK = auto()  # <<
    KLEISLI = auto()  # >=>
    LARROW = auto()  # <-
    PRODUCT = auto()  # *
    COPRODUCT = auto()  # +
    TENSOR = auto()  # @
    BACKSLASH = auto()  # \.
    EQUALS = auto()  # =
    DOT = auto()  # .
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACKET = auto()  # [
    RBRACKET = auto()  # ]
    COMMA = auto()  # ,
    TILDE = auto()  # ~
    MINUS = auto()  # -
    SLASH = auto()  # /
    WHERE = auto()  # where

    # special
    NEWLINE = auto()
    EOF = auto()


# keyword lookup table
KEYWORDS: dict[str, TokenType] = {
    "quantale": TokenType.QUANTALE,
    "category": TokenType.CATEGORY,
    "object": TokenType.OBJECT,
    "rule": TokenType.RULE,
    "latent": TokenType.LATENT,
    "observed": TokenType.OBSERVED,
    "let": TokenType.LET,
    "output": TokenType.OUTPUT,
    "identity": TokenType.IDENTITY,
    "marginalize": TokenType.MARGINALIZE,
    "space": TokenType.SPACE,
    "continuous": TokenType.CONTINUOUS,
    "stochastic": TokenType.STOCHASTIC,
    "discretize": TokenType.DISCRETIZE,
    "embed": TokenType.EMBED,
    "program": TokenType.PROGRAM,
    "draw": TokenType.DRAW,
    "observe": TokenType.OBSERVE,
    "return": TokenType.RETURN,
    "fan": TokenType.FAN,
    "repeat": TokenType.REPEAT,
    "stack": TokenType.STACK,
    "scan": TokenType.SCAN,
    "parser": TokenType.PARSER,
    "ccg": TokenType.CCG,
    "lambek": TokenType.LAMBEK,
    "type": TokenType.TYPE,
    "where": TokenType.WHERE,
}


@dataclass(frozen=True)
class Token:
    """A single token from the lexer.

    Parameters
    ----------
    type : TokenType
        The token's type.
    value : str
        The raw string value.
    line : int
        Source line number (1-indexed).
    col : int
        Source column number (1-indexed).
    """

    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"
