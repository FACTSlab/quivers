"""Per-document state for the QVR language server.

A `DocumentState` holds the latest source bytes, the parsed
`Module`, the elaborated environment, and the diagnostic list
for one open document. Re-analysis runs synchronously on every
``didOpen`` / ``didChange`` so subsequent capabilities (hover,
documentSymbol, completion) read from a coherent snapshot.
"""

from __future__ import annotations

import re
from typing import Protocol

from quivers.cli.repl_session import Diagnostic
from quivers.dsl import Compiler, CompileError, ParseError, parse
from quivers.dsl.ast_nodes import DefineDecl, Module, Statement
from quivers.dsl.constraints import check_constraints


class EnvBinding(Protocol):
    """Any value the compiler binds in the environment.

    The language server never calls into a binding; it only reports
    the binding's class name in hover output, so the protocol is
    intentionally empty.
    """


class DocumentState:
    """Per-document mutable cache for the language server.

    Each LSP document open by an editor has one of these. ``update``
    re-parses and re-elaborates in place; the resulting snapshot is
    what subsequent hover / definition / diagnostic capabilities read.
    """

    uri: str
    version: int
    source: str
    module: Module
    compiler: Compiler | None
    env: dict[str, EnvBinding]
    diagnostics: list[Diagnostic]

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.version = 0
        self.source = ""
        self.module = Module(statements=())
        self.compiler = None
        self.env = {}
        self.diagnostics = []

    def update(self, *, source: str, version: int) -> None:
        """Re-parse + re-elaborate after a text change."""
        self.source = source
        self.version = version
        self.diagnostics = []
        try:
            module = parse(source, file_path=self.uri)
        except ParseError as e:
            line, col = _extract_position(str(e))
            self.diagnostics.append(
                Diagnostic(
                    message=str(e),
                    severity="error",
                    code="parse",
                    line=line,
                    col=col,
                )
            )
            return
        self.module = module
        for v in check_constraints(module):
            self.diagnostics.append(
                Diagnostic(
                    message=v.message,
                    severity="error",
                    line=v.line,
                    col=v.col,
                    code=v.code,
                )
            )
        compiler = Compiler(module)
        try:
            self.env = compiler.compile_env()
            self.compiler = compiler
        except CompileError as e:
            self.diagnostics.append(
                Diagnostic(
                    message=str(e),
                    severity="error",
                    line=getattr(e, "line", 0),
                    col=getattr(e, "col", 0),
                    code="compile",
                )
            )
            self.compiler = compiler

    def find_decl(self, name: str) -> Statement | None:
        """Return the statement binding ``name``, or ``None``.

        A plural-name declaration (``morphism f, g : ...``) binds every
        name in its ``names`` tuple, so several names may resolve to
        the same statement. ``define`` where-blocks are searched after
        their enclosing statement.
        """
        return _find_decl_in(self.module.statements, name)

    def name_at_position(self, line: int, col: int) -> str | None:
        """Return the identifier covering ``(line, col)`` in the source."""
        lines = self.source.splitlines()
        if line >= len(lines):
            return None
        text = lines[line]
        if col > len(text):
            return None
        # Expand left and right to grab the identifier characters.
        i = col
        while i > 0 and _is_ident_char(text[i - 1]):
            i -= 1
        j = col
        while j < len(text) and _is_ident_char(text[j]):
            j += 1
        if j <= i:
            return None
        return text[i:j] or None


def decl_names(stmt: Statement) -> tuple[str, ...]:
    """Every name bound by ``stmt``.

    Plural-name declarations (``morphism f, g : ...``, ``object A, B :
    ...``, ``category NP, S``) carry a ``names`` tuple and bind one
    name per entry; single-name declarations carry ``name``;
    expression-only statements (``export``) bind nothing.
    """
    names = getattr(stmt, "names", None)
    if isinstance(names, tuple):
        return tuple(n for n in names if isinstance(n, str))
    single = getattr(stmt, "name", None)
    if isinstance(single, str):
        return (single,)
    return ()


def _find_decl_in(statements: tuple[Statement, ...], name: str) -> Statement | None:
    for stmt in statements:
        if name in decl_names(stmt):
            return stmt
        if isinstance(stmt, DefineDecl):
            nested = _find_decl_in(stmt.where, name)
            if nested is not None:
                return nested
    return None


def _is_ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


_POSITION_PATTERN = re.compile(r"line\s+(\d+),\s*col\s+(\d+)")


def _extract_position(msg: str) -> tuple[int, int]:
    """``(line, col)`` extraction from a `ParseError` message.

    The parser's syntax-error messages carry ``line L, col C`` with a
    1-based line and a 0-based column; walker-invariant messages carry
    no position, in which case ``(0, 0)`` selects the whole-file span.
    """
    m = _POSITION_PATTERN.search(msg)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0
