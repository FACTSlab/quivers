"""Per-document state for the QVR language server.

A `DocumentState` holds the latest source bytes, the parsed
`Module`, the elaborated environment, and the diagnostic list
for one open document. Re-analysis runs synchronously on every
``didOpen`` / ``didChange`` so subsequent capabilities (hover,
documentSymbol, completion) read from a coherent snapshot.
"""

from __future__ import annotations

import re
from typing import Any

from quivers.cli.repl_session import Diagnostic
from quivers.dsl import Compiler, CompileError, ParseError, parse
from quivers.dsl.ast_nodes import Module
from quivers.dsl.constraints import check_constraints


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
    env: dict[str, Any]
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
            self.diagnostics.append(
                Diagnostic(
                    message=str(e),
                    severity="error",
                    code="parse",
                    line=_extract_line(str(e)),
                    col=0,
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

    def find_decl(self, name: str):  # type: ignore[no-untyped-def]
        for stmt in self.module.statements:
            if getattr(stmt, "name", None) == name:
                return stmt
        return None

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


def _is_ident_char(c: str) -> bool:
    return c.isalnum() or c == "_"


def _extract_line(msg: str) -> int:
    """Best-effort line extraction from a panproto ParseError message."""
    m = re.search(r"line\s+(\d+)", msg)
    if m:
        return int(m.group(1))
    return 0
