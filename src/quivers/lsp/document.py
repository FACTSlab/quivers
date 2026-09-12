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
from quivers.dsl import Compiler, ParseError, parse
from quivers.dsl.ast_nodes import DefineDecl, Module, Statement
from quivers.dsl.ast_nodes.qiec import (
    QiecEffectInstanceDecl,
    QiecHandlerDecl,
)
from quivers.dsl.constraints import check_constraints
from quivers.dsl.qiec_tooling import (
    QiecBinding,
    analyze_module,
    qiec_binding_candidates,
    qiec_binding_map,
    qiec_module_name,
)
from quivers.qiec import QiecModule


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
    qiec_module: QiecModule | None
    env: dict[str, EnvBinding]
    diagnostics: list[Diagnostic]

    def __init__(self, uri: str) -> None:
        self.uri = uri
        self.version = 0
        self.source = ""
        self.module = Module(statements=())
        self.compiler = None
        self.qiec_module = None
        self.env = {}
        self.diagnostics = []

    def update(self, *, source: str, version: int) -> None:
        """Re-parse + re-elaborate after a text change."""
        self.source = source
        self.version = version
        self.diagnostics = []
        self.module = Module(statements=())
        self.compiler = None
        self.qiec_module = None
        self.env = {}
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
        analysis = analyze_module(
            module,
            module_name=qiec_module_name(self.uri),
            file_path=self.uri,
        )
        for v in check_constraints(analysis.non_qiec_module):
            self.diagnostics.append(
                Diagnostic(
                    message=v.message,
                    severity="error",
                    line=v.line,
                    col=v.col,
                    code=v.code,
                )
            )
        for diagnostic in analysis.qiec_diagnostics:
            self.diagnostics.append(
                Diagnostic(
                    message=diagnostic.message,
                    severity="error",
                    line=diagnostic.line,
                    col=diagnostic.col,
                    code=diagnostic.code,
                )
            )
        if analysis.compile_error is not None:
            error = analysis.compile_error
            self.diagnostics.append(
                Diagnostic(
                    message=str(error),
                    severity="error",
                    line=getattr(error, "line", 0),
                    col=getattr(error, "col", 0),
                    code="compile",
                )
            )
        self.env = analysis.env
        self.compiler = analysis.compiler
        self.qiec_module = analysis.qiec_module

    def find_decl(
        self,
        name: str,
        *,
        line: int | None = None,
        col: int | None = None,
    ) -> object | None:
        """Return the statement binding ``name``, or ``None``.

        A plural-name declaration (``morphism f, g : ...``) binds every
        name in its ``names`` tuple, so several names may resolve to
        the same statement. ``define`` where-blocks are searched after
        their enclosing statement.
        """
        binding = self.find_qiec_binding(name, line=line, col=col)
        if binding is not None:
            return binding.declaration
        return _find_non_qiec_decl_in(self.module.statements, name)

    def find_qiec_binding(
        self,
        name: str,
        *,
        line: int | None = None,
        col: int | None = None,
    ) -> QiecBinding | None:
        """Resolve a QIEC identity, using source context when needed.

        Operation names are scoped by their effect interfaces. At a source
        position, a lexical ``instance.operation`` request, an operation
        declaration, or a handler clause supplies the missing owner. Without
        that context, ambiguous bare member names deliberately do not resolve.
        """

        candidates = qiec_binding_candidates(self.module, name)
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        if line is None:
            return qiec_binding_map(self.module).get(name)

        source_lines = self.source.splitlines()
        source_line = source_lines[line] if line < len(source_lines) else ""
        if col is not None:
            name_start = min(col, len(source_line))
            while name_start > 0 and _is_ident_char(source_line[name_start - 1]):
                name_start -= 1
            prefix = source_line[:name_start]
            qualifier = re.search(
                r"([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*$",
                prefix,
            )
            if qualifier is not None:
                owner = qualifier.group(1)
                qualified = qiec_binding_map(self.module).get(f"{owner}.{name}")
                if qualified is not None:
                    return qualified
                instance = qiec_binding_map(self.module).get(owner)
                if instance is not None and isinstance(
                    instance.declaration, QiecEffectInstanceDecl
                ):
                    effect_name = instance.declaration.effect.name
                    qualified = qiec_binding_map(self.module).get(
                        f"{effect_name}.{name}"
                    )
                    if qualified is not None:
                        return qualified

        source_line_number = line + 1
        declared_here = [
            binding
            for binding in candidates
            if getattr(binding.declaration, "line", 0) == source_line_number
        ]
        if len(declared_here) == 1:
            return declared_here[0]

        owner = _top_level_statement_at(self.module.statements, source_line_number)
        if owner is not None:
            owner_name = getattr(owner, "name", None)
            if isinstance(owner, QiecHandlerDecl):
                owner_name = owner.effect.name
            if isinstance(owner_name, str):
                qualified = qiec_binding_map(self.module).get(f"{owner_name}.{name}")
                if qualified is not None:
                    return qualified
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


def _find_non_qiec_decl_in(
    statements: tuple[Statement, ...], name: str
) -> object | None:
    for stmt in statements:
        if name in decl_names(stmt):
            return stmt
        if isinstance(stmt, DefineDecl):
            nested = _find_non_qiec_decl_in(stmt.where, name)
            if nested is not None:
                return nested
    return None


def _top_level_statement_at(
    statements: tuple[Statement, ...], source_line: int
) -> Statement | None:
    """Return the top-level statement whose source span contains ``source_line``."""

    located = sorted(
        (statement for statement in statements if getattr(statement, "line", 0)),
        key=lambda statement: getattr(statement, "line", 0),
    )
    current: Statement | None = None
    for statement in located:
        if getattr(statement, "line", 0) > source_line:
            break
        current = statement
    return current


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
