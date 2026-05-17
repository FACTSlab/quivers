"""pygls-backed LSP server for QVR.

Capabilities (LSP 3.17):

- ``textDocument/didOpen``, ``didChange``, ``didSave``, ``didClose``
- ``textDocument/publishDiagnostics``
- ``textDocument/semanticTokens/full``
- ``textDocument/hover``
- ``textDocument/definition``
- ``textDocument/references``
- ``textDocument/documentSymbol``
- ``textDocument/completion``
- ``textDocument/formatting``

The server holds per-URI :class:`DocumentState` and re-analyses on
every change. All token classification, completion, and rendering
logic is shared with the REPL so the in-editor experience matches the
TUI exactly.
"""

from __future__ import annotations

from typing import Any

from pygls.lsp.server import LanguageServer
from lsprotocol import types as lsp

from quivers.cli.repl_complete import all_completions
from quivers.cli.repl_highlight import (
    SEMANTIC_TOKEN_MODIFIERS,
    SEMANTIC_TOKEN_TYPES,
    to_semantic_token_data,
)
from quivers.cli.repl_session import Diagnostic, ReplSession
from quivers.dsl.ast_nodes import MorphismDecl, ObjectDecl, SpaceDecl
from quivers.dsl.emit import module_to_source
from quivers.lsp.document import DocumentState

SERVER_NAME = "qvr-lsp"
SERVER_VERSION = "0.1.0"


def build_server() -> LanguageServer:
    """Return a configured :class:`pygls.server.LanguageServer`."""
    server = LanguageServer(name=SERVER_NAME, version=SERVER_VERSION)
    docs: dict[str, DocumentState] = {}

    # ----- lifecycle ----------------------------------------------------

    @server.feature(lsp.TEXT_DOCUMENT_DID_OPEN)
    def _did_open(ls: LanguageServer, params: lsp.DidOpenTextDocumentParams) -> None:
        doc = DocumentState(uri=params.text_document.uri)
        doc.update(
            source=params.text_document.text,
            version=params.text_document.version,
        )
        docs[doc.uri] = doc
        _publish(ls, doc)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CHANGE)
    def _did_change(ls: LanguageServer, params: lsp.DidChangeTextDocumentParams) -> None:
        uri = params.text_document.uri
        doc = docs.get(uri)
        if doc is None:
            return
        # Apply incremental changes if present; full-replace otherwise.
        for change in params.content_changes:
            if isinstance(change, lsp.TextDocumentContentChangePartial):
                doc.source = _apply_partial(doc.source, change)
            else:
                doc.source = change.text
        doc.update(source=doc.source, version=params.text_document.version)
        _publish(ls, doc)

    @server.feature(lsp.TEXT_DOCUMENT_DID_SAVE)
    def _did_save(ls: LanguageServer, params: lsp.DidSaveTextDocumentParams) -> None:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return
        if params.text is not None:
            doc.update(source=params.text, version=doc.version)
        _publish(ls, doc)

    @server.feature(lsp.TEXT_DOCUMENT_DID_CLOSE)
    def _did_close(_ls: LanguageServer, params: lsp.DidCloseTextDocumentParams) -> None:
        docs.pop(params.text_document.uri, None)

    # ----- semantic tokens ---------------------------------------------

    @server.feature(
        lsp.TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
        lsp.SemanticTokensLegend(
            token_types=list(SEMANTIC_TOKEN_TYPES),
            token_modifiers=list(SEMANTIC_TOKEN_MODIFIERS),
        ),
    )
    def _semantic_tokens(
        _ls: LanguageServer, params: lsp.SemanticTokensParams
    ) -> lsp.SemanticTokens:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return lsp.SemanticTokens(data=[])
        env_kinds = _env_kinds_for(doc)
        data = to_semantic_token_data(doc.source, env_kinds=env_kinds)
        return lsp.SemanticTokens(data=data)

    # ----- hover --------------------------------------------------------

    @server.feature(lsp.TEXT_DOCUMENT_HOVER)
    def _hover(
        _ls: LanguageServer, params: lsp.HoverParams
    ) -> lsp.Hover | None:
        doc = docs.get(params.text_document.uri)
        if doc is None or doc.compiler is None:
            return None
        name = doc.name_at_position(params.position.line, params.position.character)
        if name is None:
            return None
        body = _render_hover(doc, name)
        if body is None:
            return None
        return lsp.Hover(
            contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=body)
        )

    # ----- definition / references / documentSymbol --------------------

    @server.feature(lsp.TEXT_DOCUMENT_DEFINITION)
    def _definition(
        _ls: LanguageServer, params: lsp.DefinitionParams
    ) -> list[lsp.Location] | None:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return None
        name = doc.name_at_position(params.position.line, params.position.character)
        if name is None:
            return None
        decl = doc.find_decl(name)
        if decl is None:
            return None
        line = max(0, getattr(decl, "line", 1) - 1)
        col = max(0, getattr(decl, "col", 0))
        return [
            lsp.Location(
                uri=doc.uri,
                range=lsp.Range(
                    start=lsp.Position(line=line, character=col),
                    end=lsp.Position(line=line, character=col + len(name)),
                ),
            )
        ]

    @server.feature(lsp.TEXT_DOCUMENT_REFERENCES)
    def _references(
        _ls: LanguageServer, params: lsp.ReferenceParams
    ) -> list[lsp.Location] | None:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return None
        name = doc.name_at_position(params.position.line, params.position.character)
        if name is None:
            return None
        return list(_find_references(doc, name))

    @server.feature(lsp.TEXT_DOCUMENT_DOCUMENT_SYMBOL)
    def _symbols(
        _ls: LanguageServer, params: lsp.DocumentSymbolParams
    ) -> list[lsp.DocumentSymbol]:
        doc = docs.get(params.text_document.uri)
        if doc is None:
            return []
        out: list[lsp.DocumentSymbol] = []
        for stmt in doc.module.statements:
            name = getattr(stmt, "name", None)
            if name is None:
                continue
            line = max(0, getattr(stmt, "line", 1) - 1)
            col = max(0, getattr(stmt, "col", 0))
            rng = lsp.Range(
                start=lsp.Position(line=line, character=col),
                end=lsp.Position(line=line, character=col + len(name)),
            )
            out.append(
                lsp.DocumentSymbol(
                    name=name,
                    kind=_symbol_kind(stmt),
                    range=rng,
                    selection_range=rng,
                )
            )
        return out

    # ----- completion ---------------------------------------------------

    @server.feature(
        lsp.TEXT_DOCUMENT_COMPLETION,
        lsp.CompletionOptions(trigger_characters=[":", " "]),
    )
    def _completion(
        _ls: LanguageServer, params: lsp.CompletionParams
    ) -> lsp.CompletionList:
        doc = docs.get(params.text_document.uri)
        if doc is None or doc.compiler is None:
            return lsp.CompletionList(is_incomplete=False, items=[])
        session = ReplSession()
        session._module = doc.module  # noqa: SLF001
        session._compiler = doc.compiler  # noqa: SLF001
        session._env = doc.env  # noqa: SLF001
        prefix = _prefix_at(doc.source, params.position.line, params.position.character)
        items = [
            lsp.CompletionItem(
                label=c.text,
                kind=_completion_kind(c.kind),
                detail=c.detail,
            )
            for c in all_completions(session, prefix)
        ]
        return lsp.CompletionList(is_incomplete=False, items=items)

    # ----- formatting ---------------------------------------------------

    @server.feature(lsp.TEXT_DOCUMENT_FORMATTING)
    def _formatting(
        _ls: LanguageServer, params: lsp.DocumentFormattingParams
    ) -> list[lsp.TextEdit] | None:
        doc = docs.get(params.text_document.uri)
        if doc is None or not doc.module.statements:
            return None
        try:
            canonical = module_to_source(doc.module)
        except NotImplementedError:
            # Module contains a variant the canonical emitter doesn't cover;
            # leave the buffer alone rather than corrupting it.
            return None
        if canonical == doc.source:
            return []
        end_line = doc.source.count("\n")
        end_col = len(doc.source.splitlines()[-1]) if doc.source else 0
        return [
            lsp.TextEdit(
                range=lsp.Range(
                    start=lsp.Position(line=0, character=0),
                    end=lsp.Position(line=end_line, character=end_col),
                ),
                new_text=canonical,
            )
        ]

    return server


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _env_kinds_for(doc: DocumentState) -> dict[str, str]:
    """Build the name -> semantic-token-type map from a document's env."""
    compiler = doc.compiler
    if compiler is None:
        return {}
    kinds: dict[str, str] = {}
    for name in getattr(compiler, "objects", {}):
        kinds[name] = "type"
    for name in getattr(compiler, "spaces", {}):
        kinds[name] = "type"
    for name in getattr(compiler, "morphisms", {}):
        kinds[name] = "function"
    for name in getattr(compiler, "rules", {}):
        kinds[name] = "namespace"
    return kinds


def _publish(ls: LanguageServer, doc: DocumentState) -> None:
    ls.publish_diagnostics(
        doc.uri,
        [_to_lsp_diag(d, doc) for d in doc.diagnostics],
    )


def _to_lsp_diag(d: Diagnostic, doc: DocumentState) -> lsp.Diagnostic:
    line = max(0, d.line - 1) if d.line else 0
    col = max(0, d.col)
    end_line = max(0, (d.end_line or d.line or 1) - 1)
    end_col = max(col + 1, d.end_col or col + 1)
    if end_line == 0 and not d.line:
        # Whole-file diagnostic when position is unknown.
        end_line = doc.source.count("\n")
        end_col = len(doc.source.splitlines()[-1]) if doc.source else 0
    severity_map = {
        "error": lsp.DiagnosticSeverity.Error,
        "warning": lsp.DiagnosticSeverity.Warning,
        "info": lsp.DiagnosticSeverity.Information,
        "ok": lsp.DiagnosticSeverity.Information,
    }
    return lsp.Diagnostic(
        range=lsp.Range(
            start=lsp.Position(line=line, character=col),
            end=lsp.Position(line=end_line, character=end_col),
        ),
        message=d.message,
        severity=severity_map.get(d.severity, lsp.DiagnosticSeverity.Error),
        code=d.code or None,
        source=SERVER_NAME,
    )


def _apply_partial(
    source: str, change: Any
) -> str:
    """Apply one incremental change to ``source``."""
    rng = change.range
    lines = source.split("\n")

    def offset(line: int, character: int) -> int:
        line = min(line, len(lines) - 1)
        return sum(len(line_text) + 1 for line_text in lines[:line]) + character

    start = offset(rng.start.line, rng.start.character)
    end = offset(rng.end.line, rng.end.character)
    return source[:start] + change.text + source[end:]


def _render_hover(doc: DocumentState, name: str) -> str | None:
    decl = doc.find_decl(name)
    if decl is None:
        if name in doc.env:
            return f"```\n{name} :: {type(doc.env[name]).__name__}\n```"
        return None
    try:
        rendered = module_to_source(
            type(doc.module)(statements=(decl,))
        ).rstrip()
    except NotImplementedError:
        rendered = repr(decl)
    docs = getattr(decl, "docs", ())
    body = f"```qvr\n{rendered}\n```"
    if docs:
        body = "\n".join(docs) + "\n\n" + body
    return body


def _find_references(doc: DocumentState, name: str):  # type: ignore[no-untyped-def]
    """Locate every textual occurrence of ``name`` in the source."""
    for lineno, line in enumerate(doc.source.splitlines()):
        start = 0
        while True:
            idx = line.find(name, start)
            if idx == -1:
                break
            # Word-boundary check.
            left_ok = idx == 0 or not (line[idx - 1].isalnum() or line[idx - 1] == "_")
            right_ok = (
                idx + len(name) == len(line)
                or not (
                    line[idx + len(name)].isalnum() or line[idx + len(name)] == "_"
                )
            )
            if left_ok and right_ok:
                yield lsp.Location(
                    uri=doc.uri,
                    range=lsp.Range(
                        start=lsp.Position(line=lineno, character=idx),
                        end=lsp.Position(line=lineno, character=idx + len(name)),
                    ),
                )
            start = idx + len(name)


def _prefix_at(source: str, line: int, character: int) -> str:
    lines = source.splitlines()
    if line >= len(lines):
        return ""
    text = lines[line][:character]
    i = len(text)
    while i > 0 and (text[i - 1].isalnum() or text[i - 1] in "_:"):
        i -= 1
    return text[i:]


def _symbol_kind(stmt: Any) -> lsp.SymbolKind:
    if isinstance(stmt, ObjectDecl):
        return lsp.SymbolKind.Class
    if isinstance(stmt, SpaceDecl):
        return lsp.SymbolKind.Struct
    if isinstance(stmt, MorphismDecl):
        return lsp.SymbolKind.Function
    return lsp.SymbolKind.Variable


def _completion_kind(kind: str) -> lsp.CompletionItemKind:
    return {
        "command": lsp.CompletionItemKind.Operator,
        "env": lsp.CompletionItemKind.Variable,
        "keyword": lsp.CompletionItemKind.Keyword,
        "type": lsp.CompletionItemKind.Class,
        "function": lsp.CompletionItemKind.Function,
        "namespace": lsp.CompletionItemKind.Module,
        "path": lsp.CompletionItemKind.File,
    }.get(kind, lsp.CompletionItemKind.Text)


__all__ = ["build_server", "SERVER_NAME", "SERVER_VERSION"]
