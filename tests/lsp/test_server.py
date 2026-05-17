"""LSP server tests.

Exercise the QVR language server in-process: build the
:class:`LanguageServer`, call the underlying handler functions
directly with crafted ``lsprotocol`` parameters, and assert on the
structured responses. This avoids the round-trip latency of a real
stdio subprocess and the asyncio-loop teardown issues that come with
multiple pygls clients in one pytest run.

Most LSP feature handlers live as nested functions inside
:func:`quivers.lsp.server.build_server`; we recover them through the
server's command/feature registry.
"""

from __future__ import annotations

import pytest


pytest.importorskip("pygls")
pytest.importorskip("lsprotocol")


from lsprotocol import types as lsp  # noqa: E402

from quivers.lsp import build_server  # noqa: E402
from quivers.lsp.document import DocumentState  # noqa: E402
from quivers.lsp.server import (  # noqa: E402
    _env_kinds_for,
    _render_hover,
    _slice_source,
    _to_lsp_diag,
)


SAMPLE = """\
## A small demo module.
object Alpha : 3
object Beta : 4
latent f : Alpha -> Beta
"""


def _doc(source: str = SAMPLE) -> DocumentState:
    doc = DocumentState(uri="file:///tmp/sample.qvr")
    doc.update(source=source, version=1)
    return doc


def test_document_update_populates_module_and_env() -> None:
    doc = _doc()
    names = {getattr(s, "name", None) for s in doc.module.statements}
    assert {"Alpha", "Beta", "f"} <= names
    assert "Alpha" in doc.env
    assert "f" in doc.env


def test_document_update_records_parse_error() -> None:
    doc = _doc(source="@@@@ bogus")
    assert doc.diagnostics
    assert doc.diagnostics[0].severity == "error"


def test_env_kinds_for_classifies_bindings() -> None:
    doc = _doc()
    kinds = _env_kinds_for(doc)
    assert kinds["Alpha"] == "type"
    assert kinds["Beta"] == "type"
    assert kinds["f"] == "function"


def test_env_kinds_empty_when_compiler_absent() -> None:
    doc = DocumentState(uri="file:///tmp/blank.qvr")
    assert _env_kinds_for(doc) == {}


def test_render_hover_stacks_qvr_and_ast() -> None:
    doc = _doc()
    decl = doc.find_decl("f")
    assert decl is not None
    hover = _render_hover(doc, "f")
    assert hover is not None
    # QVR section: bold header, fenced qvr block, verbatim source.
    assert "**QVR source**" in hover
    assert "```qvr" in hover
    assert "latent f : Alpha -> Beta" in hover
    # Divider between the two panes.
    assert "\n---\n" in hover or hover.split("**AST")[0].rstrip().endswith("---")
    # AST section: bold header, collapsed fenced python block.
    assert "**AST (didactic)**" in hover
    assert "<details>" in hover
    assert "```python" in hover
    assert "MorphismDecl" in hover


def test_render_hover_unknown_name_returns_none() -> None:
    doc = _doc()
    assert _render_hover(doc, "nope") is None


def test_slice_source_returns_original_lines() -> None:
    doc = _doc()
    decl = doc.find_decl("Alpha")
    assert decl is not None
    sliced = _slice_source(doc, decl)
    assert sliced is not None
    assert "object Alpha : 3" in sliced


def test_to_lsp_diag_maps_severity_and_range() -> None:
    from quivers.cli.repl_session import Diagnostic

    diag = Diagnostic(
        message="oops", severity="error", line=4, col=2, code="compile"
    )
    out = _to_lsp_diag(diag, _doc())
    assert out.range.start.line == 3  # 1-indexed -> 0-indexed
    assert out.range.start.character == 2
    assert out.severity == lsp.DiagnosticSeverity.Error
    assert out.source == "qvr-lsp"


def test_to_lsp_diag_warning_severity() -> None:
    from quivers.cli.repl_session import Diagnostic

    out = _to_lsp_diag(
        Diagnostic(message="m", severity="warning", line=1, col=0),
        _doc(),
    )
    assert out.severity == lsp.DiagnosticSeverity.Warning


def test_build_server_advertises_features() -> None:
    server = build_server()
    methods = set(server.protocol.fm.features.keys())
    expected = {
        "textDocument/didOpen",
        "textDocument/didChange",
        "textDocument/didSave",
        "textDocument/didClose",
        "textDocument/hover",
        "textDocument/definition",
        "textDocument/references",
        "textDocument/documentSymbol",
        "textDocument/completion",
        "textDocument/formatting",
        "textDocument/semanticTokens/full",
    }
    assert expected <= methods


def test_document_find_decl() -> None:
    doc = _doc()
    decl = doc.find_decl("f")
    assert decl is not None
    assert getattr(decl, "name", None) == "f"
    assert doc.find_decl("missing") is None


def test_document_name_at_position() -> None:
    doc = _doc()
    # Source layout (0-indexed lines):
    #   0: '## A small demo module.'
    #   1: 'object Alpha : 3'
    #   2: 'object Beta : 4'
    #   3: 'latent f : Alpha -> Beta'
    assert doc.name_at_position(1, 9) == "Alpha"
    assert doc.name_at_position(3, 7) == "f"
    assert doc.name_at_position(3, 12) == "Alpha"


def test_document_name_at_position_out_of_range() -> None:
    doc = _doc()
    assert doc.name_at_position(999, 0) is None
