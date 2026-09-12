"""User-facing tooling over the QVR v0.19 QIEC surface."""

from __future__ import annotations

from pathlib import Path

from pygments.token import Name

from quivers.cli.check import _check_one
from quivers.cli.repl_complete import all_completions
from quivers.cli.repl_highlight import tokenize
from quivers.cli.repl_session import ReplSession
from quivers.dsl import parse
from quivers.dsl.pygments_lexer import QvrLexer
from quivers.dsl.qiec_tooling import qiec_binding_map
from quivers.lsp.document import DocumentState
from quivers.lsp.server import (
    _completion_kind,
    _env_kinds_for,
    _render_hover,
    _statement_symbols,
)
from lsprotocol import types as lsp


SOURCE = """\
index Nat = Z | S(Nat)

effect State[S : Type] [version=1, evolution=sealed]
    get : Unit -> S
    put : S -> Unit

instance cell : State[Int]
"""


SCOPED_OPERATIONS = """\
effect Reader[T : Type] [version=1, evolution=sealed]
    get : Unit -> T

effect Writer[T : Type] [version=1, evolution=sealed]
    get : T -> Unit

instance reader : Reader[Int]
instance writer : Writer[Int]

define read() : Int !{reader} =
    let value <- perform reader.get()
    return value
"""


def test_check_accepts_a_qiec_module(tmp_path: Path) -> None:
    path = tmp_path / "state.qvr"
    path.write_text(SOURCE)
    assert _check_one(path) == []


def test_check_exposes_stable_qiec_diagnostic_code(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.qvr"
    path.write_text(
        "effect E [version=1, evolution=sealed]\n"
        "    go : Unit -> Unit\n"
        "    go : Unit -> Unit\n"
    )
    diagnostics = _check_one(path)
    assert [(diagnostic.code, diagnostic.line) for diagnostic in diagnostics] == [
        ("qiec-handler", 1)
    ]


def test_cli_and_repl_keep_parse_error_locations(tmp_path: Path) -> None:
    path = tmp_path / "parse-error.qvr"
    path.write_text("effect E [version=1 evolution=sealed]\n    op : Unit -> Unit\n")

    check_diagnostic = _check_one(path)[0]
    assert (check_diagnostic.code, check_diagnostic.line, check_diagnostic.col) == (
        "parse",
        1,
        10,
    )

    load_diagnostic = ReplSession().load_file(path).diagnostics[0]
    assert (load_diagnostic.code, load_diagnostic.line, load_diagnostic.col) == (
        "parse",
        1,
        10,
    )


def test_repl_installs_qiec_bindings_and_reports_kinds() -> None:
    session = ReplSession()
    response = session.dispatch(SOURCE)
    assert response.ok, response.diagnostics
    assert session._qiec_module is not None  # noqa: SLF001
    assert {"Nat", "State", "cell", "get", "put"} <= set(session.env)
    assert session.env_kinds()["Nat"] == "type"
    assert session.env_kinds()["State"] == "type"
    assert session.kind_of("State").body.startswith("effect State")
    assert session.type_of("get").body == "get : Unit -> S"
    assert "operations: {get, put}" in session.effects("State").body


def test_completion_includes_qiec_members_and_qualified_operations() -> None:
    session = ReplSession()
    assert session.dispatch(SOURCE).ok
    state_items = {item.text: item for item in all_completions(session, "Sta")}
    assert state_items["State"].kind == "type"
    assert state_items["State.get"].kind == "function"
    assert {"cell.get", "cell.put"} == {
        item.text for item in all_completions(session, "cell.")
    }
    assert (
        next(
            item
            for item in all_completions(session, "cell.")
            if item.text == "cell.get"
        ).kind
        == "function"
    )
    assert _completion_kind(state_items["State"].kind) == lsp.CompletionItemKind.Class
    assert (
        _completion_kind(state_items["State.get"].kind)
        == lsp.CompletionItemKind.Function
    )


def test_completion_omits_ambiguous_bare_qiec_members() -> None:
    session = ReplSession()
    response = session.dispatch(
        "effect Reader [version=1, evolution=sealed]\n"
        "    get : Unit -> Int\n\n"
        "effect Writer [version=1, evolution=sealed]\n"
        "    get : Unit -> Int\n"
    )
    assert response.ok, response.diagnostics

    names = {item.text for item in all_completions(session, "")}
    assert {"Reader.get", "Writer.get"} <= names
    assert "get" not in names


def test_document_state_retains_checked_qiec_and_nested_declarations() -> None:
    document = DocumentState(uri="file:///tmp/state.qvr")
    document.update(source=SOURCE, version=1)
    assert document.diagnostics == []
    assert document.qiec_module is not None
    assert type(document.find_decl("put")).__name__ == "QiecOperationDecl"
    assert _env_kinds_for(document)["State"] == "type"
    hover = _render_hover(document, "put")
    assert hover is not None and "put : S -> Unit" in hover


def test_document_symbols_nest_qiec_signature_members() -> None:
    document = DocumentState(uri="file:///tmp/state.qvr")
    document.update(source=SOURCE, version=1)
    symbols = _statement_symbols(document, document.module.statements)
    effect = next(symbol for symbol in symbols if symbol.name == "State")
    assert effect.children is not None
    assert [child.name for child in effect.children] == ["get", "put"]


def test_repeated_operation_names_keep_qualified_identities() -> None:
    module = parse(SCOPED_OPERATIONS)
    bindings = qiec_binding_map(module)
    assert "get" not in bindings
    assert bindings["Reader.get"].declaration.line == 2
    assert bindings["Writer.get"].declaration.line == 5

    session = ReplSession()
    assert session.dispatch(SCOPED_OPERATIONS).ok
    ambiguous = session.type_of("get")
    assert not ambiguous.ok
    assert "Reader.get" in ambiguous.diagnostics[0].message
    assert "Writer.get" in ambiguous.diagnostics[0].message
    assert session.type_of("Reader.get").body == "Reader.get : Unit -> T"


def test_document_resolves_repeated_operations_from_source_context() -> None:
    document = DocumentState(uri="file:///tmp/scoped.qvr")
    document.update(source=SCOPED_OPERATIONS, version=1)

    reader_decl = document.find_decl("get", line=1, col=5)
    writer_decl = document.find_decl("get", line=4, col=5)
    request_decl = document.find_decl("get", line=10, col=34)
    assert getattr(reader_decl, "line", 0) == 2
    assert getattr(writer_decl, "line", 0) == 5
    assert request_decl == reader_decl
    assert document.find_decl("get") is None

    hover = _render_hover(document, "get", line=10, col=34)
    assert hover is not None
    assert "Reader.get : Unit -> T" in hover


def test_qiec_owner_source_slice_includes_signature_members(tmp_path: Path) -> None:
    path = tmp_path / "scoped.qvr"
    path.write_text(SCOPED_OPERATIONS)
    session = ReplSession()
    assert session.load_file(path).ok
    info = session.info("Reader")
    assert "effect Reader" in info.body
    assert "get : Unit -> T" in info.body
    member_info = session.info("Reader.get")
    assert "get : Unit -> T" in member_info.body
    assert "effect Writer" not in member_info.body

    document = DocumentState(uri=path.as_uri())
    document.update(source=SCOPED_OPERATIONS, version=1)
    reader = document.find_decl("Reader")
    assert reader is not None
    from quivers.lsp.server import _slice_source

    assert "get : Unit -> T" in (_slice_source(document, reader) or "")
    reader_get = document.find_decl("Reader.get")
    assert reader_get is not None
    assert (_slice_source(document, reader_get) or "").strip() == "get : Unit -> T"


def test_qiec_semantic_highlighting_is_context_sensitive() -> None:
    pairs = {(span.token, span.text) for span in tokenize(SOURCE) if span.text.strip()}
    assert ("type", "Nat") in pairs
    assert ("type", "State") in pairs
    assert ("function", "get") in pairs
    assert ("type", "Unit") in pairs


def test_qiec_pygments_lexer_marks_types_and_operations() -> None:
    tokens = list(QvrLexer().get_tokens(SOURCE))
    assert any(token in Name.Class and text == "Nat" for token, text in tokens)
    assert any(token in Name.Function and text == "get" for token, text in tokens)


def test_tui_environment_tree_includes_qiec_bindings() -> None:
    from quivers.cli.repl_tui import _populate_scope_tree

    class Node:
        def __init__(self, label: str, *, data: str | None = None) -> None:
            self.label = label
            self.data = data
            self.children: list[Node] = []

        def add(
            self, label: str, *, data: str | None = None, expand: bool = False
        ) -> "Node":
            del expand
            child = Node(label, data=data)
            self.children.append(child)
            return child

        def add_leaf(self, label: str, *, data: str | None = None) -> "Node":
            return self.add(label, data=data)

    session = ReplSession()
    assert session.dispatch(SOURCE).ok
    root = Node("environment")
    _populate_scope_tree(root, session, lambda _name: True, filter_text="")
    leaves = {
        child.data
        for category in root.children
        for child in category.children
        if child.data is not None
    }
    assert {"Nat", "State", "cell", "State.get", "State.put"} <= leaves
