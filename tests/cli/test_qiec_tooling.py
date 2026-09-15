"""User-facing tooling over the QVR QIEC surface."""

from __future__ import annotations

from pathlib import Path
from argparse import Namespace
import json

from pygments.token import Name

from quivers.cli.check import _check_one
from quivers.cli.run import main as run_main
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

effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

instance cell : State[Int]
"""


SCOPED_OPERATIONS = """\
effect Reader[T : Type]
    get : Unit -> T

effect Writer[T : Type]
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
    path.write_text("effect E\n    go : Unit -> Unit\n    go : Unit -> Unit\n")
    diagnostics = _check_one(path)
    assert [(diagnostic.code, diagnostic.line) for diagnostic in diagnostics] == [
        ("qiec-handler", 1)
    ]


def test_cli_and_repl_keep_parse_error_locations(tmp_path: Path) -> None:
    """One malformed source, one location, reported the same way twice.

    The operation here is missing its `:`, so the parse fails inside the
    effect body rather than at the header. What the test pins is that
    `qvr check` and a REPL load agree on where, since a diagnostic that
    moves between the two sends a user to the wrong line.
    """
    path = tmp_path / "parse-error.qvr"
    path.write_text("effect E\n    op Unit -> Unit\n")
    expected = ("parse", 2, 4)

    check_diagnostic = _check_one(path)[0]
    assert (
        check_diagnostic.code,
        check_diagnostic.line,
        check_diagnostic.col,
    ) == expected

    load_diagnostic = ReplSession().load_file(path).diagnostics[0]
    assert (
        load_diagnostic.code,
        load_diagnostic.line,
        load_diagnostic.col,
    ) == expected


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
        "effect Reader\n    get : Unit -> Int\n\neffect Writer\n    get : Unit -> Int\n"
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


def test_qvr_run_and_repl_execute_named_polymorphic_computation(
    tmp_path: Path, capsys
) -> None:
    path = tmp_path / "identity.qvr"
    path.write_text(
        "define identity[A : Type](value : A) : A !{} =\n    return value\n"
    )
    args = Namespace(
        file=str(path),
        computation="identity",
        arguments=["7"],
        static=["A=Int"],
        runtime=None,
        trace=False,
        json=True,
    )
    assert run_main(args) == 0
    assert json.loads(capsys.readouterr().out)["value"] == 7

    session = ReplSession()
    assert session.load_file(path).ok
    response = session.dispatch(":run identity --static A=Int 7")
    assert response.ok
    assert json.loads(response.body)["value"] == 7
    assert session.last_run is not None
    assert session.runtime_label == "core"
    assert session.dispatch(":detach").ok
    assert session.runtime_label == "detached"
    detached = session.dispatch(":run identity --static A=Int 7")
    assert not detached.ok
    assert detached.diagnostics[0].code == "qiec-run-validator"
    assert session.dispatch(":runtime core").ok


def test_lsp_resolves_qiec_parameters_and_let_binders() -> None:
    source = (
        "effect Reader\n"
        "    get : Unit -> Int\n\n"
        "instance reader : Reader\n\n"
        "define keep(x : Int) : Int !{reader} =\n"
        "    let y <- perform reader.get()\n"
        "    return y\n"
    )
    document = DocumentState(uri="file:///tmp/locals.qvr")
    document.update(source=source, version=1)
    parameter = document.find_decl("x", line=6, col=20)
    local = document.find_decl("y", line=7, col=12)
    assert getattr(parameter, "name", None) == "x"
    assert getattr(parameter, "line", 0) == 6
    assert getattr(local, "name", None) == "y"
    assert getattr(local, "line", 0) == 7
    hover = _render_hover(document, "y", line=7, col=12)
    assert hover is not None and "inferred by QIEC" in hover


def test_lsp_reports_selected_target_qiec_capabilities() -> None:
    document = DocumentState(uri="file:///tmp/target.qvr", target="stan")
    document.update(source=SCOPED_OPERATIONS, version=1)
    capability = [
        diagnostic
        for diagnostic in document.diagnostics
        if diagnostic.code.startswith("qiec:capability:")
    ]
    assert capability
    assert {diagnostic.severity for diagnostic in capability} == {"error"}
    assert any(
        "Stan" in diagnostic.message or "stan" in diagnostic.message
        for diagnostic in capability
    )


def test_effectful_execution_crosses_cli_repl_and_tui_boundaries(
    tmp_path: Path, capsys
) -> None:
    source = (
        "effect Echo\n"
        "    ping : Int -> Int\n\n"
        "instance echo : Echo\n\n"
        "handler pass for Echo : Int -> Int [coverage=total, implementation=foreign]\n"
        "    ping resumes 1\n\n"
        "define effectful(x : Int) : Int !{} =\n"
        "    handle echo with pass in\n"
        "        let y <- perform echo.ping(x)\n"
        "        return y\n"
    )
    path = tmp_path / "effectful.qvr"
    path.write_text(source)
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "providers": [
                    {
                        "name": "core",
                        "options": {"handlers": {"pass": {"kind": "passthrough"}}},
                    }
                ]
            }
        )
    )
    args = Namespace(
        file=str(path),
        computation="effectful",
        arguments=["9"],
        static=[],
        runtime=str(runtime_path),
        trace=False,
        json=True,
    )
    assert run_main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["value"] == 9
    assert "operation.handled" in {event["event"] for event in payload["trace"]}

    session = ReplSession()
    assert session.load_file(path).ok
    assert session.dispatch(f":runtime {runtime_path}").ok
    assert session.dispatch(":run effectful 9").ok
    from quivers.cli.repl_tui import _runtime_status

    assert _runtime_status(session) == "runtime:core last:effectful=9:Int"


def test_qvr_run_and_repl_honor_a_fuel_budget(tmp_path: Path, capsys) -> None:
    """A diverging computation stops at its budget on both surfaces."""
    path = tmp_path / "spin.qvr"
    path.write_text("define spin(seed : Int) : Int !{} =\n    spin(seed)\n")
    args = Namespace(
        file=str(path),
        computation="spin",
        arguments=["1"],
        static=[],
        runtime=None,
        trace=False,
        json=True,
        fuel=200,
    )
    assert run_main(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["diagnostics"][0]["code"] == "qiec-run-fuel"

    session = ReplSession()
    assert session.load_file(path).ok
    response = session.dispatch(":run spin 1 --fuel 200")
    assert not response.ok
    assert response.diagnostics[0].code == "qiec-run-fuel"
    rejected = session.dispatch(":run spin 1 --fuel 0")
    assert not rejected.ok
    assert rejected.diagnostics[0].code == "qiec-run-config"
