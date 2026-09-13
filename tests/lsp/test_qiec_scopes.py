"""Position-aware QIEC navigation, completion, and target diagnostics."""

from __future__ import annotations

from lsprotocol import types as lsp

from quivers.lsp.document import DocumentState
from quivers.lsp.server import (
    _find_references,
    _qiec_local_completion_items,
    build_server,
)


SOURCE = """\
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

effect Reader[T : Type] [version=1, evolution=sealed]
    get[X : Type] : X -> T

effect Writer[T : Type] [version=1, evolution=sealed]
    get[X : Type] : X -> Unit

instance reader : Reader[Int]
instance writer : Writer[Int]

define head[A : Type, n : Nat](xs : Vec[A](S(n))) : A !{} =
    case xs motive (k : Nat) => A
        Cons[tail_index](head, tail : Vec[A](tail_index)) =>
            return head

define echo[A : Type](head : A) : A !{} =
    return head

define read() : Int !{reader} =
    let value <- perform reader.get[Int](0)
    return value

define write() : Unit !{writer} =
    perform writer.get[Int](0)
    return unit
"""


def _doc(*, target: str | None = None) -> DocumentState:
    document = DocumentState(uri="file:///tmp/qiec-scopes.qvr", target=target)
    document.update(source=SOURCE, version=1)
    assert not [
        diagnostic
        for diagnostic in document.diagnostics
        if not diagnostic.code.startswith("qiec:capability:")
    ]
    return document


def _reference_positions(
    document: DocumentState, name: str, *, line: int, col: int
) -> set[tuple[int, int]]:
    return {
        (location.range.start.line, location.range.start.character)
        for location in _find_references(
            document,
            name,
            line=line,
            col=col,
        )
    }


def test_static_telescope_and_case_binders_resolve_by_position() -> None:
    document = _doc()

    family_parameter = document.find_decl("A", line=4, col=35)
    constructor_static = document.find_decl("m", line=4, col=48)
    effect_parameter = document.find_decl("T", line=7, col=30)
    operation_static = document.find_decl("X", line=7, col=26)
    computation_static = document.find_decl("A", line=16, col=35)
    motive_index = document.find_decl("k", line=16, col=36)
    branch_static = document.find_decl("tail_index", line=17, col=48)
    branch_field = document.find_decl("head", line=18, col=19)

    assert (type(family_parameter).__name__, getattr(family_parameter, "line", 0)) == (
        "QiecTypeBinder",
        3,
    )
    assert (
        type(constructor_static).__name__,
        getattr(constructor_static, "line", 0),
    ) == (
        "QiecIndexBinder",
        5,
    )
    assert (type(effect_parameter).__name__, getattr(effect_parameter, "line", 0)) == (
        "QiecTypeBinder",
        7,
    )
    assert (type(operation_static).__name__, getattr(operation_static, "line", 0)) == (
        "QiecTypeBinder",
        8,
    )
    assert (
        type(computation_static).__name__,
        getattr(computation_static, "line", 0),
    ) == (
        "QiecTypeBinder",
        16,
    )
    assert (type(motive_index).__name__, getattr(motive_index, "line", 0)) == (
        "QiecIndexBinder",
        17,
    )
    assert (type(branch_static).__name__, getattr(branch_static, "line", 0)) == (
        "QiecTypeName",
        18,
    )
    assert (type(branch_field).__name__, getattr(branch_field, "line", 0)) == (
        "QiecLocalBinding",
        18,
    )


def test_local_completion_contains_only_bindings_visible_at_cursor() -> None:
    document = _doc()
    motive = {
        item.label
        for item in _qiec_local_completion_items(document, "", line=16, col=38)
    }
    branch_before_static = {
        item.label
        for item in _qiec_local_completion_items(document, "", line=17, col=11)
    }
    branch_body = {
        item.label
        for item in _qiec_local_completion_items(document, "", line=18, col=23)
    }

    assert {"A", "n", "xs", "k"} <= motive
    assert "tail_index" not in branch_before_static
    assert {"A", "n", "xs", "tail_index", "head", "tail"} <= branch_body
    assert "k" not in branch_body


def test_lsp_completion_feature_exposes_position_visible_locals() -> None:
    server = build_server()
    server.text_document_publish_diagnostics = lambda _params: None  # type: ignore[method-assign]
    uri = "file:///tmp/qiec-completion.qvr"
    server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_OPEN](
        lsp.DidOpenTextDocumentParams(
            text_document=lsp.TextDocumentItem(
                uri=uri,
                language_id="qvr",
                version=1,
                text=SOURCE,
            )
        )
    )

    completion = server.protocol.fm.features[lsp.TEXT_DOCUMENT_COMPLETION](
        lsp.CompletionParams(
            text_document=lsp.TextDocumentIdentifier(uri=uri),
            position=lsp.Position(line=18, character=12),
        )
    )
    items = {item.label: item for item in completion.items}

    assert {"A", "n", "xs", "tail_index", "head", "tail"} <= items.keys()
    assert items["tail_index"].kind == lsp.CompletionItemKind.TypeParameter
    assert items["head"].kind == lsp.CompletionItemKind.Variable
    assert "k" not in items


def test_references_preserve_static_local_and_operation_identity() -> None:
    document = _doc()

    family_a = _reference_positions(document, "A", line=4, col=35)
    computation_a = _reference_positions(document, "A", line=16, col=35)
    branch_head = _reference_positions(document, "head", line=18, col=19)
    echo_head = _reference_positions(document, "head", line=21, col=11)
    reader_get = _reference_positions(document, "get", line=7, col=4)
    writer_get = _reference_positions(document, "get", line=10, col=4)

    assert family_a and computation_a and family_a.isdisjoint(computation_a)
    assert branch_head == {(17, 25), (18, 19)}
    assert echo_head == {(20, 22), (21, 11)}
    assert len(reader_get) == 2
    assert len(writer_get) == 2
    assert reader_get.isdisjoint(writer_get)
    assert (7, 4) in reader_get
    assert (10, 4) in writer_get


def test_server_publishes_and_dynamically_refreshes_target_diagnostics() -> None:
    server = build_server()
    published: list[lsp.PublishDiagnosticsParams] = []
    server.text_document_publish_diagnostics = published.append  # type: ignore[method-assign]

    did_open = server.protocol.fm.features[lsp.TEXT_DOCUMENT_DID_OPEN]
    did_open(
        lsp.DidOpenTextDocumentParams(
            text_document=lsp.TextDocumentItem(
                uri="file:///tmp/qiec-scopes.qvr",
                language_id="qvr",
                version=1,
                text=SOURCE,
            )
        )
    )
    assert published[-1].diagnostics == []

    change_configuration = server.protocol.fm.features[
        lsp.WORKSPACE_DID_CHANGE_CONFIGURATION
    ]
    change_configuration(
        lsp.DidChangeConfigurationParams(settings={"qvr": {"transpileTarget": "stan"}})
    )
    capability = [
        diagnostic
        for diagnostic in published[-1].diagnostics
        if str(diagnostic.code).startswith("qiec:capability:")
    ]
    assert capability

    change_configuration(
        lsp.DidChangeConfigurationParams(settings={"qvr": {"transpileTarget": ""}})
    )
    assert not [
        diagnostic
        for diagnostic in published[-1].diagnostics
        if str(diagnostic.code).startswith("qiec:capability:")
    ]
