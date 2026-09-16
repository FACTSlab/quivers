"""Regression gates for distributable QVR editor support."""

from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]
VSCODE = ROOT / "editors" / "vscode-qvr"
ZED = ROOT / "editors" / "zed-extension-qvr"


def _repository() -> dict[str, object]:
    grammar = json.loads((VSCODE / "syntaxes/qvr.tmLanguage.json").read_text())
    return grammar["repository"]


def test_textmate_recognizes_the_complete_qiec_vocabulary() -> None:
    repository = _repository()
    source = (
        "index Nat = Z | S(Nat)\n"
        "family Vec[A : Type](n : Nat) : Type\n"
        "effect State[S : Type]\n"
        "instance cell : State[Int]\n"
        "handler run for State[Int] : Int -> Int [coverage=total, implementation=foreign]\n"
        "define read() : Int !{cell} = handle cell with run in\n"
        "    let x <- perform cell.get()\n"
        "    return x\n"
    )
    declaration = re.compile(repository["keyword-declaration"]["match"])
    control = re.compile(repository["keyword-control"]["match"])
    modifier = re.compile(repository["keyword-qiec-modifier"]["match"])
    assert {
        "index",
        "family",
        "effect",
        "instance",
        "handler",
        "define",
    } <= set(declaration.findall(source))
    assert {"handle", "with", "let", "perform", "return"} <= set(
        control.findall(source)
    )
    assert {"Type", "coverage", "total"} <= set(modifier.findall(source))


def test_zed_uses_an_immutable_grammar_revision() -> None:
    manifest = tomllib.loads((ZED / "extension.toml").read_text())
    revision = manifest["grammars"]["qvr"]["commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", revision)


def test_vscode_archive_matches_release_sources() -> None:
    from tools.check_editor_release import main

    assert main() == 0


def test_vscode_forwards_live_target_configuration_to_the_lsp() -> None:
    source = (VSCODE / "src" / "extension.ts").read_text()
    compiled = (VSCODE / "out" / "extension.js").read_text()
    assert 'configurationSection: "qvr"' in source
    assert 'configurationSection: "qvr"' in compiled


def test_textual_input_registers_the_wheel_qvr_grammar() -> None:
    from textual.widgets import TextArea

    from quivers.cli.repl_tui import _enable_qvr_highlighting

    area = TextArea()
    assert _enable_qvr_highlighting(area)
    assert area.language == "qvr"


# ---------------------------------------------------------------------------
# One token corpus, classified alike by every highlighter.
# ---------------------------------------------------------------------------


_CORPUS = (
    "effect Echo\n"
    "    ping : Int -> Int\n\n"
    "define twice(x : Int) : Int !{} =\n"
    "    let y <- twice(x)\n"
    "    return y\n\n"
    "handler doubling for Echo : Int -> Int [coverage=total, implementation=authored]\n"
    "    ping(n : Int) resumes 1 =>\n"
    "        resume(n)\n"
)

#: Each token of interest by position, with the class every highlighter
#: must give it: (line, text) -> class.
_EXPECTED: dict[tuple[int, str], str] = {
    (0, "effect"): "keyword",
    (1, "ping"): "function",
    (3, "twice"): "function",
    (4, "twice"): "function",
    (7, "handler"): "keyword",
    (7, "[coverage"): "keyword",
    (7, "total"): "keyword",
    (7, "implementation"): "keyword",
    (8, "ping"): "function",
    (8, "resumes"): "keyword",
    (8, "1"): "number",
    (9, "resume"): "keyword",
    (1, "Int"): "type",
}

#: The tree-sitter capture names and Pygments token families that stand
#: for each class.
_TREE_SITTER_CLASSES = {
    "keyword": {"keyword"},
    "function": {"function", "function.call", "function.method"},
    "number": {"constant.builtin", "number"},
    "type": {"type", "type.builtin"},
}


def _repl_classes() -> dict[tuple[int, str], str]:
    from quivers.cli.repl_highlight import tokenize

    return {
        (span.line, span.text): span.token
        for span in tokenize(_CORPUS)
        if (span.line, span.text) in _EXPECTED
    }


def _pygments_classes() -> dict[tuple[int, str], str]:
    from pygments.token import Keyword, Name, Number

    from quivers.dsl.pygments_lexer import QvrLexer

    families = (
        (Keyword, "keyword"),
        (Number, "number"),
        (Name.Class, "type"),
        (Name.Function, "function"),
    )
    out: dict[tuple[int, str], str] = {}
    for index, token, text in QvrLexer().get_tokens_unprocessed(_CORPUS):
        line = _CORPUS.count("\n", 0, index)
        if (line, text) not in _EXPECTED:
            continue
        for family, name in families:
            if token in family:
                out[(line, text)] = name
                break
    return out


def _tree_sitter_classes() -> dict[tuple[int, str], str]:
    import tree_sitter

    from quivers.dsl.pygments_lexer import _load_parser

    parser, language, _ = _load_parser()
    source = _CORPUS.encode()
    tree = parser.parse(source)
    query = tree_sitter.Query(
        language, (ROOT / "grammars/qvr/queries/highlights.scm").read_text()
    )
    captures = tree_sitter.QueryCursor(query).captures(tree.root_node)
    out: dict[tuple[int, str], str] = {}
    for capture, nodes in captures.items():
        for node in nodes:
            key = (
                node.start_point.row,
                source[node.start_byte : node.end_byte].decode(),
            )
            if key not in _EXPECTED:
                continue
            for name, names in _TREE_SITTER_CLASSES.items():
                if capture in names:
                    out[key] = name
    return out


def _textmate_classes() -> dict[tuple[int, str], str]:
    repository = _repository()
    rules = {
        "keyword": ("keyword-control", "keyword-declaration", "keyword-qiec-modifier"),
        "function": ("builtin-function",),
        "type": ("builtin-type",),
    }
    out: dict[tuple[int, str], str] = {}
    for (line, text), expected in _EXPECTED.items():
        word = text.lstrip("[")
        for name, rule_names in rules.items():
            if any(
                re.fullmatch(repository[rule]["match"].replace("\\b", ""), f"({word})")
                or re.search(repository[rule]["match"], word)
                for rule in rule_names
            ):
                out[(line, text)] = name
                break
    return out


def test_highlighters_classify_the_corpus_alike() -> None:
    """The REPL tokenizer, the Pygments lexer, and the tree-sitter query
    give every token of the corpus the same class, and the TextMate
    grammar agrees on every keyword and builtin it names."""
    assert _repl_classes() == _EXPECTED
    assert _pygments_classes() == _EXPECTED
    assert _tree_sitter_classes() == _EXPECTED
    textmate = _textmate_classes()
    for key, expected in _EXPECTED.items():
        if key in textmate:
            assert textmate[key] == expected, key
    assert {key for key, value in _EXPECTED.items() if value == "keyword"} <= set(
        textmate
    )
