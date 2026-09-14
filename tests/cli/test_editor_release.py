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
        "handler run for State[Int] : Int -> Int [coverage=total]\n"
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
