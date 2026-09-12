"""Generated-asset gates for the QVR v0.19 QIEC grammar."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from quivers.cli.migrations import _manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
GRAMMAR_ROOT = REPO_ROOT / "grammars/qvr"


def test_packaged_grammar_json_matches_generated_source() -> None:
    generated = GRAMMAR_ROOT / "src/grammar.json"
    packaged = REPO_ROOT / "src/quivers/dsl/_grammar_data/grammar.json"
    assert packaged.read_bytes() == generated.read_bytes()


def test_highlight_query_is_generated_and_zed_uses_the_same_bytes() -> None:
    generator_path = GRAMMAR_ROOT / "queries/_generate.py"
    spec = importlib.util.spec_from_file_location(
        "qvr_highlight_generator", generator_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    canonical = GRAMMAR_ROOT / "queries/highlights.scm"
    zed = REPO_ROOT / "editors/zed-extension-qvr/languages/qvr/highlights.scm"
    expected = module.render()
    assert canonical.read_text(encoding="utf-8") == expected
    assert zed.read_text(encoding="utf-8") == expected
    assert "(indexed_family_decl name:" in expected
    assert "(effect_decl name:" in expected
    assert "(handler_decl name:" in expected


def test_head_snapshot_is_current_and_matches_manifest() -> None:
    generated = GRAMMAR_ROOT / "src"
    snapshot = GRAMMAR_ROOT / "vcs/parsers/HEAD/source/src"
    for relative in (
        "grammar.json",
        "node-types.json",
        "parser.c",
        "scanner.c",
        "tree_sitter/alloc.h",
        "tree_sitter/array.h",
        "tree_sitter/parser.h",
    ):
        assert (snapshot / relative).read_bytes() == (generated / relative).read_bytes()
    _manifest.verify_snapshot("HEAD", snapshot)
