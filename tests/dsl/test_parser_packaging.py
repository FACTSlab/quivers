"""Native parser wheel integrity and fail-closed selection tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from quivers.cli.migrations import _grammar as migration_grammar
from quivers.dsl import _grammar_build as grammar_build


def test_verified_bundled_parser_wins_before_source_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library = tmp_path / f"qvr_grammar{grammar_build._shared_lib_extension()}"
    library.write_bytes(b"native")
    monkeypatch.setattr(grammar_build, "_bundled_shared_lib", lambda: library)
    monkeypatch.setattr(
        grammar_build,
        "_build_shared_lib",
        lambda _grammar: pytest.fail("source compiler fallback was reached"),
    )
    grammar_dir, selected = grammar_build._parser_artifacts()
    assert grammar_dir == grammar_build._packaged_grammar_dir()
    assert selected == library


def test_incomplete_installed_package_does_not_compile_on_first_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packaged = grammar_build._packaged_grammar_dir()
    monkeypatch.setattr(grammar_build, "_bundled_shared_lib", lambda: None)
    monkeypatch.setattr(grammar_build, "_grammar_dir", lambda: packaged)
    monkeypatch.setattr(
        grammar_build,
        "_build_shared_lib",
        lambda _grammar: pytest.fail("installed-package compiler fallback was reached"),
    )
    with pytest.raises(FileNotFoundError, match="matching platform wheel"):
        grammar_build._parser_artifacts()


def test_native_manifest_binds_library_to_generated_grammar(tmp_path: Path) -> None:
    source_grammar = Path(__file__).resolve().parents[2] / "grammars" / "qvr"
    grammar = tmp_path / "qvr"
    shutil.copytree(source_grammar / "src", grammar / "src")
    library = tmp_path / f"qvr_grammar{grammar_build._shared_lib_extension()}"
    library.write_bytes(b"native parser bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(grammar_build._native_manifest_data(grammar, library)),
        encoding="utf-8",
    )

    grammar_build._validate_native_manifest(grammar, library, manifest)
    library.write_bytes(b"changed native parser bytes")
    with pytest.raises(ValueError, match="library_sha256"):
        grammar_build._validate_native_manifest(grammar, library, manifest)
    library.write_bytes(b"native parser bytes")
    with (grammar / "src" / "parser.c").open("ab") as stream:
        stream.write(b"\n/* changed */\n")
    with pytest.raises(ValueError, match="grammar_sha256"):
        grammar_build._validate_native_manifest(grammar, library, manifest)


def test_native_manifest_binds_parser_metadata(tmp_path: Path) -> None:
    source_grammar = Path(__file__).resolve().parents[2] / "grammars" / "qvr"
    grammar = tmp_path / "qvr"
    shutil.copytree(source_grammar / "src", grammar / "src")
    library = tmp_path / f"qvr_grammar{grammar_build._shared_lib_extension()}"
    library.write_bytes(b"native parser bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(grammar_build._native_manifest_data(grammar, library)),
        encoding="utf-8",
    )

    with (grammar / "src" / "node-types.json").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="grammar_sha256"):
        grammar_build._validate_native_manifest(grammar, library, manifest)


def test_verified_migration_parser_wins_before_source_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_grammar = Path(__file__).resolve().parents[2] / "grammars" / "qvr"
    snapshot = tmp_path / "v0.2.0"
    shutil.copytree(source_grammar / "src", snapshot / "source" / "src")
    library = snapshot / f"qvr{grammar_build._shared_lib_extension()}"
    library.write_bytes(b"native parser bytes")
    manifest = snapshot / "native-manifest.json"
    manifest.write_text(
        json.dumps(grammar_build._native_manifest_data(snapshot / "source", library)),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        migration_grammar,
        "_compile_parser",
        lambda _snapshot, _revision: pytest.fail(
            "migration source compiler fallback was reached"
        ),
    )

    assert migration_grammar._library_for(snapshot, "v0.2.0") == library


def test_incomplete_installed_migration_snapshot_does_not_compile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = tmp_path / "v0.2.0"
    (snapshot / "source" / "src").mkdir(parents=True)
    monkeypatch.setattr(migration_grammar, "is_packaged_vcs_path", lambda _path: True)
    monkeypatch.setattr(
        migration_grammar,
        "_compile_parser",
        lambda _snapshot, _revision: pytest.fail(
            "installed migration source compiler fallback was reached"
        ),
    )

    with pytest.raises(
        migration_grammar.HistoricalGrammarUnavailable,
        match="matching platform wheel",
    ):
        migration_grammar._library_for(snapshot, "v0.2.0")
