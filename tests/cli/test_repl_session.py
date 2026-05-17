"""ReplSession unit tests.

Drive the session directly so the meta-command surface is exercised
without spinning up a TUI or LSP server. Loading from a tmp .qvr file
covers :load / :reload; the bare statement path covers the
appended-statement evaluator; :type / :kind / :browse / :info /
:dump / :set / :help round out the surface.
"""

from __future__ import annotations

from pathlib import Path

from quivers.cli.repl_session import ReplSession


SOURCE = """\
object X : 3
object Y : 4
latent f : X -> Y
"""


def _populated() -> ReplSession:
    s = ReplSession()
    # Bare statement path; the session compiles incrementally.
    response = s.dispatch(SOURCE)
    assert response.ok, response.diagnostics
    return s


def test_help_lists_commands() -> None:
    s = ReplSession()
    r = s.dispatch(":help")
    assert ":load" in r.body
    assert ":type" in r.body
    assert r.ok


def test_unknown_command() -> None:
    s = ReplSession()
    r = s.dispatch(":nope")
    assert not r.ok
    assert "unknown command" in r.diagnostics[0].message


def test_type_of_object() -> None:
    s = _populated()
    r = s.dispatch(":type X")
    assert r.ok, r.diagnostics
    # :type renders the binding as its source-form QVR declaration so
    # the grammar can classify identifiers in type position.
    assert r.body.startswith("object X")


def test_type_of_morphism() -> None:
    s = _populated()
    r = s.dispatch(":type f")
    assert r.ok, r.diagnostics
    assert " -> " in r.body
    # Morphism signatures lead with the declaration kind keyword so
    # the QVR grammar can colour the domain/codomain as types.
    assert r.body.split()[0] in {"latent", "observed", "program", "kernel"}


def test_kind_reports_ast_variant() -> None:
    s = _populated()
    r = s.dispatch(":kind X")
    assert r.ok, r.diagnostics
    assert "TypeName" in r.body
    assert "TypeProduct" in r.body  # variant enumeration


def test_browse_lists_namespaces() -> None:
    s = _populated()
    r = s.dispatch(":browse")
    assert "objects:" in r.body
    assert "X" in r.body and "Y" in r.body
    assert "morphisms:" in r.body
    assert "f" in r.body


def test_browse_filters_namespace() -> None:
    s = _populated()
    r = s.dispatch(":browse objects")
    assert "objects:" in r.body
    assert "morphisms" not in r.body


def test_info_includes_signature() -> None:
    s = _populated()
    r = s.dispatch(":info f")
    assert r.ok
    assert "latent f" in r.body
    assert "X -> Y" in r.body


def test_dump_renders_ast() -> None:
    s = _populated()
    r = s.dispatch(":dump f")
    assert "MorphismDecl" in r.body
    assert "name='f'" in r.body


def test_dump_json() -> None:
    s = _populated()
    r = s.dispatch(":dump f --json")
    assert r.body.lstrip().startswith("{")
    assert "morphism_decl" in r.body


def test_set_toggles_option() -> None:
    s = ReplSession()
    r = s.dispatch(":set highlight=false")
    assert r.ok
    assert s.options.highlight is False
    r = s.dispatch(":set highlight=on")
    assert s.options.highlight is True


def test_set_rejects_unknown() -> None:
    s = ReplSession()
    r = s.dispatch(":set bogus=42")
    assert not r.ok


def test_load_and_reload(tmp_path: Path) -> None:
    s = ReplSession()
    path = tmp_path / "demo.qvr"
    path.write_text(SOURCE)
    r = s.dispatch(f":load {path}")
    assert r.ok, r.diagnostics
    assert "loaded" in r.body
    assert "X" in s.env and "Y" in s.env
    # Mutate the file: drop Y and reload, expect a removal diff.
    path.write_text("object X : 3\nlatent g : X -> X\n")
    r2 = s.dispatch(":reload")
    assert r2.ok, r2.diagnostics
    assert "removed:" in r2.body and "Y" in r2.body
    assert "added:" in r2.body and "g" in r2.body


def test_load_missing_file() -> None:
    s = ReplSession()
    r = s.dispatch(":load /nonexistent/path.qvr")
    assert not r.ok


def test_parse_error_yields_diagnostic() -> None:
    s = ReplSession()
    r = s.dispatch("@@@ totally bogus @@@")
    assert not r.ok
    # The session must not crash; the error must reach diagnostics.
    assert any(d.severity == "error" for d in r.diagnostics)


def test_quit_returns_sentinel() -> None:
    s = ReplSession()
    r = s.dispatch(":quit")
    assert r.body == "__quit__"
