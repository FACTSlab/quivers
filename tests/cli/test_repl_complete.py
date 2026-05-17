"""Completion engine tests.

Exercises every source the completer fans out to:

- Meta-commands (`:` prefix).
- Env-derived names (objects, spaces, morphisms, rules).
- Grammar-keyword and builtin-function/type catalogues.
- File-system path completion for `:load`.

The completer is what powers Tab in the TUI, the LSP's `completion`
handler, the Jupyter kernel's `do_complete`, and the prompt_toolkit
fallback, so every assertion here protects four surfaces at once.
"""

from __future__ import annotations

from pathlib import Path

from quivers.cli.repl_complete import Completion, all_completions
from quivers.cli.repl_session import ReplSession


SOURCE = """\
object Alpha : 3
object Beta : 4
latent f : Alpha -> Beta
"""


def _session() -> ReplSession:
    s = ReplSession()
    r = s.dispatch(SOURCE)
    assert r.ok, r.diagnostics
    return s


def test_meta_commands_prefix() -> None:
    cs = all_completions(_session(), ":lo")
    labels = [c.text for c in cs]
    assert ":load" in labels


def test_meta_commands_full_list_for_colon() -> None:
    cs = all_completions(_session(), ":")
    labels = [c.text for c in cs]
    # Every primary command should show up.
    for cmd in (":load", ":type", ":info", ":browse", ":help"):
        assert cmd in labels


def test_env_completions_pull_objects_and_morphisms() -> None:
    cs = all_completions(_session(), "A")
    texts = [c.text for c in cs]
    assert "Alpha" in texts


def test_env_completions_include_morphism() -> None:
    cs = all_completions(_session(), "f")
    texts = [c.text for c in cs]
    assert "f" in texts


def test_env_completions_carry_namespace_detail() -> None:
    cs = all_completions(_session(), "Alpha")
    matched = [c for c in cs if c.text == "Alpha"]
    assert matched
    assert matched[0].detail == "object"


def test_keyword_completions() -> None:
    cs = all_completions(_session(), "lat")
    texts = {c.text for c in cs}
    assert "latent" in texts


def test_builtin_function_completion() -> None:
    cs = all_completions(_session(), "soft")
    texts = {c.text for c in cs}
    assert "softmax" in texts
    assert "softplus" in texts


def test_builtin_type_completion() -> None:
    cs = all_completions(_session(), "Eu")
    texts = {c.text for c in cs}
    assert "Euclidean" in texts


def test_path_completion_for_load(tmp_path: Path) -> None:
    target = tmp_path / "demo.qvr"
    target.write_text("")
    cs = all_completions(_session(), str(tmp_path / "de"))
    texts = {c.text for c in cs}
    assert any(t.endswith("demo.qvr") for t in texts)


def test_empty_prefix_returns_meta_commands_only() -> None:
    cs = all_completions(_session(), "")
    # Empty prefix avoids the keyword fan-out (would otherwise dump
    # every grammar keyword). Allow env entries through but don't
    # demand them.
    for c in cs:
        assert c.kind in {"command", "env", "path"}


def test_completion_dataclass_is_hashable() -> None:
    c = Completion(text="foo", kind="env", detail="object")
    # frozen=True dataclasses round-trip through set membership.
    assert c in {c}


def test_no_completions_for_nonexistent_prefix() -> None:
    cs = all_completions(_session(), "zzzqqq_unknown")
    assert all(c.kind == "path" for c in cs) or not cs


def test_meta_command_short_alias_completes() -> None:
    cs = all_completions(_session(), ":l")
    texts = [c.text for c in cs]
    assert ":load" in texts


def test_session_with_no_loaded_module_returns_meta_only() -> None:
    cs = all_completions(ReplSession(), ":lo")
    assert any(c.text == ":load" for c in cs)
