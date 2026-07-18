"""Dispatch-level tests for the REPL ``:transpile`` command.

Drives [`ReplSession.dispatch`][quivers.cli.repl_session.ReplSession.dispatch]
directly so the suite has zero Textual dependency. The structural
correctness of each backend's emitted output is exercised by
[`tests/transpile/test_structural.py`][tests.transpile.test_structural];
this file only verifies the in-session plumbing: error paths, target
dispatch, tab completion.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from quivers.cli.repl_complete import all_completions
from quivers.cli.repl_session import ReplSession
from quivers.transpile import available_targets


_BETA_BERNOULLI_FIXTURE = """\
object Resp : FinSet 1
program flip : Resp -> Resp
    sample theta <- Beta(2.0, 2.0)
    observe y <- Bernoulli(theta)
    return y
"""


@pytest.fixture
def loaded_session() -> ReplSession:
    """A `ReplSession` with the beta-Bernoulli fixture loaded."""
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="qvr-tui-"))
    fixture = tmp_dir / "flip.qvr"
    fixture.write_text(_BETA_BERNOULLI_FIXTURE)
    session = ReplSession()
    response = session.dispatch(f":load {fixture}")
    assert response.ok, response.diagnostics
    return session


def test_transpile_requires_module_loaded() -> None:
    """Calling `:transpile` on a fresh session without `:load` errors."""
    session = ReplSession()
    response = session.dispatch(":transpile stan")
    assert not response.ok
    assert "no environment loaded" in response.diagnostics[0].message


def test_transpile_requires_target(loaded_session: ReplSession) -> None:
    """Calling `:transpile` with no argument prints the usage line."""
    response = loaded_session.dispatch(":transpile")
    assert not response.ok
    assert "usage: :transpile" in response.diagnostics[0].message


def test_transpile_unknown_target_lists_available(
    loaded_session: ReplSession,
) -> None:
    """An unknown target lists the registered backends in the error."""
    response = loaded_session.dispatch(":transpile pumc")  # typo of pymc
    assert not response.ok
    msg = response.diagnostics[0].message
    assert "unknown target 'pumc'" in msg
    for target in available_targets():
        assert target in msg


@pytest.mark.parametrize("target", sorted(available_targets()))
def test_transpile_dispatches_to_every_backend(
    loaded_session: ReplSession, target: str
) -> None:
    """`:transpile <target>` returns non-empty bytes for every registered
    backend. Structural correctness lives in
    [`test_structural.py`][tests.transpile.test_structural].
    """
    response = loaded_session.dispatch(f":transpile {target}")
    assert response.ok, (
        f"{target!r} dispatch failed: "
        + "; ".join(d.message for d in response.diagnostics)
    )
    if target == "church" and not response.body.strip():
        pytest.xfail(
            reason=(
                "panproto/panproto#172: scheme `emit_pretty` returns "
                "empty bytes for every input."
            )
        )
    assert response.body.strip(), f"{target!r} produced empty body"


def test_tab_completion_after_transpile_lists_backends(
    loaded_session: ReplSession,
) -> None:
    """``:transpile <TAB>`` completes against every registered backend."""
    hits = all_completions(loaded_session, ":transpile ")
    texts = {h.text for h in hits}
    for target in available_targets():
        assert f":transpile {target}" in texts


def test_tab_completion_filters_by_prefix(
    loaded_session: ReplSession,
) -> None:
    """``:transpile p<TAB>`` returns only backends starting with ``p``."""
    hits = all_completions(loaded_session, ":transpile p")
    texts = {h.text for h in hits}
    for target in available_targets():
        candidate = f":transpile {target}"
        if target.startswith("p"):
            assert candidate in texts
        else:
            assert candidate not in texts


def test_help_lists_transpile() -> None:
    """``:help`` includes the transpile summary."""
    response = ReplSession().dispatch(":help")
    assert response.ok
    assert "transpile" in response.body


def test_help_transpile_has_detail() -> None:
    """``:help transpile`` returns the detailed entry."""
    response = ReplSession().dispatch(":help transpile")
    assert response.ok
    assert "available_targets" in response.body
