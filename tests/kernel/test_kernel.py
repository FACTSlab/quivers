"""Jupyter kernel unit tests.

The kernel is a thin adapter over :class:`ReplSession`; the bulk of
the behaviour is covered by ``tests/cli/test_repl_session.py``. Here
we exercise the kernel-side glue: cell splitting, prefix extraction
for completion, word extraction for inspect, and the do_execute /
do_complete / do_inspect surface against a directly-instantiated
``QuiversKernel`` (no jupyter_client round-trip required).

Skipped when ``ipykernel`` is unavailable.
"""

from __future__ import annotations

import pytest


pytest.importorskip("ipykernel")


from quivers.kernel.quivers_kernel import (  # noqa: E402
    _prefix_at,
    _split_cell,
    _word_at,
)


SOURCE = """\
object X : 3
object Y : 4
latent f : X -> Y
"""


def _kernel():  # type: ignore[no-untyped-def]
    """Build a kernel without going through IPKernelApp.

    The Kernel base class is happy with no kwargs at construction time
    if we never start its channels; for the do_* method tests that's
    sufficient.
    """
    from quivers.kernel.quivers_kernel import QuiversKernel

    # Disable Kernel's normal session-creation by passing dummy
    # iopub_socket / shell_socket attributes after construction.
    k = QuiversKernel()
    # Capture iopub messages by monkey-patching send_response.
    k._captured = []  # type: ignore[attr-defined]
    k.send_response = lambda socket, msg_type, content: k._captured.append(  # type: ignore[assignment]
        (msg_type, content)
    )
    k.iopub_socket = None
    return k


# ----- splitter -----------------------------------------------------


def test_split_cell_groups_blank_separated_blocks() -> None:
    cell = "object X : 3\nobject Y : 4\n\nlatent f : X -> Y\n"
    out = _split_cell(cell)
    assert "object X : 3\nobject Y : 4" in out
    assert "latent f : X -> Y" in out


def test_split_cell_meta_lines_isolated() -> None:
    cell = ":load file.qvr\n:type X\n"
    out = _split_cell(cell)
    assert ":load file.qvr" in out
    assert ":type X" in out
    # Each meta is its own chunk so the dispatcher sees them
    # individually.
    assert len(out) == 2


def test_split_cell_empty_input() -> None:
    assert _split_cell("") == []
    assert _split_cell("\n\n") == []


def test_split_cell_preserves_multi_line_block() -> None:
    cell = "program p : X -> Y\n    z <- f\n    return z\n"
    out = _split_cell(cell)
    assert out == ["program p : X -> Y\n    z <- f\n    return z"]


# ----- helpers ------------------------------------------------------


def test_prefix_at_walks_back_through_identifier() -> None:
    assert _prefix_at(":lo", 3) == ":lo"
    assert _prefix_at("hello world", 11) == "world"
    assert _prefix_at(":info Source", 12) == "Source"


def test_word_at_extends_left_and_right() -> None:
    assert _word_at(":info Source", 8) == "Source"
    assert _word_at("foo bar baz", 0) == "foo"


# ----- do_execute ---------------------------------------------------


def test_do_execute_runs_meta_and_statement() -> None:
    k = _kernel()
    out = k.do_execute(SOURCE, silent=False)
    assert out["status"] == "ok"
    # Subsequent :type query against the freshly-loaded env.
    k._captured.clear()  # type: ignore[attr-defined]
    out2 = k.do_execute(":type f", silent=False)
    assert out2["status"] == "ok"
    streams = [c for kind, c in k._captured if kind == "stream"]  # type: ignore[attr-defined]
    body = "".join(c["text"] for c in streams)
    assert "latent f" in body or " -> " in body


def test_do_execute_silent_suppresses_stream() -> None:
    k = _kernel()
    k.do_execute(SOURCE, silent=True)
    streams = [c for kind, c in k._captured if kind == "stream"]  # type: ignore[attr-defined]
    assert streams == []


def test_do_execute_error_status() -> None:
    k = _kernel()
    out = k.do_execute("@@@ definitely not qvr @@@", silent=False)
    assert out["status"] == "error"


# ----- do_complete --------------------------------------------------


def test_do_complete_meta_prefix() -> None:
    k = _kernel()
    out = k.do_complete(":lo", 3)
    assert out["status"] == "ok"
    assert ":load" in out["matches"]
    assert out["cursor_start"] == 0
    assert out["cursor_end"] == 3


def test_do_complete_env_after_load() -> None:
    k = _kernel()
    k.do_execute(SOURCE, silent=True)
    out = k.do_complete("f", 1)
    assert "f" in out["matches"]


# ----- do_inspect ---------------------------------------------------


def test_do_inspect_returns_decl_info() -> None:
    k = _kernel()
    k.do_execute(SOURCE, silent=True)
    out = k.do_inspect("f", 1)
    assert out["status"] == "ok"
    assert out["found"] is True
    rendered = out["data"]["text/plain"]
    assert "latent f : X -> Y" in rendered


def test_do_inspect_unknown_name_not_found() -> None:
    k = _kernel()
    k.do_execute(SOURCE, silent=True)
    out = k.do_inspect("nope_unknown", 12)
    assert out["found"] is False
