"""Compile every ``qvr`` code block in the docs and run every
``python`` code block under ``docs/tutorials/``.

QVR blocks
----------

Each fenced ```qvr block under ``docs/`` (and ``README.md``) is treated
as a test case. The block's *compile mode* is chosen by an HTML comment
on the line immediately preceding the opening fence:

* ``<!-- compile: false -->``      — skip; the block is illustrative
  prose (e.g. a bind step shown outside any program body).
* ``<!-- compile: cumulative -->`` — concatenate this block with every
  prior cumulative block in the same file before compiling. Use when a
  guide walks the reader through one model in incrementally-elaborated
  fragments.
* (no marker)                      — ``standalone``: the block must
  compile on its own.

Python blocks
-------------

Each fenced ```python block under ``docs/tutorials/`` is executed in
order, with per-file shared state: every block sees the namespace
built by the previous blocks in the same file (matching the way a
reader copies the chapter into a REPL). The block's mode is chosen by
the same HTML-comment surface, scoped to a separate marker so QVR and
Python markers don't collide:

* ``<!-- python: skip -->`` — skip this block. Use for blocks that
  reference on-disk files (``open("foo.qvr")``), shell commands, or
  illustrative fragments that aren't meant to run.
* (no marker)                — run; failures fail the test.

Run with the rest of the suite::

    QVR_USE_LOCAL_GRAMMAR=1 pytest tests/test_doc_blocks.py

CI gates the doc surface on this file: a release-blocking failure here
means a published example doesn't compile or doesn't run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from quivers.dsl import loads

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC_ROOTS = (_REPO_ROOT / "docs", _REPO_ROOT / "README.md")
_TUTORIAL_ROOT = _REPO_ROOT / "docs" / "tutorials"

_QVR_FENCE_RE = re.compile(r"^([ \t]*)```qvr\n(.*?)^\1```", re.M | re.S)
_PY_FENCE_RE = re.compile(r"^([ \t]*)```python\n(.*?)^\1```", re.M | re.S)


def _dedent_fence_body(indent: str, body: str) -> str:
    """Strip ``indent`` from the start of every line in ``body``.

    Material-for-MkDocs requires fenced blocks inside ``=== "..."`` tabs
    to be indented (typically by four spaces). The fence regex captures
    that indent so we can strip it before compiling.
    """
    if not indent:
        return body
    out_lines: list[str] = []
    for line in body.splitlines(keepends=True):
        if line.startswith(indent):
            out_lines.append(line[len(indent):])
        elif line.strip() == "":
            out_lines.append(line)
        else:
            # A line shorter than ``indent`` that isn't blank: leave it
            # alone so the compiler reports an accurate column.
            out_lines.append(line)
    return "".join(out_lines)
_QVR_MARKER_RE = re.compile(
    r"<!--\s*compile:\s*(false|standalone|cumulative)\s*-->"
)
_PY_MARKER_RE = re.compile(r"<!--\s*python:\s*(skip|run)\s*-->")


def _qvr_marker_before(text: str, fence_start: int) -> str:
    """Return the compile-mode marker on the line preceding ``fence_start``."""
    line_start = text.rfind("\n", 0, fence_start - 1)
    prev_line = text[line_start + 1 : fence_start].strip()
    m = _QVR_MARKER_RE.search(prev_line)
    return m.group(1) if m else "standalone"


def _py_marker_before(text: str, fence_start: int) -> str:
    line_start = text.rfind("\n", 0, fence_start - 1)
    prev_line = text[line_start + 1 : fence_start].strip()
    m = _PY_MARKER_RE.search(prev_line)
    return m.group(1) if m else "run"


def _iter_md_files() -> list[Path]:
    files: list[Path] = []
    for root in _DOC_ROOTS:
        if root.is_dir():
            files.extend(sorted(root.rglob("*.md")))
        elif root.is_file():
            files.append(root)
    return files


def _collect_qvr_blocks() -> list[tuple[str, int, str, str]]:
    out: list[tuple[str, int, str, str]] = []
    for md in _iter_md_files():
        text = md.read_text()
        rel = str(md.relative_to(_REPO_ROOT))
        cumulative_prefix = ""
        for idx, m in enumerate(_QVR_FENCE_RE.finditer(text)):
            indent, body = m.group(1), m.group(2)
            body = _dedent_fence_body(indent, body)
            mode = _qvr_marker_before(text, m.start())
            if mode == "cumulative":
                source = cumulative_prefix + body
                cumulative_prefix = source + "\n"
            else:
                source = body
            out.append((rel, idx, mode, source))
    return out


def _collect_py_blocks() -> list[tuple[str, int, str, str]]:
    """Return ``(rel_path, block_index, mode, source)`` for every
    python block under ``docs/tutorials/``.
    """
    out: list[tuple[str, int, str, str]] = []
    for md in sorted(_TUTORIAL_ROOT.rglob("*.md")):
        text = md.read_text()
        rel = str(md.relative_to(_REPO_ROOT))
        for idx, m in enumerate(_PY_FENCE_RE.finditer(text)):
            indent, body = m.group(1), m.group(2)
            body = _dedent_fence_body(indent, body)
            mode = _py_marker_before(text, m.start())
            out.append((rel, idx, mode, body))
    return out


_QVR_BLOCKS = _collect_qvr_blocks()
_PY_BLOCKS = _collect_py_blocks()


@pytest.mark.parametrize(
    ("path", "index", "mode", "source"),
    _QVR_BLOCKS,
    ids=[f"{p}:blk{i}:{m}" for p, i, m, _ in _QVR_BLOCKS],
)
def test_qvr_doc_block(path: str, index: int, mode: str, source: str) -> None:
    del path, index  # carried only for readable test ids
    if mode == "false":
        pytest.skip("block marked compile: false (illustrative fragment)")
    loads(source)


# Per-file namespaces accumulated as python blocks execute in file order.
_PY_NAMESPACES: dict[str, dict] = {}


@pytest.mark.parametrize(
    ("path", "index", "mode", "source"),
    _PY_BLOCKS,
    ids=[f"{p}:pyblk{i}:{m}" for p, i, m, _ in _PY_BLOCKS],
)
def test_python_doc_block(
    path: str, index: int, mode: str, source: str
) -> None:
    if mode == "skip":
        pytest.skip("block marked python: skip")
    ns = _PY_NAMESPACES.setdefault(path, {"__name__": "__doc_block__"})
    try:
        exec(compile(source, f"{path}:pyblk{index}", "exec"), ns)
    except Exception:
        # Make subsequent blocks in the same file see a clean slate so
        # one failure doesn't cascade.
        _PY_NAMESPACES[path] = {"__name__": "__doc_block__"}
        raise
