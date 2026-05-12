"""Compile every ``qvr`` code block in the docs.

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

Run with the rest of the suite::

    QVR_USE_LOCAL_GRAMMAR=1 pytest tests/test_doc_blocks.py

CI gates the doc surface on this file: a release-blocking failure here
means a published example doesn't compile.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from quivers.dsl import loads

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC_ROOTS = (_REPO_ROOT / "docs", _REPO_ROOT / "README.md")

_FENCE_RE = re.compile(r"^```qvr\n(.*?)^```", re.M | re.S)
_MARKER_RE = re.compile(r"<!--\s*compile:\s*(false|standalone|cumulative)\s*-->")


def _marker_before(text: str, fence_start: int) -> str:
    """Return the compile-mode marker on the line preceding ``fence_start``."""
    line_start = text.rfind("\n", 0, fence_start - 1)
    prev_line = text[line_start + 1 : fence_start].strip()
    m = _MARKER_RE.search(prev_line)
    return m.group(1) if m else "standalone"


def _iter_md_files() -> list[Path]:
    files: list[Path] = []
    for root in _DOC_ROOTS:
        if root.is_dir():
            files.extend(sorted(root.rglob("*.md")))
        elif root.is_file():
            files.append(root)
    return files


def _collect_blocks() -> list[tuple[str, int, str, str]]:
    """Return ``(rel_path, block_index, mode, source)`` for every qvr block.

    ``cumulative`` blocks have their sources prepended with every prior
    cumulative block in the same file.
    """
    out: list[tuple[str, int, str, str]] = []
    for md in _iter_md_files():
        text = md.read_text()
        rel = str(md.relative_to(_REPO_ROOT))
        cumulative_prefix = ""
        for idx, m in enumerate(_FENCE_RE.finditer(text)):
            body = m.group(1)
            mode = _marker_before(text, m.start())
            if mode == "cumulative":
                source = cumulative_prefix + body
                cumulative_prefix = source + "\n"
            else:
                source = body
            out.append((rel, idx, mode, source))
    return out


_BLOCKS = _collect_blocks()


@pytest.mark.parametrize(
    ("path", "index", "mode", "source"),
    _BLOCKS,
    ids=[f"{p}:blk{i}:{m}" for p, i, m, _ in _BLOCKS],
)
def test_qvr_doc_block(path: str, index: int, mode: str, source: str) -> None:
    del path, index  # carried only for readable test ids
    if mode == "false":
        pytest.skip("block marked compile: false (illustrative fragment)")
    loads(source)
