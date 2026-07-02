"""Canonical-fixed-point tests for the ``.qvr`` source emitter.

Every ``.qvr`` file in the repo corpus and every compiled fenced
```qvr block in the docs is pushed through ``parse -> emit -> parse ->
emit``. The second parse must succeed and the two emitted texts must
be byte-identical: ``module_to_source`` is a canonical fixed point of
the parse/emit pair.

Requires ``QVR_USE_LOCAL_GRAMMAR=1`` (set by ``tests/conftest.py``,
and defaulted again here for direct invocation) so parsing picks up
the in-tree grammar at ``grammars/qvr/``::

    QVR_USE_LOCAL_GRAMMAR=1 pytest tests/test_emit_roundtrip.py

QVR corpus
----------

* ``docs/examples/source/**/*.qvr``
* ``tests/benchmarks/models/**/*.qvr``
* ``regression.qvr`` at the repo root

Doc blocks
----------

Each fenced ```qvr block under ``docs/`` (and ``README.md``) is a
test case, following the discovery convention of
``tests/test_doc_blocks.py``: a ``<!-- compile: false -->`` marker on
the line immediately preceding the fence excludes the block
(illustrative fragment); a ``<!-- compile: cumulative -->`` marker
concatenates the block with every prior cumulative block in the same
file, so the round-tripped source matches what the doc-block suite
compiles.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QVR_USE_LOCAL_GRAMMAR", "1")

from quivers.dsl import parse
from quivers.dsl.emit import module_to_source

_REPO_ROOT = Path(__file__).resolve().parent.parent
_QVR_ROOTS = (
    _REPO_ROOT / "docs" / "examples" / "source",
    _REPO_ROOT / "tests" / "benchmarks" / "models",
)
_QVR_FILES = (_REPO_ROOT / "regression.qvr",)
_DOC_ROOTS = (_REPO_ROOT / "docs", _REPO_ROOT / "README.md")

_QVR_FENCE_RE = re.compile(r"^([ \t]*)```qvr\n(.*?)^\1```", re.M | re.S)
_QVR_MARKER_RE = re.compile(r"<!--\s*compile:\s*(false|standalone|cumulative)\s*-->")


def _collect_qvr_files() -> list[Path]:
    files: list[Path] = []
    for root in _QVR_ROOTS:
        files.extend(sorted(root.rglob("*.qvr")))
    files.extend(f for f in _QVR_FILES if f.is_file())
    return files


def _dedent_fence_body(indent: str, body: str) -> str:
    if not indent:
        return body
    out_lines: list[str] = []
    for line in body.splitlines(keepends=True):
        if line.startswith(indent):
            out_lines.append(line[len(indent) :])
        else:
            out_lines.append(line)
    return "".join(out_lines)


def _qvr_marker_before(text: str, fence_start: int) -> str:
    line_start = text.rfind("\n", 0, fence_start - 1)
    prev_line = text[line_start + 1 : fence_start].strip()
    m = _QVR_MARKER_RE.search(prev_line)
    return m.group(1) if m else "standalone"


def _iter_md_files() -> list[Path]:
    files: list[Path] = []
    for root in _DOC_ROOTS:
        if root.is_dir():
            files.extend(sorted(root.rglob("*.md")))
        elif root.is_file():
            files.append(root)
    return files


def _collect_qvr_blocks() -> list[tuple[str, int, str]]:
    """Return ``(rel_path, block_index, source)`` for every compiled
    qvr block (blocks marked ``compile: false`` are excluded)."""
    out: list[tuple[str, int, str]] = []
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
            if mode == "false":
                continue
            out.append((rel, idx, source))
    return out


_CORPUS_FILES = _collect_qvr_files()
_DOC_QVR_BLOCKS = _collect_qvr_blocks()


def _assert_roundtrip(source: str) -> None:
    """parse -> emit -> parse -> emit; the second parse must succeed
    and the two emitted texts must be byte-identical."""
    first = module_to_source(parse(source))
    second = module_to_source(parse(first))
    assert second == first


@pytest.mark.parametrize(
    "path",
    _CORPUS_FILES,
    ids=[str(p.relative_to(_REPO_ROOT)) for p in _CORPUS_FILES],
)
def test_qvr_file_roundtrip(path: Path) -> None:
    _assert_roundtrip(path.read_text())


@pytest.mark.parametrize(
    ("path", "index", "source"),
    _DOC_QVR_BLOCKS,
    ids=[f"{p}:blk{i}" for p, i, _ in _DOC_QVR_BLOCKS],
)
def test_qvr_doc_block_roundtrip(path: str, index: int, source: str) -> None:
    del path, index  # carried only for readable test ids
    _assert_roundtrip(source)
