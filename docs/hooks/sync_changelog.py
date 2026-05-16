"""MkDocs hook that mirrors the repo-root ``CHANGELOG.md`` into
``docs/developer/changelog.md`` at build time.

The repo-root CHANGELOG is the single source of truth. This hook
copies it into the docs tree before MkDocs collects files, so the
docs site never lags behind a freshly tagged release.
"""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "CHANGELOG.md"
_DST = _ROOT / "docs" / "developer" / "changelog.md"


def on_pre_build(config, **kwargs):  # noqa: ARG001
    if not _SRC.is_file():
        return
    src_text = _SRC.read_text(encoding="utf-8")
    if _DST.is_file() and _DST.read_text(encoding="utf-8") == src_text:
        return
    _DST.parent.mkdir(parents=True, exist_ok=True)
    _DST.write_text(src_text, encoding="utf-8")
