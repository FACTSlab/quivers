"""Locate repository and installed-package grammar migration assets."""

from __future__ import annotations

from pathlib import Path


_PACKAGED_VCS_ROOT = Path(__file__).resolve().parent / "_grammar_vcs"
_REPOSITORY_VCS_ROOT = Path(__file__).resolve().parents[4] / "grammars" / "qvr" / "vcs"


def candidate_vcs_roots() -> tuple[Path, ...]:
    """Return package-local then repository-local asset locations."""
    return (_PACKAGED_VCS_ROOT, _REPOSITORY_VCS_ROOT)


def is_packaged_vcs_path(path: Path) -> bool:
    """Return whether ``path`` belongs to the installed asset directory."""
    try:
        path.resolve().relative_to(_PACKAGED_VCS_ROOT.resolve())
    except ValueError:
        return False
    return True


def vcs_root() -> Path:
    """Return the first available panproto grammar VCS root."""
    for root in candidate_vcs_roots():
        if (root / ".panproto").is_dir():
            return root
    searched = ", ".join(str(root) for root in candidate_vcs_roots())
    raise FileNotFoundError(
        "QVR migration schema history is unavailable; searched " + searched,
    )


__all__ = ["candidate_vcs_roots", "is_packaged_vcs_path", "vcs_root"]
