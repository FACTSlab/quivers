"""The changelog is published twice, and the copies must agree.

`CHANGELOG.md` at the repository root is the file contributors edit.
`docs/developer/changelog.md` is what the documentation site serves,
and it is a byte-identical copy rather than an include. Nothing else
checks it: no test reads either file, and `mkdocs --strict` builds the
copy without comparing it to anything, so an edit to one alone
publishes a changelog that disagrees with the repository and fails
silently in the direction users see.
"""

from __future__ import annotations

import pathlib


_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "CHANGELOG.md"
_PUBLISHED = _ROOT / "docs" / "developer" / "changelog.md"


def test_the_published_changelog_matches_the_repository_one() -> None:
    """Both copies hold the same bytes."""
    source = _SOURCE.read_text()
    published = _PUBLISHED.read_text()
    if source == published:
        return

    source_versions = [ln for ln in source.splitlines() if ln.startswith("## [")]
    published_versions = [ln for ln in published.splitlines() if ln.startswith("## [")]
    missing = [v for v in source_versions if v not in published_versions]
    extra = [v for v in published_versions if v not in source_versions]
    raise AssertionError(
        f"{_SOURCE.name} and {_PUBLISHED.relative_to(_ROOT)} differ. "
        f"Release headings only in {_SOURCE.name}: {missing!r}. Only in "
        f"the published copy: {extra!r}. If those are both empty the "
        f"difference is inside an entry. The published copy is not an "
        f"include, so an edit to one has to be made to the other: "
        f"`cp CHANGELOG.md docs/developer/changelog.md`."
    )
