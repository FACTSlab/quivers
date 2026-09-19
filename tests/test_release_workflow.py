"""Release-workflow regression tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_wheels_match_binary_grammar_dependency_platforms() -> None:
    """Build only platforms served by the binary grammar dependencies."""
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    assert "CIBW_ARCHS: auto64" in workflow
    assert 'CIBW_SKIP: "*-musllinux_*"' in workflow
