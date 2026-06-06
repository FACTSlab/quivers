"""Run every QVR example in `docs/examples/source/` through every
backend's target compiler / parser, asserting the transpiled bytes
are syntactically valid in the target language.

This is the gallery-level extension of
[`test_external_syntax.py`][tests.transpile.test_external_syntax]:
that test only exercises the canonical Beta-Bernoulli fixture; this
one drives all 37 examples in the documentation gallery through every
backend whose syntax-check tool is on PATH.

A cell that raises `UnsupportedConstruct` is skipped (the backend
does not support every QVR construct, and the construct-matrix /
family-matrix tests already exercise the support boundary); a cell
that transpiles but fails the target compiler is a real failure.

The four-tier verification hierarchy:

1. Walker structural assertions ([`test_structural.py`][tests.transpile.test_structural]).
2. Mapping composition laws ([`test_lens_laws.py`][tests.transpile.test_lens_laws]).
3. Target compiler acceptance (this test for the gallery; [`test_external_syntax.py`][tests.transpile.test_external_syntax] for the canonical fixture).
4. Measure equivalence in Docker
   ([`test_numeric_equivalence.py`][tests.transpile.test_numeric_equivalence]).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, transpile


_GALLERY = Path(__file__).resolve().parents[2] / "docs" / "examples" / "source"


def _gallery_examples() -> list[Path]:
    return sorted(_GALLERY.glob("*.qvr"))


# Per-backend external syntax checker: (binary, argv-builder, stdin?).
# Backends with no canonical lint-only tool are skipped (Church has
# no standard interpreter we can lint against; Gen and Turing share
# Julia's `Meta.parse`).
_SYNTAX_CHECKS: dict[str, tuple[str, list[str], bool]] = {
    "stan":    ("stanc",  ["stanc", "--no-output", "-"], True),
    "numpyro": ("python", ["python", "-c",
                           "import ast, sys; ast.parse(sys.stdin.read())"], True),
    "pyro":    ("python", ["python", "-c",
                           "import ast, sys; ast.parse(sys.stdin.read())"], True),
    "pymc":    ("python", ["python", "-c",
                           "import ast, sys; ast.parse(sys.stdin.read())"], True),
    "edward2": ("python", ["python", "-c",
                           "import ast, sys; ast.parse(sys.stdin.read())"], True),
    "webppl":  ("node",   ["node", "--check", "/dev/stdin"], True),
    "turing":  ("julia",  ["julia", "--startup-file=no", "--quiet", "-e",
                           "src = read(stdin, String); Meta.parse(src; raise=true)"], True),
    "gen":     ("julia",  ["julia", "--startup-file=no", "--quiet", "-e",
                           "src = read(stdin, String); Meta.parse(src; raise=true)"], True),
}


@pytest.mark.parametrize(
    "example", _gallery_examples(), ids=lambda p: p.stem
)
@pytest.mark.parametrize("backend", sorted(_SYNTAX_CHECKS))
def test_gallery_example_compiles(example: Path, backend: str) -> None:
    """Transpile a gallery example to `backend` and run its target
    compiler / parser as a syntax check."""
    binary, argv, _uses_stdin = _SYNTAX_CHECKS[backend]
    if shutil.which(binary) is None:
        pytest.skip(f"{binary!r} not on PATH")

    source = example.read_text()
    try:
        emitted = transpile(parse(source), target=backend)
    except UnsupportedConstruct as exc:
        # The backend's walker does not handle every construct in the
        # gallery example. The construct-matrix test owns the gap;
        # skip this cell with the unsupported kinds in the message.
        pytest.skip(
            f"backend {backend!r} does not support a construct in "
            f"{example.name}: {exc.kinds!r}"
        )

    completed = subprocess.run(
        argv,
        input=emitted,
        capture_output=True,
        timeout=60.0,
    )
    assert completed.returncode == 0, (
        f"{backend!r} compiler rejected {example.name}: "
        f"stdout={completed.stdout.decode('utf-8', errors='replace')!r} "
        f"stderr={completed.stderr.decode('utf-8', errors='replace')!r}\n"
        f"emitted source:\n{emitted.decode('utf-8', errors='replace')}"
    )
