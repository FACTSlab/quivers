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
    "stan":    ("stanc",  ["stanc", "--info", "-"], True),
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
                           "src = read(stdin, String); Meta.parseall(src)"], True),
    "gen":     ("julia",  ["julia", "--startup-file=no", "--quiet", "-e",
                           "src = read(stdin, String); Meta.parseall(src)"], True),
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
        pytest.xfail(
            f"{binary!r} not on PATH; install it in the local toolchain "
            f"or add the install step to CI"
        )

    source = example.read_text()
    try:
        emitted = transpile(parse(source), target=backend)
    except UnsupportedConstruct as exc:
        # The backend's walker does not handle every construct in the
        # gallery example. The construct-matrix test owns the gap;
        # xfail this cell with the unsupported kinds in the message.
        pytest.xfail(
            f"backend {backend!r} does not support a construct in "
            f"{example.name}: {exc.kinds!r}"
        )

    completed = subprocess.run(
        argv,
        input=emitted,
        capture_output=True,
        timeout=60.0,
    )
    if completed.returncode != 0:
        cell = (backend, example.stem)
        known = _KNOWN_RENDERER_EMIT_GAPS.get(cell)
        if known is not None:
            pytest.xfail(
                f"{backend!r} on {example.name}: known renderer-emit "
                f"gap -- {known}"
            )
    assert completed.returncode == 0, (
        f"{backend!r} compiler rejected {example.name}: "
        f"stdout={completed.stdout.decode('utf-8', errors='replace')!r} "
        f"stderr={completed.stderr.decode('utf-8', errors='replace')!r}\n"
        f"emitted source:\n{emitted.decode('utf-8', errors='replace')}"
    )


#: Cells where the QVR→<backend> transpile succeeds but the emitted
#: source is rejected by the target's syntax check. Each entry pairs
#: the (backend, fixture-stem) cell with a one-line explanation of
#: the specific renderer bug. When the renderer is fixed, the
#: emit re-passes and the entry comes out.
#:
#: This list is the residue after the categorical-metadata gate
#: (`composition_decl` etc.) stopped hiding deeper renderer gaps:
#: those programs now reach the renderer + syntax check and trip on
#: per-fixture emit bugs the categorical gate had previously hidden.
_KNOWN_RENDERER_EMIT_GAPS: dict[tuple[str, str], str] = {
    ("stan", "hmm"): (
        "stan renderer emits `categorical(emission_rows)` where "
        "emission_rows is `array[State] vector[Obs]`. The fixture's "
        "`observe obs <- Categorical(emission_rows)` has no per-row "
        "state index in the call site; the renderer needs to either "
        "pick up the latent state from the surrounding scan or thread "
        "it via a `[via=state]` fibration"
    ),
    ("stan", "mixture_model"): (
        "fixture declares `sample idx : Resp <- HalfNormal(1.0)` "
        "(real-valued) but uses `[via=idx]` as an integer index. The "
        "renderer correctly emits `array[100] real idx` but the "
        "Stan grammar then rejects the use of a real-valued idx in "
        "`lps_cls[idx[n], k]`. Either the fixture should declare idx "
        "as integer-valued (e.g. DiscreteUniform), or the via-fibration "
        "renderer should reject real-valued fibrations at lower time"
    ),
    ("gen", "montague_nli"): (
        "deduction emission for the Montague grammar's chart-parser "
        "primitives is not yet wired in the Gen.jl renderer"
    ),
    ("turing", "montague_nli"): (
        "deduction emission for the Montague grammar's chart-parser "
        "primitives is not yet wired in the Turing renderer"
    ),
}
