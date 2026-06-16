"""Tier 2: external compiler syntax checks.

For each backend whose real compiler / runtime is installed locally,
pipe the transpiled bytes through the compiler's syntax-only
invocation and assert it exits 0.

The check is gated by a per-backend ``requires_tool`` marker in
[`conftest.py`][tests.transpile.conftest]; when the binary is absent,
the cell skips with a clear reason naming the binary.

This tier is the second layer of correctness: the tree-sitter parse
in `test_roundtrip.py` is permissive; this tier asserts the real
target compiler accepts the output. The third layer
(`test_numeric_equivalence.py`) drives the runtime and checks
log-density equivalence.

The check is syntax-only (no execution). Per backend:

| Backend | Tool | Invocation |
|---|---|---|
| stan | ``stanc`` | ``stanc --info /dev/stdin`` |
| numpyro / pyro / pymc / edward2 | ``python`` | ``python -m py_compile /dev/stdin`` |
| webppl | ``node`` | ``node --check /dev/stdin`` |
| turing / gen | ``julia`` | ``julia --eval 'Meta.parse(read(stdin,String))'`` |
| jags | ``jags`` | model-file check |
| bugs | ``jags`` | reuses JAGS (BUGS-compatible) |
| church | (omitted: no canonical interpreter) | — |
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import transpile


_BETA_BERNOULLI = """\
object Resp : FinSet 4
program flip : Resp -> Resp
    sample theta <- Beta(2.0, 2.0)
    observe y <- Bernoulli(theta)
    return theta
export flip
"""


def _run_syntax_check(
    binary: str,
    argv: list[str],
    *,
    input_bytes: bytes,
    timeout: float = 30.0,
) -> tuple[int, str, str]:
    """Run the ``binary`` with ``argv``, feeding ``input_bytes`` on
    stdin. Returns (returncode, stdout, stderr)."""
    if shutil.which(binary) is None:
        pytest.skip(f"{binary!r} not on PATH")
    completed = subprocess.run(
        argv,
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
    )
    return (
        completed.returncode,
        completed.stdout.decode("utf-8", errors="replace"),
        completed.stderr.decode("utf-8", errors="replace"),
    )


def test_stan_external_syntax() -> None:
    """``stanc --info -`` accepts the transpiled Stan output."""
    source = transpile(parse(_BETA_BERNOULLI), target="stan")
    rc, out, err = _run_syntax_check(
        "stanc", ["stanc", "--info", "-"], input_bytes=source
    )
    assert rc == 0, (
        f"stanc exited {rc}: stdout={out!r} stderr={err!r}\n"
        f"source:\n{source.decode()}"
    )


@pytest.mark.parametrize("backend", ["numpyro", "pyro", "pymc", "edward2"])
def test_python_external_syntax(backend: str) -> None:
    """``python -m py_compile`` accepts each python-grammar backend's
    output."""
    source = transpile(parse(_BETA_BERNOULLI), target=backend)
    rc, out, err = _run_syntax_check(
        "python",
        ["python", "-c", "import ast, sys; ast.parse(sys.stdin.read())"],
        input_bytes=source,
    )
    assert rc == 0, (
        f"{backend!r} python ast.parse failed (rc={rc}): "
        f"stdout={out!r} stderr={err!r}\n"
        f"source:\n{source.decode()}"
    )


def test_webppl_external_syntax() -> None:
    """``node --check`` accepts the transpiled WebPPL output.

    WebPPL is a JavaScript subset; ``node --check`` validates JS
    syntax without executing.
    """
    source = transpile(parse(_BETA_BERNOULLI), target="webppl")
    # `node --check` reads from a file, not stdin; use process
    # substitution via stdin trick.
    rc, out, err = _run_syntax_check(
        "node",
        ["node", "--check", "/dev/stdin"],
        input_bytes=source,
    )
    assert rc == 0, (
        f"node --check exited {rc}: stdout={out!r} stderr={err!r}\n"
        f"source:\n{source.decode()}"
    )


@pytest.mark.parametrize("backend", ["turing", "gen"])
def test_julia_external_syntax(backend: str) -> None:
    """``julia -e 'Meta.parse(read(stdin,String))'`` parses the
    transpiled Julia source.

    `Meta.parse` returns an `Expr` for valid Julia or throws
    `ParseError` for invalid input. The script exits 1 on parse
    failure.
    """
    source = transpile(parse(_BETA_BERNOULLI), target=backend)
    rc, out, err = _run_syntax_check(
        "julia",
        [
            "julia", "--startup-file=no", "--quiet", "-e",
            "src = read(stdin, String); Meta.parse(src; raise=true)",
        ],
        input_bytes=source,
    )
    assert rc == 0, (
        f"{backend!r} julia Meta.parse failed (rc={rc}): "
        f"stdout={out!r} stderr={err!r}\n"
        f"source:\n{source.decode()}"
    )


@pytest.mark.parametrize("backend", ["bugs", "jags"])
def test_jags_external_syntax(backend: str, tmp_path) -> None:
    """``jags`` accepts the model-file syntax.

    JAGS' compile path requires a script file with `model in
    "<modelfile>"` plus data; for a syntax-only check we run the
    interactive form with `exit` and check that the model compile
    didn't error.
    """
    source = transpile(parse(_BETA_BERNOULLI), target=backend)
    model_path = tmp_path / f"model.{backend}"
    model_path.write_bytes(source)
    script_path = tmp_path / "check.cmd"
    script_path.write_text(
        f'model in "{model_path}"\nexit\n'
    )
    rc, out, err = _run_syntax_check(
        "jags",
        ["jags", str(script_path)],
        input_bytes=b"",
    )
    # JAGS prints "compiling model graph" on success and "ERROR" on
    # failure; check stderr for ERROR rather than relying on exit
    # codes (JAGS exits 0 even on parse errors in some builds).
    assert "ERROR" not in err and "Error" not in err, (
        f"{backend!r} jags compile failed: stderr={err!r}\n"
        f"source:\n{source.decode()}"
    )
