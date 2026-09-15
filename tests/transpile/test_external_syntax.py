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
| numpyro / pyro / pymc / edward2 | ``python`` | builtin ``compile(..., 'exec')`` |
| webppl | ``node`` | ``node --check /dev/stdin`` |
| turing / gen | ``julia`` | ``julia --eval 'Meta.parse(read(stdin,String))'`` |
| jags | ``jags`` | model-file check |
| bugs | ``jags`` | reuses JAGS (BUGS-compatible) |
| church | Chez (``scheme``/``chez``/``petite``/``chezscheme``) | ``<chez> --script`` |
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import tempfile

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

_QIEC_PURE = """\
define answer() : Int !{} =
    return 42
"""

_QIEC_MIXED = _QIEC_PURE + "\n" + _BETA_BERNOULLI

_QIEC_DECLARATION_ONLY = "index Nat = Z | S(Nat)\n"

_QIEC_KEYWORD_PARAMETERS = """\
define collision(class : Int, qiec_handlers : Int, end : Int) : Int !{} =
    return class
"""

_SCHEME_EXECUTABLE = next(
    (
        executable
        for name in ("scheme", "chez", "petite", "chezscheme")
        if (executable := shutil.which(name)) is not None
    ),
    None,
)

_EXTERNAL_PROGRAMS = (
    pytest.param(_BETA_BERNOULLI, id="probabilistic"),
    pytest.param(_QIEC_PURE, id="qiec"),
    pytest.param(_QIEC_MIXED, id="mixed"),
    pytest.param(_QIEC_DECLARATION_ONLY, id="qiec-declarations"),
)


def _run_syntax_check(
    binary: str,
    argv: list[str],
    *,
    input_bytes: bytes,
    timeout: float = 300.0,
) -> tuple[int, str, str]:
    """Run the ``binary`` with ``argv``, feeding ``input_bytes`` on
    stdin. Returns (returncode, stdout, stderr).

    The timeout is generous because the cost being waited on is a
    cold start rather than the parse: Julia compiles its own runtime
    on first invocation, and on a fresh machine that alone can run
    past half a minute. A parser that has actually hung still fails
    here, only later.
    """
    if shutil.which(binary) is None:
        pytest.xfail(
            f"{binary!r} not on PATH; install it in the local toolchain "
            f"or add the install step to CI"
        )
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


@pytest.mark.parametrize("program", _EXTERNAL_PROGRAMS)
def test_stan_external_syntax(program: str) -> None:
    """``stanc --info -`` accepts the transpiled Stan output."""
    source = transpile(parse(program), target="stan")
    rc, out, err = _run_syntax_check(
        "stanc", ["stanc", "--info", "-"], input_bytes=source
    )
    assert rc == 0, (
        f"stanc exited {rc}: stdout={out!r} stderr={err!r}\nsource:\n{source.decode()}"
    )


@pytest.mark.parametrize("program", _EXTERNAL_PROGRAMS)
@pytest.mark.parametrize("backend", ["numpyro", "pyro", "pymc", "edward2"])
def test_python_external_syntax(backend: str, program: str) -> None:
    """Python bytecode compilation accepts each Python backend's output."""
    source = transpile(parse(program), target=backend)
    rc, out, err = _run_syntax_check(
        "python",
        [
            "python",
            "-c",
            "import sys; compile(sys.stdin.buffer.read(), '<qvr>', 'exec')",
        ],
        input_bytes=source,
    )
    assert rc == 0, (
        f"{backend!r} python compile failed (rc={rc}): "
        f"stdout={out!r} stderr={err!r}\n"
        f"source:\n{source.decode()}"
    )


@pytest.mark.parametrize("backend", ["numpyro", "pyro", "pymc", "edward2"])
def test_python_qiec_keyword_parameter_syntax(backend: str) -> None:
    source = transpile(parse(_QIEC_KEYWORD_PARAMETERS), target=backend)
    rc, out, err = _run_syntax_check(
        "python",
        [
            "python",
            "-c",
            "import sys; compile(sys.stdin.buffer.read(), '<qvr>', 'exec')",
        ],
        input_bytes=source,
    )
    assert rc == 0, (
        f"{backend!r} keyword hygiene failed (rc={rc}): "
        f"stdout={out!r} stderr={err!r}\nsource:\n{source.decode()}"
    )


@pytest.mark.parametrize("program", _EXTERNAL_PROGRAMS)
def test_webppl_external_syntax(program: str) -> None:
    """``node --check`` accepts the transpiled WebPPL output.

    WebPPL is a JavaScript subset; ``node --check`` validates JS
    syntax without executing.
    """
    source = transpile(parse(program), target="webppl")
    # `node --check` opens its argument as a file. Handed
    # `/dev/stdin` it resolves that through `/proc/<pid>/fd/0`, which
    # is a pipe when the source arrives on stdin, and a pipe is not
    # something it can open: the check fails with ENOENT before it has
    # read a byte of JavaScript. The source goes to a real file.
    with tempfile.TemporaryDirectory() as tmp:
        script = pathlib.Path(tmp) / "emitted.js"
        script.write_bytes(source)
        rc, out, err = _run_syntax_check(
            "node",
            ["node", "--check", str(script)],
            input_bytes=b"",
        )
    assert rc == 0, (
        f"node --check exited {rc}: stdout={out!r} stderr={err!r}\n"
        f"source:\n{source.decode()}"
    )


@pytest.mark.parametrize("program", _EXTERNAL_PROGRAMS)
@pytest.mark.parametrize("backend", ["turing", "gen"])
def test_julia_external_syntax(backend: str, program: str) -> None:
    """Julia's ``Meta.parse`` loop parses the complete emitted source.

    Passing only a string to `Meta.parse` accepts exactly one top-level
    expression. QIEC output intentionally has several definitions, so the
    syntax gate advances through every expression and retains ``raise=true``.
    """
    source = transpile(parse(program), target=backend)
    rc, out, err = _run_syntax_check(
        "julia",
        [
            "julia",
            "--startup-file=no",
            "--quiet",
            "-e",
            (
                "let src = read(stdin, String), pos = 1; "
                "while pos <= ncodeunits(src); expr, pos = "
                "Meta.parse(src, pos; raise=true); expr === nothing && break; end; end"
            ),
        ],
        input_bytes=source,
    )
    assert rc == 0, (
        f"{backend!r} julia Meta.parse failed (rc={rc}): "
        f"stdout={out!r} stderr={err!r}\n"
        f"source:\n{source.decode()}"
    )


@pytest.mark.parametrize("program", _EXTERNAL_PROGRAMS)
@pytest.mark.parametrize("backend", ["bugs", "jags"])
def test_jags_external_syntax(backend: str, program: str, tmp_path) -> None:
    """``jags`` accepts the model-file syntax.

    JAGS' compile path requires a script file with `model in
    "<modelfile>"` plus data; for a syntax-only check we run the
    interactive form with `exit` and check that the model compile
    didn't error.
    """
    source = transpile(parse(program), target=backend)
    model_path = tmp_path / f"model.{backend}"
    model_path.write_bytes(source)
    script_path = tmp_path / "check.cmd"
    script_path.write_text(f'model in "{model_path}"\nexit\n')
    rc, out, err = _run_syntax_check(
        "jags",
        ["jags", str(script_path)],
        input_bytes=b"",
    )
    # JAGS prints "compiling model graph" on success and "ERROR" on
    # failure; check stderr for ERROR rather than relying on exit
    # codes (JAGS exits 0 even on parse errors in some builds).
    assert "ERROR" not in err and "Error" not in err, (
        f"{backend!r} jags compile failed: stderr={err!r}\nsource:\n{source.decode()}"
    )


@pytest.mark.parametrize("program", _EXTERNAL_PROGRAMS)
def test_church_external_syntax(program: str, tmp_path) -> None:
    """Chez loads the complete generated Church module."""
    if _SCHEME_EXECUTABLE is None:
        pytest.xfail(
            "Chez Scheme is not on PATH under scheme, chez, petite, or chezscheme"
        )
    source = transpile(parse(program), target="church")
    script = tmp_path / "model.scm"
    script.write_bytes(source)
    rc, out, err = _run_syntax_check(
        _SCHEME_EXECUTABLE,
        [_SCHEME_EXECUTABLE, "--script", str(script)],
        input_bytes=b"",
    )
    assert rc == 0, (
        f"Church Chez load failed (rc={rc}): stdout={out!r} stderr={err!r}\n"
        f"source:\n{source.decode()}"
    )
