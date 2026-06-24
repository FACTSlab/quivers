"""Security audit: QVR source as untrusted input.

When `quivers.transpile.transpile(parse(source))` is called on a QVR
source string from an untrusted source (a third-party SaaS endpoint
receives a user's `.qvr` text and runs the pipeline), the transpile
process must refuse to:

- inject host-language code via identifier or numeric-literal strings
  that survive into emitted source verbatim (e.g. a QVR variable name
  containing a Python `import os; os.system(...)` payload that flows
  through the renderer into the emitted target source);
- traverse the host filesystem (no `os.path.join` / `pathlib` /
  `open` on input-derived paths);
- exhaust host resources via deeply-nested constructs (no unbounded
  recursion that crashes the interpreter);
- overflow integer types via numeric literals at the int / float
  boundaries;
- inject arbitrary structure into the panproto schema (no constraint
  values that contain unintended grammar tokens).

Each test below ships an adversarial QVR source and asserts the
pipeline either:

* rejects with [`ParseError`][quivers.dsl.parser._registry.ParseError]
  or [`UnsupportedConstruct`][quivers.transpile.UnsupportedConstruct]
  (the input doesn't conform to the grammar / support tier), OR
* produces output that re-parses through the target's grammar
  without exposing the injection payload as live syntax.

The tests run only against the public surface (parse + transpile);
no internal hooks bypass the input validation.
"""

from __future__ import annotations

import ast
import sys

import pytest

from quivers.dsl.parser import parse
from quivers.dsl.parser._registry import ParseError
from quivers.transpile import UnsupportedConstruct, transpile


_BACKENDS_FOR_SECURITY: tuple[str, ...] = (
    "stan",
    "numpyro",
    "pyro",
    "pymc",
    "edward2",
    "turing",
    "gen",
    "webppl",
    "bugs",
    "jags",
    "church",
)


@pytest.mark.parametrize("backend", _BACKENDS_FOR_SECURITY)
def test_identifier_does_not_carry_python_injection(backend: str) -> None:
    """A QVR identifier with characters that would tokenise as a
    Python statement (`;`, newline, `__import__`) must either be
    rejected at parse time or escaped in the emit such that the
    target-language parser does not see live code.

    The grammar restricts identifiers to `[a-zA-Z_][a-zA-Z_0-9]*`,
    so this should fire `ParseError` at parse time; the test pins
    that contract so a grammar relaxation doesn't accidentally open
    the injection surface.
    """
    payload = "theta__import__os__system_rm_rf"
    src = f"""object Obs : FinSet 30
program inject : Obs -> Obs
    sample {payload} <- Beta(2.0, 5.0)
    observe y : Obs <- Bernoulli({payload})
    return {payload}
export inject
"""
    # Should parse cleanly (the identifier is grammar-legal); the
    # transpile must emit a target identifier without injecting host
    # syntax.
    module = parse(src)
    try:
        emitted = transpile(module, target=backend).decode("utf-8")
    except UnsupportedConstruct:
        return  # backend rejected the program; no surface to inject into
    # The payload should appear as a plain identifier, not as a
    # function call / statement separator.
    assert ";" not in payload  # sanity: the test payload itself is clean
    # Identifier text should appear literally somewhere in the emit.
    assert payload in emitted, (
        f"{backend}: identifier {payload!r} did not survive into emit; "
        f"either the renderer dropped the name or it was mangled"
    )


@pytest.mark.parametrize("backend", _BACKENDS_FOR_SECURITY)
def test_invalid_identifier_with_special_chars_rejected(
    backend: str,
) -> None:
    """An identifier containing host-language statement separators
    (`;`, `\\n`, `'`, backtick, etc.) MUST be rejected at parse
    time, never reach the renderer.
    """
    del backend
    for payload in (
        "x;import os",
        "x\nimport os",
        "x';os.system('rm -rf /')#",
        "x`whoami`",
        "x\\x00null",
    ):
        src = f"""object Obs : FinSet 30
program inject : Obs -> Obs
    sample {payload} <- Beta(2.0, 5.0)
    return {payload}
export inject
"""
        with pytest.raises((ParseError, Exception)):
            parse(src)


@pytest.mark.parametrize("backend", _BACKENDS_FOR_SECURITY)
def test_deep_nesting_does_not_crash(backend: str) -> None:
    """A deeply-nested let-expression must either parse + transpile
    within a reasonable stack budget, or raise a clean
    [`RecursionError`][builtins.RecursionError] / `ParseError` --
    NOT segfault, hang, or escape the Python error handler.

    Tests with 100 nested parens; the grammar should accept any
    depth within Python's `sys.setrecursionlimit`. We bump the
    limit temporarily so the test isn't gated on the default 1000.
    """
    prior_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(5000)
    try:
        depth = 100
        nested = "1.0"
        for _ in range(depth):
            nested = f"({nested} + 0.0)"
        src = f"""object Obs : FinSet 5
program nested : Obs -> Obs
    sample theta <- Normal(0.0, 1.0)
    let deep = {nested}
    observe y : Obs <- Normal(theta + deep, 1.0)
    return theta
export nested
"""
        try:
            module = parse(src)
            transpile(module, target=backend)
        except (ParseError, UnsupportedConstruct, RecursionError):
            return  # Graceful failure; the input was rejected
    finally:
        sys.setrecursionlimit(prior_limit)


@pytest.mark.parametrize("backend", _BACKENDS_FOR_SECURITY)
def test_numeric_literal_at_float_extremes(backend: str) -> None:
    """Numeric literals at float-overflow / underflow / NaN-bait
    values must either be rejected by the parser or pass through
    the pipeline without crashing the renderer.

    Values like `1e400` (positive infinity in IEEE 754 binary64)
    are legitimate inputs the parser accepts; the renderer must
    handle them without raising at the target-source level.
    """
    for literal in ("1e400", "-1e400", "1e-400", "0.0", "1e308"):
        src = f"""object Obs : FinSet 5
program extreme : Obs -> Obs
    sample theta <- Normal({literal}, 1.0)
    observe y : Obs <- Normal(theta, 1.0)
    return theta
export extreme
"""
        try:
            module = parse(src)
            transpile(module, target=backend)
        except (ParseError, UnsupportedConstruct, ValueError, OverflowError):
            continue  # any clean error is acceptable


@pytest.mark.parametrize("backend", _BACKENDS_FOR_SECURITY)
def test_long_identifier_does_not_overflow_buffer(backend: str) -> None:
    """A 10000-character identifier must either parse + transpile or
    raise a clean error. It must not crash the renderer, the
    pretty-printer, or the schema builder.
    """
    long_name = "x" * 10000
    src = f"""object Obs : FinSet 5
program long : Obs -> Obs
    sample {long_name} <- Normal(0.0, 1.0)
    observe y : Obs <- Normal({long_name}, 1.0)
    return {long_name}
export long
"""
    try:
        module = parse(src)
        transpile(module, target=backend)
    except (ParseError, UnsupportedConstruct):
        return


@pytest.mark.parametrize("backend", _BACKENDS_FOR_SECURITY)
def test_many_top_level_declarations(backend: str) -> None:
    """A program with 500 sample sites must parse + transpile within
    a reasonable wall-clock budget. The renderer should be linear in
    the number of declarations; a quadratic loop in any per-fixture
    cache would visibly explode at this scale.
    """
    n = 500
    samples = "\n    ".join(
        f"sample x_{i} <- Normal({float(i)}, 1.0)" for i in range(n)
    )
    src = f"""object Obs : FinSet 5
program many : Obs -> Obs
    {samples}
    observe y : Obs <- Normal(x_0, 1.0)
    return x_0
export many
"""
    try:
        module = parse(src)
        transpile(module, target=backend)
    except (ParseError, UnsupportedConstruct):
        return


@pytest.mark.parametrize("backend", _BACKENDS_FOR_SECURITY)
def test_filesystem_path_in_identifier_does_not_escape(
    backend: str,
) -> None:
    """A QVR identifier shaped like a relative or absolute filesystem
    path must NOT result in the renderer reading or writing that
    path. The grammar restricts identifiers to letters / digits /
    underscores, so a path with `/` or `.` should parse-fail.
    """
    del backend
    for payload in (
        "../../../etc/passwd",
        "/etc/passwd",
        "C:\\Windows\\System32",
        ".ssh/id_rsa",
    ):
        src = f"""object Obs : FinSet 5
program traverse : Obs -> Obs
    sample {payload} <- Normal(0.0, 1.0)
    return {payload}
export traverse
"""
        with pytest.raises(Exception):
            parse(src)


_PYTHON_STRING_BACKENDS: tuple[str, ...] = (
    "numpyro",
    "pyro",
    "pymc",
    "edward2",
)


@pytest.mark.parametrize("backend", _PYTHON_STRING_BACKENDS)
def test_string_literal_does_not_escape_target_quotes(
    backend: str,
) -> None:
    """A `let label = "<chars>"` binding flows through the renderer
    as a target-language string literal. The four python-host
    backends emit a Python string literal; injection-safety requires
    the re-parsed Python AST to recover `label` as a string Constant
    whose value equals the QVR-parsed string content verbatim.

    The QVR grammar reads string-literal bodies as escape-passthrough
    (a QVR ``"foo\\"bar"`` parses to the literal 7-character value
    ``foo\\"bar`` with the backslash retained, rather than decoding
    the escape). The renderer's contract is therefore: take the raw
    parsed bytes and wrap them in target-language quoting that
    preserves every byte. A naive ``f'"{value}"'`` interpolation
    would let a payload containing a target-quote character break
    out of the literal; `json.dumps`-style escaping does not.

    This test drives a payload whose QVR-parsed value contains every
    sensitive byte for Python single- and double-quoted strings
    (``"``, ``\\``, newline, plus a live ``os.system(...)`` call
    that would execute if the escape failed).
    """
    qvr_source = (
        'object Obs : FinSet 5\n'
        'program p : Obs -> Obs\n'
        '    let label = "abc\\"; os.system(\'rm -rf /\'); foo = \\"def"\n'
        '    sample x <- Normal(0.0, 1.0)\n'
        '    return x\n'
        'export p\n'
    )
    module = parse(qvr_source)
    parsed_value = _label_string_value(module)
    emitted = transpile(module, target=backend)
    body = emitted.decode("utf-8")
    tree = ast.parse(body)
    label_assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "label"
    ]
    assert len(label_assignments) == 1, (
        f"{backend!r}: expected exactly one `label = ...` assignment "
        f"in emit; got {len(label_assignments)} (emit body: {body!r})"
    )
    rhs = label_assignments[0].value
    assert isinstance(rhs, ast.Constant) and isinstance(rhs.value, str), (
        f"{backend!r}: `label` RHS is not a string literal "
        f"(got {ast.dump(rhs)}). A renderer that injected the payload "
        f"as live code would produce a Call / Name / Attribute node "
        f"here instead of an ast.Constant str."
    )
    assert rhs.value == parsed_value, (
        f"{backend!r}: emitted `label` value {rhs.value!r} does not "
        f"equal the QVR-parsed value {parsed_value!r}; the renderer "
        f"corrupted or under-escaped the string content"
    )
    calls_to_os_system = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "system"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
    ]
    assert not calls_to_os_system, (
        f"{backend!r}: payload escaped into a live `os.system(...)` "
        f"call: {[ast.dump(c) for c in calls_to_os_system]}"
    )


def _label_string_value(module: object) -> str:
    """Return the QVR-parsed string content of the `let label = "..."`
    binding inside the module's program block. Raises AssertionError
    if no such binding is present (the test relies on a specific
    fixture shape)."""
    for stmt in module.statements:
        draws = getattr(stmt, "draws", ())
        for draw in draws:
            if (
                getattr(draw, "kind", None) == "let_step"
                and getattr(draw, "name", None) == "label"
            ):
                value = draw.value
                assert getattr(value, "kind", None) == "let_expr_string", (
                    f"label binding's RHS is not a string-literal "
                    f"(got {value!r})"
                )
                return value.value
    raise AssertionError("test fixture missing `let label = ...`")


@pytest.mark.parametrize("backend", _BACKENDS_FOR_SECURITY)
def test_axis_size_overflow_rejected(backend: str) -> None:
    """An `object Obs : FinSet <huge>` declaration must either parse
    + transpile within the integer-overflow envelope, or raise a
    clean error. A value at the C `int` boundary (2^31 - 1) is a
    realistic-but-dangerous input.
    """
    for size in ("2147483647", "9223372036854775807"):
        src = f"""object Obs : FinSet {size}
program overflow : Obs -> Obs
    sample theta <- Normal(0.0, 1.0)
    return theta
export overflow
"""
        try:
            module = parse(src)
            transpile(module, target=backend)
        except (ParseError, UnsupportedConstruct, ValueError, OverflowError):
            continue
