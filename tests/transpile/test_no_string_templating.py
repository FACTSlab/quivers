"""Static guardrail: no module under `src/quivers/transpile/` may
build its output bytes via string templating.

Allowed:
- Building a panproto.Schema via SchemaBuilder, with vertex kinds and
  edge labels matching the target tree-sitter grammar.
- Setting `literal-value` constraints (those are the identifier text
  that the grammar walker substitutes into output bytes).

Forbidden:
- Returning `b"..."` or `f"...".encode()` constructed by concatenation
  of program-derived strings.
- `string.Template` substitution.
- Writing source bytes to a buffer via `IndentWriter` or similar.

Implementation: walk every transpile module's AST and look for
suspicious patterns. Failures point at the offending file + line.
"""

from __future__ import annotations

import ast
import pathlib


_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "quivers" / "transpile"
_FORBIDDEN_MODULES = frozenset({"string"})  # for `string.Template`


def _modules() -> list[pathlib.Path]:
    return sorted(p for p in _ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_string_template_imports() -> None:
    """No transpile module may import `string.Template`."""
    bad: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in _FORBIDDEN_MODULES:
                bad.append(f"{path}:{node.lineno}: imports from {node.module!r}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _FORBIDDEN_MODULES:
                        bad.append(
                            f"{path}:{node.lineno}: imports {alias.name!r}"
                        )
    assert not bad, "\n".join(bad)


def test_no_indent_writer() -> None:
    """No transpile module may import didactic's IndentWriter (an
    explicit string-buffer-with-indentation helper)."""
    bad: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == "IndentWriter":
                        bad.append(
                            f"{path}:{node.lineno}: imports IndentWriter "
                            f"from {node.module!r}"
                        )
            if isinstance(node, ast.Attribute) and node.attr == "IndentWriter":
                bad.append(f"{path}:{node.lineno}: references .IndentWriter")
    assert not bad, "\n".join(bad)


def test_no_bytes_format_or_concat_with_str() -> None:
    """No transpile module may build `bytes` via `str.format(...).encode()`
    or analogous templating idioms.

    The rule the test enforces: any `.encode(...)` call whose receiver
    is a `str.format(...)` call, an f-string with conversions that are
    user-supplied values, or a `%`-format expression, is forbidden.
    """
    bad: list[str] = []
    for path in _modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "encode"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr in {"format", "format_map"}
            ):
                bad.append(
                    f"{path}:{node.lineno}: builds bytes via str.format().encode()"
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "encode"
                and isinstance(node.func.value, ast.BinOp)
                and isinstance(node.func.value.op, ast.Mod)
            ):
                bad.append(
                    f"{path}:{node.lineno}: builds bytes via percent-format.encode()"
                )
    assert not bad, "\n".join(bad)
