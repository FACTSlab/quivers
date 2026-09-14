"""Report callables whose docstring omits a section their signature needs.

A docstring is judged against the code it documents rather than against a
fixed template. A callable taking arguments needs ``Parameters``; one
returning a value needs ``Returns``; a generator needs ``Yields``; one that
can raise needs ``Raises``. A callable that does none of those needs only a
summary.

Constructor parameters are documented on the **class**, per this
repository's convention, so ``__init__`` is skipped here and its class is
judged instead.

Run over the whole package, or over the paths being edited:

.. code-block:: shell

    python tools/docstring_audit.py                  # src/quivers
    python tools/docstring_audit.py src/quivers/qiec
    python tools/docstring_audit.py --list src/quivers/qiec/checking.py
    python tools/docstring_audit.py --check src/quivers/qiec

``--check`` exits non-zero when anything is incomplete, which is the form a
CI gate would use once a package is clean.
"""

from __future__ import annotations

import argparse
import ast
import collections
import pathlib
import sys


SECTIONS = ("Parameters", "Returns", "Yields", "Raises")


def documented_sections(doc: str) -> set[str]:
    """The numpydoc section headers a docstring carries.

    Parameters
    ----------
    doc : str
        The docstring body, already dedented by :func:`ast.get_docstring`.

    Returns
    -------
    set[str]
        The subset of :data:`SECTIONS` present. Indented and unindented
        forms both count, since a docstring's indentation depends on
        whether it documents a module, a class, or a method.
    """
    return {
        section
        for section in SECTIONS
        if f"\n    {section}\n" in doc or f"\n{section}\n" in doc
    }


def _own_nodes(node: ast.AST) -> list[ast.AST]:
    """Every node belonging to one callable, excluding nested callables.

    A nested helper has its own contract, so its `return`, `yield`, and
    `raise` must not be attributed to the function enclosing it. Walking
    blindly reports an enclosing function as returning a value when only
    its inner helper does.

    Parameters
    ----------
    node : ast.AST
        The callable whose own body to collect.

    Returns
    -------
    list[ast.AST]
        Nodes reachable from `node` without entering a nested function,
        async function, or class definition.
    """
    own: list[ast.AST] = []
    stack: list[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        current = stack.pop()
        if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        own.append(current)
        stack.extend(ast.iter_child_nodes(current))
    return own


def required_sections(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """The sections a callable's signature and body oblige it to document.

    Parameters
    ----------
    node : ast.FunctionDef or ast.AsyncFunctionDef
        The callable to inspect.

    Returns
    -------
    list[str]
        Required section names. ``Yields`` displaces ``Returns`` for a
        generator, since a generator's ``return`` ends iteration rather
        than supplying the caller's value.
    """
    arguments = [
        argument.arg
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        if argument.arg not in ("self", "cls")
    ]
    if node.args.vararg is not None or node.args.kwarg is not None:
        arguments.append("*")
    body = _own_nodes(node)
    returns = any(
        isinstance(inner, ast.Return) and inner.value is not None for inner in body
    )
    yields = any(isinstance(inner, ast.Yield | ast.YieldFrom) for inner in body)
    raises = any(
        isinstance(inner, ast.Raise) and inner.exc is not None for inner in body
    )
    required = []
    if arguments:
        required.append("Parameters")
    if yields:
        required.append("Yields")
    elif returns:
        required.append("Returns")
    if raises:
        required.append("Raises")
    return required


def audit(path: pathlib.Path) -> list[tuple[str, int, list[str]]]:
    """Find every incomplete docstring in one module.

    Parameters
    ----------
    path : pathlib.Path
        The ``.py`` file to read.

    Returns
    -------
    list[tuple[str, int, list[str]]]
        One ``(name, line, missing)`` triple per incomplete callable, where
        ``missing`` is either the absent section names or the single entry
        ``"<no docstring>"``. A file that cannot be parsed yields nothing,
        since a syntax error is the linter's report to make, not this
        one's.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError, UnicodeDecodeError:
        return []
    findings: list[tuple[str, int, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name == "__init__":
            continue
        doc = ast.get_docstring(node)
        if doc is None:
            findings.append((node.name, node.lineno, ["<no docstring>"]))
            continue
        have = documented_sections(doc)
        missing = [
            section for section in required_sections(node) if section not in have
        ]
        if missing:
            findings.append((node.name, node.lineno, missing))
    return sorted(findings, key=lambda finding: finding[1])


def _modules(roots: list[pathlib.Path]) -> list[pathlib.Path]:
    """Expand the requested paths to the Python modules under them.

    Parameters
    ----------
    roots : list[pathlib.Path]
        Files or directories named on the command line.

    Returns
    -------
    list[pathlib.Path]
        Every ``.py`` file, sorted so the report is reproducible.
    """
    modules: list[pathlib.Path] = []
    for root in roots:
        modules.extend([root] if root.is_file() else sorted(root.rglob("*.py")))
    return modules


def main(argv: list[str] | None = None) -> int:
    """Report incomplete docstrings under the requested paths.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments, defaulting to :data:`sys.argv`.

    Returns
    -------
    int
        ``0`` when nothing is incomplete, or when ``--check`` was not
        passed. ``1`` under ``--check`` when something is incomplete, so
        the command can gate a build.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=pathlib.Path)
    parser.add_argument("--list", action="store_true", help="name every callable")
    parser.add_argument("--check", action="store_true", help="exit 1 if incomplete")
    arguments = parser.parse_args(argv)

    roots = arguments.paths or [pathlib.Path("src/quivers")]
    per_file: collections.Counter[str] = collections.Counter()
    total = 0
    for module in _modules(roots):
        findings = audit(module)
        if not findings:
            continue
        per_file[str(module)] = len(findings)
        total += len(findings)
        if arguments.list:
            for name, line, missing in findings:
                print(f"{module}:{line}: {name} missing {', '.join(missing)}")

    print(f"{total} incomplete docstring(s) across {len(per_file)} file(s)")
    if not arguments.list:
        for name, count in per_file.most_common():
            print(f"  {count:>4}  {name}")
    return 1 if arguments.check and total else 0


if __name__ == "__main__":
    sys.exit(main())
