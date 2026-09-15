"""Build a tree-sitter shared library for every historical QVR
grammar revision tracked by :mod:`build_schemas`.

For each tag returned by :func:`build_schemas._distinct_grammar_revisions`
this script:

1. Materialises the tag's ``grammars/qvr/`` subtree into
   ``grammars/qvr/vcs/parsers/<tag>/source/`` (a fresh copy each run).
2. Regenerates ``parser.c`` from the tag's ``grammar.js`` with the
   ``tree-sitter`` CLI instead of using the checked-in parser.
3. Compiles ``parser.c`` together with any ``src/scanner.c`` into the
   platform shared-library format via MSVC, Clang, GCC, or ``cc``.

The migration package loads the resulting libraries through
``quivers.cli.migrations._grammar.registry_for``.

Usage::

    python grammars/qvr/vcs/build_parsers.py [--force]
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from build_schemas import _distinct_grammar_revisions, _REPO_ROOT, _VCS_ROOT

_PARSERS_DIR = _VCS_ROOT / "parsers"


def _shared_lib_extension(system: str | None = None) -> str:
    system = system or platform.system()
    if system == "Darwin":
        return ".dylib"
    if system == "Windows":
        return ".dll"
    return ".so"


_LIB_EXT = _shared_lib_extension()


def _materialise_tag(tag: str, dest: Path) -> None:
    """Extract the tag's ``grammars/qvr/`` subtree into ``dest``."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(_REPO_ROOT),
            "archive",
            "--format=tar",
            tag,
            "grammars/qvr",
        ],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(
        ["tar", "-x", "-C", str(dest), "--strip-components=2"],
        input=archive,
        check=True,
    )
    # Tagged grammar trees contain the VCS itself.  A parser snapshot
    # needs only the grammar source; retaining ``vcs/`` recursively
    # embeds every prior parser and panproto object in each snapshot.
    nested_vcs = dest / "vcs"
    if nested_vcs.exists():
        shutil.rmtree(nested_vcs)


def _generate_parser(grammar_dir: Path) -> None:
    """Run ``tree-sitter generate`` inside ``grammar_dir``.

    The CLI writes ``src/parser.c`` (and ``src/grammar.json``,
    ``src/node-types.json``) next to ``grammar.js``.
    """
    subprocess.run(
        ["tree-sitter", "generate"],
        cwd=str(grammar_dir),
        check=True,
        capture_output=True,
    )


def _compile_parser(grammar_dir: Path, out_path: Path) -> None:
    """Compile the regenerated ``parser.c`` and optional
    ``scanner.c`` into a shared library at ``out_path``."""
    src_dir = grammar_dir / "src"
    sources = [src_dir / "parser.c"]
    scanner = src_dir / "scanner.c"
    if scanner.exists():
        sources.append(scanner)
    system = platform.system()
    compiler_names = (
        ("cl", "clang", "gcc", "cc")
        if system == "Windows"
        else (
            "cc",
            "clang",
            "gcc",
        )
    )
    compiler = next(
        (path for name in compiler_names if (path := shutil.which(name))),
        None,
    )
    if compiler is None:
        raise RuntimeError("no supported C compiler found (MSVC cl, clang, gcc, or cc)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if system == "Windows" and Path(compiler).name.lower() in {"cl", "cl.exe"}:
        cmd = [
            compiler,
            "/nologo",
            "/O2",
            "/LD",
            f"/I{src_dir}",
            *(str(path) for path in sources),
            "/link",
            f"/OUT:{out_path}",
        ]
    else:
        flags = ["-O2", "-shared"]
        if system != "Windows":
            flags.append("-fPIC")
        cmd = [
            compiler,
            *flags,
            "-I",
            str(src_dir),
            *(str(path) for path in sources),
            "-o",
            str(out_path),
        ]
    subprocess.run(cmd, check=True, capture_output=True)


def _build_revision(tag: str, *, force: bool) -> Path:
    out_path = _PARSERS_DIR / tag / f"qvr{_LIB_EXT}"
    if out_path.exists() and not force:
        return out_path
    workdir = _PARSERS_DIR / tag / "source"
    _materialise_tag(tag, workdir)
    _generate_parser(workdir)
    _compile_parser(workdir, out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if the output library already exists.",
    )
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        help="Build only this tagged revision (repeatable), or HEAD.",
    )
    args = parser.parse_args(argv)

    revisions = _distinct_grammar_revisions()
    requested = set(args.revision)
    known = {tag for tag, _ in revisions} | {"HEAD"}
    unknown = requested - known
    if unknown:
        parser.error(
            "unknown revision(s): " + ", ".join(sorted(unknown)),
        )
    for tag, _ in revisions:
        if requested and tag not in requested:
            continue
        path = _build_revision(tag, force=args.force)
        print(f"  {tag}: {path.relative_to(_REPO_ROOT)}", flush=True)

    # HEAD is the working-tree grammar; build it from the live source
    # so migrations can target a HEAD that includes uncommitted changes.
    if requested and "HEAD" not in requested:
        return 0
    head_dir = _PARSERS_DIR / "HEAD" / "source"
    if head_dir.exists():
        shutil.rmtree(head_dir)
    head_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        _REPO_ROOT / "grammars" / "qvr",
        head_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("vcs", "__pycache__", "*.dylib", "*.so", "*.dll"),
    )
    _generate_parser(head_dir)
    head_out = _PARSERS_DIR / "HEAD" / f"qvr{_LIB_EXT}"
    _compile_parser(head_dir, head_out)
    print(f"  HEAD: {head_out.relative_to(_REPO_ROOT)}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
