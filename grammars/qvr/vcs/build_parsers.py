"""Build a tree-sitter shared library for every historical QVR
grammar revision tracked by :mod:`build_schemas`.

For each tag returned by :func:`build_schemas._distinct_grammar_revisions`
this script:

1. Materialises the tag's ``grammars/qvr/`` subtree into
   ``grammars/qvr/vcs/parsers/<tag>/source/`` (a fresh copy each run).
2. Regenerates ``parser.c`` from the tag's ``grammar.js`` using the
   ``tree-sitter`` CLI (the checked-in ``src/parser.c`` is not
   reused; running the CLI guarantees the parser matches the
   grammar even when the tag's checked-in artefacts are stale).
3. Compiles ``parser.c`` together with any ``src/scanner.c`` into
   ``grammars/qvr/vcs/parsers/<tag>/qvr.dylib`` (or ``.so`` on
   Linux) via the platform C compiler.

The resulting shared libraries are loadable by the same
``ctypes`` path :mod:`quivers.dsl._dev_grammar` already uses for
HEAD; :mod:`quivers.dsl._historical_grammar` exposes a uniform
``language_for(revision)`` helper.

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
_LIB_EXT = ".dylib" if platform.system() == "Darwin" else ".so"


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
    cc = "cc"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cc,
        "-O2",
        "-fPIC",
        "-shared",
        "-I",
        str(src_dir),
        *(str(p) for p in sources),
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
    args = parser.parse_args(argv)

    revisions = _distinct_grammar_revisions()
    for tag, _ in revisions:
        path = _build_revision(tag, force=args.force)
        print(f"  {tag}: {path.relative_to(_REPO_ROOT)}", flush=True)

    # HEAD is the working-tree grammar; build it from the live source
    # so migrations can target a HEAD that includes uncommitted changes.
    head_dir = _PARSERS_DIR / "HEAD" / "source"
    if head_dir.exists():
        shutil.rmtree(head_dir)
    head_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        _REPO_ROOT / "grammars" / "qvr",
        head_dir,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("vcs", "__pycache__", "*.dylib", "*.so"),
    )
    _generate_parser(head_dir)
    head_out = _PARSERS_DIR / "HEAD" / f"qvr{_LIB_EXT}"
    _compile_parser(head_dir, head_out)
    print(f"  HEAD: {head_out.relative_to(_REPO_ROOT)}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
