"""Compile the in-tree QVR tree-sitter grammar to a shared library.

The Pygments lexer ([`quivers.dsl.pygments_lexer`][quivers.dsl.pygments_lexer])
tokenizes with the real tree-sitter parser rather than a regex
approximation, so it needs a loadable ``TSLanguage``. These helpers
locate ``grammars/qvr/src/parser.c`` in an editable checkout and
compile it to a cached shared library keyed by source mtime.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _grammar_dir() -> Path:
    """Return the path to the in-tree QVR grammar directory.

    Searches upward from this file for ``grammars/qvr/`` so the helper
    works whether quivers is installed editable from a checkout or from
    a build directory.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "grammars" / "qvr"
        if (candidate / "src" / "parser.c").is_file():
            return candidate
    raise FileNotFoundError(
        "grammars/qvr/src/parser.c not found above "
        f"{here}; building the in-tree grammar requires an editable "
        "checkout of the quivers repository."
    )


def _shared_lib_extension() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _build_shared_lib(grammar_dir: Path) -> Path:
    """Compile ``parser.c`` to a shared library, cached by mtime."""
    cache_root = (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "quivers"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    out = cache_root / f"qvr_grammar{_shared_lib_extension()}"
    src = grammar_dir / "src" / "parser.c"

    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out

    cc = os.environ.get("CC", "cc")
    scanner = grammar_dir / "src" / "scanner.c"
    sources = [str(src)]
    if scanner.is_file():
        sources.append(str(scanner))
    cmd = [
        cc,
        "-shared",
        "-fPIC",
        "-O2",
        "-I",
        str(grammar_dir / "src"),
        *sources,
        "-o",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out
