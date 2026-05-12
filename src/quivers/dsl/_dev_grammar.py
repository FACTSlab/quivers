"""Local-grammar override for the QVR tree-sitter parser.

The QVR grammar lives in-tree at ``grammars/qvr/`` and will be vendored
into ``panproto-grammars-all`` once it stabilises. In the meantime, this
module compiles ``grammars/qvr/src/parser.c`` to a shared library on
demand, loads the resulting ``TSLanguage*`` via ``ctypes``, and installs
it through panproto's :meth:`AstParserRegistry.override_grammar` API so
the standard registry serves the in-tree grammar in place of whatever
``panproto-grammars-all`` currently ships for ``qvr``.

Activation: set the environment variable ``QVR_USE_LOCAL_GRAMMAR=1``
before importing :mod:`quivers.dsl.parser`. With the variable unset,
the standard panproto registry is used.

The build step requires a working C compiler in ``$PATH``; cached at
``$XDG_CACHE_HOME/quivers/qvr_grammar.dylib`` (or the platform-specific
extension) and rebuilt only when ``parser.c`` is newer than the cache.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import warnings
from pathlib import Path

import panproto


_ENV_FLAG = "QVR_USE_LOCAL_GRAMMAR"


def is_active() -> bool:
    """True when the local-grammar override is requested."""
    return os.environ.get(_ENV_FLAG, "") not in ("", "0", "false", "False")


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
        f"{here}; the local-grammar override requires an editable "
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
    cmd = [
        cc,
        "-shared",
        "-fPIC",
        "-O2",
        "-I",
        str(grammar_dir / "src"),
        str(src),
        "-o",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


_REGISTRY: object | None = None
_LIB_KEEPALIVE: ctypes.CDLL | None = None


def registry() -> object:
    """Return a panproto registry whose ``qvr`` grammar is the local build.

    The standard :func:`panproto.AstParserRegistry` constructor populates
    the registry with everything ``panproto-grammars-all`` and other
    installed companion packages contribute, then
    :meth:`override_grammar` swaps in the locally-compiled QVR grammar.
    """
    global _REGISTRY, _LIB_KEEPALIVE
    if _REGISTRY is not None:
        return _REGISTRY

    grammar_dir = _grammar_dir()
    lib_path = _build_shared_lib(grammar_dir)

    lib = ctypes.CDLL(str(lib_path))
    lib.tree_sitter_qvr.restype = ctypes.c_void_p
    language_ptr = lib.tree_sitter_qvr()

    grammar_json = (grammar_dir / "src" / "grammar.json").read_bytes()
    node_types = (grammar_dir / "src" / "node-types.json").read_bytes()

    # panproto's `AstParserRegistry()` constructor emits RuntimeWarnings
    # for every companion grammar it can't register at import time
    # (currently `al`, `csharp`, `erlang` on this environment). Those
    # are upstream-packaging issues that quivers cannot fix here; the
    # standard non-dev path suppresses them via a `catch_warnings`
    # block in `quivers.dsl.parser._registry`, and the dev path does
    # the same so test output stays clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reg = panproto.AstParserRegistry()
    reg.override_grammar(
        name="qvr",
        extensions=["qvr"],
        language_ptr=language_ptr,
        node_types=node_types,
        grammar_json=grammar_json,
    )

    # Keep the shared library alive for the lifetime of the registry;
    # the TSLanguage* is owned by ``lib`` and panproto holds it by raw
    # pointer.
    _LIB_KEEPALIVE = lib
    _REGISTRY = reg
    return _REGISTRY
