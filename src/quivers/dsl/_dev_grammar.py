"""Local-grammar override for the QVR tree-sitter parser.

The QVR grammar lives in-tree at ``grammars/qvr/`` and is normally
vendored into ``panproto-grammars-all`` for use through
``panproto.AstParserRegistry``. During grammar development the
upstream wheel lags this repository's ``grammars/qvr/`` source by one
or more releases, so the standard registry returns the previous
grammar's vertex kinds and breaks downstream walker / compiler tests.

This module compiles ``grammars/qvr/src/parser.c`` to a shared library
on demand, loads the resulting ``TSLanguage*`` via ``ctypes``, and
constructs a ``panproto._native.AstParserRegistry`` whose ``qvr``
protocol uses the locally-compiled grammar instead of the one shipped
by ``panproto-grammars-all``.

Activation: set the environment variable ``QVR_USE_LOCAL_GRAMMAR=1``
before importing :mod:`quivers.dsl.parser`. With the variable unset,
the standard panproto registry is used.

The build step requires a working C compiler in ``$PATH``; cached at
``$XDG_CACHE_HOME/quivers/qvr_grammar.dylib`` (or the platform-specific
extension) and rebuilt only when ``parser.c`` is newer than the cache.

Tracking issue: panproto/panproto#89 (request for first-class runtime
grammar override).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import warnings
from importlib.resources import files
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


# Module-level state to keep ctypes buffers alive while the registry
# holds raw pointers into them. A module-level reference is sufficient.
_REGISTRY: object | None = None
_KEEPALIVE: tuple[object, ...] = ()


def registry() -> object:
    """Return a panproto registry whose ``qvr`` grammar is the local build.

    Callers should hold the registry for as long as parsing is needed;
    the underlying ctypes buffers and shared library remain alive via
    a module-level reference.
    """
    global _REGISTRY, _KEEPALIVE
    if _REGISTRY is not None:
        return _REGISTRY

    grammar_dir = _grammar_dir()
    lib_path = _build_shared_lib(grammar_dir)

    lib = ctypes.CDLL(str(lib_path))
    lib.tree_sitter_qvr.restype = ctypes.c_void_p
    language_ptr = lib.tree_sitter_qvr()

    grammar_json = (grammar_dir / "src" / "grammar.json").read_bytes()
    node_types = (grammar_dir / "src" / "node-types.json").read_bytes()
    gj_buf = ctypes.create_string_buffer(grammar_json)
    nt_buf = ctypes.create_string_buffer(node_types)

    extra = {
        "name": "qvr",
        "extensions": ["qvr"],
        "language_ptr": language_ptr,
        "node_types_ptr": ctypes.addressof(nt_buf),
        "node_types_len": len(node_types),
        "tags_query_ptr": None,
        "tags_query_len": 0,
        "grammar_json_ptr": ctypes.addressof(gj_buf),
        "grammar_json_len": len(grammar_json),
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _REGISTRY = panproto._native.AstParserRegistry(extra_grammars=[extra])

    # Keep ctypes buffers and the shared library alive for the lifetime
    # of the registry; without these references, garbage collection
    # would invalidate the pointers panproto holds.
    _KEEPALIVE = (lib, gj_buf, nt_buf)
    _ = files  # silence import-not-used in case importlib.resources is
    # later removed; retained for future packaged-grammar support
    return _REGISTRY
