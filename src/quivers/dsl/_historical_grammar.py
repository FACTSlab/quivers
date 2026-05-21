"""Load any historical QVR grammar revision built by
``grammars/qvr/vcs/build_parsers.py``.

Mirrors [`quivers.dsl._dev_grammar`][quivers.dsl._dev_grammar] but lets the caller pick
which revision (``v0.2.0`` ... ``v0.9.0`` or ``HEAD``) to load.
Each revision lives at
``grammars/qvr/vcs/parsers/<revision>/qvr.{dylib,so,dll}`` together
with its regenerated ``source/src/grammar.json`` and
``source/src/node-types.json``.

The returned `panproto.AstParserRegistry` is keyed by
revision so the same revision can be reused without recompilation
or repeated ``override_grammar`` calls.
"""

from __future__ import annotations

import ctypes
import sys
import warnings
from pathlib import Path

import panproto


_REPO_ROOT = Path(__file__).resolve().parents[3]
_PARSERS_DIR = _REPO_ROOT / "grammars" / "qvr" / "vcs" / "parsers"


def _shared_lib_extension() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


_LIB_KEEPALIVE: dict[str, ctypes.CDLL] = {}
_REGISTRIES: dict[str, object] = {}


def available_revisions() -> tuple[str, ...]:
    """Return every revision under ``parsers/`` for which a compiled
    library and its ``grammar.json`` / ``node-types.json`` exist."""
    if not _PARSERS_DIR.is_dir():
        return ()
    ext = _shared_lib_extension()
    revs: list[str] = []
    for child in sorted(_PARSERS_DIR.iterdir()):
        if not child.is_dir():
            continue
        lib = child / f"qvr{ext}"
        grammar_json = child / "source" / "src" / "grammar.json"
        node_types = child / "source" / "src" / "node-types.json"
        if lib.is_file() and grammar_json.is_file() and node_types.is_file():
            revs.append(child.name)
    return tuple(revs)


def registry_for(revision: str) -> object:
    """Return a panproto registry whose ``qvr`` grammar is the
    ``revision`` build.

    ``revision`` must match a directory under
    ``grammars/qvr/vcs/parsers/`` produced by
    `grammars.qvr.vcs.build_parsers` (e.g. ``"v0.5.0"`` or
    ``"HEAD"``).
    """
    if revision in _REGISTRIES:
        return _REGISTRIES[revision]

    rev_dir = _PARSERS_DIR / revision
    lib_path = rev_dir / f"qvr{_shared_lib_extension()}"
    if not lib_path.is_file():
        raise FileNotFoundError(
            f"no compiled parser at {lib_path}; run "
            "grammars/qvr/vcs/build_parsers.py first",
        )

    lib = ctypes.CDLL(str(lib_path))
    lib.tree_sitter_qvr.argtypes = []
    lib.tree_sitter_qvr.restype = ctypes.c_void_p
    language_ptr = lib.tree_sitter_qvr()
    if not language_ptr:
        raise RuntimeError(
            f"tree_sitter_qvr() returned NULL for {revision}",
        )

    grammar_json = (rev_dir / "source" / "src" / "grammar.json").read_bytes()
    node_types = (rev_dir / "source" / "src" / "node-types.json").read_bytes()

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

    _LIB_KEEPALIVE[revision] = lib
    _REGISTRIES[revision] = reg
    return reg


def parse(revision: str, source: bytes) -> object:
    """Parse ``source`` with ``revision``'s grammar; return the
    panproto `Schema` instance produced by tree-sitter."""
    reg = registry_for(revision)
    lens = reg.lens("qvr")  # type: ignore[attr-defined]
    return lens.parse(source)
