"""Locate or build the authoritative QVR tree-sitter native parser.

The Pygments lexer ([`quivers.dsl.pygments_lexer`][quivers.dsl.pygments_lexer])
tokenizes with the real tree-sitter parser rather than a regex
approximation, so it needs a loadable ``TSLanguage``. Platform wheels bundle
that parser and validate it against the packaged generated grammar before
loading it. Editable source checkouts retain a content-addressed development
build fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any


_NATIVE_MANIFEST_VERSION = 1
_NATIVE_LIBRARY_STEM = "qvr_grammar"


def _packaged_grammar_dir() -> Path:
    return Path(__file__).resolve().parent / "_grammar_data"


def _development_grammar_dir() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "grammars" / "qvr"
        if (candidate / "src" / "parser.c").is_file():
            return candidate
    return None


def _grammar_dir() -> Path:
    """Return the path to the in-tree QVR grammar directory.

    An editable checkout's generated source is authoritative during grammar
    development. An installed wheel instead uses the package-local source
    snapshot under ``_grammar_data/src``.
    """
    development = _development_grammar_dir()
    if development is not None:
        return development
    packaged = _packaged_grammar_dir()
    if (packaged / "src" / "parser.c").is_file():
        return packaged
    here = Path(__file__).resolve()
    raise FileNotFoundError(
        "the packaged QVR parser source and grammars/qvr/src/parser.c "
        f"were both unavailable from {here}; install a complete quivers "
        "wheel or use an editable checkout"
    )


def _shared_lib_extension() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform == "win32":
        return ".dll"
    return ".so"


def _compiler() -> str:
    configured = os.environ.get("CC")
    if configured:
        return configured
    candidates = (
        ("cl", "clang", "gcc", "cc")
        if sys.platform == "win32"
        else ("cc", "clang", "gcc")
    )
    compiler = next((path for name in candidates if (path := shutil.which(name))), None)
    if compiler is None:
        raise FileNotFoundError("no C compiler available for the QVR parser build")
    return compiler


def _compile_command(
    compiler: str, source_dir: Path, sources: list[Path], output: Path
) -> list[str]:
    if sys.platform == "win32" and Path(compiler).name.lower() in {"cl", "cl.exe"}:
        return [
            compiler,
            "/nologo",
            "/O2",
            "/LD",
            *shlex.split(os.environ.get("CL", ""), posix=False),
            f"/I{source_dir}",
            *(str(source) for source in sources),
            "/link",
            f"/OUT:{output}",
        ]
    flags = [
        *shlex.split(os.environ.get("CPPFLAGS", "")),
        *shlex.split(os.environ.get("CFLAGS", "")),
        *shlex.split(os.environ.get("ARCHFLAGS", "")),
        "-O2",
        "-shared",
    ]
    if sys.platform != "win32":
        flags.append("-fPIC")
    return [
        compiler,
        *flags,
        "-I",
        str(source_dir),
        *(str(source) for source in sources),
        *shlex.split(os.environ.get("LDFLAGS", "")),
        "-o",
        str(output),
    ]


def _grammar_source_files(grammar_dir: Path) -> tuple[Path, ...]:
    source_dir = grammar_dir / "src"
    parser = source_dir / "parser.c"
    if not parser.is_file():
        raise FileNotFoundError(f"QVR generated parser source is missing: {parser}")
    sources = [parser]
    scanner = source_dir / "scanner.c"
    if scanner.is_file():
        sources.append(scanner)
    for name in ("grammar.json", "node-types.json"):
        metadata = source_dir / name
        if metadata.is_file():
            sources.append(metadata)
    sources.extend(sorted((source_dir / "tree_sitter").rglob("*.h")))
    return tuple(sources)


def _grammar_source_digest(grammar_dir: Path) -> str:
    """Return a path-sensitive digest of generated parser inputs."""

    digest = hashlib.sha256()
    for source in _grammar_source_files(grammar_dir):
        digest.update(source.relative_to(grammar_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compile_shared_lib_to(grammar_dir: Path, output: Path) -> None:
    """Compile the generated parser into exactly ``output``."""

    source_files = _grammar_source_files(grammar_dir)
    sources = [source for source in source_files if source.suffix == ".c"]
    output.parent.mkdir(parents=True, exist_ok=True)
    command = _compile_command(_compiler(), grammar_dir / "src", sources, output)
    subprocess.run(command, cwd=output.parent, check=True)
    if not output.is_file():
        raise FileNotFoundError(f"QVR parser compiler did not produce {output}")


def _native_manifest_data(grammar_dir: Path, library: Path) -> dict[str, Any]:
    """Construct the manifest bound to one generated source/native pair."""

    return {
        "format": _NATIVE_MANIFEST_VERSION,
        "grammar_sha256": _grammar_source_digest(grammar_dir),
        "library_sha256": _file_digest(library),
        "library": library.name,
        "platform": sysconfig.get_platform(),
    }


def _validate_native_manifest(
    grammar_dir: Path,
    library: Path,
    manifest_path: Path,
) -> None:
    """Fail closed unless the bundled parser matches packaged generated source."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid bundled QVR parser manifest: {error}") from error
    if (
        not isinstance(payload, dict)
        or payload.get("format") != _NATIVE_MANIFEST_VERSION
    ):
        raise ValueError("unsupported bundled QVR parser manifest format")
    expected = _native_manifest_data(grammar_dir, library)
    # The wheel tag and the platform's dynamic loader enforce binary
    # compatibility. ``sysconfig.get_platform()`` reflects the interpreter's
    # build target, which may legitimately differ from a compatible wheel's
    # deployment target (notably on macOS), so it remains informational.
    for key in ("grammar_sha256", "library_sha256", "library"):
        if payload.get(key) != expected[key]:
            raise ValueError(f"bundled QVR parser manifest mismatch for {key}")


def _bundled_shared_lib() -> Path | None:
    """Return the verified wheel-native parser, or ``None`` in source trees."""

    native_dir = Path(__file__).resolve().parent / "_grammar_native"
    library = native_dir / f"{_NATIVE_LIBRARY_STEM}{_shared_lib_extension()}"
    manifest = native_dir / "manifest.json"
    if not library.exists() and not manifest.exists():
        return None
    if not library.is_file() or not manifest.is_file():
        raise FileNotFoundError("the bundled QVR parser or its manifest is missing")
    grammar_dir = _packaged_grammar_dir()
    _validate_native_manifest(grammar_dir, library, manifest)
    return library


def _parser_artifacts() -> tuple[Path, Path]:
    """Return ``(grammar_dir, native_library)`` for the current platform.

    A verified wheel-native parser always wins. Compilation is permitted only
    from an editable source checkout; an incomplete installed wheel fails
    rather than compiling on the user's machine or selecting another grammar.
    """

    bundled = _bundled_shared_lib()
    if bundled is not None:
        return _packaged_grammar_dir(), bundled
    grammar_dir = _grammar_dir()
    if grammar_dir.resolve() == _packaged_grammar_dir().resolve():
        raise FileNotFoundError(
            "this quivers installation has generated QVR grammar data but no "
            "native parser for the current platform; install a matching "
            "platform wheel"
        )
    return grammar_dir, _build_shared_lib(grammar_dir)


def _build_shared_lib(grammar_dir: Path) -> Path:
    """Compile ``parser.c`` to a content-addressed shared library."""
    cache_root = (
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "quivers"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = _grammar_source_digest(grammar_dir)
    out = cache_root / (f"qvr_grammar-{digest[:24]}{_shared_lib_extension()}")

    if out.exists():
        return out

    with tempfile.TemporaryDirectory(prefix="qvr-build-", dir=cache_root) as tmp:
        temporary = Path(tmp) / out.name
        _compile_shared_lib_to(grammar_dir, temporary)
        try:
            os.replace(temporary, out)
        except FileNotFoundError:
            # A concurrent process may have published this exact parser.
            if not out.is_file():
                raise
    return out
