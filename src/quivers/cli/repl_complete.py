"""Tab-completion sources for the REPL and LSP.

Completions merge three sources:

1. Env names from the active [`quivers.cli.repl_session.ReplSession`][quivers.cli.repl_session.ReplSession]
   (objects, spaces, morphisms, rules).
2. Keywords pulled live from the QVR Pygments lexer's keyword and
   builtin tables, so adding a new grammar keyword automatically lights
   it up.
3. Meta-command names from [`quivers.cli.repl_session`][quivers.cli.repl_session].

Each completion carries an optional one-line documentation string;
prompt_toolkit and the LSP both expose this to the user.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import TYPE_CHECKING

from quivers.dsl.pygments_lexer import (
    _ALGEBRA_NAMES,
    _BUILTIN_FUNCTION_TOKENS,
    _BUILTIN_TYPE_TOKENS,
    _KEYWORD_TOKENS,
)

if TYPE_CHECKING:
    from quivers.cli.repl_session import ReplSession


@dataclass(frozen=True)
class Completion:
    """One completion candidate."""

    text: str
    kind: str  # "env", "keyword", "type", "function", "namespace", "command", "path"
    detail: str = ""


_META_COMMANDS = (
    "load",
    "reload",
    "type",
    "kind",
    "info",
    "doc",
    "browse",
    "dump",
    "edit",
    "trace",
    "set",
    "help",
    "quit",
)


def all_completions(session: "ReplSession", prefix: str) -> list[Completion]:
    """Return every candidate whose text starts with ``prefix``.

    The caller (prompt_toolkit Completer, LSP completion handler)
    decides which slice to show.
    """
    out: list[Completion] = []
    out.extend(_meta_completions(prefix))
    out.extend(_env_completions(session, prefix))
    out.extend(_keyword_completions(prefix))
    out.extend(_path_completions(prefix))
    return out


def _meta_completions(prefix: str) -> list[Completion]:
    out: list[Completion] = []
    if prefix.startswith(":"):
        p = prefix[1:]
        for name in _META_COMMANDS:
            if name.startswith(p):
                out.append(
                    Completion(text=":" + name, kind="command", detail="meta-command")
                )
    return out


def _env_completions(session: "ReplSession", prefix: str) -> list[Completion]:
    compiler = session._compiler  # noqa: SLF001 — internal but stable
    if compiler is None:
        return []
    out: list[Completion] = []
    for label, mapping in (
        ("object", compiler.objects),
        ("space", compiler.spaces),
        ("morphism", compiler.morphisms),
        ("rule", compiler.rules),
        ("program", compiler.programs),
        ("deduction", compiler.deductions),
        ("signature", compiler.signatures),
        ("encoder", compiler.encoders),
        ("decoder", compiler.decoders),
        ("loss", compiler.losses),
        ("bundle", compiler.bundles),
        ("contraction", compiler.contractions),
    ):
        for name in mapping:
            if name.startswith(prefix):
                out.append(Completion(text=name, kind="env", detail=label))
    return out


def _keyword_completions(prefix: str) -> list[Completion]:
    if not prefix:
        return []
    out: list[Completion] = []
    for kw in sorted(_KEYWORD_TOKENS):
        if kw.startswith(prefix):
            out.append(Completion(text=kw, kind="keyword", detail="keyword"))
    for fn in sorted(_BUILTIN_FUNCTION_TOKENS):
        if fn.startswith(prefix):
            out.append(Completion(text=fn, kind="function", detail="builtin"))
    for ty in sorted(_BUILTIN_TYPE_TOKENS):
        if ty.startswith(prefix):
            out.append(Completion(text=ty, kind="type", detail="builtin type"))
    for ns in sorted(_ALGEBRA_NAMES):
        if ns.startswith(prefix):
            out.append(Completion(text=ns, kind="namespace", detail="algebra"))
    return out


def _path_completions(prefix: str) -> list[Completion]:
    """File-path completions for :load."""
    if "/" not in prefix and not prefix.endswith(".qvr"):
        return []
    out: list[Completion] = []
    for match in sorted(glob.glob(prefix + "*")):
        out.append(Completion(text=match, kind="path", detail="path"))
    return out


__all__ = ["Completion", "all_completions"]
