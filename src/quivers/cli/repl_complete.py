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
    """Complete bindings from every populated env bucket.

    Two modes:

    * **Bare-name prefix**: ``ld`` -> ``lda`` (program), with detail
      naming the container kind. Also surfaces deep paths whose
      *final* segment matches the prefix, so typing ``thet`` suggests
      ``lda::theta``.
    * **Scope-path prefix**: ``lda::`` lists every child of the
      ``lda`` binding's scope; ``lda::z::`` walks one level deeper.
      Each candidate's text is the full ``::``-path so accepting it
      keeps the path complete.
    """
    from quivers.analysis.scope import (
        SCOPE_SEPARATOR,
        resolve_scoped_path,
        scope_children,
    )

    compiler = session._compiler  # noqa: SLF001 — internal but stable
    if compiler is None:
        return []
    out: list[Completion] = []

    # Mode B: scope-path completion. The prefix is ``a::b::`` (or
    # ``a::b::c``); enumerate the children of ``a::b``'s scope
    # whose name starts with the trailing segment.
    if SCOPE_SEPARATOR in prefix:
        head, _, tail = prefix.rpartition(SCOPE_SEPARATOR)
        if head:
            parent = resolve_scoped_path(compiler, head)
            if parent is not None:
                for child_name, child_ref in scope_children(parent).items():
                    if child_name.startswith(tail):
                        out.append(
                            Completion(
                                text=child_ref.path,
                                kind="env",
                                detail=child_ref.kind,
                            )
                        )
        return out

    # Mode A: bare-name completion. Surface top-level bindings whose
    # name starts with the prefix, *and* any deep path whose final
    # segment matches (so users find scoped bindings without
    # knowing the prefix).
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

    # Surface scope paths whose final segment matches the prefix.
    # Skips duplicates of bare-name matches.
    seen = {c.text for c in out}
    if prefix:
        from quivers.analysis.scope import ScopedRef

        for kind, mapping in (
            ("program", compiler.programs),
            ("deduction", compiler.deductions),
            ("signature", compiler.signatures),
            ("encoder", compiler.encoders),
            ("decoder", compiler.decoders),
            ("bundle", compiler.bundles),
            ("contraction", compiler.contractions),
        ):
            for top_name, top_node in mapping.items():
                top = ScopedRef(
                    name=top_name,
                    kind=kind,  # type: ignore[arg-type]
                    path=top_name,
                    parent_kind=None,
                    node=top_node,
                )
                _walk_scope_for_prefix(top, prefix, out, seen)
    return out


def _walk_scope_for_prefix(  # type: ignore[no-untyped-def]
    ref, prefix: str, out: list, seen: set
) -> None:
    """Walk ``ref``'s scope subtree; emit a completion for every
    descendant whose final-segment name starts with ``prefix``."""
    from quivers.analysis.scope import scope_children

    for child_name, child_ref in scope_children(ref).items():
        if child_name.startswith(prefix) and child_ref.path not in seen:
            out.append(
                Completion(text=child_ref.path, kind="env", detail=child_ref.kind)
            )
            seen.add(child_ref.path)
        _walk_scope_for_prefix(child_ref, prefix, out, seen)


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
