"""One-hop migrator: v0.14.0 source to v0.15.0 source.

Grammar delta (panproto VCS diff): the target removes
``composition_level``, ``let_decl``, and ``vocab_literal``, and adds
``define_decl``, ``constructor_options``, and ``constructor_kwarg``.
Alongside the vertex-level changes, several anonymous tokens move:

* ``rule_decl`` conclusions use the ``|-`` turnstile instead of ``=>``.
* ``bundle N : [...]`` replaces ``bundle N = [...]``.
* ``decoder N : SIG`` replaces ``decoder N over SIG``.
* ``composition N [level=L]`` replaces ``composition N as L``.
* Top-level ``define`` replaces top-level ``let`` (recursively inside
  ``where`` blocks); program-step ``let`` is untouched.
* ``sample (a, b) <- m`` replaces ``sample [a, b] <- m``: variable
  tuples are parenthesized.
* Constructor keyword options use braces (``Real 1 {low=0.0}``)
  instead of a bracketed option block; a paren wrapper whose sole
  purpose was to bind the bracket to the constructor
  (``(Real 1 [low=0.0])``) is dropped when it wraps exactly one
  constructor-with-options, and kept otherwise (parens around an
  object expression are always valid).
* Encoder op rules gain a leading ``op`` keyword.
* The compose operators ``>=>``, ``*>``, ``~>``, ``||>``, ``?>``,
  ``&&>``, ``+>``, ``$>``, and ``%>`` are cut with no rewrite; any
  occurrence raises `MigrationError` naming the operator and its
  location.

Migration strategy: every rewrite is token-local, so the hop is a
span-edit pass rather than a full re-emit. Each top-level declaration
is parsed with the source revision's snapshot, its subtree is walked
via `SchemaView`, and per-kind collectors record byte-span edits
located through the parser's interstitial-token constraints. The
edited slice replaces the declaration's span in the original bytes,
so comments, doc comments, blank lines, and all untouched formatting
survive byte-for-byte. Each edited declaration is validated through
the target lens (`validate_decl`), and the assembled file is parsed
whole as the final grammar-binding gate.
"""

from __future__ import annotations

import re
from typing import Callable

from quivers.cli.migrations._common import (
    MigrationError,
    SchemaView,
    validate_decl,
)
from quivers.dsl._historical_grammar import registry_for


_SOURCE_REV = "v0.14.0"
# The target grammar is the working-tree surface; its parser snapshot
# lives under ``grammars/qvr/vcs/parsers/HEAD/`` until the release is
# tagged, at which point the snapshot directory gains the tag name.
_TARGET_REV = "HEAD"

_COMMENT_KINDS = frozenset({"line_comment", "doc_comment", "block_comment"})

_CUT_COMPOSE_OPS: frozenset[str] = frozenset(
    {">=>", "*>", "~>", "||>", "?>", "&&>", "+>", "$>", "%>"}
)

# One edit: replace ``source[start:end]`` with ``replacement`` (a
# zero-width span inserts).
_Edit = tuple[int, int, str]

_EditCollector = Callable[[SchemaView, str, list[_Edit]], None]


def _line_col(source: bytes, pos: int) -> tuple[int, int]:
    prefix = source[:pos]
    line = prefix.count(b"\n") + 1
    col = pos - (prefix.rfind(b"\n") + 1) + 1
    return line, col


def _located(view: SchemaView, vid: str, message: str) -> MigrationError:
    start, _end = view.span(vid)
    line, col = _line_col(view.source, start)
    return MigrationError(f"line {line}, column {col}: {message}")


def _token_edit(
    view: SchemaView,
    vid: str,
    token: str,
    replacement: str,
    lo: int,
    hi: int,
) -> _Edit:
    """Locate ``token`` among ``vid``'s interstitial runs within the
    byte window ``[lo, hi)`` and return the edit replacing it. The
    interstitial constraints carry exactly the anonymous tokens of
    the vertex's own production, so the window plus a word-boundary
    search pins the occurrence unambiguously."""
    pattern = re.compile(
        rf"\b{re.escape(token)}\b" if token.isalpha() else re.escape(token)
    )
    for start, text in view.interstitials(vid):
        for m in pattern.finditer(text):
            pos = start + m.start()
            if lo <= pos < hi:
                return (pos, pos + len(token), replacement)
    raise _located(
        view,
        vid,
        f"{view.kind(vid)}: expected token {token!r} between bytes "
        f"{lo} and {hi}; the source may not be valid {_SOURCE_REV} surface",
    )


# ---------------------------------------------------------------------------
# Per-kind edit collectors
# ---------------------------------------------------------------------------


def _collect_rule_decl(view: SchemaView, vid: str, edits: list[_Edit]) -> None:
    """``rule N(vars) : premises => conclusion`` gains the ``|-``
    turnstile in place of ``=>``."""
    premises = view.fields(vid, "premises")
    conclusion = view.field(vid, "conclusion")
    if not premises or conclusion is None:
        raise _located(view, vid, "rule declaration missing premises or conclusion")
    lo = max(view.span(p)[1] for p in premises)
    hi, _ = view.span(conclusion)
    edits.append(_token_edit(view, vid, "=>", "|-", lo, hi))


def _collect_bundle_decl(view: SchemaView, vid: str, edits: list[_Edit]) -> None:
    """``bundle N = [...]`` becomes ``bundle N : [...]``."""
    name = view.field(vid, "name")
    if name is None:
        raise _located(view, vid, "bundle declaration missing name")
    _, name_end = view.span(name)
    _, decl_end = view.span(vid)
    edits.append(_token_edit(view, vid, "=", ":", name_end, decl_end))


def _collect_decoder_decl(view: SchemaView, vid: str, edits: list[_Edit]) -> None:
    """``decoder N over SIG`` becomes ``decoder N : SIG``."""
    name = view.field(vid, "name")
    signature = view.field(vid, "signature")
    if name is None or signature is None:
        raise _located(view, vid, "decoder declaration missing name or signature")
    _, name_end = view.span(name)
    sig_start, _ = view.span(signature)
    edits.append(_token_edit(view, vid, "over", ":", name_end, sig_start))


def _collect_composition_decl(view: SchemaView, vid: str, edits: list[_Edit]) -> None:
    """``composition N as LEVEL`` becomes ``composition N
    [level=LEVEL]``: the ``as`` clause turns into an option block on
    the declaration header."""
    level = view.field(vid, "level")
    if level is None:
        return
    name = view.field(vid, "name")
    if name is None:
        raise _located(view, vid, "composition declaration missing name")
    _, name_end = view.span(name)
    level_start, level_end = view.span(level)
    as_start, _as_end, _ = _token_edit(view, vid, "as", "", name_end, level_start)
    edits.append((as_start, level_end, f"[level={view.text(level)}]"))


def _collect_let_decl(view: SchemaView, vid: str, edits: list[_Edit]) -> None:
    """Top-level ``let`` becomes ``define``. Nested ``where``-block
    bindings are separate ``let_decl`` vertices, so the subtree walk
    rewrites each one; program-step ``let`` is a different vertex
    kind (``let_step``) and never reaches this collector."""
    name = view.field(vid, "name")
    if name is None:
        raise _located(view, vid, "let declaration missing name")
    name_start, _ = view.span(name)
    keyword = re.compile(r"\blet\b")
    best: int | None = None
    for start, text in view.interstitials(vid):
        for m in keyword.finditer(text):
            pos = start + m.start()
            if pos < name_start and (best is None or pos > best):
                best = pos
    if best is None:
        raise _located(view, vid, "let declaration: 'let' keyword not found")
    edits.append((best, best + len("let"), "define"))


def _collect_var_tuple(view: SchemaView, vid: str, edits: list[_Edit]) -> None:
    """``[a, b]`` variable tuples become ``(a, b)``."""
    start, end = view.span(vid)
    if view.source[start : start + 1] != b"[" or view.source[end - 1 : end] != b"]":
        raise _located(view, vid, "variable tuple is not bracket-delimited")
    edits.append((start, start + 1, "("))
    edits.append((end - 1, end, ")"))


# The constructor keyword parameters. Every other key on an option
# block the source parser attached to a constructor belongs to the
# enclosing declaration: the source grammar's GLR conflict let a
# declaration's trailing ``[...]`` bind to a constructor codomain,
# so attachment alone does not settle intent; the key set does.
_CONSTRUCTOR_KWARG_KEYS = frozenset({"low", "high"})


def _option_block_keys(view: SchemaView, options_vid: str) -> list[str]:
    """The keys of every ``option_entry`` in an option block. A
    valueless (flag) entry contributes its bare key."""
    keys: list[str] = []
    for entry_vid in view.outgoing_vids(options_vid):
        if view.kind(entry_vid) != "option_entry":
            continue
        key_vid = view.field(entry_vid, "key")
        if key_vid is not None:
            keys.append(view.text(key_vid))
    return keys


def _constructor_options_to_braces(view: SchemaView, ctor_vid: str) -> bool:
    """Whether the option block the source parser attached to
    ``ctor_vid`` is a genuine constructor-kwarg block (rewritten to
    braces) rather than a mis-attached declaration block (left in
    place so the target grammar rebinds it to the declaration).
    Raises on a mixed block: no single attachment honors it."""
    options = view.field(ctor_vid, "options")
    if options is None:
        return False
    keys = _option_block_keys(view, options)
    ctor_keys = [k for k in keys if k in _CONSTRUCTOR_KWARG_KEYS]
    if not ctor_keys:
        return False
    if len(ctor_keys) < len(keys):
        raise _located(
            view,
            options,
            f"option block mixes constructor keys {sorted(ctor_keys)!r} with "
            f"declaration keys; split it into a brace block on the "
            f"constructor and a bracket block on the declaration",
        )
    return True


def _collect_continuous_constructor(
    view: SchemaView,
    vid: str,
    edits: list[_Edit],
) -> None:
    """A constructor's keyword-option block moves from ``[k=v, ...]``
    to braces: ``Real 1 {low=0.0, high=1.0}``. A block carrying
    declaration keys stays byte-identical: the target grammar always
    binds a trailing ``[...]`` to the enclosing declaration."""
    if not _constructor_options_to_braces(view, vid):
        return
    options = view.field(vid, "options")
    if options is None:
        return
    start, end = view.span(options)
    if view.source[start : start + 1] != b"[" or view.source[end - 1 : end] != b"]":
        raise _located(view, options, "constructor option block is not bracketed")
    edits.append((start, start + 1, "{"))
    edits.append((end - 1, end, "}"))


def _collect_object_paren(view: SchemaView, vid: str, edits: list[_Edit]) -> None:
    """Drop a paren wrapper whose sole content is one constructor
    whose option block becomes braces: with braces, the options can
    no longer leak to the enclosing declaration, and the brace block
    seals the constructor against absorbing following tokens, so the
    wrapper does nothing. Any other paren shape is kept: parens
    around an object expression are always valid."""
    kids = [
        k for k in view.outgoing_vids(vid) if view.kind(k) not in _COMMENT_KINDS
    ]
    if len(kids) != 1:
        return
    inner = kids[0]
    if view.kind(inner) != "continuous_constructor":
        return
    if not _constructor_options_to_braces(view, inner):
        return
    start, end = view.span(vid)
    if view.source[start : start + 1] != b"(" or view.source[end - 1 : end] != b")":
        return
    edits.append((start, start + 1, ""))
    edits.append((end - 1, end, ""))


def _collect_encoder_op_rule(view: SchemaView, vid: str, edits: list[_Edit]) -> None:
    """Encoder op rules gain the leading ``op`` keyword."""
    op = view.field(vid, "op")
    if op is None:
        raise _located(view, vid, "encoder op rule missing op name")
    start, _ = view.span(op)
    edits.append((start, start, "op "))


def _collect_compose_expr(view: SchemaView, vid: str, edits: list[_Edit]) -> None:
    """The cut compose operators admit no rewrite; raise naming the
    operator and its location. ``>>`` and ``<<`` pass through."""
    del edits
    op = view.consts(vid).get("field:op")
    if op is None or op not in _CUT_COMPOSE_OPS:
        return
    pos, _ = view.span(vid)
    for start, text in view.interstitials(vid):
        idx = text.find(op)
        if idx >= 0:
            pos = start + idx
            break
    line, col = _line_col(view.source, pos)
    raise MigrationError(
        f"line {line}, column {col}: compose operator {op!r} was removed "
        f"from the surface with no rewrite; express algebra-tagged "
        f"composition via .change_base(...) or a composition declaration"
    )


_EDIT_COLLECTORS: dict[str, _EditCollector] = {
    "rule_decl": _collect_rule_decl,
    "bundle_decl": _collect_bundle_decl,
    "decoder_decl": _collect_decoder_decl,
    "composition_decl": _collect_composition_decl,
    "let_decl": _collect_let_decl,
    "var_tuple": _collect_var_tuple,
    "continuous_constructor": _collect_continuous_constructor,
    "object_paren": _collect_object_paren,
    "encoder_op_rule": _collect_encoder_op_rule,
    "compose_expr": _collect_compose_expr,
}


# ---------------------------------------------------------------------------
# Parse gates
# ---------------------------------------------------------------------------


def _gate_parse(view: SchemaView, rev: str, message: str) -> None:
    """Reject a parse containing an ERROR vertex anywhere (the
    parser does not always attach recovery vertices under the root)
    or a zero-width vertex (tree-sitter's MISSING recovery inserts
    empty tokens instead of ERROR nodes, e.g. for a constructor
    without its size argument)."""
    for vertex in view.schema.vertices:
        if vertex.kind == "ERROR":
            raise _located(
                view,
                vertex.id,
                f"{message} {rev}; fix the input before migrating",
            )
        consts = view.consts(vertex.id)
        start = consts.get("start-byte")
        if (
            start is not None
            and start == consts.get("end-byte")
            and vertex.id != "parse_emit_lens"
        ):
            raise _located(
                view,
                vertex.id,
                f"{message} {rev} (missing {vertex.kind!r} token); "
                "fix the input before migrating",
            )


# ---------------------------------------------------------------------------
# Declaration conversion: subtree walk + span edits
# ---------------------------------------------------------------------------


def _subtree_vids(view: SchemaView, root_vid: str) -> list[str]:
    out: list[str] = []
    stack = [root_vid]
    while stack:
        vid = stack.pop()
        out.append(vid)
        stack.extend(view.outgoing_vids(vid))
    return out


def _apply_edits(source: bytes, lo: int, hi: int, edits: list[_Edit]) -> str:
    """Apply ``edits`` to ``source[lo:hi]`` and return the result as
    text. Edits must be non-overlapping; a collision means two
    collectors claimed the same bytes, which is an internal bug worth
    a loud failure rather than silent corruption."""
    ordered = sorted(edits, key=lambda edit: (edit[0], edit[1]))
    parts: list[bytes] = []
    cursor = lo
    for start, end, replacement in ordered:
        if start < cursor:
            raise MigrationError(
                f"internal error: overlapping edits at byte {start}",
            )
        parts.append(source[cursor:start])
        parts.append(replacement.encode("utf-8"))
        cursor = end
    parts.append(source[cursor:hi])
    return b"".join(parts).decode("utf-8")


def _convert_decl(view: SchemaView, decl_vid: str) -> str:
    """Collect and apply every span edit under one top-level
    declaration; return the migrated declaration text (trailing
    newline included)."""
    edits: list[_Edit] = []
    for vid in _subtree_vids(view, decl_vid):
        kind = view.kind(vid)
        if kind == "ERROR":
            raise _located(
                view,
                vid,
                f"source does not parse under {_SOURCE_REV}; "
                "fix the input before migrating",
            )
        collector = _EDIT_COLLECTORS.get(kind)
        if collector is not None:
            collector(view, vid, edits)
    lo, hi = view.span(decl_vid)
    text = _apply_edits(view.source, lo, hi, edits)
    if not text.endswith("\n"):
        text += "\n"
    return text


# Every top-level statement kind of the source grammar dispatches to
# the shared subtree converter: the rewrites are construct-local
# (turnstiles, keywords, brackets), not declaration-shape-local, so
# one walk covers declarations of every kind. Kinds with no matching
# constructs come back byte-identical.
_DECL_CONVERTERS: dict[str, Callable[[SchemaView, str], str]] = {
    kind: _convert_decl
    for kind in (
        "composition_decl",
        "category_decl",
        "rule_decl",
        "schema_decl",
        "object_decl",
        "morphism_decl",
        "bundle_decl",
        "program_decl",
        "contraction_decl",
        "let_decl",
        "export_decl",
        "deduction_decl",
        "signature_decl",
        "encoder_decl",
        "decoder_decl",
        "loss_decl",
        "pragma_outer",
        "pragma_inner",
    )
}


# Source-side rule names this hop semantically translates: the
# vertex kinds carrying an edit collector, plus the rules consumed
# through them. Consumed by the chain-coverage check to verify that
# every rule removed in the panproto schema diff between this hop's
# source and target has a handler here.
SOURCE_RULE_COVERAGE: frozenset[str] = frozenset(_EDIT_COLLECTORS.keys()) | frozenset(
    {
        # Read by the composition_decl collector: the level literal
        # moves into the header option block as ``[level=...]``.
        "composition_level",
        # Declared but unreferenced in the source grammar: no rule's
        # RHS produces it, so no source text can reach it. Its
        # removal at the target needs no converter.
        "vocab_literal",
    }
)


def migrate(source: bytes) -> bytes:
    """Migrate one file's bytes from the v0.14.0 surface to the
    v0.15.0 surface.

    Parses with the source snapshot, rewrites each top-level
    declaration by local span edits (validating each through the
    target lens), and splices the results back over the original
    byte spans, so everything between declarations (blank lines,
    free-standing comments) survives untouched. The assembled file
    is parsed whole through the target lens as the final gate."""
    src_lens = registry_for(_SOURCE_REV).lens("qvr")
    schema = src_lens.parse(source)
    view = SchemaView(schema, source)
    _gate_parse(view, _SOURCE_REV, "source does not parse under")

    replacements: list[_Edit] = []
    for decl_vid in view.top_level_decls():
        kind = view.kind(decl_vid)
        if kind in _COMMENT_KINDS:
            continue
        converter = _DECL_CONVERTERS.get(kind)
        if converter is None:
            raise _located(
                view,
                decl_vid,
                f"no converter for top-level declaration kind {kind!r}",
            )
        text = converter(view, decl_vid)
        validate_decl(_TARGET_REV, text)
        lo, hi = view.span(decl_vid)
        replacements.append((lo, hi, text))

    result = _apply_edits(source, 0, len(source), replacements).encode("utf-8")

    tgt_lens = registry_for(_TARGET_REV).lens("qvr")
    final_view = SchemaView(tgt_lens.parse(result), result)
    _gate_parse(
        final_view,
        _TARGET_REV,
        "assembled migration output does not parse under",
    )
    return result
