"""Source-level migration table for QVR's grammar evolution.

The table records, for every pair of adjacent grammar revisions in
the in-tree panproto VCS chain, the surface rewrites that lower a
``.qvr`` file written at the older grammar to one parseable at the
newer grammar. Rewrites are regex-based; each rule names the
grammar move it implements so the test suite and ``qvr migrate``'s
audit trail can refer to it by symbol.

The composite migration from any FROM revision to any TO revision
is the *categorical composition* of the per-adjacent-pair rule
lists, applied in revision order. ``qvr migrate`` walks the VCS
chain to determine the in-order list, concatenates the rule lists,
and applies them to each input file.

The rewrites here are intentionally textual, not AST-level: every
``.qvr`` source under quivers is a small Python-style file, the
surface-level rewrites are local and unambiguous, and the cost of
running them through an AST visitor (one bespoke parser per
historical grammar) is wildly disproportionate to the size of the
problem. The downside is that pathological cases (string literals
containing keyword-shaped substrings, unusual whitespace) need
manual review; the standard quivers code style does not exercise
any of them.
"""

from __future__ import annotations

import re
from collections.abc import Callable


class Rewrite:
    """One surface rewrite rule from a historical grammar to its
    successor.

    Not a :class:`didactic.api.Model` because the ``replacement``
    field is a :class:`Callable`, which the didactic schema
    translator cannot lift into a panproto sort. The rewrite list
    is internal compiler data, not a serialised structure, so a
    plain immutable class with typed attributes is the right fit.

    Attributes
    ----------
    name : str
        Mnemonic identifier for diagnostics and tests.
    pattern : str
        Regex pattern; applied with :data:`re.MULTILINE` so ``^``
        and ``$`` anchor per-line.
    replacement : str or Callable[[re.Match], str]
        Replacement string (with the usual ``\\1`` / ``\\2``
        backrefs) or a callable taking the match object and
        returning the replacement.
    description : str
        Short human-readable description; surfaced by
        ``qvr migrate --explain`` (and the test suite).
    """

    __slots__ = ("name", "pattern", "replacement", "description")

    name: str
    pattern: str
    replacement: str | Callable[[re.Match[str]], str]
    description: str

    def __init__(
        self,
        *,
        name: str,
        pattern: str,
        replacement: str | Callable[[re.Match[str]], str],
        description: str,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "pattern", pattern)
        object.__setattr__(self, "replacement", replacement)
        object.__setattr__(self, "description", description)

    def __setattr__(self, key: str, value: object) -> None:
        raise AttributeError(
            f"Rewrite is immutable; cannot reassign {key!r}",
        )


def apply_rewrites(text: str, rules: tuple[Rewrite, ...]) -> str:
    """Apply each rewrite to ``text`` in order, returning the
    final string."""
    for rule in rules:
        text = re.sub(
            rule.pattern,
            rule.replacement,
            text,
            flags=re.MULTILINE,
        )
    return text


# ---------------------------------------------------------------------------
# v0.2.0 -> v0.3.0: type-expression namespace rename + new declaration forms
# ---------------------------------------------------------------------------
#
# v0.2.0 used a ``cat`` namespace for category-typed declarations
# (``cat_atom``, ``cat_paren``, ``cat_product``, ``cat_slash``).
# v0.3.0 unified these into the type-expression vocabulary
# (``type_atom``, ``type_paren``, ``type_product``, ``type_slash``)
# and added the alias / bundle / schema decls + free-residuated /
# free-monoid object initializers. None of the additions break the
# v0.2.0 surface; only the ``cat`` namespace itself disappears, and
# no v0.2.0 code in this repo declared anything in that namespace.
_V0_2_0_TO_V0_3_0: tuple[Rewrite, ...] = ()


# ---------------------------------------------------------------------------
# v0.3.0 -> v0.4.0: output -> export, draw_step keyword rationalization
# ---------------------------------------------------------------------------
#
# * ``output X`` becomes ``export X`` (the v0.4.0 surface generalised
#   single-output to multi-export).
# * ``arrow_draw_step`` / ``draw_step`` collapse into the unified
#   ``bind_step`` shape; the surface keyword is unchanged for the
#   common ``x <- F(args)`` form, but optional explicit ``draw``
#   prefixes are dropped.
_V0_3_0_TO_V0_4_0: tuple[Rewrite, ...] = (
    Rewrite(
        name="output_to_export",
        pattern=r"^(\s*)output\b",
        replacement=r"\1export",
        description="rename the single-output decl keyword to export",
    ),
    Rewrite(
        name="drop_draw_keyword",
        pattern=r"^(\s*)draw\s+([A-Za-z_][A-Za-z0-9_]*\s*<-)",
        replacement=r"\1\2",
        description=(
            "drop the optional ``draw`` prefix on bind steps; the "
            "v0.4.0 surface accepts the bare ``x <- F(args)`` form"
        ),
    ),
)


# ---------------------------------------------------------------------------
# v0.4.0 -> v0.5.0: continuous/stochastic split collapses into kernel
# ---------------------------------------------------------------------------
#
# * ``continuous f : A -> B ~ Family(args)`` and
#   ``stochastic f : A -> B`` both fold into ``kernel f : A -> B
#   [~ Family(args)]`` (parametric or lookup-table kernel).
_V0_4_0_TO_V0_5_0: tuple[Rewrite, ...] = (
    Rewrite(
        name="continuous_to_kernel",
        pattern=r"^(\s*)continuous\b",
        replacement=r"\1kernel",
        description=(
            "rename the parametric-kernel keyword from "
            "``continuous`` to ``kernel``"
        ),
    ),
    Rewrite(
        name="stochastic_to_kernel",
        pattern=r"^(\s*)stochastic\b",
        replacement=r"\1kernel",
        description=(
            "rename the lookup-table-kernel keyword from "
            "``stochastic`` to ``kernel``"
        ),
    ),
)


# ---------------------------------------------------------------------------
# v0.5.0 -> v0.6.0: pure additions (let-factor expressions)
# ---------------------------------------------------------------------------
_V0_5_0_TO_V0_6_0: tuple[Rewrite, ...] = ()


# ---------------------------------------------------------------------------
# v0.6.0 -> v0.7.0: ``quantale`` -> ``algebra`` rename
# ---------------------------------------------------------------------------
_V0_6_0_TO_V0_7_0: tuple[Rewrite, ...] = (
    Rewrite(
        name="quantale_to_algebra",
        pattern=r"^(\s*)quantale\b",
        replacement=r"\1algebra",
        description=(
            "rename the ``quantale`` declaration keyword to "
            "``algebra``; the underlying mathematical structure is "
            "unchanged (still a complete idempotent semiring)"
        ),
    ),
)


# ---------------------------------------------------------------------------
# v0.7.0 -> v0.9.0: pure additions (einsum-style contraction wiring)
# ---------------------------------------------------------------------------
_V0_7_0_TO_V0_9_0: tuple[Rewrite, ...] = ()


# ---------------------------------------------------------------------------
# v0.9.0 -> HEAD: the homogenization (twelve coordinated moves)
# ---------------------------------------------------------------------------
#
# Every declaration adopts ``KEYWORD NAME(params) : SIG [options]``.
# Twelve renames + structural moves:
#
#   1. ``algebra X``           -> ``composition X at algebra``
#   2. ``semigroupoid X``      -> ``composition X at semigroupoid``
#   3. ``bilinear_form X``     -> ``composition X at bilinear_form``
#   4. ``composition_rule X``  -> ``composition X at rule``
#   5. ``object N : ...``      -> ``type N : ...``
#   6. ``object N = init``     -> ``type N : init``
#   7. ``space S : ...``       -> ``type S : ...``
#   8. ``alias A = ...``       -> ``type A : ...``
#   9. ``type A = ...`` (alias) -> ``type A : ...``
#  10. ``latent f : A -> B``   -> ``morphism f : A -> B [role=latent]``
#  11. ``observed f : A -> B = E``
#                              -> ``morphism f : A -> B [role=observed] ~ E``
#  12. ``kernel f : A -> B ~ F(...)`` (parametric)
#                              -> ``morphism f : A -> B [role=kernel] ~ F(...)``
#  13. ``kernel f : A -> B`` (lookup, no ~)
#                              -> ``morphism f : A -> B [role=kernel]``
#  14. ``embed e : A -> B``    -> ``morphism e : A -> B [role=embed]``
#  15. ``discretize d : SP bins N``
#                              -> ``morphism d : SP -> _Bins_N [role=discretize, bins=N]``
#  16. ``program p : A -> B ! Effects`` (decl-line effects)
#                              -> ``program p : A -> B [effects=[Effects]]``
#  17. Bare draw steps ``name <- F(args)`` inside a program body
#       get a leading ``sample`` keyword.
#
# Each transform is anchored at line-start so trailing-content rules
# (option blocks, ``~ init``) are not corrupted on re-application.


def _wrap_program_effects(match: re.Match[str]) -> str:
    """``program ... ! E1, E2`` ->
    ``program ... [effects=[E1, E2]]:``.

    The trailing colon is mandatory at HEAD: program declarations
    open their indented body with ``:`` rather than a bare newline,
    so the rewrite both wraps the effect list in the option block
    and threads the colon onto the end of the signature line.
    """
    signature = match.group(1)
    raw_effects = match.group(2).strip()
    parts = [p.strip() for p in raw_effects.split(",") if p.strip()]
    return f"{signature} [effects=[" + ", ".join(parts) + "]]:"


def _discretize_to_morphism(match: re.Match[str]) -> str:
    """``discretize d : SP bins N`` ->
    ``morphism d : SP -> _Bins_N [role=discretize, bins=N]``.

    The synthesised codomain ``_Bins_N`` names the implicit
    discrete fibration; downstream code that needs the actual
    codomain object should declare ``type _Bins_N : N`` explicitly.
    """
    indent = match.group(1)
    name = match.group(2)
    space = match.group(3)
    nbins = match.group(4)
    return (
        f"{indent}morphism {name} : {space} -> _Bins_{nbins} "
        f"[role=discretize, bins={nbins}]"
    )


_V0_9_0_TO_HEAD: tuple[Rewrite, ...] = (
    # Composition keyword tower -> unified ``composition NAME at LEVEL``.
    Rewrite(
        name="algebra_to_composition",
        pattern=r"^(\s*)algebra\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        replacement=r"\1composition \2 at algebra",
        description="algebra X -> composition X at algebra",
    ),
    Rewrite(
        name="semigroupoid_to_composition",
        pattern=r"^(\s*)semigroupoid\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        replacement=r"\1composition \2 at semigroupoid",
        description="semigroupoid X -> composition X at semigroupoid",
    ),
    Rewrite(
        name="bilinear_form_to_composition",
        pattern=r"^(\s*)bilinear_form\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        replacement=r"\1composition \2 at bilinear_form",
        description="bilinear_form X -> composition X at bilinear_form",
    ),
    Rewrite(
        name="composition_rule_to_composition",
        pattern=r"^(\s*)composition_rule\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        replacement=r"\1composition \2 at rule",
        description="composition_rule X -> composition X at rule",
    ),
    # Type-decl tower: object/space/alias/type-alias -> type.
    # Order matters: ``object X = init`` is matched before
    # ``object X : T`` so the ``=`` form (enum / free-residuated /
    # free-monoid initializer) lands as ``type X : init`` and not
    # ``type X = init``.
    Rewrite(
        name="object_init_to_type",
        pattern=r"^(\s*)object\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*",
        replacement=r"\1type \2 : ",
        description="object X = init -> type X : init",
    ),
    Rewrite(
        name="object_typed_to_type",
        pattern=r"^(\s*)object\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*",
        replacement=r"\1type \2 : ",
        description="object X : T -> type X : T",
    ),
    Rewrite(
        name="space_to_type",
        pattern=r"^(\s*)space\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*",
        replacement=r"\1type \2 : ",
        description="space S : E -> type S : E",
    ),
    Rewrite(
        name="alias_to_type",
        pattern=r"^(\s*)alias\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*",
        replacement=r"\1type \2 : ",
        description="alias A = E -> type A : E",
    ),
    Rewrite(
        name="type_alias_assign_to_type_colon",
        # The prior surface accepted ``type A = E`` as a type-alias
        # form; HEAD requires ``type A : E``. We must NOT match the
        # already-correct ``type A : E`` rows, and we must NOT match
        # ``let X = E``.
        pattern=(
            r"^(\s*)type\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
        ),
        replacement=r"\1type \2 : ",
        description="type A = E -> type A : E (type-alias form)",
    ),
    # Morphism keyword tower -> ``morphism ... [role=...]``.
    Rewrite(
        name="latent_to_morphism",
        pattern=(
            r"^(\s*)latent\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
            r"(.+?)\s*->\s*(\S+(?:\s*\*\s*\S+)*)\s*$"
        ),
        replacement=r"\1morphism \2 : \3 -> \4 [role=latent]",
        description="latent f : A -> B -> morphism f : A -> B [role=latent]",
    ),
    Rewrite(
        name="observed_init_to_morphism",
        pattern=(
            r"^(\s*)observed\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
            r"(.+?)\s*->\s*(.+?)\s*=\s*(.+)$"
        ),
        replacement=(
            r"\1morphism \2 : \3 -> \4 [role=observed] ~ \5"
        ),
        description=(
            "observed f : A -> B = expr -> "
            "morphism f : A -> B [role=observed] ~ expr"
        ),
    ),
    Rewrite(
        name="observed_bare_to_morphism",
        pattern=(
            r"^(\s*)observed\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
            r"(.+?)\s*->\s*(.+?)\s*$"
        ),
        replacement=r"\1morphism \2 : \3 -> \4 [role=observed]",
        description="observed f : A -> B -> morphism f : A -> B [role=observed]",
    ),
    Rewrite(
        name="kernel_with_init_to_morphism",
        pattern=(
            r"^(\s*)kernel\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
            r"(.+?)\s*->\s*(.+?)\s*~\s*(.+)$"
        ),
        replacement=(
            r"\1morphism \2 : \3 -> \4 [role=kernel] ~ \5"
        ),
        description=(
            "kernel f : A -> B ~ F(args) -> "
            "morphism f : A -> B [role=kernel] ~ F(args)"
        ),
    ),
    Rewrite(
        name="kernel_bare_to_morphism",
        pattern=(
            r"^(\s*)kernel\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
            r"(.+?)\s*->\s*(.+?)\s*$"
        ),
        replacement=r"\1morphism \2 : \3 -> \4 [role=kernel]",
        description=(
            "kernel f : A -> B (lookup-table) -> "
            "morphism f : A -> B [role=kernel]"
        ),
    ),
    Rewrite(
        name="embed_to_morphism",
        pattern=(
            r"^(\s*)embed\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
            r"(.+?)\s*->\s*(.+?)\s*$"
        ),
        replacement=r"\1morphism \2 : \3 -> \4 [role=embed]",
        description=(
            "embed e : A -> B -> morphism e : A -> B [role=embed]"
        ),
    ),
    Rewrite(
        name="discretize_to_morphism",
        pattern=(
            r"^(\s*)discretize\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
            r"(\S+)\s+bins\s+(\d+)\s*$"
        ),
        replacement=_discretize_to_morphism,
        description=(
            "discretize d : SP bins N -> "
            "morphism d : SP -> _Bins_N [role=discretize, bins=N]"
        ),
    ),
    # Program-decl effects clause: ``! E1, E2`` -> ``[effects=[E1, E2]]:``.
    # Must run before ``program_decl_trailing_colon`` so the latter
    # does not double-up the colon on programs that carried effects.
    Rewrite(
        name="program_effects_clause",
        pattern=(
            r"(^\s*program\s+[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\([^)]*\))?\s*:\s*\S+(?:\s*\*\s*\S+)*"
            r"\s*->\s*\S+(?:\s*\*\s*\S+)*)\s*!\s*([^\n]+?)\s*$"
        ),
        replacement=_wrap_program_effects,
        description=(
            "program p : A -> B ! E1, E2 -> "
            "program p : A -> B [effects=[E1, E2]]:"
        ),
    ),
    # Programs without an effects clause need the trailing ``:``
    # added: HEAD's program rule mandates ``program ... :`` before
    # the indented body. Match only signatures that do NOT already
    # end with ``:`` (covers the effects-already-rewritten case).
    Rewrite(
        name="program_decl_trailing_colon",
        pattern=(
            r"^(\s*program\s+[A-Za-z_][A-Za-z0-9_]*"
            r"(?:\([^)]*\))?\s*:\s*\S+(?:\s*\*\s*\S+)*"
            r"\s*->\s*\S+(?:\s*\*\s*\S+)*"
            r"(?:\s*\[[^\]]*\])?)\s*$"
        ),
        replacement=r"\1:",
        description=(
            "append the mandatory ``:`` to a program declaration "
            "signature line when it is missing"
        ),
    ),
    # Bare draw step inside a program body -> ``sample x <- F(args)``.
    # Indent-anchored to avoid matching let-arith RHS expressions
    # that happen to contain ``<-`` substrings (none of which exist
    # in the prior surface).
    Rewrite(
        name="bare_draw_to_sample",
        pattern=(
            r"^(\s{4,})([A-Za-z_][A-Za-z0-9_]*)(\s*<-\s*)"
        ),
        replacement=r"\1sample \2\3",
        description=(
            "bare ``x <- F(args)`` program step gets a leading "
            "``sample`` keyword"
        ),
    ),
    # Indexed bare draw: ``var : Index <- F(args)`` -> ``sample var : Index <- F(args)``.
    Rewrite(
        name="bare_indexed_draw_to_sample",
        pattern=(
            r"^(\s{4,})([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"
            r"([A-Za-z_][A-Za-z0-9_]*)\s*<-\s*"
        ),
        replacement=r"\1sample \2 : \3 <- ",
        description=(
            "indexed bare draw ``var : Index <- F(args)`` gets a "
            "leading ``sample`` keyword"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Pairwise table + composite-chain helper
# ---------------------------------------------------------------------------


MIGRATIONS: dict[tuple[str, str], tuple[Rewrite, ...]] = {
    ("v0.2.0", "v0.3.0"): _V0_2_0_TO_V0_3_0,
    ("v0.3.0", "v0.4.0"): _V0_3_0_TO_V0_4_0,
    ("v0.4.0", "v0.5.0"): _V0_4_0_TO_V0_5_0,
    ("v0.5.0", "v0.6.0"): _V0_5_0_TO_V0_6_0,
    ("v0.6.0", "v0.7.0"): _V0_6_0_TO_V0_7_0,
    ("v0.7.0", "v0.9.0"): _V0_7_0_TO_V0_9_0,
    ("v0.9.0", "HEAD"): _V0_9_0_TO_HEAD,
}


# Linear chain of recorded grammar revisions, oldest-first. Matches
# the order of tags in the panproto VCS at grammars/qvr/vcs/.
CHAIN_ORDER: tuple[str, ...] = (
    "v0.2.0",
    "v0.3.0",
    "v0.4.0",
    "v0.5.0",
    "v0.6.0",
    "v0.7.0",
    "v0.9.0",
    "HEAD",
)


class MigrationLookupError(Exception):
    """Raised when no path through the chain connects FROM to TO."""


def composite_rewrites(from_ref: str, to_ref: str) -> tuple[Rewrite, ...]:
    """Concatenate the per-adjacent-pair rule lists between two
    revisions in :data:`CHAIN_ORDER`.

    The result is the migration that, applied to a ``.qvr`` file
    valid under FROM, produces a file valid under TO. FROM == TO
    yields the empty rule list (identity migration). FROM > TO
    (backward migration) is not yet wired; the lookup raises
    :class:`MigrationLookupError` rather than guessing an inverse.
    """
    if from_ref not in CHAIN_ORDER:
        raise MigrationLookupError(
            f"FROM revision {from_ref!r} is not in CHAIN_ORDER",
        )
    if to_ref not in CHAIN_ORDER:
        raise MigrationLookupError(
            f"TO revision {to_ref!r} is not in CHAIN_ORDER",
        )
    src_idx = CHAIN_ORDER.index(from_ref)
    tgt_idx = CHAIN_ORDER.index(to_ref)
    if tgt_idx < src_idx:
        raise MigrationLookupError(
            f"backward migration {from_ref} -> {to_ref} is not "
            "wired; the table is forward-only",
        )
    if tgt_idx == src_idx:
        return ()
    composite: list[Rewrite] = []
    for src, tgt in zip(
        CHAIN_ORDER[src_idx:tgt_idx],
        CHAIN_ORDER[src_idx + 1:tgt_idx + 1],
    ):
        composite.extend(MIGRATIONS[(src, tgt)])
    return tuple(composite)


__all__ = [
    "CHAIN_ORDER",
    "MIGRATIONS",
    "MigrationLookupError",
    "Rewrite",
    "apply_rewrites",
    "composite_rewrites",
]
