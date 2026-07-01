"""One-hop migrator: v0.11.0 source to v0.14.0 source.

Grammar delta: v0.14.0 adds two productions to the `_draw_arg`
choice, `family_call_arg` (nested `Family(...)` at draw-arg
position, e.g. ``Mixture([0.3, 0.7], [PointMass(0), Poisson(rate)])``)
and `list_arg` (bracketed list literal at draw-arg position).
Every other rule keeps the exact shape it had under v0.11.0. No
rule was renamed, no rule was removed, no field kind changed.

Migration semantics: because the two new alternatives are pure
extensions of the source grammar, every v0.11.0 draw arg
(``bracket_index_arg``, ``identifier``, ``signed_number``) parses
under v0.14.0 without modification. Source bytes pass through
unchanged; the hop is a byte-identity.

The empty
[`SOURCE_RULE_COVERAGE`][quivers.cli.migrations.v0_11_0_to_v0_14_0.SOURCE_RULE_COVERAGE]
is the correct declaration for an extension-only hop: the
chain-coverage check reports "no removed rules to cover" against
the panproto VCS diff.
"""

from __future__ import annotations


def migrate(source: bytes) -> bytes:
    """Byte-identity migrator.

    Every draw-arg shape that parses under v0.11.0 also parses
    under v0.14.0 because the two new `_draw_arg` alternatives
    (`family_call_arg`, `list_arg`) are strict grammar extensions.
    """
    return source


SOURCE_RULE_COVERAGE: frozenset[str] = frozenset()
