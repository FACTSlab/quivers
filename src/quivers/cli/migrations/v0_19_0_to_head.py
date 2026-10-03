"""One-hop migrator from the released v0.19 surface to QVR HEAD.

QVR HEAD accepts hanging-indented argument and list forms plus named draw
arguments. Categorical draw applications now state their parameterization, so
this hop rewrites the former positional probability form to ``probs=``. Other
source remains byte-preserving.
"""

from __future__ import annotations

from quivers.cli.migrations._common import SchemaView, parse_validated_source


_SOURCE_REV = "v0.19.0"
_TARGET_REV = "HEAD"


_CATEGORICAL_CALL_KINDS: dict[str, str] = {
    "sample_step": "morphism",
    "observe_step": "morphism",
    "marginalize_step": "morphism",
    "morphism_init_family": "family",
    "morphism_call": "callee",
    "family_call_arg": "family",
}


def _categorical_probability_edits(view: SchemaView) -> list[tuple[int, str]]:
    """Insert ``probs=`` around positional categorical draw arguments."""
    edits: list[tuple[int, str]] = []
    pending = list(reversed(view.top_level_decls()))
    while pending:
        vid = pending.pop()
        pending.extend(reversed(view.outgoing_vids(vid)))
        family_field = _CATEGORICAL_CALL_KINDS.get(view.kind(vid))
        if family_field is None:
            continue
        family = view.field(vid, family_field)
        if family is None or view.text(family) != "Categorical":
            continue
        args = view.fields(vid, "args")
        if not args:
            continue
        first_start, _ = view.span(args[0])
        if len(args) == 1:
            edits.append((first_start, "probs="))
            continue
        _, last_end = view.span(args[-1])
        edits.extend(((first_start, "probs=["), (last_end, "]")))
    return edits


def migrate(source: bytes) -> bytes:
    """Name positional categorical probabilities and validate the result."""
    schema = parse_validated_source(_SOURCE_REV, source)
    view = SchemaView(schema, source)
    migrated = bytearray(source)
    for position, insertion in sorted(
        _categorical_probability_edits(view), reverse=True
    ):
        migrated[position:position] = insertion.encode("utf-8")
    result = bytes(migrated)
    parse_validated_source(_TARGET_REV, result, role="migration target")
    return result


SOURCE_RULE_COVERAGE: frozenset[str] = frozenset()
