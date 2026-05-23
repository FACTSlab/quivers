"""End-to-end round-trip: parse QVR, transpile to every backend, then
re-parse the output through the target grammar and assert the result is
a non-empty schema."""

from __future__ import annotations

import pytest

import panproto
from quivers.dsl.parser import parse
from quivers.transpile import available_targets, transpile


# Minimal Beta-Bernoulli fixture every backend must handle.
_FIXTURE = """\
program flip : Resp -> Resp:
    sample theta <- Beta(2.0, 2.0)
    observe y <- Bernoulli(theta)
    return y
"""

# Backend → tree-sitter grammar name (for the round-trip re-parse).
# `webppl`, `turing`, and `gen` are registered but omitted from this
# suite: they hit upstream panproto grammar bugs in `emit_pretty`
# (panproto/panproto#150 for Julia macrocall_expression; webppl's
# JavaScript output is missing automatic-semicolon-insertion
# delimiters between adjacent expression statements). Their output is
# structurally informed but does not round-trip cleanly through the
# tree-sitter parser; they will join this matrix once those upstreams
# land.
_GRAMMARS = {
    "stan": "stan",
    "numpyro": "python",
    "pyro": "python",
    "pymc": "python",
    "edward2": "python",
    "church": "scheme",
}


def _registry() -> panproto.AstParserRegistry:
    return panproto.AstParserRegistry()


@pytest.mark.parametrize("target", sorted(_GRAMMARS))
def test_round_trip(target: str) -> None:
    """Every backend's output must re-parse cleanly through its target
    grammar.

    Not a structural-equivalence check (the AST shape diverges across
    languages by construction); just a syntax-validity guard. If
    `emit_pretty` produces bytes the grammar rejects, the backend has a
    bug in its vertex-kind or edge-label choices.
    """
    if target not in available_targets():
        pytest.skip(f"backend {target!r} not registered")
    module = parse(_FIXTURE)
    output = transpile(module, target=target)
    assert output, f"empty bytes from {target}"
    grammar = _GRAMMARS[target]
    back = _registry().parse_with_protocol(grammar, output, f"rt.{target}")
    assert back.vertex_count > 0, (
        f"{target!r} output re-parsed to zero vertices; bytes = {output!r}"
    )


def test_target_registration() -> None:
    """Every backend listed in `_GRAMMARS` must be registered."""
    registered = set(available_targets())
    expected = set(_GRAMMARS)
    missing = expected - registered
    assert not missing, f"unregistered backends: {missing}"
