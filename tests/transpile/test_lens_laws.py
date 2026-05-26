"""Tier 1 (continued): lens-law property tests on Mapping
composition.

The transpile pipeline composes two
[`didactic.api.Mapping`][didactic.api.Mapping] instances:

- per-backend `SchemaTransform` (`Module` → `panproto.Schema`)
- `EmitPretty(grammar)` (`panproto.Schema` → `bytes`)

For every backend we check:

1. **Determinism**: `transform.forward(module)` returns structurally
   identical schemas across two consecutive calls.
2. **Composition equivalence**: `EmitPretty(SchemaTransform(module))`
   equals `(SchemaTransform >> EmitPretty)(module)`.
3. **Re-emit fixed point**: re-parsing the emit and re-emitting
   produces the same bytes as the original emit.
"""

from __future__ import annotations

import pytest

import panproto

from quivers.dsl.parser import parse
from quivers.transpile import available_targets, transpile
from quivers.transpile._pipeline import EmitPretty


_BACKENDS = sorted(available_targets())

_FIXTURE = """\
object Resp : FinSet 4
program flip : Resp -> Resp
    sample theta <- Beta(2.0, 2.0)
    observe y <- Bernoulli(theta)
    return theta
export flip
"""


def _walker_for(backend: str):
    """Load the backend module and return a fresh `SchemaTransform`."""
    import importlib

    module = importlib.import_module(
        f"quivers.transpile.backends.{backend}"
    )
    walker_cls_name_candidates = [
        f"_{backend.title()}Walker",
        f"_{backend.capitalize()}Walker",
        f"_StanWalker" if backend == "stan" else None,
        f"_NumPyroWalker" if backend == "numpyro" else None,
        f"_PyroWalker" if backend == "pyro" else None,
        f"_PyMCWalker" if backend == "pymc" else None,
        f"_Edward2Walker" if backend == "edward2" else None,
        f"_ChurchWalker" if backend == "church" else None,
        f"_WebPPLWalker" if backend == "webppl" else None,
        f"_TuringWalker" if backend == "turing" else None,
        f"_GenWalker" if backend == "gen" else None,
        f"_BugsWalker" if backend == "bugs" else None,
        f"_JagsWalker" if backend == "jags" else None,
    ]
    for name in walker_cls_name_candidates:
        if name and hasattr(module, name):
            return getattr(module, name)()
    raise AssertionError(
        f"backend {backend!r}: cannot locate walker class in module "
        f"{module.__name__!r}; tried {walker_cls_name_candidates}"
    )


def _grammar_for(backend: str) -> str:
    return {
        "stan": "stan", "numpyro": "python", "pyro": "python",
        "pymc": "python", "edward2": "python", "church": "scheme",
        "webppl": "javascript", "turing": "julia", "gen": "julia",
        "bugs": "bugs", "jags": "jags",
    }[backend]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_walker_determinism(backend: str) -> None:
    """Two consecutive `forward()` calls on the same Module return
    schemas with the same vertex / edge multisets."""
    module = parse(_FIXTURE)
    walker = _walker_for(backend)
    schema_a = walker.forward(module)
    schema_b = walker.forward(module)
    a_kinds = sorted(v.kind for v in schema_a.vertices)
    b_kinds = sorted(v.kind for v in schema_b.vertices)
    assert a_kinds == b_kinds, (
        f"{backend}: vertex-kind multisets differ between calls"
    )
    a_edges = sorted((e.kind, _vertex_kind(schema_a, e.src), _vertex_kind(schema_a, e.tgt))
                     for e in schema_a.edges)
    b_edges = sorted((e.kind, _vertex_kind(schema_b, e.src), _vertex_kind(schema_b, e.tgt))
                     for e in schema_b.edges)
    assert a_edges == b_edges, (
        f"{backend}: edge structure differs between calls"
    )


def _vertex_kind(schema: panproto.Schema, vid: str) -> str:
    for v in schema.vertices:
        if v.id == vid:
            return v.kind
    raise AssertionError(f"missing vertex {vid!r}")


@pytest.mark.parametrize("backend", _BACKENDS)
def test_mapping_composition_equivalence(backend: str) -> None:
    """``EmitPretty(SchemaTransform(module))`` equals
    ``(SchemaTransform >> EmitPretty)(module)``."""
    module = parse(_FIXTURE)
    walker = _walker_for(backend)
    grammar = _grammar_for(backend)
    emit = EmitPretty(grammar)
    direct = emit.forward(walker.forward(module))
    composed = (walker >> emit)(module)
    assert direct == composed, (
        f"{backend}: composition law violated; direct emit and "
        f"composed pipeline produce different bytes"
    )


# Per-backend grammar emit_pretty fixed-point status. False marks
# backends whose underlying tree-sitter `emit_pretty` re-emit cycle
# is known not to be a fixed point on the canonical fixture; the
# strict-xfail flips when panproto fixes the upstream non-determinism.
_REEMIT_IS_FIXED_POINT: dict[str, bool] = {
    "stan": True,
    "numpyro": False,  # panproto py emit normalises `;` → `\n` on reparse
    "pyro": False,     # same
    "pymc": False,     # same + as_pattern emit drops the alias identifier
    "edward2": False,  # same
    "church": True,
    "webppl": True,
    "turing": True,
    "gen": True,
    "bugs": True,
    "jags": True,
}


@pytest.mark.parametrize("backend", _BACKENDS)
def test_reemit_fixed_point(backend: str, request: pytest.FixtureRequest) -> None:
    """Re-parsing the emit and re-emitting produces the same bytes.

    Strict-xfail for cells whose grammar's `emit_pretty` is known not
    to be a fixed point on the canonical fixture; flips when panproto
    fixes the upstream re-emit non-determinism.
    """
    if not _REEMIT_IS_FIXED_POINT[backend]:
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    f"panproto `emit_pretty({backend!r}, ...)` round-trip is "
                    f"not a fixed point on the canonical fixture; first emit "
                    f"normalises differently than the re-emit of the parsed "
                    f"schema. Upstream tree-sitter Python emit-pretty bug."
                ),
            )
        )
    module = parse(_FIXTURE)
    grammar = _grammar_for(backend)
    reg = panproto.AstParserRegistry()

    first = transpile(module, target=backend)
    reparsed = reg.parse_with_protocol(grammar, first, f"first.{backend}")
    second = bytes(reg.emit_pretty(grammar, reparsed))
    assert first == second, (
        f"{backend}: re-emit is not a fixed point; "
        f"first={first!r}, second={second!r}"
    )
