"""Tier 1 (continued): lens-law property tests on Mapping
composition.

The transpile pipeline composes three
[`didactic.api.Mapping`][didactic.api.Mapping] arrows:

- [`Lower`][quivers.transpile.lower.Lower] (`Module` → `IRProgram`)
- per-backend `Renderer` wrapped as a Mapping (`IRProgram` →
  `panproto.Schema`)
- `EmitPretty(grammar)` (`panproto.Schema` → `bytes`)

For every backend we check:

1. **Determinism**: rendering the same module twice returns
   structurally identical schemas.
2. **Composition equivalence**: `EmitPretty(Renderer(Lower(module)))`
   equals `(Lower >> Renderer >> EmitPretty)(module)`.
3. **Re-emit fixed point**: re-parsing the emit and re-emitting
   produces the same bytes as the original emit.
"""

from __future__ import annotations

import didactic.api as dx
import pytest

import panproto

from quivers.dsl.ast_nodes import Module
from quivers.dsl.parser import parse
from quivers.transpile import (
    _RENDERERS,
    available_targets,
    transpile,
)
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile._pipeline import EmitPretty
from quivers.transpile.ir import IRProgram
from quivers.transpile.lower import Lower
from quivers.transpile.renderers._base import RendererBase


_BACKENDS = sorted(available_targets())

_FIXTURE = """\
object Resp : FinSet 4
program flip : Resp -> Resp
    sample theta <- Beta(2.0, 2.0)
    observe y <- Bernoulli(theta)
    return theta
export flip
"""


class _ExpandAndLower(dx.Mapping):
    """`Module` -> `IRProgram`: per-target composite-let expansion
    followed by [`Lower`][quivers.transpile.lower.Lower].

    The production `transpile()` calls `expand_composite_lets(...,
    target=backend)` before `Lower().forward(...)`; the lens-law
    tests compose the same pair so the law applies to the full
    target-specific frontend, not just the target-independent
    `Lower` step.
    """

    def __init__(self, target: str) -> None:
        self._target = target
        self._lower = Lower()

    def forward(self, module: Module) -> IRProgram:  # type: ignore[override]
        expanded = expand_composite_lets(module, target=self._target)
        return self._lower.forward(expanded)


class _RenderMapping(dx.Mapping):
    """`IRProgram` -> `panproto.Schema`: wrap a
    [`RendererBase`][quivers.transpile.renderers._base.RendererBase]
    instance's `render` method as a Mapping so it composes with
    `Lower` and `EmitPretty` via `>>`."""

    def __init__(self, renderer: RendererBase) -> None:
        self._renderer = renderer

    def forward(self, ir: IRProgram) -> panproto.Schema:  # type: ignore[override]
        return self._renderer.render(ir)


def _renderer_for(backend: str) -> RendererBase:
    """Return a fresh renderer instance for `backend`."""
    renderer_cls = _RENDERERS[backend][0]
    return renderer_cls()


def _grammar_for(backend: str) -> str:
    return _RENDERERS[backend][1]


def _pipeline_for(backend: str) -> tuple[_ExpandAndLower, _RenderMapping, EmitPretty]:
    """Build the three Mapping arrows for `backend`."""
    return (
        _ExpandAndLower(backend),
        _RenderMapping(_renderer_for(backend)),
        EmitPretty(_grammar_for(backend)),
    )


@pytest.mark.parametrize("backend", _BACKENDS)
def test_walker_determinism(backend: str) -> None:
    """Two consecutive renders on the same Module return schemas
    with the same vertex / edge multisets."""
    module = parse(_FIXTURE)
    front, render_a, _ = _pipeline_for(backend)
    _, render_b, _ = _pipeline_for(backend)
    ir = front.forward(module)
    schema_a = render_a.forward(ir)
    schema_b = render_b.forward(ir)
    a_kinds = sorted(v.kind for v in schema_a.vertices)
    b_kinds = sorted(v.kind for v in schema_b.vertices)
    assert a_kinds == b_kinds, (
        f"{backend}: vertex-kind multisets differ between calls"
    )
    a_edges = sorted(
        (e.kind, _vertex_kind(schema_a, e.src), _vertex_kind(schema_a, e.tgt))
        for e in schema_a.edges
    )
    b_edges = sorted(
        (e.kind, _vertex_kind(schema_b, e.src), _vertex_kind(schema_b, e.tgt))
        for e in schema_b.edges
    )
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
    """``EmitPretty(Renderer(Lower(module)))`` equals
    ``(Lower >> Renderer >> EmitPretty)(module)``."""
    module = parse(_FIXTURE)
    front, render, emit = _pipeline_for(backend)
    direct = emit.forward(render.forward(front.forward(module)))
    composed = (front >> render >> emit)(module)
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
    "numpyro": True,
    "pyro": True,
    "pymc": True,
    "edward2": True,
    "church": True,    # vacuous: panproto/panproto#172 makes both emits empty
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
