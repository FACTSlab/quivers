"""Per-backend renderers consuming the transpile IR.

Each backend implements a [`Renderer`][quivers.transpile.renderers._base.Renderer]
class with one public method `render(ir: IRProgram) -> panproto.Schema`
plus four private dispatch points (`declare`, `sample`, `marginalize`,
`broadcast`). The IR-walk dispatch, index-substitution helpers, the
explicit-latent rewrite helper for marginalize, and the structural
invariants (`assert_no_dangling_refs`, `assert_no_dropped_param_map`,
`assert_no_lists`) live on
[`RendererBase`][quivers.transpile.renderers._base.RendererBase].
"""

from __future__ import annotations

from quivers.transpile.renderers._base import (
    BlockKind,
    IRArgTransform,
    Renderer,
    RendererBase,
    SchemaFragment,
    assert_no_dangling_refs,
    assert_no_dropped_param_map,
    assert_no_lists,
)
from quivers.transpile.renderers.edward2 import Edward2Renderer


__all__ = [
    "BlockKind",
    "Edward2Renderer",
    "IRArgTransform",
    "Renderer",
    "RendererBase",
    "SchemaFragment",
    "assert_no_dangling_refs",
    "assert_no_dropped_param_map",
    "assert_no_lists",
]
