"""Per-backend renderers consuming the transpile IR.

Each backend implements the [`Renderer`][quivers.transpile.renderers.Renderer]
protocol: one public `render(ir: IRProgram) -> panproto.Schema` method
plus four dispatch points (`declare`, `sample`, `marginalize`,
`broadcast`) that receive the call's
[`RenderContext`][quivers.transpile.renderers.RenderContext]. The IR-walk
dispatch, index-substitution helpers, the atom enumeration a marginalize
block is scored under, and the structural invariants
(`assert_no_dangling_refs`, `assert_no_dropped_param_map`,
`assert_no_lists`) live on
[`RendererBase`][quivers.transpile.renderers.RendererBase], which every
in-tree renderer subclasses.
"""

from __future__ import annotations

from quivers.transpile.renderers._base import (
    BlockKind,
    IRArgTransform,
    IRMarginalAtom,
    RenderContext,
    Renderer,
    RendererBase,
    SchemaFragment,
    assert_no_dangling_refs,
    assert_no_dropped_param_map,
    assert_no_lists,
)
from quivers.transpile.renderers.bugs import BUGSRenderer
from quivers.transpile.renderers.church import ChurchRenderer
from quivers.transpile.renderers.edward2 import Edward2Renderer
from quivers.transpile.renderers.gen import GenRenderer
from quivers.transpile.renderers.jags import JAGSRenderer
from quivers.transpile.renderers.numpyro import NumPyroRenderer
from quivers.transpile.renderers.pymc import PyMCRenderer
from quivers.transpile.renderers.pyro import PyroRenderer
from quivers.transpile.renderers.stan import StanRenderer
from quivers.transpile.renderers.turing import TuringRenderer
from quivers.transpile.renderers.webppl import WebPPLRenderer


__all__ = [
    "BUGSRenderer",
    "BlockKind",
    "ChurchRenderer",
    "Edward2Renderer",
    "GenRenderer",
    "IRArgTransform",
    "IRMarginalAtom",
    "JAGSRenderer",
    "NumPyroRenderer",
    "PyMCRenderer",
    "PyroRenderer",
    "RenderContext",
    "Renderer",
    "RendererBase",
    "SchemaFragment",
    "StanRenderer",
    "TuringRenderer",
    "WebPPLRenderer",
    "assert_no_dangling_refs",
    "assert_no_dropped_param_map",
    "assert_no_lists",
]
