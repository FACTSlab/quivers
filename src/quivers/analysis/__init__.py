"""Algebra-guided training tooling.

This package inspects a compiled QVR program and returns chain metadata,
initialization recipes, and source-keyed diagnostics without rewriting the
program. It accepts either a
[`quivers.dsl.ast_nodes.Module`][quivers.dsl.ast_nodes.Module] AST or its
runtime [`quivers.continuous.programs.MonadicProgram`][quivers.continuous.programs.MonadicProgram].

The pieces:

* `ChainShape` ([`quivers.analysis.chain_shape`][quivers.analysis.chain_shape]): walks
  a program's let / latent steps and tags each one with its
  source location, governing algebra, and intermediate
  dimensionality. This metadata supplies the initialization and saturation
  analyses.
* `recommend_init` ([`quivers.analysis.init_spec`][quivers.analysis.init_spec]): given
  a program, produces a per-latent `InitSpec` from each
  algebra's initialization recipe. Pair with
  `apply_init_spec` to materialize the initial values onto
  the program's learnable parameters.
* `saturation_warnings`
  ([`quivers.analysis.saturation`][quivers.analysis.saturation]): given a program, returns
  source-keyed warnings about latents that, under the
  recommended init, would saturate the surrounding algebra's
  value range.

See [*Analysis Pipelines: Fitting and Diagnostics*](https://factslab.github.io/quivers/guides/analysis-fitting-and-diagnostics/)
for the user-facing workflow.
"""

from __future__ import annotations

from quivers.analysis.chain_shape import ChainShape, StepShape
from quivers.analysis.init_spec import (
    InitSpec,
    apply_init_spec,
    recommend_init,
)
from quivers.analysis.saturation import SaturationWarning, saturation_warnings

__all__ = [
    "ChainShape",
    "StepShape",
    "InitSpec",
    "recommend_init",
    "apply_init_spec",
    "SaturationWarning",
    "saturation_warnings",
]
