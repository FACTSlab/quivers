"""Algebra-guided training tooling.

This package collects the static-analysis / metadata-derivation /
numerics-preserving-recomputation passes that read off a compiled
QVR program (a [`quivers.dsl.ast_nodes.Module`][quivers.dsl.ast_nodes.Module] AST or its
runtime [`quivers.continuous.programs.MonadicProgram`][quivers.continuous.programs.MonadicProgram]) and
return structured data the user can act on. None of the passes here
rewrite the program; they only derive metadata, diagnostics, or
sampler / init parameters that respect the source as the canonical
specification.

The pieces:

* `ChainShape` ([`quivers.analysis.chain_shape`][quivers.analysis.chain_shape]): walks
  a program's let / latent steps and tags each one with its
  source location, governing algebra, and intermediate
  dimensionality. The foundation downstream tooling reads off.
* `recommend_init` ([`quivers.analysis.init_spec`][quivers.analysis.init_spec]): given
  a program, produces a per-latent `InitSpec` from each
  algebra's saturation-free init recipe. Pair with
  `apply_init_spec` to materialise the initial values onto
  the program's learnable parameters.
* `saturation_warnings`
  ([`quivers.analysis.saturation`][quivers.analysis.saturation]): given a program, returns
  source-keyed warnings about latents that, under the
  recommended init, would saturate the surrounding algebra's
  value range.

See ``notes/algebra-guided-training-tooling.md`` for the broader
roadmap.
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
