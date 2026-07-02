"""Algebraic effect handlers for probabilistic programs.

Effect handlers in the Pyro `poutine` / NumPyro `handlers` shape,
grounded in the algebraic-effects calculus of
[Plotkin and Pretnar 2009](https://doi.org/10.1007/978-3-642-00590-9_7)
and its probabilistic application in
[Scibior et al. 2018](https://doi.org/10.1145/3236778) and
[Nguyen et al. 2023](https://doi.org/10.1145/3609026.3609729).

The `EffectHandler` ABC and the thread-local handler stack live in
`quivers.effects.base`. Concrete handlers (`TraceHandler`,
`ClampHandler`, `DoHandler`, `MaskHandler`, `ScaleHandler`,
`BlockHandler`, `ReplayHandler`, `LiftHandler`, `CollapseHandler`)
live in per-file modules and are re-exported here along with their
short-name factories. The `reparam` subpackage collects the
reparameterisation strategies.

The `clamp` handler is the Pyro-poutine-style analogue of Pyro's
`condition`; the name avoids collision with the top-level
[`quivers.inference.conditioning.condition`][quivers.inference.conditioning.condition]
factory, which returns a
[`Conditioned`][quivers.inference.conditioning.Conditioned]
model wrapper. Both compose observations onto a model; pick
`clamp` when writing a handler stack, `condition` when writing a
top-level `Conditioned` object.

The handler-aware interpreter `run_program` lives in
`quivers.effects.interpreter`; the thin `quivers.inference.trace.trace`
wrapper stacks a `TraceHandler` on and returns the recorded trace.
"""

from __future__ import annotations

from quivers.effects.base import (
    EffectHandler,
    Message,
    apply_stack,
)
from quivers.effects.block import BlockHandler, block
from quivers.effects.collapse import CollapseHandler, collapse
from quivers.effects.clamp import ClampHandler, clamp
from quivers.effects.do import DoHandler, do
from quivers.effects.interpreter import run_program
from quivers.effects.lift import LiftHandler, lift
from quivers.effects.mask import MaskHandler, mask
from quivers.effects.replay import ReplayHandler, replay
from quivers.effects.scale import ScaleHandler, scale
from quivers.effects.trace_handler import TraceHandler
from quivers.effects.reparam import (
    ConjugateReparam,
    LocScaleReparam,
    NeuTraReparam,
    Reparam,
    ReparamOrchestrator,
    TransformReparam,
    reparam,
)


__all__ = [
    "EffectHandler",
    "Message",
    "apply_stack",
    "run_program",
    "TraceHandler",
    "ClampHandler",
    "clamp",
    "DoHandler",
    "do",
    "MaskHandler",
    "mask",
    "ScaleHandler",
    "scale",
    "BlockHandler",
    "block",
    "ReplayHandler",
    "replay",
    "LiftHandler",
    "lift",
    "CollapseHandler",
    "collapse",
    "Reparam",
    "ReparamOrchestrator",
    "reparam",
    "LocScaleReparam",
    "TransformReparam",
    "NeuTraReparam",
    "ConjugateReparam",
]
