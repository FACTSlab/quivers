"""QVR reference [`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe].

Computes ``log p(θ, y)`` at each test point by walking the QVR
[`MonadicProgram`][quivers.continuous.programs.MonadicProgram] with
[`trace`][quivers.inference.trace.trace], clamping every site to the
corresponding entry in ``Point.params`` / ``Point.data``. The
resulting [`Trace.log_joint`][quivers.inference.trace.Trace] is the
sum of log-densities across every stochastic site — exactly the
joint log-density the numeric-equivalence test asserts on.

This probe is always available: it does not need an external
runtime, just the in-process QVR machinery.
"""

from __future__ import annotations

import dataclasses
import pathlib

import torch

from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse
from quivers.inference.trace import trace
from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


@dataclasses.dataclass(frozen=True)
class QvrProbe:
    """Reference probe: evaluates the QVR-source program in-process."""

    backend: str = "qvr"

    def available(self) -> bool:
        return True

    def evaluate(
        self,
        source: bytes,
        fixture_name: str,
        points: list[Point],
        *,
        scratch: pathlib.Path,
    ) -> ProbeResult:
        """Trace the QVR program at each clamped (θ, y) point.

        ``source`` here is the *original* QVR `.qvr` source (the
        harness passes it through unchanged for the QVR probe); other
        backend probes receive transpiled source. The QVR probe does
        not write to ``scratch``; it lives entirely in-process.
        """
        del scratch  # in-process; no scratch files needed
        module = parse(source.decode("utf-8"))
        compiler = Compiler(module)
        program = compiler.compile()
        monadic = program._morphism

        # The trace function expects a ``x`` input of shape
        # ``(batch, _x_dim)``. For programs with no scalar params
        # (the common case in fixtures), a zero tensor of shape
        # (1, 1) suffices: the input is the categorical bracket
        # token, not the latent parameters. Programs with
        # ``program(alpha : Real, ...)`` style params need real
        # input slices; those come from `points[i].params` and the
        # harness splices them into the first axis.
        log_densities: list[float] = []
        for pt in points:
            observations = _observations_from_point(pt)
            x = _x_input_from_point(pt, monadic)
            tr = trace(monadic, x, observations=observations)
            if tr.log_joint is None:
                msg = (
                    f"qvr probe on {fixture_name!r}: trace returned "
                    f"None log_joint"
                )
                raise RuntimeError(msg)
            log_densities.append(float(tr.log_joint.item()))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": "quivers in-process"},
        )


def _observations_from_point(pt: Point) -> dict[str, torch.Tensor]:
    """Merge ``params`` + ``data`` into the clamping dict
    [`trace`][quivers.inference.trace.trace] consumes."""
    merged: dict[str, torch.Tensor] = {}
    for k, v in pt.params.items():
        merged[k] = _as_tensor(v)
    for k, v in pt.data.items():
        merged[k] = _as_tensor(v)
    return merged


def _x_input_from_point(pt: Point, monadic) -> torch.Tensor:
    """Construct the ``x`` input tensor for the
    [`MonadicProgram`][quivers.continuous.programs.MonadicProgram].

    Programs without scalar parameters get a placeholder (1, 1)
    tensor. Programs declared as
    ``program prog(alpha : Real, ...) : ...`` consume their scalar
    params from ``pt.params`` via the program's ``_param_dims`` /
    ``_params`` slicing convention (see
    [`trace`][quivers.inference.trace.trace] for the splitting).
    """
    if monadic._params is None or not monadic._params:
        return torch.zeros(1, 1)
    slices: list[torch.Tensor] = []
    for pname, pdim in zip(monadic._params, monadic._param_dims):
        if pname not in pt.params:
            raise KeyError(
                f"program param {pname!r} missing from point.params; "
                f"available: {list(pt.params)}"
            )
        value = pt.params[pname]
        chunk = torch.tensor(
            [float(value)] if isinstance(value, (int, float))
            else [float(v) for v in value],
            dtype=torch.get_default_dtype(),
        )
        if chunk.numel() != pdim:
            raise ValueError(
                f"program param {pname!r}: expected {pdim} value(s); "
                f"got {chunk.numel()}"
            )
        slices.append(chunk)
    return torch.cat(slices, dim=-1).unsqueeze(0)


def _as_tensor(value) -> torch.Tensor:
    """Wrap a Python scalar or list into a 1-D `torch.Tensor`."""
    if isinstance(value, (int, float)):
        return torch.tensor([float(value)])
    return torch.tensor([float(x) for x in value])


# `LogDensityProbe` is `runtime_checkable`; assert at import time so
# regressions in the dataclass shape surface immediately.
_PROBE: LogDensityProbe = QvrProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["QvrProbe"]
