"""QVR reference [`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe].

Computes ``log p(theta, y)`` at each test point by walking the QVR
[`MonadicProgram`][quivers.continuous.programs.MonadicProgram] with
[`trace`][quivers.inference.trace.trace], clamping every site to the
corresponding entry in ``Point.params`` / ``Point.data``. The
resulting [`Trace.log_joint`][quivers.inference.trace.Trace] is the
sum of log-densities across every stochastic site, exactly the
joint log-density the numeric-equivalence test asserts on.

This probe is always available: it does not need an external
runtime, just the in-process QVR machinery.
"""

from __future__ import annotations

import dataclasses
import pathlib

import torch

from quivers.continuous.programs import MonadicProgram
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
        monadic: MonadicProgram | None = None,
        x_input: torch.Tensor | None = None,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> ProbeResult:
        """Trace the QVR program at each clamped (theta, y) point.

        ``source`` here is the *original* QVR `.qvr` source (the
        harness passes it through unchanged for the QVR probe); other
        backend probes receive transpiled source. The QVR probe does
        not write to ``scratch``; it lives entirely in-process.

        Parameters
        ----------
        source
            QVR `.qvr` source bytes. Parsed and compiled when
            ``monadic`` is not supplied.
        fixture_name
            Stem used in diagnostic messages.
        points
            One [`Point`][tests.transpile.probes._protocol.Point]
            per evaluation; produces one log-density per point in
            the same order.
        scratch
            Per-cell scratch directory; unused by the in-process
            probe (kept for protocol parity with out-of-process
            backends).
        monadic
            Pre-compiled
            [`MonadicProgram`][quivers.continuous.programs.MonadicProgram]
            instance. When provided, takes precedence over the
            source-parse path. Lets the harness pass a template
            instantiation (e.g. ``prog.gmm(alpha=1.0).morphism``)
            built by the synthetic-data block so the probe walks the
            same template instance the data was generated under.
        x_input
            Program-input tensor of shape ``(batch, *event)``. When
            provided, takes precedence over the per-point param
            slicing; lets state-space / sequence / transformer
            programs declare the per-step input directly rather than
            forcing the probe to derive it from a `Point`.
        observations
            Pre-shaped observation dict, name to tensor. When
            provided, takes precedence over the per-point flat-list
            inflation; preserves the multi-dim shapes
            (`(T, state_dim)`, `(B, L)`, ...) that the per-point
            flattening discards. Used by the gallery harness for
            sequence / state-space examples whose observations are
            inherently multi-axis.
        """
        del scratch  # in-process; no scratch files needed
        program_monadic = monadic if monadic is not None else _compile_to_monadic(
            source, fixture_name,
        )

        log_densities: list[float] = []
        for pt in points:
            obs = _observations_from_point(pt)
            if observations is not None:
                # Pre-shaped observations take precedence: they
                # preserve the multi-axis shapes the per-point
                # flat-list inflation discards. `pt.params` keeps
                # contributing the latent-site clamps that aren't
                # carried in the dataset's observation dict.
                for k, v in observations.items():
                    obs[k] = v
            x = _x_input(pt, program_monadic, x_input)
            tr = trace(program_monadic, x, observations=obs)
            if tr.log_joint is None:
                msg = (
                    f"qvr probe on {fixture_name!r}: trace returned "
                    f"None log_joint"
                )
                raise RuntimeError(msg)
            log_densities.append(float(tr.log_joint.sum().item()))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": "quivers in-process"},
        )


def _compile_to_monadic(source: bytes, fixture_name: str) -> MonadicProgram:
    """Parse and compile `source`; return the exported
    [`MonadicProgram`][quivers.continuous.programs.MonadicProgram].

    A `Program` wrapping a `MonadicProgram` exposes the morphism via
    `_morphism`. A `Program` whose export is a parametric template
    (the `Program(None)` shape with a `templates` dict) has no root
    morphism; the probe rejects it with a user-shaped error pointing
    the caller at the in-process template-instantiation idiom.
    A `Program` with no exported morphism at all (a module that only
    declares signatures, encoders, decoders, losses, deductions) is
    rejected likewise; the gallery-numeric tier is only meaningful
    for probabilistic programs.
    """
    module = parse(source.decode("utf-8"))
    compiler = Compiler(module)
    program = compiler.compile()
    morphism = program._morphism
    if isinstance(morphism, MonadicProgram):
        return morphism
    templates = getattr(program, "templates", None)
    if templates:
        names = sorted(templates)
        raise RuntimeError(
            f"qvr probe on {fixture_name!r}: program exports a "
            f"parametric template ({names!r}) with no concrete "
            f"instantiation. Pass the instantiated MonadicProgram "
            f"via the `monadic` keyword (the synthetic-data block "
            f"typically binds it to `model = fit.morphism` after "
            f"`fit = prog.{names[0]}(...)`)."
        )
    if morphism is None:
        raise RuntimeError(
            f"qvr probe on {fixture_name!r}: the module has no "
            f"exported morphism, so it carries no probabilistic "
            f"program for QvrProbe to evaluate."
        )
    raise RuntimeError(
        f"qvr probe on {fixture_name!r}: exported morphism is "
        f"{type(morphism).__name__!r}, not a MonadicProgram; "
        f"only monadic probabilistic programs have a joint "
        f"log-density the probe can trace."
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


def _x_input(
    pt: Point,
    monadic: MonadicProgram,
    explicit: torch.Tensor | None,
) -> torch.Tensor:
    """Resolve the program-input tensor.

    Resolution order:

    1. An ``explicit`` tensor supplied by the harness (typical
       gallery path: the synthetic-data block builds ``x_in`` /
       ``x`` / ``state_prev`` and the harness threads it through).
    2. A param-slice tensor synthesised from ``pt.params`` when the
       program declares scalar parameters via
       ``program prog(alpha : Real, ...) : ...`` (the program's
       ``_param_dims`` / ``_params`` define the splitting; see
       [`trace`][quivers.inference.trace.trace]).
    3. A `(1, 1)` placeholder for non-parametric programs that read
       a bracket token rather than a real input slice.
    """
    if explicit is not None:
        return explicit
    if monadic._params is None or not monadic._params:
        return torch.zeros(1, 1)
    if monadic._param_dims is None:
        raise RuntimeError(
            f"qvr probe: program declares params {list(monadic._params)} "
            f"but exposes no `_param_dims` slice plan; cannot synthesise "
            f"input tensor from point.params"
        )
    slices: list[torch.Tensor] = []
    for pname, pdim in zip(monadic._params, monadic._param_dims):
        if pname not in pt.params:
            raise RuntimeError(
                f"qvr probe: program param {pname!r} missing from "
                f"point.params; available: {sorted(pt.params)}"
            )
        value = pt.params[pname]
        chunk = torch.tensor(
            [float(value)] if isinstance(value, (int, float))
            else [float(v) for v in value],
            dtype=torch.get_default_dtype(),
        )
        if chunk.numel() != pdim:
            raise RuntimeError(
                f"qvr probe: program param {pname!r}: expected "
                f"{pdim} value(s); got {chunk.numel()}"
            )
        slices.append(chunk)
    return torch.cat(slices, dim=-1).unsqueeze(0)


def _as_tensor(value: float | int | list[float] | list[int]) -> torch.Tensor:
    """Wrap a Python scalar or list into a 1-D `torch.Tensor`."""
    if isinstance(value, (int, float)):
        return torch.tensor([float(value)])
    return torch.tensor([float(x) for x in value])


# `LogDensityProbe` is `runtime_checkable`; assert at import time so
# regressions in the dataclass shape surface immediately.
_PROBE: LogDensityProbe = QvrProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["QvrProbe"]
