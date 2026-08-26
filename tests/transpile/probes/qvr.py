"""QVR reference [`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe].

Computes ``log p(theta, y)`` at each test point by walking the QVR
[`MonadicProgram`][quivers.continuous.programs.MonadicProgram] with
[`trace`][quivers.inference.trace.trace], clamping every site to the
corresponding entry in ``Point.params`` / ``Point.data``. The
resulting [`Trace.log_joint`][quivers.inference.trace.Trace] is the
sum of log-densities across every stochastic site, exactly the
joint log-density the numeric-equivalence test asserts on.

Every reference evaluation passes two independent guards before its
value is returned:

1. [`assert_all_latents_clamped`][tests.transpile.probes.qvr.assert_all_latents_clamped],
   a *structural* guard reading
   [`Trace.latent_sites`][quivers.effects.trace_types.Trace]. It
   names the offending site, so it is the guard that tells a caller
   which ground truth the point failed to bind.
2. [`assert_reference_joint_deterministic`][tests.transpile.probes.qvr.assert_reference_joint_deterministic],
   a *behavioural* guard. The program is traced once per entry of
   [`DETERMINISM_SEEDS`][tests.transpile.probes.qvr.DETERMINISM_SEEDS],
   each under a freshly-seeded global torch RNG, and every joint (and
   every per-site log-density summand) must agree bit for bit.

The second guard subsumes the first as a *guarantee* and does not
replace it as a *diagnostic*. `Trace.latent_sites` enumerates only
sites recorded in `Trace.sites`, so a latent living inside a
`SampledComposition` (an RNN scan cell, an attention chain, a
decoder marginalisation) is invisible to it: the structural guard
passes while the joint is still redrawn on every call. Bitwise
equality across distinct RNG states has no such blind spot, because
a resampled quantity anywhere in the computation moves the bits it
feeds.

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
from quivers.inference.trace import Trace, trace
from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


# Global torch RNG seeds the behavioural determinism guard traces
# under. Two suffice to expose any site that consumes randomness,
# because a deterministic joint is by definition invariant to the
# generator state; a caller that wants a wider sweep (the dedicated
# oracle-determinism tier does) passes its own `seeds`. The values are
# fixed constants so a probe evaluation is reproducible run to run.
DETERMINISM_SEEDS: tuple[int, ...] = (0x5EED0001, 0x5EED0002)


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

        Each point is traced once per entry of
        [`DETERMINISM_SEEDS`][tests.transpile.probes.qvr.DETERMINISM_SEEDS]
        and both guards run before the value is recorded, so a
        non-deterministic joint can never leave this method as a
        measurement. The caller's global torch RNG state is restored
        on exit, which makes the probe RNG-neutral: evaluating it
        cannot shift any draw the surrounding test makes.

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
            traces = reference_traces(
                program_monadic,
                pt,
                x_input=x_input,
                observations=observations,
            )
            tr = traces[0]
            if tr.log_joint is None:
                msg = (
                    f"qvr probe on {fixture_name!r}: trace returned "
                    f"None log_joint"
                )
                raise RuntimeError(msg)
            assert_all_latents_clamped(tr, fixture_name)
            assert_reference_joint_deterministic(
                traces, fixture_name, DETERMINISM_SEEDS,
            )
            log_densities.append(float(tr.log_joint.sum().item()))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": "quivers in-process"},
        )


def clamping_observations(
    pt: Point, observations: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Build the dict [`trace`][quivers.inference.trace.trace] clamps with.

    ``pt.params`` and ``pt.data`` are merged first, then any
    pre-shaped ``observations`` entry overwrites its per-point
    counterpart: the pre-shaped dict is the only channel that
    preserves the multi-axis shapes (`(T, state_dim)`, `(B, L)`, ...)
    the flat per-point payload discards, while ``pt.params`` keeps
    contributing the latent-site clamps the dataset's observation
    dict does not carry.
    """
    merged: dict[str, torch.Tensor] = {}
    for k, v in pt.params.items():
        merged[k] = _as_tensor(v)
    for k, v in pt.data.items():
        merged[k] = _as_tensor(v)
    if observations is not None:
        for k, v in observations.items():
            merged[k] = v
    return merged


def reference_traces(
    monadic: MonadicProgram,
    pt: Point,
    *,
    x_input: torch.Tensor | None = None,
    observations: dict[str, torch.Tensor] | None = None,
    seeds: tuple[int, ...] = DETERMINISM_SEEDS,
) -> list[Trace]:
    """Trace `monadic` at `pt` once per entry of `seeds`, unjudged.

    Each trace runs under a freshly-seeded global torch RNG, and the
    caller's RNG state is restored before returning, so the sweep
    perturbs nothing outside this call. The traces come back
    unexamined: judging them is
    [`assert_reference_joint_deterministic`][tests.transpile.probes.qvr.assert_reference_joint_deterministic]'s
    job, and separating the two lets a caller measure the disagreement
    itself rather than only learn that some guard raised.

    Parameters
    ----------
    monadic
        Compiled program to trace.
    pt
        Point supplying the clamped ``params`` / ``data`` values.
    x_input
        Explicit program-input tensor; see
        [`_x_input`][tests.transpile.probes.qvr._x_input] for the
        resolution order when it is omitted.
    observations
        Pre-shaped observation dict taking precedence over the
        per-point flat payload.
    seeds
        Global torch RNG seeds, one trace each. At least two are
        required: a single trace cannot exhibit a disagreement, so a
        one-seed sweep would make every determinism assertion
        vacuous.

    Returns
    -------
    list[Trace]
        One trace per seed, in `seeds` order.
    """
    if len(seeds) < 2:
        raise ValueError(
            f"reference_traces needs at least two seeds to observe a "
            f"disagreement; got {seeds!r}. A single-seed sweep makes "
            f"the determinism assertion vacuous."
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"reference_traces seeds must be pairwise distinct; got "
            f"{seeds!r}. Repeating a seed compares a computation "
            f"against itself under the same generator state, which "
            f"holds for a non-deterministic program too."
        )
    obs = clamping_observations(pt, observations)
    x = _x_input(pt, monadic, x_input)
    saved_rng_state = torch.get_rng_state()
    try:
        traces: list[Trace] = []
        for seed in seeds:
            torch.manual_seed(seed)
            traces.append(trace(monadic, x, observations=obs))
    finally:
        torch.set_rng_state(saved_rng_state)
    return traces


def assert_all_latents_clamped(tr: Trace, fixture_name: str) -> None:
    """Fail loudly when the point leaves a free latent site unclamped.

    A reference joint is only meaningful when every unobserved,
    non-deterministic sample site is pinned to its ground-truth
    value. Any such site the point does not clamp is resampled fresh
    on each call, so [`trace`][quivers.inference.trace.trace] returns
    a different (and wrong) joint every evaluation while still passing
    a finiteness check. The guard reads
    [`Trace.latent_sites`][quivers.effects.trace_types.Trace], which
    excludes both observed sites (clamped to data) and deterministic
    sites (let bindings, score / marginalize bodies), so it fires
    only on genuinely-resampled latents and never on a legitimately
    marginalized or observed site.

    This is the *diagnostic* half of the determinism contract: it
    names the site whose ground truth is missing. It is not the
    guarantee, because `Trace.latent_sites` sees only sites recorded
    in `Trace.sites`, and a latent internal to a `SampledComposition`
    is recorded nowhere. Such a program passes here and is caught by
    [`assert_reference_joint_deterministic`][tests.transpile.probes.qvr.assert_reference_joint_deterministic].
    """
    free = sorted(tr.latent_sites)
    if free:
        raise RuntimeError(
            f"qvr probe on {fixture_name!r}: free latent sample "
            f"site(s) {free!r} were not clamped by the point, so "
            f"they are resampled on every call and the joint "
            f"log-density is nondeterministic. Every unobserved, "
            f"non-deterministic sample site must be clamped to its "
            f"ground-truth value; bind the missing ground truth in "
            f"the example's synthetic-data block under the site's "
            f"name (a template-inlined site 'outer$inner' is spelled "
            f"'outer_inner' in Python)."
        )


def assert_reference_joint_deterministic(
    traces: list[Trace],
    fixture_name: str,
    seeds: tuple[int, ...] = DETERMINISM_SEEDS,
) -> None:
    """Require every trace in `traces` to carry the same joint, bit for bit.

    The traces come from
    [`reference_traces`][tests.transpile.probes.qvr.reference_traces],
    one per entry of `seeds`, each run under a distinct global torch
    RNG state. A deterministic joint is invariant to that state by
    definition, so any difference in the bit pattern of `log_joint`,
    or of any per-site `log_prob` summand feeding it, means some
    quantity inside the computation was redrawn and the "reference"
    is a sample rather than a density.

    Comparison is on raw bytes rather than on a tolerance. A
    tolerance would be a second, silent threshold sitting underneath
    the equivalence tolerance the transpile tier actually asserts on,
    and a resampled latent whose effect happens to land inside it
    would pass. Determinism is exact or it is absent.

    Per-site `log_prob` is compared alongside the joint because the
    joint is their sum: two redrawn summands can cancel at one point
    and not at another, so summing first would let a real
    nondeterminism hide behind an accidental cancellation. Site
    *values* are deliberately not compared. A marginalized or
    enumerated site (`hmm`'s `state`, `zip_regression`'s `z`) records
    a drawn representative value while its `log_prob` carries the
    reduction over the whole support, so its value moves with the
    generator while the density it contributes does not. The density
    is what the reference reports and what the guard protects.
    """
    if len(traces) != len(seeds):
        raise ValueError(
            f"assert_reference_joint_deterministic on {fixture_name!r}: "
            f"{len(traces)} trace(s) against {len(seeds)} seed(s); the "
            f"two must correspond one to one for the message to name "
            f"the seed a disagreement came from."
        )
    for index, tr in enumerate(traces):
        if tr.log_joint is None:
            raise RuntimeError(
                f"qvr probe on {fixture_name!r}: the trace under seed "
                f"{seeds[index]} returned a None log_joint, so the "
                f"reference carries no density to compare."
            )
    first = traces[0]
    first_joint = first.log_joint
    assert first_joint is not None
    for index in range(1, len(traces)):
        other = traces[index]
        other_joint = other.log_joint
        assert other_joint is not None
        moving = _sites_with_moving_log_prob(first, other)
        if _bitwise_equal(first_joint, other_joint) and not moving:
            continue
        left = float(first_joint.sum().item())
        right = float(other_joint.sum().item())
        raise RuntimeError(
            f"qvr probe on {fixture_name!r}: the reference joint is "
            f"not deterministic. Tracing the program under global "
            f"torch RNG seed {seeds[0]} and again under seed "
            f"{seeds[index]} produced joints that differ bit for "
            f"bit: {left!r} (hex {left.hex()}) versus {right!r} "
            f"(hex {right.hex()}), a gap of {right - left!r} nats. "
            f"Site log-densities that moved between the two runs: "
            f"{moving!r}. Some quantity in this program is redrawn on "
            f"every call, so the value is a sample from the joint's "
            f"estimator rather than the joint, and no backend "
            f"comparison against it means anything. A `sample` site "
            f"named above needs its ground truth bound in the "
            f"example's synthetic-data block; an empty list means the "
            f"redrawn latent lives inside a composition and is "
            f"recorded at no site at all, so the composition's "
            f"marginalisation has to become deterministic (or expose "
            f"its inner sites) before this program has a reference "
            f"joint."
        )


def _sites_with_moving_log_prob(first: Trace, second: Trace) -> list[str]:
    """Names whose recorded log-density is not bit-identical across two traces.

    A name present in one trace and absent from the other counts as
    moving: a control-flow path taken under one generator state and
    not the other is nondeterminism of the sharpest kind.
    """
    moving: set[str] = set()
    for name, site in first.sites.items():
        other = second.sites.get(name)
        if other is None or not _bitwise_equal(site.log_prob, other.log_prob):
            moving.add(name)
    for name in second.sites:
        if name not in first.sites:
            moving.add(name)
    return sorted(moving)


def _bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Whether two tensors carry identical dtype, shape, and raw bytes.

    Reinterpreting through `uint8` sidesteps floating-point equality
    entirely: `nan != nan` under `torch.equal`, while two traces that
    both produced the same `nan` bit pattern are in fact in
    agreement, and two subnormals that compare equal under a
    tolerance are not.
    """
    if left.dtype is not right.dtype or left.shape != right.shape:
        return False
    left_bytes = _raw_bytes(left)
    right_bytes = _raw_bytes(right)
    return torch.equal(left_bytes, right_bytes)


def _raw_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Flat `uint8` reinterpretation of `tensor`'s storage.

    The `clone` is what makes the `view` total: a tensor sliced out
    of a larger buffer carries a storage offset, and reinterpreting
    the dtype of such a view is rejected unless the offset happens to
    divide evenly into the target element size.
    """
    flat = tensor.detach().cpu().reshape(-1).clone()
    return flat.view(torch.uint8)


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


__all__ = [
    "DETERMINISM_SEEDS",
    "QvrProbe",
    "assert_all_latents_clamped",
    "assert_reference_joint_deterministic",
    "clamping_observations",
    "reference_traces",
]
