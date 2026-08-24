"""Extract synthetic-data snippets from `docs/examples/*.md`.

Each gallery example carries a fenced ```python code block under a
`### Generating synthetic data` heading. The snippet seeds torch,
loads the QVR program, samples ground-truth parameters, forward-
generates observations, and builds an `observations` dict the
runtime consumes at trace time.

This module exposes [`load_gallery_data`][tests.transpile._gallery_data.load_gallery_data]
which:

1. Locates the matching `.md` for a QVR source file.
2. Extracts the `### Generating synthetic data` Python block.
3. Executes it in an isolated namespace (with `torch` pre-imported).
4. Returns the resulting `observations` mapping plus every captured
   `true_*` ground-truth parameter (used as a representative latent
   point for the numeric-equivalence test).

The extraction is fail-soft: examples whose data-gen block is
absent or whose snippet raises return None, and the caller skips
the cell with a clear reason.
"""

from __future__ import annotations

import dataclasses
import math
import re
from pathlib import Path

import torch
from torch.distributions import constraints

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.programs import (
    MonadicProgram,
    _LetSpec,
    _ScoreSpec,
)
from tests.transpile.probes._protocol import Point
from tests.transpile.probes.qvr import QvrProbe


_GALLERY_DOCS = (
    Path(__file__).resolve().parents[2] / "docs" / "examples"
)
_GALLERY_SOURCE = _GALLERY_DOCS / "source"

# The synthetic-data block is delimited by `### Generating synthetic
# data` on the start and the next `### ` or `## ` heading on the
# end (capturing the fenced ```python block within).
_DATA_BLOCK_RE = re.compile(
    r"###\s+Generating synthetic data\b.*?```python\n(.*?)```",
    re.DOTALL,
)


@dataclasses.dataclass(frozen=True)
class GalleryDataset:
    """One example's synthetic data + ground-truth parameters."""

    observations: dict[str, torch.Tensor]
    params: dict[str, torch.Tensor]
    """Captured `true_*` variables (with the `true_` prefix stripped)
    representing a single point in latent space at which to evaluate
    the joint log-density."""

    x_input: torch.Tensor | None
    """The program-input tensor the synthetic-data block prepared.
    Recognised under any of the names `x_in`, `x`, or `state_prev`;
    the first one bound in the namespace wins. Programs that declare
    `(alpha : Real, ...)` scalar parameters consume this directly;
    state-space and sequence-model programs consume the per-step
    input slice from it."""

    monadic: MonadicProgram | None
    """The compiled [`MonadicProgram`][quivers.continuous.programs.MonadicProgram]
    the synthetic-data block bound to `model` (or, for state-space
    examples, `inner`). Parametric program templates are exported as
    a `Program` with `_morphism=None`; the block invokes the template
    at concrete arguments (e.g. `prog.gmm(alpha=1.0)`) and binds the
    instantiated MonadicProgram. The QVR probe consumes this directly
    rather than re-parsing the source, so it sees the same template
    instance the data was generated under."""


def md_path_for(source_qvr: Path) -> Path:
    """The docs `.md` corresponding to a `docs/examples/source/<stem>.qvr`."""
    md_stem = source_qvr.stem.replace("_", "-")
    return _GALLERY_DOCS / f"{md_stem}.md"


_OBSERVE_NAME_RE = re.compile(
    r"^\s*observe\s+([A-Za-z_][A-Za-z_0-9]*)\b",
    re.MULTILINE,
)

_SAMPLE_NAME_RE = re.compile(
    r"^\s*sample\s+([A-Za-z_][A-Za-z_0-9]*)\b",
    re.MULTILINE,
)


def _qvr_observe_names(source_qvr: Path) -> list[str]:
    """Extract every `observe <name>` binder from the QVR source.
    Surface read rather than full parse; the regex is conservative
    (matches the `observe IDENT` prefix of an observe step), so it
    intentionally returns nothing on a source it can't recognise.
    """
    try:
        text = source_qvr.read_text(encoding="utf-8")
    except OSError:
        return []
    return _OBSERVE_NAME_RE.findall(text)


def _qvr_sample_names(source_qvr: Path) -> list[str]:
    """Extract every `sample <name>` binder from the QVR source.

    Used to pair `Point.params` with the program's latent-variable
    names: a `.md` snippet may bind the ground-truth value to either
    ``<name>``, ``<name>_true``, or ``true_<name>``, and the loader
    needs the list of latent names to recognise any of those three
    spellings as the same parameter.
    """
    try:
        text = source_qvr.read_text(encoding="utf-8")
    except OSError:
        return []
    return _SAMPLE_NAME_RE.findall(text)


def _sample_site_names(
    source_qvr: Path, monadic: MonadicProgram | None,
) -> list[str]:
    """Return the program's actual latent sample-site names.

    Observed steps (`observe <name>`) are excluded: their values come
    from the observations dict, not the captured ground truth. When
    the compiled
    [`MonadicProgram`][quivers.continuous.programs.MonadicProgram] is
    available it is the authority: template inlining alpha-renames an
    inner draw (``sample z`` inside a ``sample theta <- template(...)``
    step) to ``theta$z``, and folds the outer name into a
    deterministic ``let``. The compiled step specs carry the
    post-inline names the trace actually clamps, whereas the raw
    ``sample <name>`` regex reports the pre-inline source names and so
    would key the ground truth to a name no site answers to. Falls
    back to the source-level regex only when no compiled program was
    captured (the snippet bound neither ``model`` nor ``inner``).
    """
    if monadic is not None:
        names: list[str] = []
        for spec in monadic._step_specs:
            if isinstance(spec, (_LetSpec, _ScoreSpec)) or spec.is_observed:
                continue
            names.extend(spec.vars)
        return names
    return _qvr_sample_names(source_qvr)


def _is_tensor_like(value: object) -> bool:
    """True iff `value` is a torch.Tensor or a list of numerics
    convertible to one (the loader's two accepted shapes)."""
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(
            isinstance(x, (int, float, list, tuple)) for x in value
        )
    return False


def _observations_from_namespace(
    source_qvr: Path, ns: dict[str, object],
) -> dict[str, object]:
    """Build an observations dict by matching the QVR program's
    `observe <name>` binders against tensors in `ns` by name.

    Strategy, in order:

    1. Direct hit: `ns[<observe_name>]` is a tensor.
    2. Prefixed hit: `ns["obs_" + <observe_name>]` is a tensor.
    3. Common alias: a single namespace tensor whose name matches a
       conventional observation alias (`counts`, `targets`, `Y`, `y`,
       `data`). Used only when the program has exactly one observe
       step; ambiguous otherwise.

    Returns an empty dict when no match is found; the caller treats
    that as "no observation data was generated."
    """
    observe_names = _qvr_observe_names(source_qvr)
    if not observe_names:
        return {}
    out: dict[str, object] = {}
    for name in observe_names:
        if name in ns and _is_tensor_like(ns[name]):
            out[name] = ns[name]
            continue
        prefixed = f"obs_{name}"
        if prefixed in ns and _is_tensor_like(ns[prefixed]):
            out[name] = ns[prefixed]
            continue
    if out:
        return out
    if len(observe_names) == 1:
        name = observe_names[0]
        for alias in ("counts", "targets", "Y", "y", "data", "obs"):
            if alias in ns and _is_tensor_like(ns[alias]):
                return {name: ns[alias]}
    return out


def load_gallery_data(source_qvr: Path) -> GalleryDataset | None:
    """Run the example's `### Generating synthetic data` block and
    return its observations + captured ground-truth `true_*` params.

    Returns None when the doc lacks the data block, the snippet
    raises, or the resulting `observations` is not a dict of
    tensors."""
    md = md_path_for(source_qvr)
    if not md.exists():
        return None
    text = md.read_text(encoding="utf-8")
    match = _DATA_BLOCK_RE.search(text)
    if match is None:
        return None
    snippet = match.group(1)

    # Build a controlled namespace. Pre-bind `torch` so the snippet
    # does not need to import it again; pre-import `load` and seed.
    ns: dict[str, object] = {
        "__name__": "__gallery_data__",
        "torch": torch,
    }
    try:
        exec(compile(snippet, str(md), "exec"), ns)
    except Exception:
        return None

    observations = ns.get("observations")
    if not isinstance(observations, dict):
        # `sites` is the alternative idiomatic name in state-space
        # examples (continuous_hmm threads observed step values via
        # `sites = {"s_new": ..., "o": ...}` rather than an
        # `observations` dict).
        sites = ns.get("sites")
        if isinstance(sites, dict):
            observations = sites
    if not isinstance(observations, dict):
        # Fall back: pull observations from the namespace by name
        # match against the QVR program's `observe <name> : ...`
        # binders. This lets a doc's synthetic-data block bind its
        # generated tensors to natural variable names (e.g.
        # `obs_counts`, `targets`, `y`) without also assembling a
        # separate `observations` dict.
        observations = _observations_from_namespace(source_qvr, ns)
        if not observations:
            return None
    obs_tensors: dict[str, torch.Tensor] = {}
    for k, v in observations.items():
        if isinstance(v, torch.Tensor):
            obs_tensors[k] = v
        else:
            try:
                obs_tensors[k] = torch.as_tensor(v, dtype=torch.float64)
            except (TypeError, ValueError):
                return None

    # Compiled MonadicProgram. Templates compile to `Program(None)`
    # with `program.templates[<name>]` invokers; the synthetic-data
    # block instantiates the template at concrete arguments and binds
    # the result to `model` (or, for examples that wire the bare
    # morphism, `inner`). Capturing either lets the QVR probe walk
    # the instantiated program directly, and pins the ground-truth
    # capture below to the program's actual (post-inline) site names.
    monadic: MonadicProgram | None = None
    for monad_name in ("model", "inner"):
        candidate = ns.get(monad_name)
        if isinstance(candidate, MonadicProgram):
            monadic = candidate
            break

    # Match every sample site against the namespace under three
    # spellings: bare `<site>`, suffixed `<site>_true`, and prefixed
    # `true_<site>`, keying the captured value under the site's real
    # name so the trace clamps it. A template-inlined site carries a
    # `$` (`theta$z`); `$` is not a legal Python identifier char, so
    # the snippet spells it with `_` (`true_theta_z`) and the matcher
    # accepts that normalized base too. The bare spelling is accepted
    # only for a real sample site, so intermediate snippet bindings
    # (`T = 64`, `model = ...`, the let-step `mu_true` / `alpha_true`
    # intermediates) never get mis-captured as ground truth.
    sample_sites = _sample_site_names(source_qvr, monadic)
    params: dict[str, torch.Tensor] = {}

    def _coerce(value: object) -> torch.Tensor | None:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=torch.float64)
        if isinstance(value, (int, float)):
            return torch.tensor(float(value), dtype=torch.float64)
        if isinstance(value, (list, tuple)):
            try:
                return torch.as_tensor(value, dtype=torch.float64)
            except (TypeError, ValueError):
                return None
        return None

    for site in sample_sites:
        if site in params:
            continue
        bases = [site]
        normalized = site.replace("$", "_")
        if normalized != site:
            bases.append(normalized)
        for base in bases:
            hit: torch.Tensor | None = None
            for spelling in (f"true_{base}", f"{base}_true", base):
                if spelling not in ns:
                    continue
                hit = _coerce(ns[spelling])
                if hit is not None:
                    break
            if hit is not None:
                params[site] = hit
                break

    # Program-input tensor: the snippet may bind any of the canonical
    # names below. Try each in order; the first tensor-typed binding
    # wins. State-space examples conventionally use `state_prev`;
    # sequence and template examples use `x_in`; transformer-style
    # examples use `x`.
    x_input: torch.Tensor | None = None
    for x_name in ("x_in", "x", "state_prev", "x_lift"):
        candidate = ns.get(x_name)
        if isinstance(candidate, torch.Tensor):
            x_input = candidate
            break

    return GalleryDataset(
        observations=obs_tensors,
        params=params,
        x_input=x_input,
        monadic=monadic,
    )


def gallery_examples_with_data() -> list[Path]:
    """Return every `docs/examples/source/*.qvr` whose `.md` carries
    a `### Generating synthetic data` block."""
    out: list[Path] = []
    for qvr in sorted(_GALLERY_SOURCE.glob("*.qvr")):
        md = md_path_for(qvr)
        if not md.exists():
            continue
        text = md.read_text(encoding="utf-8")
        if _DATA_BLOCK_RE.search(text) is not None:
            out.append(qvr)
    return out


def point_from_dataset(dataset: GalleryDataset) -> Point:
    """Build a single `Point` from a GalleryDataset.

    The Point's `params` map carries the captured `true_*` ground-
    truth parameter values (one tuple-typed entry per latent name);
    `data` carries every observation tensor as a flat list.

    When a name appears in both `dataset.params` (captured as a
    ground-truth latent) and `dataset.observations` (clamped in the
    `.md` snippet's observations dict for the SVI demo), the params
    spelling wins: the latent ground truth is the canonical value to
    score the joint at, and the entry is dropped from the data section
    so a backend that declares the name as a parameter (Stan's
    `parameters {}` block, a PyMC unobserved RV) does not also receive
    it as a data input. `dataset.observations` itself is left intact,
    so the in-process QVR trace still clamps every site."""
    return _point_from_tensors(dataset.params, dataset.observations)


def _point_from_tensors(
    params: dict[str, torch.Tensor],
    observations: dict[str, torch.Tensor],
) -> Point:
    """Flatten a (latent, observed) tensor pair into a wire
    [`Point`][tests.transpile.probes._protocol.Point].

    Every value becomes a row-major float list; a length-1 list
    collapses to a bare float so the probe's dict-to-Tensor casting
    picks the scalar shape. A name bound in both sections is emitted
    only under ``params`` (see
    [`point_from_dataset`][tests.transpile._gallery_data.point_from_dataset]).
    """
    def _flatten(t: torch.Tensor) -> list[float]:
        return t.detach().to(dtype=torch.float64).flatten().tolist()
    flat_params = {k: _flatten(v) for k, v in params.items()}
    flat_data = {
        k: _flatten(v)
        for k, v in observations.items()
        if k not in params
    }
    squeezed_params: dict[str, float | int | list[float] | list[int]] = {
        k: (v[0] if len(v) == 1 else v) for k, v in flat_params.items()
    }
    squeezed_data: dict[str, float | int | list[float] | list[int]] = {
        k: (v[0] if len(v) == 1 else v) for k, v in flat_data.items()
    }
    return Point(params=squeezed_params, data=squeezed_data)


# ---------------------------------------------------------------------------
# Multi-point evaluation.
#
# Theorem 4.1 of `docs/semantics/transpile-correctness.md` fixes the
# transpile contract as CONSTANT-spread equivalence: the pointwise
# difference `log p_QVR - log p_backend` must be the same constant `c`
# at every (theta, y) in the support. A one-point comparison cannot
# test that -- the spread of a single difference around its own mean is
# identically zero -- so the check is vacuous until the point set has
# real variation in BOTH the latents and the observed data. Latent-only
# variation is not enough either: a backend that drops a data-dependent
# term (Stan's `~` sampling statement discards data-only summands) keeps
# a constant offset as the latents move and only breaks constancy when
# the data moves.
# ---------------------------------------------------------------------------


PERTURB_GROUND_TRUTH = "ground-truth"
"""Label of point 0: the captured ground truth, unperturbed."""

PERTURB_LATENTS = "latents"
"""Label of a point whose latent sites moved and whose data is at
ground truth. Isolates a backend that mis-scores the prior."""

PERTURB_DATA = "data"
"""Label of a point whose observed data moved and whose latents are at
ground truth. Isolates a backend that drops a data-dependent term."""

PERTURB_BOTH = "latents+data"
"""Label of a point where both sections moved. Catches an offset that
cancels when only one section varies."""

_PERTURBATION_CYCLE: tuple[str, ...] = (
    PERTURB_LATENTS,
    PERTURB_DATA,
    PERTURB_BOTH,
)
"""Mode schedule for points 1..n-1, cycled by index. The cycle puts a
latents-only and a data-only point before the joint one, so a broken
constancy localises to the section that moved."""

_PERTURBATION_SCALE = 0.2
"""Base perturbation magnitude, in the natural unconstrained
coordinate of each value's support (log scale for positive values,
logit scale for bounded ones, additive nats for reals). Large enough
that a dropped data term or a truncated latent shifts the difference
by orders of magnitude more than float round-off; small enough that
the joint stays well inside the region where every backend's
log-density is numerically well behaved."""

_INTEGER_STEP_FRACTION = 0.25
"""Fraction of a count vector's mean magnitude used as the standard
deviation of its integer perturbation, floored at one count so a
small-count fixture still moves."""

_BOOLEAN_FLIP_PROBABILITY = 0.25
"""Per-entry flip probability for a Boolean-supported value."""

_SUPPORT_EPS = 1e-12
"""Floor applied before taking a log / logit, so a ground-truth value
sitting numerically on a support boundary perturbs to a finite
interior point instead of producing a non-finite coordinate."""

_MAX_REDRAWS = 6
"""Attempts per point before giving up. Each retry halves the
perturbation scale, so the last attempt is 1/32 of the base scale."""

_MAX_FAMILY_DEPTH = 8
"""Depth bound on the wrapper-unwrapping walk in
[`_resolve_support`][tests.transpile._gallery_data._resolve_support]."""


def perturbation_labels(n_points: int = 6) -> list[str]:
    """Per-index perturbation label for a
    [`points_from_dataset`][tests.transpile._gallery_data.points_from_dataset]
    point list of the same length.

    Index 0 is always
    [`PERTURB_GROUND_TRUTH`][tests.transpile._gallery_data.PERTURB_GROUND_TRUTH];
    every later index cycles through `_PERTURBATION_CYCLE`. The
    schedule is a pure function of the index, so a failure message can
    name the perturbation that broke constancy without threading the
    labels through the point set.
    """
    if n_points <= 0:
        raise ValueError(
            f"n_points must be positive, got {n_points!r}"
        )
    return [PERTURB_GROUND_TRUTH] + [
        _PERTURBATION_CYCLE[(i - 1) % len(_PERTURBATION_CYCLE)]
        for i in range(1, n_points)
    ]


def _resolve_support(
    morphism: ContinuousMorphism,
) -> constraints.Constraint | None:
    """The support constraint of the distribution a draw step samples.

    A plate draw / vectorised observe wraps a per-row family and
    inherits its support, but the generic
    [`ContinuousMorphism.support`][quivers.continuous.morphisms.ContinuousMorphism.support]
    default is `real`, so a wrapper that does not forward the inner
    constraint reports `real` for a positively-supported family. The
    walk therefore descends into `.family` whenever the current level
    reports the `real` default and an inner family exists, and returns
    the first non-default constraint it finds.
    """
    current: ContinuousMorphism = morphism
    for _ in range(_MAX_FAMILY_DEPTH):
        support = getattr(current, "support", None)
        inner = getattr(current, "family", None)
        if (
            inner is not None
            and isinstance(inner, ContinuousMorphism)
            and isinstance(support, type(constraints.real))
        ):
            current = inner
            continue
        return support if isinstance(support, constraints.Constraint) else None
    support = getattr(current, "support", None)
    return support if isinstance(support, constraints.Constraint) else None


def site_supports(
    dataset: GalleryDataset,
) -> dict[str, constraints.Constraint]:
    """Declared support constraint per stochastic site of the example's
    compiled program, keyed by the site's post-inline name.

    This is the authority on how a value may be perturbed and on
    whether it is integer- or real-valued: it comes from the family the
    QVR source declared for the site, not from the ground-truth value
    the synthetic-data block happened to draw. Returns an empty map
    when the example's `.md` bound no compiled
    [`MonadicProgram`][quivers.continuous.programs.MonadicProgram].
    """
    monadic = dataset.monadic
    if monadic is None:
        return {}
    out: dict[str, constraints.Constraint] = {}
    for spec in monadic._step_specs:
        if isinstance(spec, (_LetSpec, _ScoreSpec)):
            continue
        morphism = monadic._modules.get(spec.morphism_name)
        if not isinstance(morphism, ContinuousMorphism):
            continue
        support = _resolve_support(morphism)
        if support is None:
            continue
        for name in spec.vars:
            out[name] = support
    return out


def _base_constraint(
    support: constraints.Constraint,
) -> constraints.Constraint:
    """Strip the event-axis wrapper an
    [`independent`][torch.distributions.constraints.independent]
    constraint puts around a per-coordinate one (`real_vector` is
    `independent(real, 1)`), so the per-coordinate constraint is what
    decides the dtype tag and the perturbation coordinate."""
    current = support
    for _ in range(_MAX_FAMILY_DEPTH):
        inner = getattr(current, "base_constraint", None)
        if not isinstance(inner, constraints.Constraint):
            return current
        current = inner
    return current


def is_discrete_support(support: constraints.Constraint) -> bool:
    """True iff `support` admits only integer-valued points.

    Used to decide the wire dtype tag for a site: an integer-supported
    site must reach a backend that separates `int` from `real` (Stan,
    JAGS, BUGS) as an integer, and a continuous-supported site must
    reach it as a float even when its ground-truth value happens to
    have no fractional part.
    """
    return isinstance(
        _base_constraint(support),
        (
            constraints.integer_interval,
            type(constraints.nonnegative_integer),
            type(constraints.boolean),
        ),
    )


def _perturb_integer(
    work: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Move an integer-valued vector by a small integer delta.

    The result is clamped into the intersection of the declared
    support bounds and the value's own observed range. Clamping to the
    observed range is what keeps a count observation in support when
    the declared constraint is looser than the model's real alphabet:
    a categorical emission declares `IntegerGreaterThan(0)` while the
    emission row has finite width, so an unbounded upward step would
    index past the row and send the joint to `-inf`. A vector whose
    entries are all equal has an empty range and stays put.
    """
    if work.numel() == 0:
        return work
    magnitude = max(
        1.0, _INTEGER_STEP_FRACTION * float(work.abs().mean().item()),
    )
    moved = torch.round(work + torch.round(noise * magnitude))
    low = max(lower, float(work.min().item()))
    high = min(upper, float(work.max().item()))
    if low > high:
        return work
    return moved.clamp(min=low, max=high)


def _perturb_lower_cholesky(
    work: torch.Tensor, noise: torch.Tensor, scale: float,
) -> torch.Tensor:
    """Perturb a lower-triangular factor with positive diagonal.

    The diagonal moves multiplicatively (staying strictly positive)
    and the strict lower triangle moves additively, so the result is
    still a valid Cholesky factor and the covariance it induces stays
    positive definite.
    """
    tril = torch.tril(work)
    diag = torch.diagonal(tril, dim1=-2, dim2=-1)
    diag_noise = torch.diagonal(noise, dim1=-2, dim2=-1)
    moved = torch.tril(tril + scale * noise, diagonal=-1)
    new_diag = diag.clamp_min(_SUPPORT_EPS) * torch.exp(scale * diag_noise)
    return moved + torch.diag_embed(new_diag)


def _perturb_by_support(
    value: torch.Tensor,
    support: constraints.Constraint,
    generator: torch.Generator,
    scale: float,
) -> torch.Tensor | None:
    """Move `value` inside `support` by roughly `scale` in the
    support's natural unconstrained coordinate.

    Returns None for a constraint whose interior this helper cannot
    parameterise; the caller then leaves the value at ground truth
    rather than risk stepping outside the support and comparing two
    `-inf` joints (which would be a vacuous match of a different
    kind).
    """
    support = _base_constraint(support)
    work = value.detach().to(dtype=torch.float64)
    noise = torch.randn(
        work.shape, generator=generator, dtype=torch.float64,
    )
    moved: torch.Tensor
    if isinstance(support, type(constraints.simplex)):
        logits = torch.log(work.clamp_min(_SUPPORT_EPS)) + scale * noise
        moved = torch.softmax(logits, dim=-1)
    elif isinstance(support, type(constraints.boolean)):
        uniform = torch.rand(
            work.shape, generator=generator, dtype=torch.float64,
        )
        moved = torch.where(
            uniform < _BOOLEAN_FLIP_PROBABILITY, 1.0 - work, work,
        )
    elif isinstance(support, constraints.integer_interval):
        moved = _perturb_integer(
            work, noise,
            float(support.lower_bound), float(support.upper_bound),
        )
    elif isinstance(support, type(constraints.nonnegative_integer)):
        moved = _perturb_integer(
            work, noise, float(support.lower_bound), math.inf,
        )
    elif isinstance(
        support, (constraints.interval, constraints.half_open_interval),
    ):
        lower = float(support.lower_bound)
        upper = float(support.upper_bound)
        width = upper - lower
        unit = ((work - lower) / width).clamp(
            _SUPPORT_EPS, 1.0 - _SUPPORT_EPS,
        )
        logit = torch.log(unit) - torch.log1p(-unit)
        moved = lower + width * torch.sigmoid(logit + scale * noise)
    elif isinstance(
        support, (constraints.greater_than, constraints.greater_than_eq),
    ):
        lower = float(support.lower_bound)
        moved = lower + (work - lower).clamp_min(
            _SUPPORT_EPS,
        ) * torch.exp(scale * noise)
    elif isinstance(support, constraints.less_than):
        upper = float(support.upper_bound)
        moved = upper - (upper - work).clamp_min(
            _SUPPORT_EPS,
        ) * torch.exp(scale * noise)
    elif isinstance(support, type(constraints.lower_cholesky)):
        moved = _perturb_lower_cholesky(work, noise, scale)
    elif isinstance(support, type(constraints.corr_cholesky)):
        rows = torch.tril(work + scale * torch.tril(noise, diagonal=-1))
        norms = rows.norm(dim=-1, keepdim=True).clamp_min(_SUPPORT_EPS)
        moved = rows / norms
    elif isinstance(support, type(constraints.positive_definite)):
        factor, info = torch.linalg.cholesky_ex(work)
        if int(info.max().item()) != 0:
            return None
        perturbed = _perturb_lower_cholesky(factor, noise, scale)
        moved = perturbed @ perturbed.transpose(-2, -1)
    elif isinstance(support, type(constraints.real)):
        moved = work + scale * noise
    else:
        return None
    return moved.to(dtype=value.dtype)


def _covariate_support(
    value: torch.Tensor,
) -> constraints.Constraint | None:
    """Constraint inferred from a covariate's own value domain.

    A name in the observations dict that answers to no stochastic site
    is a covariate the program reads through a `let`, so no declared
    family fixes its constraint. A value carrying a fractional part is
    unconstrained real and moves additively. An integer-valued one is
    left alone: a plate subscript and a count covariate are
    indistinguishable at this level, and stepping a subscript outside
    its plate would index past the gathered parameter rather than
    produce a different in-support point.
    """
    if value.numel() == 0:
        return None
    if not value.dtype.is_floating_point:
        return None
    if torch.equal(value, value.round()):
        return None
    return constraints.real


def _perturb_section(
    section: dict[str, torch.Tensor],
    supports: dict[str, constraints.Constraint],
    generator: torch.Generator,
    scale: float,
    *,
    infer_from_value: bool,
    exclude: frozenset[str] = frozenset(),
) -> dict[str, torch.Tensor]:
    """Perturb every value in `section` whose constraint is known.

    A name with a declared site support moves under that constraint. A
    name without one falls back to
    [`_covariate_support`][tests.transpile._gallery_data._covariate_support]
    when `infer_from_value` is set (the observations dict, which mixes
    observe sites with plain covariates), and otherwise stays at ground
    truth. Names in `exclude` are copied through untouched.
    """
    out: dict[str, torch.Tensor] = {}
    for name, value in section.items():
        if name in exclude:
            out[name] = value
            continue
        support = supports.get(name)
        if support is None and infer_from_value:
            support = _covariate_support(value)
        if support is None:
            out[name] = value
            continue
        moved = _perturb_by_support(value, support, generator, scale)
        out[name] = value if moved is None else moved
    return out


def observations_for_point(
    dataset: GalleryDataset, point: Point,
) -> dict[str, torch.Tensor]:
    """Rebuild the pre-shaped observation dict the in-process
    [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe] clamps with, for
    one point of a
    [`points_from_dataset`][tests.transpile._gallery_data.points_from_dataset]
    list.

    The probe's `observations` keyword takes precedence over the flat
    per-point payload, because it is the only channel that preserves
    the multi-axis shapes flattening discards. That precedence makes it
    mandatory here: passing `dataset.observations` unchanged alongside
    a perturbed point would silently score the QVR side at the
    ground-truth data while the backend scored the perturbed data, and
    the resulting mismatch would look like a backend bug. Each entry is
    therefore inflated back from the point (from `data`, or from
    `params` for a name the point carries as a latent) into the
    reference tensor's shape and dtype.
    """
    out: dict[str, torch.Tensor] = {}
    for name, reference in dataset.observations.items():
        if name in point.data:
            flat = point.data[name]
        elif name in point.params:
            flat = point.params[name]
        else:
            out[name] = reference
            continue
        values = (
            [float(flat)]
            if isinstance(flat, (int, float))
            else [float(v) for v in flat]
        )
        out[name] = (
            torch.tensor(values, dtype=torch.float64)
            .reshape(tuple(reference.shape))
            .to(dtype=reference.dtype)
        )
    return out


def _qvr_log_density(
    dataset: GalleryDataset, point: Point, fixture: str,
) -> float:
    """Joint log-density of the compiled program at one point."""
    monadic = dataset.monadic
    if monadic is None:
        raise RuntimeError(
            f"{fixture!r}: no compiled MonadicProgram to score"
        )
    result = QvrProbe().evaluate(
        b"",
        fixture,
        [point],
        # The in-process probe writes nothing; the path is required by
        # the probe Protocol and discarded on entry.
        scratch=_GALLERY_DOCS,
        monadic=monadic,
        x_input=dataset.x_input,
        observations=observations_for_point(dataset, point),
    )
    return result.log_densities[0]


def points_from_dataset(
    dataset: GalleryDataset,
    n_points: int = 6,
    seed: int = 0,
) -> list[Point]:
    """Build a deterministic multi-point evaluation set for `dataset`.

    Point 0 is the captured ground truth. Every later point perturbs
    the latents, the observed data, or both, following the schedule
    [`perturbation_labels`][tests.transpile._gallery_data.perturbation_labels]
    reports for the same length. Each value moves inside its own
    support: a positive scale moves multiplicatively, a bounded value
    moves in logit space, a simplex row is renormalised, a Cholesky
    factor keeps its triangular / positive-diagonal shape, and an
    integer count takes an integer step clamped to the attested range.
    A value whose constraint the harness cannot establish -- an
    integer-valued covariate, which is indistinguishable from a plate
    subscript -- stays at ground truth.

    Parameters
    ----------
    dataset
        The example's synthetic data and captured ground truth.
    n_points
        Total points to return, ground truth included. The default of
        6 gives two latents-only, two data-only, and one joint
        perturbation, which is enough for the constant-spread check to
        separate a prior-scoring bug from a dropped data term.
    seed
        Seed of a local [`torch.Generator`][torch.Generator]. The
        global RNG is never touched, so the point set is reproducible
        run to run and independent of whatever the example's
        synthetic-data snippet seeded.

    Returns
    -------
    list[Point]
        Exactly `n_points` points, in schedule order.

    Raises
    ------
    AssertionError
        When a perturbed point still scores a non-finite QVR joint
        after `_MAX_REDRAWS` attempts at successively halved scales.
        A perturbation that cannot be brought back into support is a
        constraint this module models wrongly, not a tolerable point
        to drop: dropping it would shift every later index and
        silently weaken the check.
    """
    labels = perturbation_labels(n_points)
    ground_truth = point_from_dataset(dataset)
    points: list[Point] = [ground_truth]
    if n_points == 1:
        return points

    fixture = "points_from_dataset"
    # Validate against the QVR joint only when the ground-truth point
    # itself scores finitely. An example whose oracle cannot score the
    # program at all (a non-deterministic composition marginalisation,
    # a free latent the snippet never bound) gives no baseline to
    # measure a perturbation against, so its points are emitted
    # unvalidated rather than measured against a broken reference.
    validate = False
    if dataset.monadic is not None:
        try:
            baseline = _qvr_log_density(dataset, ground_truth, fixture)
        except RuntimeError:
            validate = False
        else:
            validate = math.isfinite(baseline)

    generator = torch.Generator()
    generator.manual_seed(seed)
    supports = site_supports(dataset)
    # A name bound in both sections is a latent whose value the
    # snippet also clamps for its SVI demo. The latent spelling is
    # canonical, so the data pass leaves it alone and the latent pass
    # owns it.
    shared_names = frozenset(dataset.params) & frozenset(
        dataset.observations
    )

    for index in range(1, n_points):
        mode = labels[index]
        for attempt in range(_MAX_REDRAWS):
            scale = _PERTURBATION_SCALE * (0.5 ** attempt)
            params = (
                _perturb_section(
                    dataset.params, supports, generator, scale,
                    infer_from_value=False,
                )
                if mode in (PERTURB_LATENTS, PERTURB_BOTH)
                else dict(dataset.params)
            )
            observations = (
                _perturb_section(
                    dataset.observations, supports, generator, scale,
                    infer_from_value=True, exclude=shared_names,
                )
                if mode in (PERTURB_DATA, PERTURB_BOTH)
                else dict(dataset.observations)
            )
            candidate = _point_from_tensors(params, observations)
            if not validate:
                points.append(candidate)
                break
            if math.isfinite(_qvr_log_density(dataset, candidate, fixture)):
                points.append(candidate)
                break
        else:
            raise AssertionError(
                f"point {index} ({mode}): every perturbation attempt "
                f"down to scale {_PERTURBATION_SCALE * 0.5 ** (_MAX_REDRAWS - 1):.4g} "
                f"left the QVR joint non-finite, so no in-support "
                f"point exists under the constraints this module "
                f"derived. Sites: {sorted(supports)}."
            )
    return points


__all__ = [
    "GalleryDataset",
    "PERTURB_BOTH",
    "PERTURB_DATA",
    "PERTURB_GROUND_TRUTH",
    "PERTURB_LATENTS",
    "gallery_examples_with_data",
    "is_discrete_support",
    "load_gallery_data",
    "md_path_for",
    "observations_for_point",
    "perturbation_labels",
    "point_from_dataset",
    "points_from_dataset",
    "site_supports",
]
