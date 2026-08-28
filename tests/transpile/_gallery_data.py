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
4. Relabels the rows of every plate that carries a structural
   subscript, so no subscript coincides with the row counter (see
   [`_dealias_row_order`][tests.transpile._gallery_data._dealias_row_order]).
   A row relabeling leaves the joint exactly invariant, since the
   density is a product over the plate's rows.
5. Returns the resulting `observations` mapping, the ground-truth
   value of every latent sample site (a representative point in
   latent space for the numeric-equivalence test), and the scalar
   type-parameters the snippet instantiated a parametric program
   template at.

The extraction is fail-soft: examples whose data-gen block is
absent or whose snippet raises return None, and the caller skips
the cell with a clear reason.

Alongside the data it also owns the *isolation* of the out-of-process
measurements the gallery tiers run against that data:
[`probe_scratch`][tests.transpile._gallery_data.probe_scratch] hands
out the bind-mounted directory a container reads its inputs from, and
[`probe_script_digests`][tests.transpile._gallery_data.probe_script_digests]
records the identity of the probe sources the harness copies into it.
Both exist because a measurement that shares either with anything else
stops being a measurement of the cell under test; see `probe_scratch`
for what sharing actually costs.
"""

from __future__ import annotations

import ast
import atexit
import dataclasses
import hashlib
import math
import os
import re
import shutil
import tempfile
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

PROBE_SCRIPT_DIR = Path(__file__).resolve().parent / "probes" / "_scripts"
"""Directory of the per-backend probe entrypoints and their shared
reshape helpers.

[`run_probe`][tests.transpile._docker.run_probe] copies the selected
entrypoint plus `_reshape.py` / `_reshape.jl` out of here and into the
bind-mounted scratch at launch time, so the container executes whatever
these files hold *at that instant* rather than whatever they held when
the session started."""


PROBE_SCRATCH_PARENT = Path("/tmp")
"""Parent of the per-process probe-scratch root.

`/tmp` rather than [`tempfile.gettempdir`][tempfile.gettempdir]:
`run_probe` bind-mounts the scratch into a container, and `/tmp`
(`/private/tmp` once resolved) is on the host-sharing list every Docker
installation the matrix runs against exposes by default, while the
per-user `$TMPDIR` on macOS is not guaranteed to be."""


_PROBE_SCRATCH_ROOT: Path | None = None
"""Lazily-created root holding every scratch this *process* hands out.

One per interpreter, named after the pid and a
[`tempfile`][tempfile.mkdtemp] suffix, and removed at interpreter exit.
Two concurrent pytest processes on one machine therefore never see each
other's probe inputs."""


_PROBE_LABEL_RE = re.compile(r"[^0-9A-Za-z._-]+")

_PROBE_ROOT_RE = re.compile(r"^quivers-probe-(\d+)-[0-9A-Za-z_]+$")
"""Shape of a probe-scratch root name, capturing the owning pid.

The pid is what makes an abandoned root identifiable. A root is swept
at interpreter exit, which a killed run never reaches, so the leftovers
have to be distinguishable from the roots of processes still using
them, and only the owner's pid does that."""


def sweep_abandoned_probe_roots(parent: Path | None = None) -> list[Path]:
    """Remove probe-scratch roots whose owning interpreter is gone.

    A root belonging to a live pid is in use right now, possibly by
    another session on this machine, and is never touched. A root whose
    pid names no process belongs to nobody: the run that owned it was
    killed before its exit hook could sweep it, and what it holds (a
    compiled model, a point set, a container's result) is of no use to
    anyone.

    Parameters
    ----------
    parent
        Directory to sweep, defaulting to
        [`PROBE_SCRATCH_PARENT`][tests.transpile._gallery_data.PROBE_SCRATCH_PARENT].
        A test names its own so that exercising the sweep cannot reach
        the roots of concurrent sessions, which is the one thing this
        function must never do.

    Returns
    -------
    list[Path]
        The roots removed, in sorted order, so a caller can report what
        a run reclaimed.
    """
    root_parent = PROBE_SCRATCH_PARENT if parent is None else parent
    swept: list[Path] = []
    for entry in sorted(root_parent.glob("quivers-probe-*")):
        if not entry.is_dir():
            continue
        match = _PROBE_ROOT_RE.match(entry.name)
        if match is None:
            continue
        pid = int(match.group(1))
        if pid <= 0:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # No such process: the owner is gone and cannot come back
            # under this pid without the OS having recycled it, which
            # would make the directory its own anyway.
            shutil.rmtree(entry, ignore_errors=True)
            swept.append(entry)
        except (OverflowError, PermissionError):
            # Out of pid range, or owned by another user whose live
            # process this one may not signal. Either way the root is
            # not established as abandoned, so it stays.
            continue
    return swept


def probe_scratch(label: str) -> Path:
    """A freshly-created, empty, process-private directory for one
    out-of-process probe run.

    Every caller of [`run_probe`][tests.transpile._docker.run_probe]
    must route its `scratch` through here, and must call it again for
    each run rather than reusing the path it got back.

    Why a fixed path is not merely untidy
    -------------------------------------

    `run_probe` writes the container's *inputs* (`source.<ext>`,
    `points.json`, `shapes.json`, `dtypes.json`, `probe.py`,
    `_reshape.py`, `_reshape.jl`) into the directory, launches the
    container against it, and reads the container's `result.json` back
    out. A path derived from the cell alone, `/tmp/<prefix>_<model>_<backend>`,
    is therefore a rendezvous point that is shared by

    1. every earlier run of the same cell, whose artefacts the writer
       does not remove: a compiled Stan binary, a container-written
       `__pycache__`, a `shapes.json` a later caller may not overwrite
       because it passed `None`;
    2. every *concurrent* process running that cell, including one on a
       different checkout of the tree, which may write a different
       point set into `points.json` between this process's write and
       its container's read, or leave a `result.json` this process then
       reads as its own answer.

    Neither corruption announces itself. The container still runs, the
    probe still returns finite log-densities, and the harness attributes
    the resulting mismatch to the *model*. Worse, the corruption is
    shared: a foreign point set in a scratch is not a property of any
    one backend, so the same wrong number lands on every backend of that
    model at once and reads as a genuine, model-specific finding rather
    than as a harness fault.

    The directory returned here is created fresh by
    [`tempfile.mkdtemp`][tempfile.mkdtemp], so it is empty, is not the
    path any other call returned, and cannot be guessed by another
    process. `label` only makes the path legible while a run is in
    flight; it carries no uniqueness, and repeating it is fine.

    Parameters
    ----------
    label
        Human-readable tag for the run, conventionally
        `"<tier>-<model>-<backend>"`. Characters outside
        `[0-9A-Za-z._-]` are folded to `_`.

    Returns
    -------
    Path
        A directory that exists, is empty, and is returned exactly once.

    Raises
    ------
    ValueError
        When `label` folds to the empty string, which would leave the
        run untraceable in a directory listing.
    """
    global _PROBE_SCRATCH_ROOT
    if _PROBE_SCRATCH_ROOT is None:
        sweep_abandoned_probe_roots()
        root = Path(
            tempfile.mkdtemp(
                prefix=f"quivers-probe-{os.getpid()}-",
                dir=PROBE_SCRATCH_PARENT,
            )
        )
        # A container may leave artefacts the host user cannot unlink
        # (a `__pycache__` written under a different uid mapping).
        # Interpreter shutdown is not the place to raise about that, so
        # the sweep is best-effort; what matters for correctness is that
        # nothing is ever *read* from a reused directory, which
        # `mkdtemp` already guarantees.
        atexit.register(shutil.rmtree, root, True)
        _PROBE_SCRATCH_ROOT = root
    safe = _PROBE_LABEL_RE.sub("_", label).strip("_")
    if not safe:
        raise ValueError(
            f"probe_scratch label {label!r} folds to the empty string, "
            f"so the run would be anonymous in a directory listing. "
            f"Pass a tag with at least one of [0-9A-Za-z._-]."
        )
    return Path(tempfile.mkdtemp(prefix=f"{safe}-", dir=_PROBE_SCRATCH_ROOT))


def probe_scratch_root() -> Path | None:
    """The per-process scratch root, or None before the first
    [`probe_scratch`][tests.transpile._gallery_data.probe_scratch] call.

    Exposed so a test can assert the root differs between processes,
    which is the property a fixed `/tmp/<prefix>_<cell>` path violates
    and the one that keeps two concurrent runs from trading inputs.
    """
    return _PROBE_SCRATCH_ROOT


def probe_script_digests() -> dict[str, str]:
    """SHA-256 of every file in
    [`PROBE_SCRIPT_DIR`][tests.transpile._gallery_data.PROBE_SCRIPT_DIR],
    keyed by file name.

    `run_probe` copies these into the container at launch, so they are
    read from the working tree once per cell rather than once per
    session. An edit that lands mid-session therefore splits the run:
    cells measured before it used one helper and cells measured after
    it used another, with no record of which.

    `_reshape.py` makes that split maximally deceptive. It is the one
    file every Python-side probe imports, so a change to how it inflates
    a flat point payload back into the shapes the target declares moves
    the data *every* backend scores, and the resulting failure lands on
    all of them at once for whichever model was in flight.

    Compare a baseline captured at import against a fresh call to detect
    it; see
    [`assert_probe_scripts_unchanged`][tests.transpile._gallery_data.assert_probe_scripts_unchanged].
    """
    return {
        entry.name: hashlib.sha256(entry.read_bytes()).hexdigest()
        for entry in sorted(PROBE_SCRIPT_DIR.iterdir())
        if entry.is_file()
    }


def assert_probe_scripts_unchanged(
    baseline: dict[str, str], names: frozenset[str] | None = None,
) -> None:
    """Fail when the probe sources moved since `baseline` was taken.

    Parameters
    ----------
    baseline
        A [`probe_script_digests`][tests.transpile._gallery_data.probe_script_digests]
        mapping captured earlier, conventionally at module import.
    names
        Restrict the comparison to these file names. A cell copies only
        its own backend entrypoint and the two reshape helpers into its
        container, so those are the only files whose movement can
        change what *that* cell measured; an edit to some other
        backend's entrypoint is a fault of the cells that copied it,
        and each of those reports it for itself. Passing None compares
        the whole directory.

    Raises
    ------
    RuntimeError
        When any file in scope was added, removed, or rewritten. The
        message names the files, because the interesting question is
        which cells the edit contaminated, and that is answered by
        which helper moved.
    """
    current = probe_script_digests()
    candidates = set(baseline) | set(current)
    if names is not None:
        candidates &= names
    moved = sorted(
        name
        for name in candidates
        if baseline.get(name) != current.get(name)
    )
    if not moved:
        return
    raise RuntimeError(
        f"the probe sources under {PROBE_SCRIPT_DIR} changed while this "
        f"session was measuring: {moved!r}. `run_probe` copies them into "
        f"the container at launch, so a cell measured before the edit "
        f"and a cell measured after it ran against different harnesses, "
        f"and the numbers this cell reported are of unknown provenance. "
        f"This is a fault in the session, not in any model: re-run the "
        f"tier against a tree nothing else is writing to. A change to "
        f"`_reshape.py` in particular moves the data every Python-side "
        f"backend scores, so it surfaces as a whole row of one model's "
        f"cells failing together, which is indistinguishable by eye "
        f"from a finding about that model."
    )


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
    """One point in latent space at which to evaluate the joint
    log-density, keyed by the program's post-inline sample-site names.

    Two channels fill it. The first is the snippet's captured ground
    truth, bound as `<site>`, `true_<site>`, or `<site>_true`. The
    second is the observations dict itself: a state-space example ships
    its latent trajectory there (`sites = {"s_new": ..., "o": ...}`)
    because that dict is what its inference demos clamp, and the site
    is a latent all the same. Either way the value reaches a backend
    through the parameter channel, never through a model input the
    emitted program does not declare."""

    scalar_params: dict[str, float]
    """The exported program's scalar type-parameters, at the concrete
    values the synthetic-data block instantiated the template with.

    A parametric program header (`program lda(alpha : Real, beta :
    Real) : Token -> Mix`) renders as a required *input* of the emitted
    program on every backend: numpyro emits `def model(alpha, beta,
    word_idx, w=None)`, Stan declares `real alpha; real beta;` in its
    `data` block, JAGS reads `alpha` as an unknown variable. These
    names are neither `observe` binders nor `sample` sites, so they
    reach the container through neither
    [`observations`][tests.transpile._gallery_data.GalleryDataset] nor
    [`params`][tests.transpile._gallery_data.GalleryDataset]; the point
    builder emits them into `Point.data`, which is the section every
    probe script hands to the model as an input."""

    observe_names: frozenset[str]
    """Every `observe <name>` binder the QVR source declares.

    Distinguishes an observation the program scores from a plain
    covariate the program only reads (a plate subscript such as
    `word_idx` or `out_idx`). The distinction decides whether an
    integer-dtyped entry of the observations dict may be perturbed:
    stepping an observed count to a neighbouring in-support value is a
    real data perturbation, whereas stepping a subscript would gather a
    different parameter row rather than move the data."""

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

_EXPORT_NAME_RE = re.compile(
    r"^\s*export\s+([A-Za-z_][A-Za-z_0-9]*)\b",
    re.MULTILINE,
)

_PROGRAM_HEADER_RE = re.compile(
    r"^\s*program\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(([^)]*)\)",
    re.MULTILINE,
)

_TYPE_PARAM_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z_0-9]*)\s*:\s*([A-Za-z_][A-Za-z_0-9]*)\s*$",
)

_SCALAR_PARAM_TYPE = "Real"
"""The only declared type-parameter kind that reaches a backend as a
numeric input. A `FinSet`-typed parameter names an *object* the
compiler resolves at instantiation time (`school_effects(spread :
Real, K : FinSet)`), so it has no runtime value to send."""


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


def _exported_type_parameters(
    source_qvr: Path,
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Declared type-parameters of every program the source exports.

    Keyed by program name, each value is the header's parameter list in
    declaration order as `(name, type)` pairs. Only a header whose
    parameters are *typed* contributes: `program gru_cell(x_t, h_prev)`
    binds per-step values rather than type-parameters, and its
    untyped entries are dropped.

    Restricting to exported programs is what keeps an inner parametric
    program out of the table. `parametric_pooling` declares
    `program school_effects(spread : Real, K : FinSet)` and samples it
    from within `pooled_tight`; only `pooled_tight` is exported, so the
    emitted program takes no scalar input and none is sent.
    """
    try:
        text = source_qvr.read_text(encoding="utf-8")
    except OSError:
        return {}
    exported = frozenset(_EXPORT_NAME_RE.findall(text))
    out: dict[str, tuple[tuple[str, str], ...]] = {}
    for name, raw_params in _PROGRAM_HEADER_RE.findall(text):
        if name not in exported:
            continue
        declared: list[tuple[str, str]] = []
        for chunk in raw_params.split(","):
            typed = _TYPE_PARAM_RE.match(chunk)
            if typed is not None:
                declared.append((typed.group(1), typed.group(2)))
        if declared:
            out[name] = tuple(declared)
    return out


def _literal_argument(
    node: ast.expr, program: str, param: str, md: Path,
) -> float:
    """Value of one literal argument of a template invocation.

    The synthetic-data block instantiates a parametric program at
    constants (`prog.lda(alpha=1.0, beta=0.5)`), and those constants
    are the values every emitted program expects as inputs. A
    non-literal argument would make the harness guess at a value the
    backend must be driven with, so it raises instead.
    """
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise RuntimeError(
            f"{md.name}: the synthetic-data block invokes "
            f"{program!r} with a non-literal value for the scalar "
            f"type-parameter {param!r}, so the harness cannot send "
            f"the emitted program the input it declares. Bind the "
            f"argument to a numeric literal in the snippet."
        ) from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            f"{md.name}: scalar type-parameter {param!r} of "
            f"{program!r} was given {value!r}, which is not a real "
            f"number; the emitted program declares it as a real "
            f"input."
        )
    return float(value)


def _scalar_program_arguments(
    snippet: str, md: Path, declared: dict[str, tuple[tuple[str, str], ...]],
) -> dict[str, float]:
    """Concrete values the snippet instantiated the exported
    parametric program at, keyed by type-parameter name.

    Reads the snippet's own syntax rather than the compiled program:
    template instantiation bakes the arguments into the resulting
    [`MonadicProgram`][quivers.continuous.programs.MonadicProgram]'s
    families and leaves no record of them (`model._params` is None for
    an instantiated template), so the invocation site is the only place
    the values still exist under their declared names.
    """
    if not declared:
        return {}
    tree = ast.parse(snippet, str(md))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        params = declared.get(func.attr)
        if params is None:
            continue
        scalar = {
            name for name, kind in params if kind == _SCALAR_PARAM_TYPE
        }
        bound: dict[str, float] = {}
        for (name, _kind), positional in zip(params, node.args):
            if name in scalar:
                bound[name] = _literal_argument(
                    positional, func.attr, name, md,
                )
        for keyword in node.keywords:
            if keyword.arg in scalar:
                bound[keyword.arg] = _literal_argument(
                    keyword.value, func.attr, keyword.arg, md,
                )
        missing = sorted(scalar - set(bound))
        if missing:
            raise RuntimeError(
                f"{md.name}: the synthetic-data block invokes "
                f"{func.attr!r} without a value for the scalar "
                f"type-parameter(s) {missing!r}, which every emitted "
                f"program declares as required inputs."
            )
        return bound
    return {}


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


_DEALIAS_ATTEMPTS = 8
"""Row permutations drawn per plate before the de-aliasing pass gives
up. A uniform permutation of a plate of more than a handful of rows
leaves the design untouched with vanishing probability, so a draw is
rejected only in the degenerate case, and eight independent draws
exhaust that case rather than papering over it."""


def _row_permutation(
    stem: str, length: int, attempt: int,
) -> torch.Tensor:
    """A deterministic permutation of `length` row positions.

    Seeded from the example's stem, the plate width, and the attempt
    index, so the de-aliased design is reproducible run to run and
    independent of both the global RNG and whatever the example's
    synthetic-data snippet seeded.
    """
    digest = hashlib.sha256(
        f"{stem}:{length}:{attempt}".encode("utf-8"),
    ).digest()[:8]
    generator = torch.Generator()
    generator.manual_seed(int.from_bytes(digest, "big") % (2 ** 63))
    return torch.randperm(length, generator=generator)


def _structural_subscript_names(
    observations: dict[str, torch.Tensor],
    observe_names: frozenset[str],
) -> frozenset[str]:
    """Names in `observations` that gather a parameter row rather than
    carry a coordinate of the support.

    A subscript is an integer-dtyped vector the program only reads:
    `out_idx`, `word_idx`, `item_idx`. An `observe` binder is data
    however it is typed, and a floating-point covariate is a real
    coordinate however integral its values, so both are excluded.
    """
    return frozenset(
        name
        for name, value in observations.items()
        if name not in observe_names
        and value.dim() == 1
        and not value.dtype.is_floating_point
    )


def _dealias_row_order(
    observations: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    x_input: torch.Tensor | None,
    observe_names: frozenset[str],
    stem: str,
) -> tuple[
    dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor | None
]:
    """Relabel the rows of every plate that carries a structural
    subscript, so no subscript coincides with the row counter.

    A subscript cannot be perturbed (stepping it gathers a different
    parameter row rather than moving the data), so it is frozen at
    every point of the evaluation set. That freezing is safe only if
    the subscript is not *recoverable from the row position*: a design
    written as `torch.arange(D).repeat(N)` equals `i % D` at every
    row, so a renderer that discards the supplied vector and recomputes
    the subscript from its own loop counter emits an identical density
    at every point, and the constant-spread check absorbs the defect
    into its additive constant. The same holds for
    `repeat_interleave` (`i // K`) and for a bare `arange` (`i`).

    Permuting the rows of the plate closes that hole without moving a
    single coordinate of the support. The density is a product over
    the plate's rows, so relabeling the rows leaves the joint exactly
    invariant, which is why the de-aliased design still reproduces
    every pinned reference joint. What it destroys is the coincidence:
    no rule over the row counter reproduces the permuted design, so a
    renderer that fails to read it pairs each response with the wrong
    parameter row, and the resulting discrepancy moves with the
    responses and the latents rather than holding constant.

    Every array of the plate moves together (the subscripts, the
    covariates, the responses, and the program-input rows), because a
    row relabeling is only measure-preserving when it is applied to the
    whole row. Arrays of other widths, and the per-plate latents in
    `params`, are untouched.
    """
    subscripts = _structural_subscript_names(observations, observe_names)
    if not subscripts:
        return observations, params, x_input
    dealiased = dict(observations)
    moved_params = dict(params)
    moved_input = x_input
    for length in sorted(
        {int(observations[name].shape[0]) for name in subscripts}
    ):
        rows = sorted(
            name
            for name, value in observations.items()
            if value.dim() == 1 and int(value.shape[0]) == length
        )
        design = sorted(subscripts & frozenset(rows))
        for attempt in range(_DEALIAS_ATTEMPTS):
            order = _row_permutation(stem, length, attempt)
            if all(
                not torch.equal(dealiased[name], dealiased[name][order])
                for name in design
            ):
                break
        else:
            raise RuntimeError(
                f"{stem!r}: {_DEALIAS_ATTEMPTS} row permutations of the "
                f"{length}-row plate all left {design!r} unchanged, so "
                f"the design cannot be separated from the row counter. "
                f"A plate whose subscript is invariant under every "
                f"permutation is constant, and a constant subscript "
                f"gathers one parameter row for the whole plate."
            )
        for name in rows:
            dealiased[name] = dealiased[name][order]
            if name in moved_params:
                moved_params[name] = moved_params[name][order]
        if (
            moved_input is not None
            and moved_input.dim() > 0
            and int(moved_input.shape[0]) == length
        ):
            moved_input = moved_input[order]
    return dealiased, moved_params, moved_input


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

    # A sample site the snippet ships inside its observations dict is
    # still a LATENT of the program: the dict is what the `.md`'s SVI
    # and MCMC demos clamp, not a declaration that the value is data.
    # Filing it under `params` is what routes it to the container's
    # parameter channel (Stan's constrained_params, numpyro's
    # substitution dict, WebPPL's clampedParams) instead of trying to
    # hand a latent's value in through a model input the emitted
    # program never declares. `observations` is left intact, so the
    # in-process QVR trace keeps clamping the site at full rank.
    for site in sample_sites:
        if site in params or site not in obs_tensors:
            continue
        params[site] = obs_tensors[site].detach().to(dtype=torch.float64)

    scalar_params = _scalar_program_arguments(
        snippet, md, _exported_type_parameters(source_qvr),
    )
    collisions = sorted(
        frozenset(scalar_params) & (frozenset(params) | frozenset(obs_tensors))
    )
    if collisions:
        raise RuntimeError(
            f"{md.name}: scalar type-parameter(s) {collisions!r} share "
            f"a name with a sample site or an observation, so the "
            f"point's data section cannot carry both. Rename the "
            f"program's type-parameter in the `.qvr` source."
        )

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

    observe_names = frozenset(_qvr_observe_names(source_qvr))
    obs_tensors, params, x_input = _dealias_row_order(
        obs_tensors, params, x_input, observe_names, source_qvr.stem,
    )

    return GalleryDataset(
        observations=obs_tensors,
        params=params,
        scalar_params=scalar_params,
        observe_names=observe_names,
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
    so the in-process QVR trace still clamps every site.

    The program's scalar type-parameters ride in the same `data`
    section: they are inputs of the emitted program on every backend,
    and `data` is the section each probe script binds to the model's
    formal arguments."""
    return _point_from_tensors(
        dataset.params, dataset.observations, dataset.scalar_params,
    )


def _point_from_tensors(
    params: dict[str, torch.Tensor],
    observations: dict[str, torch.Tensor],
    scalar_params: dict[str, float],
) -> Point:
    """Flatten a (latent, observed) tensor pair into a wire
    [`Point`][tests.transpile.probes._protocol.Point].

    Every value becomes a row-major float list; a length-1 list
    collapses to a bare float so the probe's dict-to-Tensor casting
    picks the scalar shape. A name bound in both sections is emitted
    only under ``params`` (see
    [`point_from_dataset`][tests.transpile._gallery_data.point_from_dataset]),
    and each entry of ``scalar_params`` joins the data section as a
    bare float.
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
    squeezed_data.update(scalar_params)
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
    """Move an integer-valued tensor by a small integer delta.

    The step is clamped into a window, and which window applies turns
    on whether the value **attests a range of its own**.

    1. A value whose entries span more than one integer attests that
       range: the fixture's own forward simulation reached every value
       between its minimum and its maximum, so the window is the
       intersection of that range with the declared bounds. This is
       what keeps a count observation in support when the declared
       constraint is looser than the model's real alphabet: a
       categorical emission declares `IntegerGreaterThan(0)` while the
       emission row has finite width, so an unbounded upward step
       would index past the row and send the joint to `-inf`.
    2. A value whose entries are all equal, a scalar most of all,
       attests nothing. Intersecting with its own degenerate range
       would pin it to the one value it holds and freeze the
       coordinate at every point of the evaluation set, which is not a
       conservative reading of an unseen support but an unconditional
       refusal to exercise the coordinate. The declared bounds
       therefore govern on their own whenever they are finite, since a
       finite declared interval is itself an attestation of where the
       value may go.
    3. A value that attests no range under bounds that are not both
       finite has nothing to bound an upward step with, and stays put.
    """
    if work.numel() == 0:
        return work
    magnitude = max(
        1.0, _INTEGER_STEP_FRACTION * float(work.abs().mean().item()),
    )
    moved = torch.round(work + torch.round(noise * magnitude))
    attested_low = float(work.min().item())
    attested_high = float(work.max().item())
    if attested_low < attested_high:
        low = max(lower, attested_low)
        high = min(upper, attested_high)
    elif math.isfinite(lower) and math.isfinite(upper):
        low, high = lower, upper
    else:
        return work
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


def _observed_count_floor(
    section: dict[str, torch.Tensor], observe_names: frozenset[str],
) -> float:
    """Smallest value a scalar count parameter of `section` may take.

    A scalar integer covariate parameterises a bounded count family:
    it is the `total_count` of a Binomial or Beta-Binomial, the size
    of a Multinomial draw, the trial budget of a Negative-Binomial
    framing. Every one of those families supports the counts in
    `[0, total]`, so the largest count the fixture actually scores is
    the point below which the parameter cannot move without dropping
    an observed response out of the support.

    The floor reads the integer-valued `observe` binders only. A
    fractional binder is a real observation that no count parameter
    bounds, and a covariate is not scored at all.
    """
    floor = 0.0
    for name, value in section.items():
        if name not in observe_names or value.numel() == 0:
            continue
        if value.dtype.is_floating_point and not torch.equal(
            value, value.round(),
        ):
            continue
        floor = max(floor, float(value.max().item()))
    return floor


def _data_section_support(
    name: str,
    value: torch.Tensor,
    observe_names: frozenset[str],
    count_floor: float,
) -> constraints.Constraint | None:
    """Constraint inferred for an observations-dict entry that answers
    to no compiled stochastic site.

    Two shapes of entry reach this helper. The first is a covariate the
    program reads through a `let` or a plate subscript, for which no
    declared family fixes a constraint. The second is a genuine
    `observe` binder the compiled program buries inside a closure: a
    grouped `marginalize` folds its body's observe into a single score
    callable (`observe w : Word <- Categorical(phi[z]) [via=word_idx]`
    becomes one `_grouped_ll_z_0` let plus a `_marg_z` score), so
    [`site_supports`][tests.transpile._gallery_data.site_supports]
    reports nothing for `w` even though `w` is the model's only
    observation.

    Resolution splits on which of the two kinds the entry is, and the
    covariate case then splits on the entry's **dtype** and **rank**:

    1. A covariate carried in a floating-point tensor is a real
       coordinate of the data and moves additively, whatever values it
       happens to hold. A time index, an exposure, a design covariate:
       each enters the density through arithmetic, so a value with no
       fractional part is a real coordinate that landed on an integer,
       not an index. Freezing it would leave the equivalence check
       blind to every backend error that is a function of that
       covariate alone, since such an error is constant across a point
       set that never moves it.
    2. A covariate carried in an integer **vector** is a structural
       subscript: it names a row of a parameter plate rather than a
       point of the support, so stepping it would gather different
       parameters rather than move the data. It stays put, and
       [`_dealias_row_order`][tests.transpile._gallery_data._dealias_row_order]
       is what keeps a frozen subscript from aliasing the row counter.
    3. A covariate carried in an integer **scalar** cannot be a
       subscript, because a subscript has to index a plate and so has
       to have one entry per row. It is a count parameter of an
       observation family, and its value enters the density through
       that family's normaliser: the Beta-Binomial's
       `lgamma(total + 1)` terms move with it exactly as the response
       does. It is therefore inside the support Theorem 4.1 quantifies
       over and has to move, within the window
       `[count_floor, 2 * value - count_floor]`. That window is the
       widest interval centred on the attested ground truth whose
       lower end is
       [`_observed_count_floor`][tests.transpile._gallery_data._observed_count_floor],
       the largest count the fixture scores; a fixture whose count
       parameter already sits at that floor has no room to move and
       the entry stays put, which the point-set strength check then
       reports as an unexercised coordinate rather than absorbing.
    4. An **observe binder** carrying a fractional part is
       unconstrained real and moves additively.
    5. An integer-valued **observe binder** takes an integer step
       inside its own attested range. The attested range is the
       conservative reading of a declared support the harness cannot
       see: every value in `[min, max]` is a point the model's own
       forward simulation reached or bracketed, so the alphabet of a
       categorical emission and the range of a count observation are
       both respected without the harness having to reconstruct the
       family's parameters.
    """
    if value.numel() == 0:
        return None
    if name not in observe_names:
        if value.dtype.is_floating_point:
            return constraints.real
        if value.dim() != 0:
            return None
        ground_truth = float(value.item())
        ceiling = 2.0 * ground_truth - count_floor
        if ceiling <= count_floor:
            return None
        return constraints.integer_interval(
            int(count_floor), int(ceiling),
        )
    if value.dtype.is_floating_point and not torch.equal(
        value, value.round(),
    ):
        return constraints.real
    lower = int(value.min().item())
    upper = int(value.max().item())
    if lower == upper:
        return None
    return constraints.integer_interval(lower, upper)


def _perturb_section(
    section: dict[str, torch.Tensor],
    supports: dict[str, constraints.Constraint],
    generator: torch.Generator,
    scale: float,
    *,
    infer_from_value: bool,
    observe_names: frozenset[str] = frozenset(),
    exclude: frozenset[str] = frozenset(),
) -> dict[str, torch.Tensor]:
    """Perturb every value in `section` whose constraint is known.

    A name with a declared site support moves under that constraint. A
    name without one falls back to
    [`_data_section_support`][tests.transpile._gallery_data._data_section_support]
    when `infer_from_value` is set (the observations dict, which mixes
    observe sites with plain covariates), and otherwise stays at ground
    truth. Names in `exclude` are copied through untouched.
    """
    out: dict[str, torch.Tensor] = {}
    count_floor = (
        _observed_count_floor(section, observe_names)
        if infer_from_value
        else 0.0
    )
    for name, value in section.items():
        if name in exclude:
            out[name] = value
            continue
        support = supports.get(name)
        if support is None and infer_from_value:
            support = _data_section_support(
                name, value, observe_names, count_floor,
            )
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


def observed_data_names(dataset: GalleryDataset) -> frozenset[str]:
    """Names a point carries as genuinely *observed* data.

    Three kinds of entry live in a point's `data` section, and only one
    of them is data the model conditions on:

    1. An `observe` binder or a covariate the program reads. These are
       the observed data.
    2. A sample site the `.md` snippet also clamps for its inference
       demo. The point files it under `params` (the latent spelling is
       canonical), so it is not observed data and is excluded here.
    3. A scalar type-parameter of the program. It is an *input* of the
       emitted program, fixed by the template instantiation, so it
       never varies and is excluded here too.
    """
    return frozenset(dataset.observations) - frozenset(dataset.params)


def structural_subscript_names(dataset: GalleryDataset) -> frozenset[str]:
    """Names of `dataset`'s observations that gather a parameter row
    rather than carry a coordinate of the support.

    The same classifier the perturber consults, published so a caller
    that wants to *exempt* a frozen coordinate has to justify the
    exemption against the harness's own reading of the entry rather
    than against a written claim. A subscript is an integer-dtyped
    vector the program only reads: it has one entry per plate row and
    each entry selects a parameter row, so stepping it gathers
    different parameters instead of moving the data. An `observe`
    binder is data however it is typed, a floating-point covariate is
    a real coordinate however integral its values, and a scalar
    integer covariate is a count parameter rather than a subscript,
    so none of the three is reported here.
    """
    return _structural_subscript_names(
        dataset.observations, dataset.observe_names,
    )


def varying_observation_names(
    dataset: GalleryDataset, points: list[Point],
) -> frozenset[str]:
    """The
    [`observed_data_names`][tests.transpile._gallery_data.observed_data_names]
    whose value actually differs somewhere in `points`.

    This is the observable form of "the data really moved", and it is
    strictly stronger than watching the joint move: a latents-only
    perturbation moves the joint while leaving every observation at
    ground truth, which is exactly the blind spot that lets a backend
    drop a data-dependent term and still hold a constant offset.
    """
    moved: set[str] = set()
    for name in observed_data_names(dataset):
        seen = {
            _wire_key(point.data[name])
            for point in points
            if name in point.data
        }
        if len(seen) > 1:
            moved.add(name)
    return frozenset(moved)


def _wire_key(
    value: float | int | list[float] | list[int],
) -> tuple[float, ...]:
    """Hashable identity of one point-section entry, for equality
    comparison across the point set."""
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(float(v) for v in value)


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
    integer count takes an integer step clamped to the attested range,
    and a scalar count parameter steps inside the window its family's
    support leaves it above the largest count the fixture scores.
    A structural subscript, the one entry kind that names a parameter
    row rather than a point of the support, stays at ground truth; its
    row order is de-aliased at load time instead, by
    [`_dealias_row_order`][tests.transpile._gallery_data._dealias_row_order].

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
                    infer_from_value=True,
                    observe_names=dataset.observe_names,
                    exclude=shared_names,
                )
                if mode in (PERTURB_DATA, PERTURB_BOTH)
                else dict(dataset.observations)
            )
            candidate = _point_from_tensors(
                params, observations, dataset.scalar_params,
            )
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
    "PROBE_SCRATCH_PARENT",
    "PROBE_SCRIPT_DIR",
    "assert_probe_scripts_unchanged",
    "gallery_examples_with_data",
    "is_discrete_support",
    "load_gallery_data",
    "md_path_for",
    "observations_for_point",
    "observed_data_names",
    "perturbation_labels",
    "point_from_dataset",
    "points_from_dataset",
    "probe_scratch",
    "probe_scratch_root",
    "probe_script_digests",
    "structural_subscript_names",
    "sweep_abandoned_probe_roots",
    "site_supports",
    "varying_observation_names",
]
