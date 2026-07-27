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
import re
from pathlib import Path

import torch

from quivers.continuous.programs import MonadicProgram
from tests.transpile.probes._protocol import Point


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

    # Match every `sample <name>` site in the QVR source against the
    # namespace under three spellings: bare `<name>`, suffixed
    # `<name>_true`, and prefixed `true_<name>`. The bare spelling
    # is accepted only when the QVR source declares the name as a
    # sample site, so intermediate bindings (`T = 64`, `model = ...`)
    # in the snippet do not get mis-captured as ground-truth.
    sample_names = set(_qvr_sample_names(source_qvr))
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

    for sample_name in sample_names:
        for spelling in (
            f"true_{sample_name}",
            f"{sample_name}_true",
            sample_name,
        ):
            if spelling not in ns:
                continue
            coerced = _coerce(ns[spelling])
            if coerced is None:
                continue
            params[sample_name] = coerced
            break

    # Also accept the bare `true_*` / `*_true` namespace bindings
    # whose name matches a QVR sample-site declaration. Bindings whose
    # stripped name does not appear in the program's sample-site set
    # are local snippet variables (the let-step intermediates
    # `mu_true`, `alpha_true`, ... in regression examples) and would
    # poison the backend probe by surfacing as `unused data`
    # variables in renderers (e.g. BUGS) that reject unknown clamps.
    for k, v in ns.items():
        if k.startswith("true_"):
            name = k[len("true_"):]
        elif k.endswith("_true"):
            name = k[: -len("_true")]
        else:
            continue
        if name in params or name not in sample_names:
            continue
        coerced = _coerce(v)
        if coerced is not None:
            params[name] = coerced

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

    # Compiled MonadicProgram. Templates compile to `Program(None)`
    # with `program.templates[<name>]` invokers; the synthetic-data
    # block instantiates the template at concrete arguments and binds
    # the result to `model` (or, for examples that wire the bare
    # morphism, `inner`). Capturing either lets the QVR probe walk
    # the instantiated program directly.
    monadic: MonadicProgram | None = None
    for monad_name in ("model", "inner"):
        candidate = ns.get(monad_name)
        if isinstance(candidate, MonadicProgram):
            monadic = candidate
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
    def _flatten(t: torch.Tensor) -> list[float]:
        return t.detach().to(dtype=torch.float64).flatten().tolist()
    params = {k: _flatten(v) for k, v in dataset.params.items()}
    data = {
        k: _flatten(v)
        for k, v in dataset.observations.items()
        if k not in dataset.params
    }
    # Squeeze scalar entries (length-1 lists) to plain floats so the
    # probe's dict-to-Tensor casting picks the right shape.
    params = {k: (v[0] if len(v) == 1 else v) for k, v in params.items()}
    data = {k: (v[0] if len(v) == 1 else v) for k, v in data.items()}
    return Point(params=params, data=data)


__all__ = [
    "GalleryDataset",
    "gallery_examples_with_data",
    "load_gallery_data",
    "md_path_for",
    "point_from_dataset",
]
