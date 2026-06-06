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
    """The `x_in` tensor when the snippet defines one (programs that
    declare `(alpha : Real, ...)` scalar parameters consume this)."""


def md_path_for(source_qvr: Path) -> Path:
    """The docs `.md` corresponding to a `docs/examples/source/<stem>.qvr`."""
    md_stem = source_qvr.stem.replace("_", "-")
    return _GALLERY_DOCS / f"{md_stem}.md"


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

    params: dict[str, torch.Tensor] = {}
    for k, v in ns.items():
        if not k.startswith("true_"):
            continue
        name = k[len("true_"):]
        if isinstance(v, torch.Tensor):
            params[name] = v.to(dtype=torch.float64)
        elif isinstance(v, (int, float)):
            params[name] = torch.tensor(float(v), dtype=torch.float64)
        elif isinstance(v, (list, tuple)):
            try:
                params[name] = torch.as_tensor(v, dtype=torch.float64)
            except (TypeError, ValueError):
                continue

    x_input = ns.get("x_in")
    if not isinstance(x_input, torch.Tensor):
        x_input = None

    return GalleryDataset(
        observations=obs_tensors, params=params, x_input=x_input,
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
    `data` carries every observation tensor as a flat list."""
    def _flatten(t: torch.Tensor) -> list[float]:
        return t.detach().to(dtype=torch.float64).flatten().tolist()
    params = {k: _flatten(v) for k, v in dataset.params.items()}
    data = {k: _flatten(v) for k, v in dataset.observations.items()}
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
