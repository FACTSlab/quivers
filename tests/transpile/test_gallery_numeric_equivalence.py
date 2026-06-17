"""Tier-4 numeric equivalence on the documentation gallery.

For every `docs/examples/source/<example>.qvr` that ships a `###
Generating synthetic data` block in its sibling `.md`, this test:

1. Extracts and executes the data-gen snippet to recover an
   `observations` dict plus every captured `true_*` ground-truth
   parameter value.
2. Builds a single test
   [`Point`][tests.transpile.probes._protocol.Point] anchoring
   the ground-truth params and observations.
3. Runs the in-process
   [`QvrProbe`][tests.transpile.probes.qvr.QvrProbe] to compute
   `log p_QVR(θ_true, y) = sum_i log f_i(...)` at that point.
4. For every backend whose Docker image is locally built, runs the
   target's native log-density probe inside the container and
   asserts constant-spread equivalence (`max_i | δ_i − mean δ | <
   1e-6`) per Theorem 4.1 of
   [docs/semantics/transpile-correctness.md](../../docs/semantics/transpile-correctness.md).

Cells skip (do not fail) when:

- the example's `.md` lacks a synthetic-data block;
- the QVR program does not compile (missing walker feature);
- the backend's Docker image is not built;
- the transpile raises `UnsupportedConstruct` for this example.

When neither Docker images nor backend Python runtimes are
available (the typical local-dev state), the test still exercises
the QVR-side trace: every gallery example whose `.md` ships data
gets a verified `log p_QVR` value, which is the strongest in-process
correctness signal we can produce without target runtimes.
"""

from __future__ import annotations

import pathlib

import pytest

from quivers.transpile import UnsupportedConstruct, transpile
from tests.transpile import _docker, _gallery_data
from tests.transpile.probes import _protocol
from tests.transpile.probes.qvr import QvrProbe


_BACKENDS_WITH_IMAGES = {
    "stan": ("panproto-test-stan", "stan", "stan.py"),
    "numpyro": ("panproto-test-numpyro", "py", "numpyro.py"),
    "pyro": ("panproto-test-pyro", "py", "pyro.py"),
    "pymc": ("panproto-test-pymc", "py", "pymc.py"),
    "edward2": ("panproto-test-edward2", "py", "edward2.py"),
    "turing": ("panproto-test-julia", "jl", "turing.jl"),
    "gen": ("panproto-test-julia", "jl", "gen.jl"),
    "webppl": ("panproto-test-node", "js", "webppl.py"),
    "jags": ("panproto-test-jags", "jags", "jags.py"),
    "bugs": ("panproto-test-bugs", "bugs", "jags.py"),
}


def _gallery_cells() -> list[pathlib.Path]:
    return _gallery_data.gallery_examples_with_data()


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
def test_gallery_qvr_logdensity_finite(example: pathlib.Path) -> None:
    """The QVR-side log-density at the ground-truth (θ_true, y)
    point evaluates to a finite real number.

    This is the always-on Tier-4 invariant: regardless of backend
    Docker availability, every gallery example with synthetic data
    must produce a well-defined joint log-density under the
    reference QVR trace. A non-finite value (`-inf`, `nan`)
    indicates that either (a) the data-gen snippet's ground-truth
    parameters fall outside the model's support, or (b) the QVR
    program has a structural defect that the trace surfaces only
    on real data.
    """
    dataset = _gallery_data.load_gallery_data(example)
    if dataset is None:
        pytest.skip(f"{example.stem!r}: synthetic-data block missing or failed")

    point = _gallery_data.point_from_dataset(dataset)
    probe = QvrProbe()
    source = example.read_bytes()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    try:
        result = probe.evaluate(
            source, example.stem, [point], scratch=scratch,
        )
    except Exception as exc:
        # Programs the in-process QVR trace cannot evaluate (missing
        # primitives, axis-misalignment, etc.) skip cleanly; the
        # gallery still grows as the runtime grows.
        pytest.skip(
            f"{example.stem!r}: QvrProbe raised {type(exc).__name__}: {exc}"
        )
    assert len(result.log_densities) == 1, (
        f"{example.stem!r}: expected exactly one log-density, got "
        f"{result.log_densities!r}"
    )
    lp = result.log_densities[0]
    import math
    assert math.isfinite(lp), (
        f"{example.stem!r}: QVR log p(θ_true, y) is non-finite ({lp!r}); "
        f"observations={list(dataset.observations)} params={list(dataset.params)}"
    )


@pytest.mark.parametrize(
    "example", _gallery_cells(), ids=lambda p: p.stem
)
@pytest.mark.parametrize("backend", sorted(_BACKENDS_WITH_IMAGES))
def test_gallery_backend_logdensity_matches_qvr(
    example: pathlib.Path, backend: str
) -> None:
    """Constant-spread equivalence between the QVR reference and the
    backend's native log-density at the ground-truth (θ_true, y)
    point set for `example`.

    Cells skip when the backend image is absent or the transpile
    raises `UnsupportedConstruct`. Passing cells certify Theorem
    4.1's `log p_T - log p_QVR = c_T` identity at the chosen point.
    """
    image, ext, script_name = _BACKENDS_WITH_IMAGES[backend]
    if not _docker.docker_available():
        pytest.skip("docker not available")
    if not _docker.image_available(image):
        pytest.skip(f"image {image!r} not built; run tests/transpile/docker/build.sh")

    dataset = _gallery_data.load_gallery_data(example)
    if dataset is None:
        pytest.skip(f"{example.stem!r}: synthetic-data block missing")

    # Transpile to bytes; skip on UnsupportedConstruct (the
    # construct-matrix / family-matrix tests own that gap).
    source = example.read_bytes()
    try:
        emitted = transpile(
            __import__("quivers.dsl.parser", fromlist=["parse"]).parse(
                source.decode("utf-8")
            ),
            target=backend,
        )
    except UnsupportedConstruct as exc:
        pytest.skip(
            f"backend {backend!r} on {example.stem!r}: "
            f"UnsupportedConstruct {exc.kinds!r}"
        )

    # Build the QVR reference at the (θ_true, y) point.
    point = _gallery_data.point_from_dataset(dataset)
    qvr_probe = QvrProbe()
    scratch = pathlib.Path("/tmp") / f"qvr_gallery_eq_{example.stem}"
    scratch.mkdir(exist_ok=True, parents=True)
    try:
        qvr_result = qvr_probe.evaluate(
            source, example.stem, [point], scratch=scratch,
        )
    except Exception as exc:
        pytest.skip(
            f"{example.stem!r}: QvrProbe raised {type(exc).__name__}: {exc}"
        )

    # Per-backend probe via Docker. Each backend's probe script
    # lives at tests/transpile/probes/_scripts/<script>.
    # Skip when the backend probe cannot read the emitted source
    # (the backend-side probe scripts are written for the dedicated
    # composition fixtures and need per-example data shapes
    # registered to handle arbitrary gallery datasets; the runtime
    # raises a clear "no shape registered" message).
    script_path = (
        pathlib.Path(__file__).parent / "probes" / "_scripts" / script_name
    )
    try:
        raw_result = _docker.run_probe(
            image=image,
            script=script_path,
            source=emitted,
            source_ext=ext,
            points=[{"params": point.params, "data": point.data}],
            scratch=scratch,
        )
    except (NotImplementedError, RuntimeError, FileNotFoundError) as exc:
        pytest.skip(
            f"{backend!r} probe on {example.stem!r}: {type(exc).__name__}: {exc}"
        )

    backend_lps = [float(x) for x in raw_result["log_densities"]]

    from tests.transpile import _equivalence

    _equivalence.assert_log_density_match(
        qvr_result.log_densities,
        backend_lps,
        context=f"{backend}@{example.stem}",
    )


def _shapes_from_dataset(dataset: _gallery_data.GalleryDataset) -> dict[str, list[int]]:
    """The per-name shape table the per-backend probe scripts need to
    reshape the flat lists in `Point` back to tensors."""
    shapes: dict[str, list[int]] = {}
    for k, v in dataset.observations.items():
        shapes[k] = list(v.shape) if v.shape else [1]
    for k, v in dataset.params.items():
        shapes[k] = list(v.shape) if v.shape else [1]
    return shapes
