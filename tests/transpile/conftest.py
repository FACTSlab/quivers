"""Shared fixtures and availability markers for the transpile test
suite.

Three families of fixtures live here:

- **`backend`** — parametrised over every name returned by
  [`available_targets`][quivers.transpile.available_targets].
- **`fixture_module`** — parametrised over every entry in the QVR
  fixture corpus (`tests/transpile/fixtures/<category>/<name>.qvr`),
  pre-parsed into a `Module` AST.
- **`probe_for`** — returns the right
  [`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
  for a given backend name, or `None` when the runtime is not
  installed. Tier 3 tests gate on `probe.available()`.

Tool availability markers:

- `pytest.mark.requires_tool("stanc")` etc. — gates on
  `shutil.which(<binary>)`.
- `pytest.mark.requires_docker` — gates on `docker_available()`.
- `pytest.mark.requires_image("<tag>")` — gates on
  `docker image inspect <tag>` succeeding.
"""

from __future__ import annotations

import pathlib
import shutil

import pytest

from tests.transpile import _docker
from tests.transpile.fixtures import _load
from tests.transpile.probes._protocol import LogDensityProbe


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_tool(binary): skip if `binary` is not on PATH",
    )
    config.addinivalue_line(
        "markers",
        "requires_docker: skip if `docker info` does not succeed",
    )
    config.addinivalue_line(
        "markers",
        "requires_image(tag): skip if `docker image inspect <tag>` fails",
    )
    config.addinivalue_line(
        "markers",
        "requires_probe(backend): skip if the backend's probe is "
        "unavailable",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply per-marker skips at collection time."""
    del config
    for item in items:
        for marker in item.iter_markers(name="requires_tool"):
            binary = marker.args[0]
            if shutil.which(binary) is None:
                item.add_marker(
                    pytest.mark.skip(reason=f"binary {binary!r} not on PATH")
                )
        if list(item.iter_markers(name="requires_docker")):
            if not _docker.docker_available():
                item.add_marker(
                    pytest.mark.skip(
                        reason="docker daemon not reachable"
                    )
                )
        for marker in item.iter_markers(name="requires_image"):
            tag = marker.args[0]
            if not _docker.image_available(tag):
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"docker image {tag!r} not built locally; "
                        f"run tests/transpile/docker/build.sh"
                    )
                )
        for marker in item.iter_markers(name="requires_probe"):
            backend = marker.args[0]
            probe = _probe_for_name(backend)
            if probe is None or not probe.available():
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{backend} probe runtime unavailable"
                    )
                )


def _probe_for_name(backend: str) -> LogDensityProbe | None:
    """Load the probe class for ``backend`` and instantiate it.

    Importing here is cheap (each probe module does only side-effect
    free imports at top level) and isolates the probe-availability
    check from the test bodies.
    """
    module_name = f"tests.transpile.probes.{backend}"
    try:
        import importlib
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    # The probe class is named ``<Backend>Probe``; the module also
    # exposes a `_PROBE` singleton.
    probe = getattr(module, "_PROBE", None)
    if not isinstance(probe, LogDensityProbe):
        return None
    return probe


@pytest.fixture
def probe_for():
    """Factory: given a backend name, return its probe or None."""
    return _probe_for_name


@pytest.fixture(scope="session")
def composition_fixtures() -> list[_load.Fixture]:
    return _load.load_compositions()


@pytest.fixture(scope="session")
def family_fixtures() -> list[_load.Fixture]:
    return _load.load_families()


@pytest.fixture(scope="session")
def statement_fixtures() -> list[_load.Fixture]:
    return _load.load_statements()


@pytest.fixture(scope="session")
def step_fixtures() -> list[_load.Fixture]:
    return _load.load_steps()


@pytest.fixture(scope="session")
def let_expression_fixtures() -> list[_load.Fixture]:
    return _load.load_let_expressions()


@pytest.fixture(scope="session")
def option_fixtures() -> list[_load.Fixture]:
    return _load.load_options()


@pytest.fixture(scope="session")
def axes_fixtures() -> list[_load.Fixture]:
    return _load.load_axes()


@pytest.fixture
def scratch(tmp_path: pathlib.Path) -> pathlib.Path:
    """Per-test scratch directory for probes that write files."""
    sub = tmp_path / "probe_scratch"
    sub.mkdir(exist_ok=True)
    return sub
