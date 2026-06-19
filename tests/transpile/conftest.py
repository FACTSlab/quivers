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

import os
import pathlib
import platform
import shutil
import subprocess
import time

import pytest

from tests.transpile import _docker
from tests.transpile.fixtures import _load
from tests.transpile.probes._protocol import LogDensityProbe


_DOCKER_IMAGE_BUILD_SCRIPT = (
    pathlib.Path(__file__).parent / "docker" / "build.sh"
)
_DOCKER_IMAGE_TAGS = (
    "panproto-test-stan",
    "panproto-test-numpyro",
    "panproto-test-pyro",
    "panproto-test-pymc",
    "panproto-test-edward2",
    "panproto-test-julia",
    "panproto-test-node",
    "panproto-test-jags",
    "panproto-test-bugs",
)


def _start_docker_daemon() -> None:
    """Start Docker Desktop on macOS or `dockerd` on Linux when the
    daemon is not reachable, then block up to 60s for it to come up.

    Raises [`RuntimeError`][builtins.RuntimeError] if the daemon does
    not become reachable within the timeout, so a test that needs it
    cannot silently skip. Setting `QUIVERS_NO_DOCKER_AUTOSTART=1`
    short-circuits the autostart attempt for CI environments where
    Docker is provisioned by the runner and shouldn't be touched
    here.
    """
    if _docker.docker_available():
        return
    if os.environ.get("QUIVERS_NO_DOCKER_AUTOSTART") == "1":
        raise RuntimeError(
            "docker daemon not reachable and `QUIVERS_NO_DOCKER_AUTOSTART=1` "
            "is set; either start the daemon or unset the variable so the "
            "harness can start it"
        )
    if platform.system() == "Darwin":
        subprocess.run(
            ["open", "-a", "Docker"], check=False, capture_output=True,
        )
    elif platform.system() == "Linux":
        subprocess.run(
            ["systemctl", "start", "docker"],
            check=False, capture_output=True,
        )
    else:
        raise RuntimeError(
            f"docker daemon not reachable and autostart is not "
            f"implemented for {platform.system()!r}"
        )
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if _docker.docker_available():
            return
        time.sleep(2.0)
    raise RuntimeError(
        "docker daemon did not become reachable within 60s; check "
        "Docker Desktop / dockerd status before re-running the suite"
    )


def _build_missing_docker_images(tags: tuple[str, ...]) -> None:
    """Run `tests/transpile/docker/build.sh` when any image in `tags`
    is missing locally. The build script is idempotent: present images
    are a fast cache hit.

    Raises [`RuntimeError`][builtins.RuntimeError] if the build script
    exits non-zero or if any image is still missing after the build,
    so a test cannot silently skip on a missing image.
    """
    missing = [t for t in tags if not _docker.image_available(t)]
    if not missing:
        return
    if not _DOCKER_IMAGE_BUILD_SCRIPT.exists():
        raise RuntimeError(
            f"docker images missing ({missing!r}) and build script "
            f"{_DOCKER_IMAGE_BUILD_SCRIPT} not found"
        )
    completed = subprocess.run(
        ["bash", str(_DOCKER_IMAGE_BUILD_SCRIPT)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"docker image build failed (exit {completed.returncode}); "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    still_missing = [t for t in tags if not _docker.image_available(t)]
    if still_missing:
        raise RuntimeError(
            f"docker image build completed but images still missing: "
            f"{still_missing!r}; inspect "
            f"{_DOCKER_IMAGE_BUILD_SCRIPT.parent}/<tag>/Dockerfile"
        )


@pytest.fixture(scope="session", autouse=True)
def _ensure_docker_environment() -> None:
    """Session-scope autouse fixture: bring Docker up and build every
    probe image before the first test runs.

    Replaces the per-test "skip when daemon down / image missing"
    pattern. Either the environment is brought into the state the
    suite needs, or a clear configuration error fires at session
    start (no silent per-test skips).

    Set `QUIVERS_SKIP_DOCKER=1` to opt out (only for environments
    where you genuinely cannot run Docker tests, e.g. a pure
    documentation build). Tests that need Docker will then raise
    a configuration error rather than skip silently.
    """
    if os.environ.get("QUIVERS_SKIP_DOCKER") == "1":
        return
    _start_docker_daemon()
    _build_missing_docker_images(_DOCKER_IMAGE_TAGS)


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
    """Apply per-marker xfails at collection time.

    Environment shortfalls (binary missing from PATH, probe runtime
    not installed) become `strict=False` xfails so the gap is visible
    in the test report rather than absorbed into the skip pile. The
    Docker daemon and probe images are guaranteed available by the
    session-scope `_ensure_docker_environment` autouse fixture, so
    the `requires_docker` / `requires_image` markers reduce to a
    declaration-of-intent here and don't introduce per-test skips.
    """
    del config
    for item in items:
        for marker in item.iter_markers(name="requires_tool"):
            binary = marker.args[0]
            if shutil.which(binary) is None:
                item.add_marker(
                    pytest.mark.xfail(
                        reason=(
                            f"binary {binary!r} not on PATH; install it "
                            f"in the local toolchain or add the install "
                            f"step to CI"
                        ),
                        strict=False,
                    )
                )
        for marker in item.iter_markers(name="requires_probe"):
            backend = marker.args[0]
            probe = _probe_for_name(backend)
            if probe is None or not probe.available():
                item.add_marker(
                    pytest.mark.xfail(
                        reason=(
                            f"{backend} probe runtime unavailable; install "
                            f"the runtime locally or add the install step "
                            f"to CI"
                        ),
                        strict=False,
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
