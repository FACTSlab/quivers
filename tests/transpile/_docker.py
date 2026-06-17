"""Docker container driver for out-of-process probes.

Each Tier-3 probe runs in its own pinned image, invoked once per
``(fixture, point_set)``. The driver here centralises the bind-mount
+ argv shape so per-backend probes only need to specify the image
tag and their entrypoint script.

The host's `docker` CLI is the dependency. When `docker` is not on
PATH or the host daemon is not running, [`docker_available`][.]
returns False and the calling test should `pytest.skip`.

Each image carries a corresponding script under
[`tests/transpile/probes/_scripts/<backend>.py`][tests.transpile.probes._scripts]
that reads `/io/source.<ext>`, `/io/points.json`, and emits
`/io/result.json` (a JSON object with one ``log_densities`` key).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess


def docker_available() -> bool:
    """True iff the `docker` CLI is on PATH and the daemon answers
    `docker info` in under three seconds."""
    if shutil.which("docker") is None:
        return False
    try:
        completed = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


def image_available(tag: str) -> bool:
    """True iff a Docker image matching ``tag`` is built or pulled
    locally.

    Uses ``docker images --filter reference=<tag>`` rather than
    ``docker image inspect`` because Docker Desktop 28+ has a daemon
    bug where ``image inspect <name>`` and ``image inspect
    <name>:latest`` both raise ``No such image`` even when ``docker
    images`` lists the image. The filter-form query goes through a
    different daemon path and reliably returns the image when it
    exists.
    """
    if not docker_available():
        return False
    completed = subprocess.run(
        [
            "docker",
            "images",
            "--filter",
            f"reference={tag}",
            "--format",
            "{{.ID}}",
        ],
        capture_output=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return False
    return bool(completed.stdout.strip())


def run_probe(
    *,
    image: str,
    script: pathlib.Path,
    source: bytes,
    source_ext: str,
    points: list[dict],
    scratch: pathlib.Path,
    timeout: float = 120.0,
) -> dict:
    """Invoke a probe image against ``source`` + ``points``.

    Layout under the bind-mounted ``scratch`` directory:

    ```
    /io/source.<ext>   # transpiled source bytes
    /io/points.json    # list[Point.dict()]
    /io/probe.py       # the entrypoint script for this backend
    /io/result.json    # probe writes this
    ```

    The container is launched read-only against /io except for the
    result file; this lets the host harness reason about side effects.

    Returns the decoded JSON object the script wrote to result.json.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    source_path = scratch / f"source.{source_ext}"
    source_path.write_bytes(source)
    (scratch / "points.json").write_text(json.dumps(points))
    (scratch / "probe.py").write_bytes(script.read_bytes())
    result_path = scratch / "result.json"
    if result_path.exists():
        result_path.unlink()

    # Each per-backend image carries `ENTRYPOINT ["python"]` (Python
    # backends) or its target-language equivalent (`julia`, `node`,
    # `jags`); the container takes the script path as the single
    # post-entrypoint argument. Do NOT also pass `python` here -- the
    # entrypoint already provides it, so prefixing would run
    # `python python /io/probe.py` and the container fails with
    # "no such file: /io/python".
    argv = [
        "docker", "run", "--rm",
        "-v", f"{scratch.resolve()}:/io",
        "-w", "/io",
        "-e", f"FIXTURE_EXT={source_ext}",
        image,
        "/io/probe.py",
    ]
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"probe container {image!r} exited {completed.returncode}\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )
    if not result_path.exists():
        raise RuntimeError(
            f"probe container {image!r} did not write /io/result.json\n"
            f"stdout: {completed.stdout}\n"
            f"stderr: {completed.stderr}"
        )
    return json.loads(result_path.read_text())


__all__ = ["docker_available", "image_available", "run_probe"]
