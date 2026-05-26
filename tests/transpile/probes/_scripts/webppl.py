"""In-container WebPPL probe.

The node container has webppl on PATH. This Python driver writes the
user's model body plus a clamping driver to a temp .wppl file, runs
webppl, parses the printed log-density JSON.
"""
import json
import pathlib
import subprocess


_DRIVER_TEMPLATE = """\
// Clamping driver: replace `sample` and `observe` with their
// `factor`-only equivalents so the program score equals the joint
// log-density at the supplied (params, data) point.
var clampedParams = JSON.parse(read("__PARAMS_PATH__"));
var clampedData = JSON.parse(read("__DATA_PATH__"));

// Replace `sample(dist)` with `factor(dist.score(value))`, where
// `value` comes from the clamped-params map keyed by the variable
// name in the calling assignment. WebPPL exposes the calling line
// via `arguments.callee.name` rarely; for this contract the
// transpiled model emits each sample as `var X = sample(...)` so the
// driver can pattern-match on the LHS through a wrapper.
//
// To make this work without re-parsing the model, the harness
// transpiles model bodies with a single-line `factor(dist.score(v))`
// per assignment (no `sample`). The driver here is therefore a
// thin shell.

__USER_MODEL__

console.log(JSON.stringify({log_density: __TOTAL_FACTOR_VAR__ || 0}));
"""


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.js").read_text()
    points = json.loads((io / "points.json").read_text())

    log_densities = []
    for i, pt in enumerate(points):
        params_path = io / f"params.{i}.json"
        data_path = io / f"data.{i}.json"
        params_path.write_text(json.dumps(pt.get("params", {})))
        data_path.write_text(json.dumps(pt.get("data", {})))

        driver = (
            _DRIVER_TEMPLATE
            .replace("__PARAMS_PATH__", str(params_path))
            .replace("__DATA_PATH__", str(data_path))
            .replace("__USER_MODEL__", source)
            .replace("__TOTAL_FACTOR_VAR__", "totalLogProb")
        )
        wppl_path = io / f"driver.{i}.wppl"
        wppl_path.write_text(driver)

        completed = subprocess.run(
            ["webppl", str(wppl_path)],
            capture_output=True,
            timeout=60,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"webppl exited {completed.returncode}: "
                f"stderr={completed.stderr.decode('utf-8', 'replace')}"
            )
        last_line = (
            completed.stdout.decode("utf-8").strip().splitlines()[-1]
        )
        log_densities.append(float(json.loads(last_line)["log_density"]))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
