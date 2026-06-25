"""In-container WebPPL probe.

The node container has the `webppl` CLI on PATH. This Python driver
takes the rendered model source, transforms each `sample(<dist>)` and
`observe(<dist>, <val>)` call into a plain JavaScript expression that
accumulates `(<dist>).score(<value>)` into a running `totalLogProb`,
then runs the program with `webppl` and parses the printed
log-density JSON.

Why a source rewrite: WebPPL's `sample` / `observe` / `factor`
primitives are CPS-transformed inside the interpreter and only have
meaning inside an inference algorithm. To compute a joint
log-density at a clamped (params, data) point we lift the call into
plain JS that uses each distribution object's `score(value)` method
directly. The renderer's emission shape is structured enough that a
balanced-paren scan suffices to find the inner argument lists.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess

from _reshape import load_tables, reshape_point


def _find_matching(source: str, open_idx: int) -> int:
    """Return the index of the `)` matching the `(` at ``open_idx``.

    Tracks paren / brace / bracket nesting and skips over string
    literals. Raises ``ValueError`` when no match is found before
    end-of-string.
    """
    assert source[open_idx] == "(", source[open_idx]
    depth = 0
    i = open_idx
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return i
        elif ch in "\"'":
            quote = ch
            i += 1
            while i < n and source[i] != quote:
                if source[i] == "\\":
                    i += 2
                    continue
                i += 1
        i += 1
    raise ValueError(
        f"unbalanced parenthesis starting at index {open_idx}"
    )


def _split_top_level_comma(text: str) -> tuple[str, str]:
    """Split ``text`` at the top-level comma into ``(left, right)``.

    Used to split an `observe(dist, value)` argument list. Tracks
    paren / brace / bracket nesting and string literals so commas
    inside nested calls / arrays / objects / strings do not split.
    Raises ``ValueError`` when no top-level comma is found.
    """
    depth = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch in "\"'":
            quote = ch
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    i += 2
                    continue
                i += 1
        elif ch == "," and depth == 0:
            return text[:i].strip(), text[i + 1:].strip()
        i += 1
    raise ValueError(f"no top-level comma in {text!r}")


_SAMPLE_DECL_RE = re.compile(
    r"\bvar\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*sample\s*\("
)
_OBSERVE_RE = re.compile(r"\bobserve\s*\(")

# Distribution families whose `score(value)` expects a JavaScript
# boolean rather than the {0, 1} integer the probe receives in the
# Point's `data` map. The Python-side
# [`Point`][tests.transpile.probes._protocol.Point] contract pins
# every observation as a Python int / float / list, so the probe
# coerces booleanly-typed obs values to true / false at the call
# site via `(<value> === 1)`.
_BOOLEAN_VALUED_DISTS: frozenset[str] = frozenset({"Bernoulli"})


def _dist_family(dist_expr: str) -> str | None:
    """Extract the distribution family name from a `<Family>({...})`
    expression. Returns ``None`` when the expression does not match
    that shape.
    """
    stripped = dist_expr.lstrip()
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
    if match is None:
        return None
    return match.group(1)


def _coerce_value(dist_expr: str, value_expr: str) -> str:
    """Wrap ``value_expr`` in a `=== 1` test when the distribution's
    `score` expects a JavaScript boolean.

    Leaves the expression unchanged for every other distribution; the
    probe's data contract is otherwise a one-to-one match between QVR
    real / positive / integer values and WebPPL distribution scoring
    arguments.
    """
    family = _dist_family(dist_expr)
    if family is not None and family in _BOOLEAN_VALUED_DISTS:
        return f"(({value_expr}) === 1)"
    return value_expr


def _rewrite_sample(source: str) -> str:
    """Replace each ``var <name> = sample(<dist>);`` with a
    clamped-value bind plus a score accumulation.

    The emitted form is::

        var <name> = clampedParams.<name>;
        totalLogProb = totalLogProb + (<dist>).score(<name>);

    The original `sample` semantics is irrelevant here: the probe's
    contract is that every latent is supplied through ``clampedParams``.
    """
    out: list[str] = []
    cursor = 0
    for match in _SAMPLE_DECL_RE.finditer(source):
        out.append(source[cursor:match.start()])
        name = match.group(1)
        open_paren = match.end() - 1
        close_paren = _find_matching(source, open_paren)
        dist_expr = source[open_paren + 1:close_paren].strip()
        # Skip an optional trailing whitespace + `;` so the rewrite
        # does not leave a stray empty statement.
        tail = close_paren + 1
        while tail < len(source) and source[tail] in " \t":
            tail += 1
        if tail < len(source) and source[tail] == ";":
            tail += 1
        coerced_name = _coerce_value(dist_expr, name)
        replacement = (
            f"var {name} = clampedParams.{name};\n"
            f"  globalStore.lp = globalStore.lp + "
            f"({dist_expr}).score({coerced_name});"
        )
        out.append(replacement)
        cursor = tail
    out.append(source[cursor:])
    return "".join(out)


def _rewrite_observe(source: str) -> str:
    """Replace each ``observe(<dist>, <val>);`` with a score
    accumulation::

        totalLogProb = totalLogProb + (<dist>).score(<val>);
    """
    out: list[str] = []
    cursor = 0
    for match in _OBSERVE_RE.finditer(source):
        out.append(source[cursor:match.start()])
        open_paren = match.end() - 1
        close_paren = _find_matching(source, open_paren)
        inner = source[open_paren + 1:close_paren]
        dist_expr, value_expr = _split_top_level_comma(inner)
        tail = close_paren + 1
        while tail < len(source) and source[tail] in " \t":
            tail += 1
        if tail < len(source) and source[tail] == ";":
            tail += 1
        coerced_value = _coerce_value(dist_expr, value_expr)
        replacement = (
            f"globalStore.lp = globalStore.lp + "
            f"({dist_expr}).score({coerced_value});"
        )
        out.append(replacement)
        cursor = tail
    out.append(source[cursor:])
    return "".join(out)


_MODEL_FN_RE = re.compile(
    r"\bvar\s+model\s*=\s*function\s*\(([^)]*)\)"
)


def _model_parameter_names(source: str) -> tuple[str, ...]:
    """Return the formal parameter names of the rendered ``model``
    function in declaration order.

    The renderer always emits a single top-level
    ``var model = function (<inputs>) { ... };`` declaration. The
    probe binds each clamped data input by name before calling
    ``model``.
    """
    match = _MODEL_FN_RE.search(source)
    if match is None:
        raise ValueError("no `var model = function (...)` declaration found")
    raw = match.group(1).strip()
    if not raw:
        return ()
    return tuple(p.strip() for p in raw.split(","))


def _json_literal(value: object) -> str:
    """Render a Python-native JSON value as a JavaScript literal.

    Uses :func:`json.dumps` so booleans / numbers / arrays / objects /
    strings round-trip through the same parser the WebPPL interpreter
    uses for `JSON.parse`.
    """
    return json.dumps(value)


def _build_driver(
    rendered: str,
    params: dict[str, object],
    data: dict[str, object],
) -> str:
    """Build the WebPPL driver source for one (params, data) point.

    Composes:

    * the rewritten model source (with `sample` / `observe` lifted to
      score accumulation),
    * clamped-params object literal,
    * data bindings as top-level `var <name> = <literal>;` decls,
    * a final `model(<args...>);` call so the rewritten sample /
      observe statements run,
    * a `console.log` of the resulting `totalLogProb`.
    """
    rewritten = _rewrite_sample(rendered)
    rewritten = _rewrite_observe(rewritten)
    param_names = _model_parameter_names(rendered)
    data_decls = "\n".join(
        f"var {name} = {_json_literal(data.get(name))};"
        for name in param_names
    )
    call_args = ", ".join(param_names)
    clamped_literal = _json_literal(params)
    return (
        "globalStore.lp = 0;\n"
        f"var clampedParams = {clamped_literal};\n"
        f"{data_decls}\n"
        f"{rewritten}\n"
        f"model({call_args});\n"
        "console.log(JSON.stringify({log_density: globalStore.lp}));\n"
    )


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.js").read_text()
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)

    log_densities: list[float] = []
    for i, pt in enumerate(points):
        reshaped = reshape_point(pt, shapes, dtypes)
        driver = _build_driver(
            source,
            reshaped.get("params", {}),
            reshaped.get("data", {}),
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
                f"stderr={completed.stderr.decode('utf-8', 'replace')}\n"
                f"driver:\n{driver}"
            )
        stdout_text = completed.stdout.decode("utf-8")
        # WebPPL writes a trailing `undefined` line after the program
        # finishes (the value of the last expression at the REPL); the
        # probe's contract is a JSON object on its own line carrying
        # the `log_density` field, so search the stdout for that
        # specific shape rather than blindly taking the last line.
        payload: dict | None = None
        for line in stdout_text.splitlines():
            text = line.strip()
            if not text or not text.startswith("{"):
                continue
            try:
                candidate = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "log_density" in candidate:
                payload = candidate
        if payload is None:
            raise RuntimeError(
                "webppl produced no parseable log-density line\n"
                f"stdout={stdout_text!r}\n"
                f"stderr={completed.stderr.decode('utf-8', 'replace')!r}\n"
                f"driver:\n{driver}"
            )
        log_densities.append(float(payload["log_density"]))

    (io / "result.json").write_text(
        json.dumps({"log_densities": log_densities})
    )


if __name__ == "__main__":
    main()
