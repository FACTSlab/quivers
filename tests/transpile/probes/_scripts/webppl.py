"""In-container WebPPL probe.

The node container has the `webppl` CLI on PATH. This Python driver
takes the rendered model source, transforms each `sample(<dist>)` and
`observe(<dist>, <val>)` call into a plain JavaScript expression that
accumulates a `score(<value>)` term into a running `globalStore.lp`,
then runs the program with `webppl` and parses the printed
log-density JSON.

Why a source rewrite: WebPPL's `sample` / `observe` / `factor`
primitives are CPS-transformed inside the interpreter and only have
meaning inside an inference algorithm. To compute a joint
log-density at a clamped (params, data) point we lift the call into
plain JS that uses each distribution object's `score(value)` method
directly. The renderer's emission shape is structured enough that a
balanced-paren scan suffices to find the inner argument lists.

The renderer spells a draw in exactly three shapes, and the probe
lifts all three:

1. scalar, `var <name> = sample(<dist>);`
2. iid plate, `var <name> = repeat(<n>, function () { return
   sample(<dist>); });`
3. indexed plate, `var <name> = mapIndexed(function (<i>, <j>) {
   return sample(<dist>); }, <plate>);` where `<dist>` reads `<i>`.

A plated draw the probe failed to lift would stay live: WebPPL would
redraw it at run time, dropping its prior term and making the
returned log-density non-deterministic. That is a wrong finite number
rather than an error, so after rewriting the probe asserts that no
`sample(` or `observe(` token survives inside the model function.

Scoring a distribution declared by the transpiler's runtime prelude
needs one extra step. WebPPL's CPS transform compiles a member call
(`dist.score(v)`) as a plain JavaScript call, which reaches a
CPS-rewritten function with the wrong arity and yields a trampoline
thunk instead of a number. Binding the method to an identifier first
routes the call back through the transform. Built-in distributions
are the mirror image: their `score` is native JavaScript that rejects
the CPS argument list. The probe therefore reads the prelude's
top-level `var <Name> = function (params) {` declarations and picks
the calling convention per family.

When `/io/export_names.json` is present the probe also reports the
program's exported value at each point. WebPPL's export surface is
the `model` function's own `return`, so the driver binds the call's
result and prints it alongside the log-density. The rewrite has
already clamped every draw, which makes that value a deterministic
function of the point.
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
from typing import TYPE_CHECKING

from _reshape import (
    export_payload,
    load_export_names,
    load_tables,
    reshape_point,
)

# The probe payload is strictly numeric: scalars and arbitrarily
# nested lists of the same. The aliases mirror `_reshape`'s and live
# under `TYPE_CHECKING` so the module still imports on the container's
# Python; `from __future__ import annotations` defers evaluation so the
# recursive `NestedNumber` reference resolves lazily.
if TYPE_CHECKING:
    Number = int | float
    NestedNumber = Number | list["NestedNumber"]
    PointSection = dict[str, NestedNumber]


_WEBPPL_BIN = shutil.which("webppl")

# WebPPL's CPS transform recurses over the whole program, so a large
# array literal (a 200-element response vector) overflows Node's
# default 984KB stack during compilation. The interpreter itself is
# plain Node, so invoking it through `node --stack-size` lifts the
# limit without changing any semantics.
_NODE_STACK_SIZE = "40000"


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
_REPEAT_DECL_RE = re.compile(
    r"\bvar\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*repeat\s*\("
)
_MAPINDEXED_DECL_RE = re.compile(
    r"\bvar\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*mapIndexed\s*\("
)
_OBSERVE_RE = re.compile(r"\bobserve\s*\(")

# `repeat`'s callback for an iid plate takes no arguments and its whole
# body is a single `return sample(<dist>)`.
_IID_CALLBACK_RE = re.compile(
    r"\Afunction\s*\(\s*\)\s*\{\s*return\s+sample\s*\("
)
# `mapIndexed`'s callback binds the plate index and the (unused) plate
# element; the distribution expression reads the index.
_INDEXED_CALLBACK_RE = re.compile(
    r"\Afunction\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,"
    r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*\{\s*return\s+sample\s*\("
)
# Everything a lifted callback is allowed to have after its inner
# `sample(...)` call: the statement terminator and the closing brace.
_CALLBACK_TAIL_RE = re.compile(r"\A\s*;?\s*\}\s*\Z")

# A distribution the transpiler's runtime prelude declares, rather than
# one WebPPL ships. Anchored at line start so the model body's nested
# `var` bindings never match.
_GRAFTED_DIST_RE = re.compile(
    r"^var\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"function\s*\(\s*params\s*\)\s*\{",
    re.MULTILINE,
)

# Distribution families whose `score(value)` expects a JavaScript
# boolean rather than the {0, 1} integer the probe receives in the
# Point's `data` map. The Python-side
# [`Point`][tests.transpile.probes._protocol.Point] contract pins
# every observation as a Python int / float / list, so the probe
# coerces booleanly-typed obs values to true / false at the call
# site via `(<value> === 1)`.
_BOOLEAN_VALUED_DISTS: frozenset[str] = frozenset({"Bernoulli"})

# Distribution families whose `score(value)` expects a WebPPL tensor
# rather than the plain JavaScript array the flat `Point` payload
# reshapes into. `Vector` is WebPPL's array-to-tensor constructor, and
# it is the same marshalling step `_reshape` performs for the array
# backends: the point's wire form is a list, the runtime's native
# container is not.
_VECTOR_VALUED_DISTS: frozenset[str] = frozenset({"Dirichlet"})

# Top-level helper injected into every driver. See the module
# docstring for why a grafted distribution's `score` cannot be reached
# through a member call.
_PROBE_SCORE_HELPER = """\
var _qvr_probe_score = function (dist, value) {
  var scoreFn = dist.score;
  return scoreFn(value);
};"""

# Fresh names the lifted plate callbacks bind. Prefixed so they cannot
# collide with a renderer-emitted binding.
_PLATE_INDEX_VAR = "_qvr_plate_i"


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


def _grafted_dist_names(source: str) -> frozenset[str]:
    """Distribution families the rendered source declares itself.

    The transpiler grafts a runtime prelude ahead of the model for
    every family WebPPL's `dists` module does not ship. Each graft is a
    top-level ``var <Name> = function (params) { ... }`` returning an
    object with `sample` / `score` / `support`.
    """
    return frozenset(_GRAFTED_DIST_RE.findall(source))


def _coerce_value(dist_expr: str, value_expr: str) -> str:
    """Marshal ``value_expr`` into the container the distribution's
    `score` expects: a JavaScript boolean for a Bernoulli-shaped
    support, a WebPPL tensor for a simplex-shaped one.

    Leaves the expression unchanged for every other distribution; the
    probe's data contract is otherwise a one-to-one match between QVR
    real / positive / integer values and WebPPL distribution scoring
    arguments.
    """
    family = _dist_family(dist_expr)
    if family is not None and family in _BOOLEAN_VALUED_DISTS:
        return f"(({value_expr}) === 1)"
    if family is not None and family in _VECTOR_VALUED_DISTS:
        return f"Vector({value_expr})"
    return value_expr


def _score_expr(
    dist_expr: str, value_expr: str, grafted: frozenset[str],
) -> str:
    """Build the JavaScript expression scoring ``value_expr`` under
    ``dist_expr``, picking the calling convention the family needs."""
    coerced = _coerce_value(dist_expr, value_expr)
    family = _dist_family(dist_expr)
    if family is not None and family in grafted:
        return f"_qvr_probe_score({dist_expr}, {coerced})"
    return f"({dist_expr}).score({coerced})"


def _statement_end(source: str, close_paren: int) -> int:
    """Index just past the `;` terminating the statement whose final
    `)` sits at ``close_paren``, so a rewrite leaves no empty
    statement behind."""
    tail = close_paren + 1
    while tail < len(source) and source[tail] in " \t":
        tail += 1
    if tail < len(source) and source[tail] == ";":
        tail += 1
    return tail


def _clamped_bind(name: str, score_term: str) -> str:
    """The replacement statement pair for one lifted draw: bind the
    name to its clamped value, then accumulate the score term."""
    return (
        f"var {name} = clampedParams.{name};\n"
        f"  globalStore.lp = globalStore.lp + {score_term};"
    )


def _lift_iid_plate(
    name: str, args: str, grafted: frozenset[str],
) -> str | None:
    """Lift ``repeat(<n>, function () { return sample(<dist>); })``.

    Returns ``None`` when the callback is not a bare draw, which is the
    deterministic use of `repeat` (a broadcast concentration vector,
    a dummy plate array) and needs no clamping. Raises when the
    callback does draw but in a shape this lift does not cover, so an
    unrecognised plate emission never scores a free latent.
    """
    n_expr, callback = _split_top_level_comma(args)
    head = _IID_CALLBACK_RE.match(callback)
    if head is None:
        return None
    open_paren = head.end() - 1
    close_paren = _find_matching(callback, open_paren)
    dist_expr = callback[open_paren + 1:close_paren].strip()
    tail = callback[close_paren + 1:]
    if _CALLBACK_TAIL_RE.match(tail) is None:
        msg = (
            f"webppl probe: `var {name} = repeat(...)` draws a sample "
            f"inside a callback whose body does more than return it; "
            f"the probe cannot clamp it. callback: {callback!r}"
        )
        raise ValueError(msg)
    element = f"{name}[{_PLATE_INDEX_VAR}]"
    score_term = (
        f"sum(mapN(function ({_PLATE_INDEX_VAR}) {{\n"
        f"    return {_score_expr(dist_expr, element, grafted)};\n"
        f"  }}, {n_expr}))"
    )
    return _clamped_bind(name, score_term)


def _lift_indexed_plate(
    name: str, args: str, grafted: frozenset[str],
) -> str | None:
    """Lift ``mapIndexed(function (<i>, <j>) { return sample(<dist>);
    }, <plate>)``.

    The distribution expression reads the plate index, so the lift
    keeps `mapIndexed` over the original plate array and indexes the
    clamped value with the same index variable. Returns ``None`` when
    the callback is a deterministic gather rather than a draw.
    """
    callback, plate_expr = _split_top_level_comma(args)
    head = _INDEXED_CALLBACK_RE.match(callback)
    if head is None:
        return None
    index_var, element_var = head.group(1), head.group(2)
    open_paren = head.end() - 1
    close_paren = _find_matching(callback, open_paren)
    dist_expr = callback[open_paren + 1:close_paren].strip()
    tail = callback[close_paren + 1:]
    if _CALLBACK_TAIL_RE.match(tail) is None:
        msg = (
            f"webppl probe: `var {name} = mapIndexed(...)` draws a "
            f"sample inside a callback whose body does more than "
            f"return it; the probe cannot clamp it. "
            f"callback: {callback!r}"
        )
        raise ValueError(msg)
    element = f"{name}[{index_var}]"
    score_term = (
        f"sum(mapIndexed(function ({index_var},{element_var}) {{\n"
        f"    return {_score_expr(dist_expr, element, grafted)};\n"
        f"  }}, {plate_expr}))"
    )
    return _clamped_bind(name, score_term)


def _rewrite_plated_samples(
    source: str, grafted: frozenset[str],
) -> str:
    """Replace every plated draw with a clamped bind plus a summed
    score accumulation over the plate."""
    lifts = (
        (_REPEAT_DECL_RE, _lift_iid_plate),
        (_MAPINDEXED_DECL_RE, _lift_indexed_plate),
    )
    for decl_re, lift in lifts:
        out: list[str] = []
        cursor = 0
        for match in decl_re.finditer(source):
            if match.start() < cursor:
                continue
            open_paren = match.end() - 1
            close_paren = _find_matching(source, open_paren)
            args = source[open_paren + 1:close_paren]
            replacement = lift(match.group(1), args, grafted)
            if replacement is None:
                continue
            out.append(source[cursor:match.start()])
            out.append(replacement)
            cursor = _statement_end(source, close_paren)
        out.append(source[cursor:])
        source = "".join(out)
    return source


def _rewrite_sample(source: str, grafted: frozenset[str]) -> str:
    """Replace each ``var <name> = sample(<dist>);`` with a
    clamped-value bind plus a score accumulation.

    The emitted form is::

        var <name> = clampedParams.<name>;
        globalStore.lp = globalStore.lp + (<dist>).score(<name>);

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
        out.append(
            _clamped_bind(name, _score_expr(dist_expr, name, grafted))
        )
        cursor = _statement_end(source, close_paren)
    out.append(source[cursor:])
    return "".join(out)


def _rewrite_observe(source: str, grafted: frozenset[str]) -> str:
    """Replace each ``observe(<dist>, <val>);`` with a score
    accumulation::

        globalStore.lp = globalStore.lp + (<dist>).score(<val>);
    """
    out: list[str] = []
    cursor = 0
    for match in _OBSERVE_RE.finditer(source):
        out.append(source[cursor:match.start()])
        open_paren = match.end() - 1
        close_paren = _find_matching(source, open_paren)
        inner = source[open_paren + 1:close_paren]
        dist_expr, value_expr = _split_top_level_comma(inner)
        out.append(
            "globalStore.lp = globalStore.lp + "
            f"{_score_expr(dist_expr, value_expr, grafted)};"
        )
        cursor = _statement_end(source, close_paren)
    out.append(source[cursor:])
    return "".join(out)


_MODEL_FN_RE = re.compile(
    r"\bvar\s+model\s*=\s*function\s*\(([^)]*)\)"
)
_LIVE_PRIMITIVE_RE = re.compile(r"\b(sample|observe)\s*\(")


def _assert_fully_lifted(rewritten: str) -> None:
    """Fail when a `sample` / `observe` primitive survives the rewrite.

    A surviving `sample(` is the dangerous case: WebPPL draws it fresh
    on every run, so the site contributes no prior term and the
    returned log-density is a random number. Checking only from the
    model declaration onward keeps the runtime prelude's `sample:`
    method definitions and its prose comments out of scope.
    """
    match = _MODEL_FN_RE.search(rewritten)
    if match is None:
        raise ValueError("no `var model = function (...)` declaration found")
    body = rewritten[match.start():]
    live = _LIVE_PRIMITIVE_RE.search(body)
    if live is None:
        return
    start = max(0, live.start() - 200)
    msg = (
        f"webppl probe: `{live.group(1)}(` survived the rewrite, so "
        f"that site would be redrawn at run time instead of scored at "
        f"the point's value. context:\n{body[start:live.end() + 200]}"
    )
    raise ValueError(msg)


def _json_literal(value: NestedNumber | PointSection | None) -> str:
    """Render a Python-native JSON value as a JavaScript literal.

    Uses :func:`json.dumps` so booleans / numbers / arrays / objects /
    strings round-trip through the same parser the WebPPL interpreter
    uses for `JSON.parse`.
    """
    return json.dumps(value)


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


def _build_driver(
    rendered: str,
    params: PointSection,
    data: PointSection,
) -> str:
    """Build the WebPPL driver source for one (params, data) point.

    Composes:

    * the rewritten model source (with every `sample` / `observe`
      lifted to score accumulation),
    * the grafted-distribution scoring helper,
    * clamped-params object literal,
    * data bindings as top-level `var <name> = <literal>;` decls,
    * a final `model(<args...>);` call so the rewritten sample /
      observe statements run,
    * a `console.log` of the resulting `globalStore.lp`.
    """
    grafted = _grafted_dist_names(rendered)
    rewritten = _rewrite_plated_samples(rendered, grafted)
    rewritten = _rewrite_sample(rewritten, grafted)
    rewritten = _rewrite_observe(rewritten, grafted)
    _assert_fully_lifted(rewritten)
    param_names = _model_parameter_names(rendered)
    data_decls = "\n".join(
        f"var {name} = {_json_literal(data.get(name))};"
        for name in param_names
    )
    call_args = ", ".join(param_names)
    clamped_literal = _json_literal(params)
    return (
        "globalStore.lp = 0;\n"
        f"{_PROBE_SCORE_HELPER}\n"
        f"var clampedParams = {clamped_literal};\n"
        f"{data_decls}\n"
        f"{rewritten}\n"
        f"var __qvr_exported = model({call_args});\n"
        "console.log(JSON.stringify({log_density: globalStore.lp, "
        "exported: __qvr_exported}));\n"
    )


def main() -> None:
    io = pathlib.Path("/io")
    source = (io / "source.js").read_text()
    points = json.loads((io / "points.json").read_text())
    shapes, dtypes = load_tables(io)

    if _WEBPPL_BIN is None:
        raise RuntimeError(
            "webppl probe: no `webppl` executable on PATH inside the "
            "container"
        )

    export_names = load_export_names(io)
    log_densities: list[float] = []
    exports: list[list[NestedNumber]] = []
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
            [
                "node",
                f"--stack-size={_NODE_STACK_SIZE}",
                _WEBPPL_BIN,
                str(wppl_path),
            ],
            capture_output=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"webppl exited {completed.returncode}: "
                f"stdout={completed.stdout.decode('utf-8', 'replace')}\n"
                f"stderr={completed.stderr.decode('utf-8', 'replace')}\n"
                f"driver:\n{driver}"
            )
        stdout_text = completed.stdout.decode("utf-8")
        # WebPPL writes a trailing `undefined` line after the program
        # finishes (the value of the last expression at the REPL); the
        # probe's contract is a JSON object on its own line carrying
        # the `log_density` field, so search the stdout for that
        # specific shape rather than blindly taking the last line.
        payload: dict[str, NestedNumber] | None = None
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
        if export_names:
            returned = payload.get("exported")
            # A JS `return` the renderer never emitted leaves the call
            # `undefined`, which `JSON.stringify` drops from the object
            # entirely; the missing key is exactly the dropped-export
            # defect, and `export_payload` names it.
            if len(export_names) > 1 and isinstance(returned, list):
                # The WebPPL renderer spells a multi-name return as a
                # JS array; `export_payload` reads an ordered return
                # as a tuple.
                returned = tuple(returned)
            exports.append(export_payload(export_names, returned))

    result: dict[str, list] = {"log_densities": log_densities}
    if export_names:
        result["exports"] = exports
    (io / "result.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
