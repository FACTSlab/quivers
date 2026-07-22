"""[`LogDensityProbe`][tests.transpile.probes._protocol.LogDensityProbe]
for the Church backend.

Church has no single canonical interpreter: the target is the
abstract stochastic-lambda-calculus semantics of Goodman, Mansinghka,
Roy, Bonawitz, and Tenenbaum (2008). The emitted program is a
self-contained Scheme module that grafts the Church reference runtime
(`sample` / `observe` / `factor` on a distribution-object protocol,
plus one distribution constructor per family), so any R6RS-capable
Scheme evaluates its joint log-density.

The probe runs the emitted module through a reachable Scheme
interpreter (Chez Scheme's ``chez`` / ``scheme``, or Petite Chez's
``petite``). It appends a *clamping driver* that rebinds ``sample`` so
each latent site scores its clamped test-point value rather than a
fresh draw, calls the emitted ``model`` on the point's data inputs,
and prints the accumulated ``*log-weight*`` -- the joint log-density
of the clamped ``(theta, y)`` point. The driver reads the clamped
scalars from a single flat cursor, one entry consumed per scalar leaf
the site's distribution draws, so a scalar site consumes one value, a
Dirichlet / multivariate-Gaussian site consumes its whole event
vector, and a matrix site consumes its whole flattened matrix.

Available iff a Scheme interpreter binary is on ``PATH``; the numeric
tier skips the ``(*, church)`` cell only when none is installed.
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import shutil
import subprocess

from tests.transpile.probes._protocol import LogDensityProbe, Point, ProbeResult


#: Scheme interpreter binaries that evaluate the grafted runtime. Each
#: accepts ``<binary> --script <file>`` and provides the R6RS
#: ``fold-left`` / ``map`` surface the runtime's broadcast arithmetic
#: relies on. Order is preference order.
_SCHEME_INTERPRETERS: tuple[str, ...] = ("chez", "scheme", "petite", "chezscheme")


def _find_interpreter() -> str | None:
    """Return the first reachable Scheme interpreter binary, or None."""
    for name in _SCHEME_INTERPRETERS:
        path = shutil.which(name)
        if path is not None:
            return path
    return None


def _scheme_literal(value: float | int) -> str:
    """Render one Python number as a Scheme literal.

    Integral values render as exact integers (so index arrays feed
    ``list-ref``, which demands an exact index); non-integral values
    render as inexact flonums.
    """
    f = float(value)
    if f.is_integer():
        return str(int(f))
    return repr(f)


def _flatten(value: float | int | list[float] | list[int]) -> list[float]:
    """Flatten a scalar or (possibly nested) list into a flat list."""
    if isinstance(value, (int, float)):
        return [float(value)]
    out: list[float] = []
    for item in value:
        out.extend(_flatten(item))
    return out


def _scheme_value(value: float | int | list) -> str:
    """Render a scalar or nested list as a Scheme datum.

    Scalars render through
    [`_scheme_literal`][tests.transpile.probes.church._scheme_literal];
    lists render as ``(list ...)`` recursively so a nested Python list
    becomes a nested Scheme list.
    """
    if isinstance(value, (int, float)):
        return _scheme_literal(value)
    inner = " ".join(_scheme_value(v) for v in value)
    return f"(list {inner})"


def _balanced_form(text: str, open_paren_idx: int) -> str:
    """Return the balanced parenthesised form starting at
    ``open_paren_idx`` (which must index a ``(``)."""
    depth = 0
    i = open_paren_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx : i + 1]
        i += 1
    raise ValueError("unbalanced Scheme form")


def _model_body(source: str) -> str:
    """Return the balanced ``(define(model ...) ...)`` form."""
    idx = source.rfind("(define(model")
    if idx < 0:
        raise ValueError("no `(define(model ...)` form in emitted Church source")
    return _balanced_form(source, idx)


def _signature_inputs(model_form: str) -> tuple[str, ...]:
    """Parse the model's formal-parameter names from its signature.

    The signature is the second element of the ``define`` form:
    ``(define (model <a> <b> ...) ...)``. Returns the input names in
    declaration order (empty for a nullary model).
    """
    open_model = model_form.index("(model")
    sig = _balanced_form(model_form, open_model)
    inner = sig[len("(model") : -1].strip()
    if not inner:
        return ()
    return tuple(inner.split())


def _sample_sites(model_form: str) -> tuple[str, ...]:
    """Return the model's latent sample-site names in source order.

    Walks the body's top-level ``(define <name> <rhs>)`` forms and
    keeps every one whose ``<rhs>`` calls ``sample``; a deterministic
    let (no ``sample``) and the Gaussian-process mean / covariance
    helpers are excluded. The result is the exact order in which the
    model's ``sample`` calls execute, so the clamp cursor's per-site
    values line up with the draws.
    """
    body_open = model_form.index("(model")
    sig = _balanced_form(model_form, body_open)
    cursor = body_open + len(sig)
    sites: list[str] = []
    depth_text = model_form
    i = cursor
    n = len(depth_text)
    while i < n:
        j = depth_text.find("(define ", i)
        if j < 0:
            break
        form = _balanced_form(depth_text, j)
        after = j + len("(define ")
        name = ""
        k = after
        while k < len(depth_text) and depth_text[k] not in "( )":
            name += depth_text[k]
            k += 1
        if name and "(sample" in form:
            sites.append(name)
        i = j + len(form)
    return tuple(sites)


#: Rewrites every ``(map ...)`` call site in the emitted source to
#: ``(qmap ...)``. The R6RS ``map`` leaves its per-element call order
#: unspecified (Chez evaluates it out of order), which would misassign
#: the clamp cursor's positional values across a batched sample site;
#: ``qmap`` calls its function strictly left-to-right so the cursor and
#: the site's index axis line up. ``map`` only ever appears in
#: call position in the runtime and emitted model, so the rewrite is
#: exact.
_MAP_CALL_RE = re.compile(r"\(map(?=[\s(])")


#: Left-to-right ``map``. Prepended verbatim (never rewritten) above
#: the runtime so every rewritten ``(qmap ...)`` resolves to it. It
#: reproduces ``map``'s value while fixing evaluation order, so the
#: runtime's pure maps are unaffected and the sample-driving maps
#: consume the cursor in index order.
_QMAP_PRELUDE = """\
;; ---- ordered map (prepended by the Church probe) ----
(define (qmap-heads ls)
  (if (null? ls) (quote ()) (cons (caar ls) (qmap-heads (cdr ls)))))
(define (qmap-tails ls)
  (if (null? ls) (quote ()) (cons (cdar ls) (qmap-tails (cdr ls)))))
(define (qmap f . ls)
  (if (null? (car ls))
      (quote ())
      (let loop ((rest ls) (acc (quote ())))
        (if (null? (car rest))
            (reverse acc)
            (loop (qmap-tails rest) (cons (apply f (qmap-heads rest)) acc))))))
"""


_DRIVER_TEMPLATE = """
;; ---- probe clamping driver (appended by the Church probe) ----
;; A flat cursor of clamped scalars, concatenated across latent sites
;; in the model's sample-call order. `sample` is rebound to score the
;; clamped value in place of a fresh draw; the site's own draw supplies
;; only the value's shape, so a scalar site consumes one cursor entry
;; and a vector / matrix site consumes its whole event structure.
(define *clamp-cursor* (list {cursor}))
(define (clamp-next!)
  (if (null? *clamp-cursor*)
      (error 'clamp "cursor exhausted: more sample leaves than clamps")
      (let ((v (car *clamp-cursor*)))
        (set! *clamp-cursor* (cdr *clamp-cursor*))
        v)))
(define (take-shaped template)
  (if (pair? template) (qmap take-shaped template) (clamp-next!)))
(set! sample
  (lambda (d)
    (let* ((tmpl (dist-draw d)) (val (take-shaped tmpl)))
      (record-score! (dist-score d val))
      val)))
(model {args})
(if (pair? *clamp-cursor*)
    (error 'clamp "cursor not exhausted: fewer sample leaves than clamps"))
(display *log-weight*)
(newline)
"""


@dataclasses.dataclass(frozen=True)
class ChurchProbe:
    backend: str = "church"

    def available(self) -> bool:
        """True iff a Scheme interpreter binary is reachable on ``PATH``.

        The numeric-equivalence layer skips the ``(*, church)`` cell
        only when no interpreter is installed; it exercises the
        executing path the moment one appears.
        """
        return _find_interpreter() is not None

    def evaluate(
        self,
        source: bytes,
        fixture_name: str,
        points: list[Point],
        *,
        scratch: pathlib.Path,
    ) -> ProbeResult:
        """Compute the joint log-density at each clamped test point.

        Writes the emitted module plus a per-point clamping driver to
        ``scratch`` and runs it through the reachable Scheme
        interpreter, reading the accumulated ``*log-weight*``.
        """
        interpreter = _find_interpreter()
        if interpreter is None:
            raise RuntimeError(
                "no Scheme interpreter on PATH (one of "
                f"{', '.join(_SCHEME_INTERPRETERS)}); ChurchProbe.available() "
                "returned False but evaluate() was called anyway"
            )
        scratch.mkdir(parents=True, exist_ok=True)
        text = source.decode("utf-8")
        model_form = _model_body(text)
        inputs = _signature_inputs(model_form)
        sites = _sample_sites(model_form)

        log_densities: list[float] = []
        for i, pt in enumerate(points):
            cursor = self._cursor_literals(pt, sites, fixture_name)
            args = self._call_args(pt, inputs, fixture_name)
            driver = _DRIVER_TEMPLATE.format(cursor=" ".join(cursor), args=args)
            program = _QMAP_PRELUDE + _MAP_CALL_RE.sub("(qmap", text) + driver
            scm_path = scratch / f"{fixture_name}.{i}.scm"
            scm_path.write_text(program)
            completed = subprocess.run(
                [interpreter, "--script", str(scm_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{interpreter} exited {completed.returncode} on "
                    f"{fixture_name!r} point {i}: "
                    f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
                )
            last_line = completed.stdout.strip().splitlines()[-1]
            log_densities.append(float(last_line))

        return ProbeResult(
            backend=self.backend,
            fixture=fixture_name,
            log_densities=log_densities,
            metadata={"runtime": f"{pathlib.Path(interpreter).name} (Scheme)"},
        )

    def _cursor_literals(
        self, pt: Point, sites: tuple[str, ...], fixture_name: str
    ) -> list[str]:
        """Build the flat cursor of Scheme literals in sample-site order.

        Each site's clamped value is flattened (row-major) and appended;
        a site whose clamp is absent from ``pt.params`` is a
        contract violation the probe surfaces rather than silently
        misaligning the cursor.
        """
        cursor: list[str] = []
        for name in sites:
            if name not in pt.params:
                raise RuntimeError(
                    f"church probe on {fixture_name!r}: sample site "
                    f"{name!r} has no clamp in point.params "
                    f"(available: {sorted(pt.params)})"
                )
            cursor.extend(_scheme_literal(v) for v in _flatten(pt.params[name]))
        return cursor

    def _call_args(self, pt: Point, inputs: tuple[str, ...], fixture_name: str) -> str:
        """Render the model call's data-input arguments.

        Each formal input name resolves against ``pt.data`` first, then
        ``pt.params`` (so a program that reads a latent as a plain input
        still binds); a missing input is a contract violation.
        """
        rendered: list[str] = []
        for name in inputs:
            if name in pt.data:
                rendered.append(_scheme_value(pt.data[name]))
            elif name in pt.params:
                rendered.append(_scheme_value(pt.params[name]))
            else:
                raise RuntimeError(
                    f"church probe on {fixture_name!r}: model input "
                    f"{name!r} absent from point.data and point.params "
                    f"(data: {sorted(pt.data)}; params: {sorted(pt.params)})"
                )
        return " ".join(rendered)


_PROBE: LogDensityProbe = ChurchProbe()
assert isinstance(_PROBE, LogDensityProbe)


__all__ = ["ChurchProbe"]
