"""Regression tests for the WebPPL runtime graft and its trigger.

WebPPL compiles its source through a CPS transform that rejects
`while` / `for` loops, so the grafted runtime helpers
([`runtime_webppl.js`][quivers.transpile.runtime_webppl]) must stay in
WebPPL's functional subset: any loop makes the whole emitted program
fail to parse (`cpsFinalStatement`) the moment the helper is grafted.
These tests pin two properties of the graft:

1. The runtime helper source is loop-free, so every emission that
   grafts it parses under WebPPL.
2. The graft fires exactly when a helper is used. A deterministic
   `let` that pivots on a data-input array lowers to a plain
   `mapIndexed` and needs no helper, so the runtime must not be
   prepended; a `let` that broadcasts an array-valued prior under a
   scalar operator does emit `_qvr_bcast` and must graft it.

Assertions run on emitted text so they stay fast and deterministic.
"""

from __future__ import annotations

import pathlib
import re

from quivers.dsl.parser import parse
from quivers.transpile import transpile

_RUNTIME_WEBPPL = (
    pathlib.Path(__import__("quivers.transpile", fromlist=["__file__"]).__file__)
    .resolve()
    .parent
    / "runtime_webppl.js"
)


def _emit(source: str) -> str:
    return transpile(parse(source), target="webppl").decode()


def _dense(text: str) -> str:
    return "".join(text.split())


# ---------------------------------------------------------------------------
# The grafted runtime must contain no `while` / `for` loop: WebPPL's CPS
# transform rejects both, so a single loop anywhere in the file makes
# every grafting emission fail to compile.
# ---------------------------------------------------------------------------


def test_runtime_webppl_has_no_loops() -> None:
    text = _RUNTIME_WEBPPL.read_text()
    # Strip line comments so a `while` / `for` mentioned in prose does
    # not trip the scan.
    code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
    assert not re.search(r"\bwhile\b", code), (
        "runtime_webppl.js contains a `while` loop; WebPPL's CPS "
        "transform rejects it and every grafting emission fails to "
        "parse"
    )
    assert not re.search(r"\bfor\s*\(", code), (
        "runtime_webppl.js contains a `for` loop; WebPPL's CPS "
        "transform rejects it and every grafting emission fails to "
        "parse"
    )


def test_runtime_webppl_has_no_reassignment() -> None:
    # WebPPL is single-assignment: `x = x + 1` is a syntax error, so a
    # loop-free helper that still mutated a local would break too. The
    # only assignment form the emit tolerates is a `var` declaration or
    # a write to the special `globalStore`.
    text = _RUNTIME_WEBPPL.read_text()
    code = "\n".join(line.split("//", 1)[0] for line in text.splitlines())
    # A bare `<ident> = ` not preceded by `var ` and not a `==` / `<=`
    # / `>=` / `!=` comparison, nor a property key (`k:`), nor a
    # `globalStore.` write.
    offenders = re.findall(
        r"(?m)^\s*(?!var\b)(?!return\b)([A-Za-z_]\w*)\s=\s(?!=)", code
    )
    assert not offenders, (
        f"runtime_webppl.js reassigns locals {offenders!r}; WebPPL is "
        f"single-assignment and rejects reassignment"
    )


# ---------------------------------------------------------------------------
# Graft trigger: a data-input-pivot `let` lowers to `mapIndexed` and
# must NOT prepend the runtime; nothing in the body needs a helper.
# ---------------------------------------------------------------------------


_DATA_INPUT_PIVOT = """object Obs : FinSet 60

program blr : Obs -> Obs
    sample a <- Normal(0.0, 1.0)
    sample b <- Normal(0.0, 1.0)
    let mu = a + b * x_design
    observe y : Obs <- Normal(mu, 0.3)
    return mu

export blr"""


def test_webppl_data_input_pivot_let_does_not_graft_runtime() -> None:
    out = _emit(_DATA_INPUT_PIVOT)
    dense = _dense(out)
    # The array-valued `let` lifts through `mapIndexed`, indexing the
    # data input per element, so no broadcast helper is emitted.
    assert "mapIndexed" in dense
    assert "x_design[__i" in dense
    # None of the runtime helpers are grafted for this model.
    assert "var _qvr_bcast = function" not in out
    assert "var LogNormal = function" not in out
    assert "var _gamma_sample = function" not in out


# ---------------------------------------------------------------------------
# Graft trigger: a `let` broadcasting an array-valued prior under a
# scalar operator DOES emit `_qvr_bcast`, so the helper is grafted.
# ---------------------------------------------------------------------------


_ARRAY_PRIOR_BROADCAST = """object O : FinSet 4

program g : O -> O
    sample w : O <- Normal(0.0, 1.0)
    let z = w * 2.0 + 1.0
    observe y : O <- Normal(z, 0.5)
    return z

export g"""


def test_webppl_array_prior_let_grafts_qvr_bcast() -> None:
    out = _emit(_ARRAY_PRIOR_BROADCAST)
    dense = _dense(out)
    # The array-valued prior `w` broadcasts through `_qvr_bcast`.
    assert '_qvr_bcast("*"' in dense
    assert '_qvr_bcast("+"' in dense
    assert "var _qvr_bcast = function" in out
    # The scalar operators are never applied to the array directly.
    assert "w*2" not in dense


# ---------------------------------------------------------------------------
# Gamma rate -> scale reciprocal: WebPPL's built-in `Gamma({shape,
# scale})` is scale-parameterised, so the torch rate is inverted.
# ---------------------------------------------------------------------------


_GAMMA = """object O : FinSet 8

program g : O -> O
    sample t <- Gamma(2.0, 5.0)
    return t

export g"""


def test_webppl_gamma_emits_reciprocal_scale() -> None:
    out = _dense(_emit(_GAMMA))
    assert "Gamma({shape:2,scale:1/5})" in out
    # The old bug aliased the rate straight into the scale slot.
    assert "scale:5}" not in out
