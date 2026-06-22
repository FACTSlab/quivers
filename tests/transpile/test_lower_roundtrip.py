"""Round-trip lowering test: every gallery `.qvr` with a
[`ProgramDecl`][quivers.dsl.ast_nodes.ProgramDecl] lowers without
error, and the lowered IR's input list covers every free name in the
source while the body step count matches the source's expanded step
count (after `expand_composite_lets` plus marginalize unrolling).

Parameterised over `docs/examples/source/*.qvr`. Gallery files that
do not declare a program (deduction-only files such as
`ccg.qvr`) are skipped at collection time. Gallery files that
exercise composition constructs the lowering does not yet handle
(`scan`, `fan`, kernel composite-let chains) are reported as
`xfail` with the precise `UnsupportedConstruct` kind so the gate
stays loud.
"""

from __future__ import annotations

import pathlib

import pytest

from quivers.dsl.ast_nodes import (
    LetStep,
    MarginalizeStep,
    ObserveStep,
    ProgramDecl,
    ProgramStep,
    ReturnStep,
    SampleStep,
    ScoreStep,
)
from quivers.dsl.ast_nodes.declarations import ExportDecl
from quivers.dsl.parser import parse
from quivers.transpile._api import UnsupportedConstruct
from quivers.transpile._expand_composites import expand_composite_lets
from quivers.transpile.ir import (
    IRDataInput,
    IRDeterministic,
    IRMarginalize,
    IRNode,
    IRObserve,
    IRProgram,
    IRReturn,
    IRSample,
    IRScore,
)
from quivers.transpile.lower import (
    Lower,
    free_vars_in_let,
)


_GALLERY_DIR = pathlib.Path(__file__).resolve().parents[2] / (
    "docs/examples/source"
)


def _gallery_paths() -> list[pathlib.Path]:
    """Return every `.qvr` gallery example with a `ProgramDecl`."""
    paths: list[pathlib.Path] = []
    for path in sorted(_GALLERY_DIR.glob("*.qvr")):
        src = path.read_text()
        try:
            module = parse(src)
        except Exception:  # noqa: BLE001
            continue
        if any(isinstance(s, ProgramDecl) for s in module.statements):
            paths.append(path)
    return paths


GALLERY = _gallery_paths()


@pytest.mark.parametrize(
    "path", GALLERY, ids=[p.stem for p in GALLERY]
)
def test_lower_roundtrip(path: pathlib.Path) -> None:
    """Lower the gallery example and verify the structural invariants."""
    src = path.read_text()
    module = parse(src)
    program = _pick_program(module)
    try:
        ir = Lower().forward(module)
    except UnsupportedConstruct as exc:
        pytest.xfail(
            f"lowering not yet supported for {path.name}: "
            f"{exc.kinds[0] if exc.kinds else exc}"
        )

    # Structural invariants on the IR shape.
    assert isinstance(ir, IRProgram)
    assert ir.name == program.name
    assert all(isinstance(inp, IRDataInput) for inp in ir.inputs)
    assert all(isinstance(node, IRNode) for node in ir.body)

    # Free names in the (expanded) source must be covered by the
    # IR's `inputs` plus the bound names in the body.
    expanded = expand_composite_lets(module, target="stan")
    expanded_program = next(
        s for s in expanded.statements
        if isinstance(s, ProgramDecl) and s.name == program.name
    )
    source_free = _source_free_names(expanded_program)
    ir_bound = _ir_bound_names(ir)
    ir_input_names = {inp.name for inp in ir.inputs}
    missing = source_free - ir_bound - ir_input_names
    assert not missing, (
        f"{path.name}: source free names not covered by IR "
        f"inputs/bindings: {sorted(missing)}"
    )

    # Body step count: every source `ProgramStep` (after composite
    # expansion + marginalize unrolling) maps to an IR node.
    expected_body_len = _expected_body_step_count(expanded_program)
    body_step_count = _ir_body_step_count(ir.body)
    assert body_step_count == expected_body_len, (
        f"{path.name}: IR body step count {body_step_count} != "
        f"expected source step count {expected_body_len}"
    )


def _source_free_names(program: ProgramDecl) -> set[str]:
    """Return the set of free identifier names referenced in the
    program body but not bound by any sample / observe / let / score
    / marginalize step or by a program parameter."""
    bound: set[str] = set()
    if program.params is not None:
        bound.update(program.params)
    if program.type_params is not None:
        bound.update(p.name for p in program.type_params)
    used: list[str] = []
    _collect_step_names(program.draws, bound, used)
    return set(used) - bound


def _collect_step_names(
    steps: tuple[ProgramStep, ...],
    bound: set[str],
    out: list[str],
) -> None:
    for step in steps:
        if isinstance(step, SampleStep):
            for v in step.vars:
                bound.add(v)
            for a in step.args or ():
                _add_arg_names(a, out)
        elif isinstance(step, ObserveStep):
            bound.add(step.var)
            for a in step.args or ():
                _add_arg_names(a, out)
            if step.via is not None:
                out.append(step.via)
        elif isinstance(step, MarginalizeStep):
            bound.add(step.var)
            for a in step.args or ():
                _add_arg_names(a, out)
            _collect_step_names(step.scope, bound, out)
        elif isinstance(step, LetStep):
            bound.add(step.name)
            out.extend(free_vars_in_let(step.value))
        elif isinstance(step, ScoreStep):
            bound.add(step.name)
            out.extend(free_vars_in_let(step.value))
        elif isinstance(step, ReturnStep):
            out.extend(step.vars)


def _add_arg_names(arg: str | float, out: list[str]) -> None:
    """Collect free names from a parser-form arg (string or float)."""
    if not isinstance(arg, str):
        return
    text = arg
    # bracket-indexed `name[i0][i1]...` form: collect the base name
    # and each index expression.
    if "[" in text:
        head, _, rest = text.partition("[")
        out.append(head)
        depth = 0
        inner = ""
        for ch in text[len(head):]:
            if ch == "[":
                depth += 1
                if depth == 1:
                    inner = ""
                    continue
            if ch == "]":
                depth -= 1
                if depth == 0:
                    _add_arg_names(inner, out)
                    inner = ""
                    continue
            if depth >= 1:
                inner += ch
        return
    # Plain identifier or numeric literal.
    try:
        float(text)
    except ValueError:
        out.append(text)


def _ir_bound_names(ir: IRProgram) -> set[str]:
    """Return every name bound by a node in the IR body (recursive)."""
    out: set[str] = set()

    def walk(body: tuple[IRNode, ...]) -> None:
        for node in body:
            if isinstance(node, (IRSample, IRObserve, IRDeterministic)):
                out.add(node.name)
            elif isinstance(node, IRScore):
                out.add(node.name)
            elif isinstance(node, IRMarginalize):
                out.add(node.latent)
                walk(node.scope)
            elif isinstance(node, IRDataInput):
                out.add(node.name)

    walk(ir.body)
    return out


def _ir_body_step_count(body: tuple[IRNode, ...]) -> int:
    """Count IR body nodes, excluding the synthesised `IRReturn`."""
    count = 0
    for node in body:
        if isinstance(node, IRReturn):
            continue
        if isinstance(node, IRMarginalize):
            count += 1 + _ir_body_step_count(node.scope)
        else:
            count += 1
    return count


def _expected_body_step_count(program: ProgramDecl) -> int:
    """Count source `ProgramStep`s, recursing into marginalize scopes,
    excluding the bare `return` slot (handled separately)."""
    count = 0
    for step in program.draws:
        if isinstance(step, ReturnStep):
            continue
        if isinstance(step, MarginalizeStep):
            count += 1 + _expected_body_step_count(
                ProgramDecl(
                    name=program.name,
                    domain=program.domain,
                    codomain=program.codomain,
                    draws=step.scope,
                ),
            )
        else:
            count += 1
    return count



def _pick_program(module) -> ProgramDecl:
    """Match `Lower._pick_program`: prefer a program named in an
    `export` declaration; otherwise return the last `ProgramDecl`.
    A fixture that declares both `program lstm_cell` and
    `program lstm_lm` with `export lstm_lm` lowers `lstm_lm`, so the
    test invariants must apply against the same program Lower picked.
    """
    programs = []
    exported_names = set()
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            programs.append(stmt)
        elif isinstance(stmt, ExportDecl):
            if hasattr(stmt, "expr") and hasattr(stmt.expr, "name"):
                exported_names.add(stmt.expr.name)
    if not programs:
        raise AssertionError("no ProgramDecl in module")
    return next(
        (p for p in programs if p.name in exported_names),
        programs[-1],
    )
