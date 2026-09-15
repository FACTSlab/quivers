"""The QIEC golden modules parse to one unambiguous tree each.

`tests/fixtures/qiec/` holds the source modules that the executable
surface is frozen against. Downstream passes are built to handle exactly
these forms, so a change that makes one of them ambiguous, or that quietly
stops parsing one, has to fail here rather than in whichever pass notices
first.

The four modules cover the surface between them: a recursive indexed
traversal, an authored state handler with a scoped instance, an effectful
probabilistic helper beside a model, and a grouped marginal over a
GADT-coded observation state.
"""

from __future__ import annotations

import pathlib

import pytest

from quivers.dsl import parse
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser._registry import _registry, _Tree


_FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "qiec"

#: The surface forms the goldens exist to pin, and the module that must
#: exercise each. A form nothing exercises is a form no downstream pass is
#: obliged to handle, which is how a frozen surface rots.
_REQUIRED_FORMS: dict[str, str] = {
    "qiec_call_computation": "recursive_indexed_traversal",
    "qiec_pure_binding": "authored_state_handler",
    "qiec_instance_computation": "authored_state_handler",
    "qiec_resume_computation": "authored_state_handler",
    "qiec_handler_return_clause": "authored_state_handler",
    "qiec_handler_operation_clause": "authored_state_handler",
    "qiec_case_computation": "recursive_indexed_traversal",
    "marginalize_step": "grouped_marginal_gadt_state",
}


def _golden_paths() -> list[pathlib.Path]:
    """Every golden module, sorted.

    Returns
    -------
    list[pathlib.Path]
        The `.qvr` sources under `tests/fixtures/qiec/`.

    Raises
    ------
    AssertionError
        If the directory holds no goldens, which would make every test
        below vacuous.
    """
    paths = sorted(_FIXTURES.glob("*.qvr"))
    assert paths, f"no golden modules under {_FIXTURES}"
    return paths


def _node_kinds(source: str) -> set[str]:
    """Every named node kind in a source's parse tree.

    The tree is built directly rather than through `parse`, because the
    checks here are about what the grammar produced and must stay
    meaningful even while a walker for a new form is still being written.

    Parameters
    ----------
    source : str
        QVR source text.

    Returns
    -------
    set[str]
        The named kinds appearing anywhere in the tree.

    Raises
    ------
    PanprotoError
        If the source cannot be parsed at all.
    """
    source_bytes = source.encode("utf-8")
    schema = _registry().parse_with_protocol("qvr", source_bytes, "<golden>")
    tree = _Tree(schema, source_bytes)
    return {tree.kind(vid) for vid in tree.vertices}


@pytest.mark.parametrize("path", _golden_paths(), ids=lambda p: p.stem)
def test_golden_module_parses_without_error_nodes(path: pathlib.Path) -> None:
    """A golden parses cleanly.

    An `ERROR` or `MISSING` node means tree-sitter recovered rather than
    parsed, which is the ambiguity this corpus exists to exclude.
    """
    kinds = _node_kinds(path.read_text())
    damaged = {kind for kind in kinds if kind in ("ERROR", "MISSING")}
    assert not damaged, (
        f"{path.name} parsed with {sorted(damaged)!r}. The grammar recovered "
        f"instead of parsing, so this module no longer pins one tree."
    )


def test_every_frozen_surface_form_is_exercised() -> None:
    """Each pinned form appears in the golden that owns it.

    Without this, a form could be dropped from the grammar and from the
    goldens together, and the corpus would still pass while the surface it
    was supposed to freeze had silently narrowed.
    """
    by_stem = {path.stem: _node_kinds(path.read_text()) for path in _golden_paths()}
    missing = {
        form: owner
        for form, owner in _REQUIRED_FORMS.items()
        if form not in by_stem.get(owner, set())
    }
    assert not missing, (
        f"these frozen surface forms are absent from the golden that should "
        f"exercise them: {missing!r}. Either the module stopped using the "
        f"form, or the grammar stopped producing it."
    )


@pytest.mark.parametrize("path", _golden_paths(), ids=lambda p: p.stem)
def test_golden_module_round_trips_through_parse_and_emit(
    path: pathlib.Path,
) -> None:
    """Emitting a parsed golden and parsing it again yields the same AST.

    This is what makes the emitter trustworthy as an inverse of the
    parser rather than merely a pretty printer: a form the emitter drops
    or spells differently enough to re-parse as something else fails
    here.
    """
    source = path.read_text()
    module = parse(source)
    assert parse(module_to_source(module)) == module


@pytest.mark.parametrize("path", _golden_paths(), ids=lambda p: p.stem)
def test_golden_module_is_already_canonical(path: pathlib.Path) -> None:
    """A golden is written in the form the emitter produces.

    Stronger than the round trip above, and the reason it is worth
    pinning: a golden that drifts from canonical form still round-trips
    through the AST while no longer showing what the emitter writes, so
    the corpus would stop documenting the surface it freezes.
    """
    source = path.read_text()
    emitted = module_to_source(parse(source))
    if not emitted.endswith("\n"):
        emitted += "\n"
    assert emitted == source, (
        f"{path.name} is not in canonical form. Rewrite it as "
        f"`module_to_source(parse(source))` produces it, or fix the "
        f"emitter if the canonical form is wrong."
    )


@pytest.mark.parametrize("path", _golden_paths(), ids=lambda p: p.stem)
def test_golden_module_lowers_serializes_and_projects(path: pathlib.Path) -> None:
    """A golden survives every stable pass downstream of the parser.

    Parsing alone proves the surface is unambiguous; it does not prove the
    forms it freezes are handled. Each golden is lowered to the kernel,
    rechecked independently of the lowerer, round-tripped through the wire
    format, projected into the transpiler IR, and analyzed for every
    target, so a form that a later pass silently cannot handle fails here
    with that pass named.
    """
    from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
    from quivers.qiec import validate_module
    from quivers.qiec.serialization import dumps, loads
    from quivers.transpile import available_targets
    from quivers.transpile.qiec_ir import analyze_qiec_capabilities, lower_qiec_ir

    module = lower_qvr_to_qiec(parse(path.read_text()), file_path=str(path))
    validate_module(module)
    assert loads(dumps(module)) == module
    ir = lower_qiec_ir(module)
    assert ir.computations, f"{path.name} lowered to no computations"
    for target in available_targets():
        diagnostics = analyze_qiec_capabilities(ir, target)
        for diagnostic in diagnostics:
            assert diagnostic.feature, diagnostic.message
