"""Tests for the qvr_program panproto protocol and Schema extraction.

For every example program, compile to populate the resolution environment,
extract the corresponding :class:`panproto.Schema`, and assert it validates
against :data:`~quivers.dsl.program_theory.QVR_PROGRAM_PROTOCOL`. Then
spot-check that ``panproto.diff_schemas`` produces a non-trivial diff
between two structurally distinct example programs (sanity that the
extracted schemas carry enough structure to be diffable).
"""

from pathlib import Path

import panproto
import pytest

from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse_file
from quivers.dsl.program_theory import QVR_PROGRAM_PROTOCOL, extract_program_schema


EXAMPLES_DIR = Path(__file__).parent.parent / "src/quivers/dsl/examples"
EXAMPLE_PATHS = sorted(EXAMPLES_DIR.glob("*.qvr"))


def _compile_to_env(path: Path) -> Compiler:
    """Parse the program and run :meth:`Compiler.compile_env` to populate
    the resolved object/space/morphism dictionaries."""
    module = parse_file(path)
    compiler = Compiler(module)
    compiler.compile_env()
    return compiler


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=[p.stem for p in EXAMPLE_PATHS])
def test_program_schema_validates(path: Path) -> None:
    """Every example produces a Schema that validates against qvr_program."""
    compiler = _compile_to_env(path)
    schema = extract_program_schema(compiler)
    assert schema.protocol == "qvr_program"
    schema.validate(QVR_PROGRAM_PROTOCOL)


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=[p.stem for p in EXAMPLE_PATHS])
def test_program_schema_has_root(path: Path) -> None:
    """Every extracted schema has the ``program`` root vertex."""
    compiler = _compile_to_env(path)
    schema = extract_program_schema(compiler)
    kinds = {v.kind for v in schema.vertices}
    assert "program" in kinds


def test_program_schema_object_decls_for_hmm() -> None:
    """hmm.qvr's two object declarations (State, Obs) appear as object_decl
    vertices with the right names and bind to finset vertices."""
    compiler = _compile_to_env(EXAMPLES_DIR / "hmm.qvr")
    schema = extract_program_schema(compiler)

    object_decl_ids = [v.id for v in schema.vertices if v.kind == "object_decl"]
    decl_names = []
    for vid in object_decl_ids:
        for c in schema.constraints_for(vid):
            if c.sort == "name":
                decl_names.append(c.value)
    assert set(decl_names) == {"State", "Obs"}

    finset_ids = [v.id for v in schema.vertices if v.kind == "finset"]
    finset_props = []
    for fid in finset_ids:
        props = {c.sort: c.value for c in schema.constraints_for(fid)}
        finset_props.append(props)
    cardinalities = sorted(int(p["cardinality"]) for p in finset_props)
    assert cardinalities == [8, 16]


def test_output_decl_recorded_for_hmm() -> None:
    """The compiler's output expression surfaces as an output_decl vertex."""
    compiler = _compile_to_env(EXAMPLES_DIR / "hmm.qvr")
    schema = extract_program_schema(compiler)
    output_ids = [v.id for v in schema.vertices if v.kind == "output_decl"]
    assert len(output_ids) == 1, "expected exactly one output_decl"
    output_constraints = {c.sort: c.value for c in schema.constraints_for(output_ids[0])}
    # hmm.qvr's `output hmm` resolves to ExprIdent(name='hmm')
    assert output_constraints["name"] == "hmm"
    # the program -> output_decl edge with kind 'output' is present
    output_edges = [
        e for e in schema.edges if e.src == "program" and e.kind == "output"
    ]
    assert len(output_edges) == 1
    assert output_edges[0].tgt == output_ids[0]


def test_diff_distinguishes_distinct_programs() -> None:
    """panproto.diff_schemas produces a non-trivial diff between two
    structurally distinct example programs."""
    compiler_a = _compile_to_env(EXAMPLES_DIR / "hmm.qvr")
    compiler_b = _compile_to_env(EXAMPLES_DIR / "pcfg.qvr")
    schema_a = extract_program_schema(compiler_a)
    schema_b = extract_program_schema(compiler_b)

    diff = panproto.diff_schemas(schema_a, schema_b).to_dict()
    # at minimum, the two schemas should differ — they declare different
    # objects (State/Obs vs N/T) and different morphisms.
    assert (
        len(diff["added_vertices"]) > 0
        or len(diff["removed_vertices"]) > 0
    )


def test_identical_compilation_produces_equal_schemas() -> None:
    """Compiling the same .qvr file twice yields identical Schemas."""
    a = extract_program_schema(_compile_to_env(EXAMPLES_DIR / "hmm.qvr"))
    b = extract_program_schema(_compile_to_env(EXAMPLES_DIR / "hmm.qvr"))
    diff = panproto.diff_schemas(a, b).to_dict()
    # vertex/edge counts should match; vertex IDs are deterministic in the
    # extractor (object_decl::Name, then sequential numeric suffixes for the
    # set-object subtrees) so the schemas should be structurally identical.
    assert a.vertex_count == b.vertex_count
    assert a.edge_count == b.edge_count
    assert len(diff["added_vertices"]) == 0
    assert len(diff["removed_vertices"]) == 0
