"""Grammar / AST round-trip and compound-arg validation tests.

Exercises the new ``draw_arg_list`` / ``draw_arg_matrix`` grammar
productions, the `DrawArg`
tagged-union variants the parser emits, and the
[`validate_family_arg_shapes`][quivers.dsl.compiler._validate.validate_family_arg_shapes]
pass (implicit-defaults warning + family-arg-shape error).
"""

from __future__ import annotations

import textwrap

from quivers.dsl.ast_nodes import (
    DrawArgList,
    DrawArgMatrix,
    DrawArgName,
    DrawArgScalar,
    MorphismDecl,
    ProgramDecl,
    SampleStep,
)
from quivers.dsl.compiler._validate import validate_family_arg_shapes
from quivers.dsl.parser import parse


def _parse(src: str):
    return parse(textwrap.dedent(src).encode())


def _first_sample(module) -> SampleStep:
    for stmt in module.statements:
        if isinstance(stmt, ProgramDecl):
            for step in stmt.draws:
                if isinstance(step, SampleStep):
                    return step
    raise AssertionError("no SampleStep in module")


def test_parser_emits_draw_arg_list_for_vector_literal():
    src = """
        object X : FinSet 3
        program p : X -> X
            sample y : X <- Categorical([0.1, 0.2, 0.7])
            return y
    """
    module = _parse(src)
    step = _first_sample(module)
    assert step.args is not None
    assert len(step.args) == 1
    arg = step.args[0]
    assert isinstance(arg, DrawArgList)
    assert arg.elements == (0.1, 0.2, 0.7)


def test_parser_emits_draw_arg_matrix_for_2d_literal():
    src = """
        object X : FinSet 2
        morphism mvn : X -> X [role=kernel] ~ MultivariateNormal([0.0, 0.0], [[1.0, 0.5], [0.5, 1.0]])
        program p : X -> X
            sample z : X <- mvn
            return z
    """
    module = _parse(src)
    morphism = next(
        s for s in module.statements if isinstance(s, MorphismDecl)
    )
    assert morphism.init_family is not None
    args = morphism.init_family.args
    assert len(args) == 2
    mean, cov = args
    assert isinstance(mean, DrawArgList)
    assert mean.elements == (0.0, 0.0)
    assert isinstance(cov, DrawArgMatrix)
    assert len(cov.rows) == 2
    assert cov.rows[0].elements == (1.0, 0.5)
    assert cov.rows[1].elements == (0.5, 1.0)


def test_parser_emits_draw_arg_scalar_for_numeric_literal():
    src = """
        object X : FinSet 3
        program p : X -> X
            sample y : X <- Categorical(0.5)
            return y
    """
    module = _parse(src)
    step = _first_sample(module)
    assert step.args is not None
    arg = step.args[0]
    assert isinstance(arg, DrawArgScalar)
    assert arg.value == 0.5


def test_parser_emits_draw_arg_name_for_identifier():
    src = """
        object X : FinSet 3
        program p(probs) : X -> X
            sample y : X <- Categorical(probs)
            return y
    """
    module = _parse(src)
    step = _first_sample(module)
    assert step.args is not None
    arg = step.args[0]
    assert isinstance(arg, DrawArgName)
    assert arg.text == "probs"


def test_implicit_family_defaults_emits_warning_diagnostic():
    src = """
        object X : Real 1
        program p : X -> X
            sample x : X <- Normal
            return x
    """
    module = _parse(src)
    diags = validate_family_arg_shapes(module)
    target = [d for d in diags if d.code == "implicit-family-defaults"]
    assert target, f"expected implicit-family-defaults warning, got {diags!r}"
    assert all(d.severity == "warning" for d in target)


def test_family_arg_shape_error_for_arity_mismatch():
    src = """
        object X : Real 1
        program p : X -> X
            sample x : X <- Normal(1, 2, 3)
            return x
    """
    module = _parse(src)
    diags = validate_family_arg_shapes(module)
    target = [d for d in diags if d.code == "family-arg-shape"]
    assert target, f"expected family-arg-shape diagnostic, got {diags!r}"
    assert any(d.severity == "error" for d in target)


def test_simplex_literal_sum_warning():
    src = """
        object X : FinSet 3
        program p : X -> X
            sample y : X <- Categorical([0.1, 0.2, 0.3])
            return y
    """
    module = _parse(src)
    diags = validate_family_arg_shapes(module)
    sim = [
        d
        for d in diags
        if d.code == "family-arg-shape" and d.severity == "warning"
    ]
    assert sim, f"expected simplex-sum warning, got {diags!r}"


def test_simplex_literal_valid_no_warning():
    src = """
        object X : FinSet 3
        program p : X -> X
            sample y : X <- Categorical([0.1, 0.2, 0.7])
            return y
    """
    module = _parse(src)
    diags = validate_family_arg_shapes(module)
    assert all(
        d.code != "family-arg-shape" or d.severity != "warning"
        for d in diags
    ), f"unexpected warning: {diags!r}"
