"""Grammar / AST round-trip and compound-arg validation tests.

Exercises the ``draw_arg_list`` grammar production (a vector literal
``[a, b]`` is a `DrawArgList` of atoms; a matrix literal
``[[a, b], [c, d]]`` is a `DrawArgList` whose items are themselves
`DrawArgList` rows), the `DrawArg` tagged-union variants the parser
emits, and the
[`validate_family_arg_shapes`][quivers.dsl.compiler._validate.validate_family_arg_shapes]
pass (implicit-defaults warning + family-arg-shape error).
"""

from __future__ import annotations

import textwrap

from quivers.dsl.ast_nodes import (
    DrawArgList,
    DrawArgName,
    DrawArgNamed,
    DrawArgScalar,
    MorphismDecl,
    ProgramDecl,
    SampleStep,
)
from quivers.dsl.compiler._validate import validate_family_arg_shapes
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.dsl.draw_args import is_matrix, list_atoms, matrix_rows


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
            sample y : X <- Categorical(probs=[0.1, 0.2, 0.7])
            return y
    """
    module = _parse(src)
    step = _first_sample(module)
    assert step.args is not None
    assert len(step.args) == 1
    arg = step.args[0]
    assert isinstance(arg, DrawArgNamed) and arg.parameter == "probs"
    assert isinstance(arg.value, DrawArgList)
    assert not is_matrix(arg.value)
    assert list_atoms(arg.value) == (0.1, 0.2, 0.7)


def test_parser_emits_nested_draw_arg_list_for_2d_literal():
    src = """
        object X : FinSet 2
        program p : X -> X
            sample z : X <- MultivariateNormal(loc=[0.0, 0.0], scale_tril=[[1.0, 0.5], [0.5, 1.0]])
            return z
    """
    module = _parse(src)
    step = _first_sample(module)
    assert step.args is not None
    assert len(step.args) == 2
    mean, cov = step.args
    assert isinstance(mean, DrawArgNamed) and mean.parameter == "loc"
    assert isinstance(cov, DrawArgNamed) and cov.parameter == "scale_tril"
    # A vector literal is a flat `DrawArgList` of scalar atoms.
    assert isinstance(mean.value, DrawArgList)
    assert not is_matrix(mean.value)
    assert list_atoms(mean.value) == (0.0, 0.0)
    # A matrix literal is a `DrawArgList` whose items are themselves
    # `DrawArgList` rows of scalar atoms.
    assert isinstance(cov.value, DrawArgList)
    assert is_matrix(cov.value)
    assert matrix_rows(cov.value) == ((1.0, 0.5), (0.5, 1.0))


def test_parser_emits_draw_arg_scalar_for_numeric_literal():
    src = """
        object X : FinSet 3
        program p : X -> X
            sample y : X <- Categorical(probs=0.5)
            return y
    """
    module = _parse(src)
    step = _first_sample(module)
    assert step.args is not None
    arg = step.args[0]
    assert isinstance(arg, DrawArgNamed) and arg.parameter == "probs"
    assert isinstance(arg.value, DrawArgScalar)
    assert arg.value.value == 0.5


def test_parser_emits_draw_arg_name_for_identifier():
    src = """
        object X : FinSet 3
        program p(probs) : X -> X
            sample y : X <- Categorical(probs=probs)
            return y
    """
    module = _parse(src)
    step = _first_sample(module)
    assert step.args is not None
    arg = step.args[0]
    assert isinstance(arg, DrawArgNamed) and arg.parameter == "probs"
    assert isinstance(arg.value, DrawArgName)
    assert arg.value.text == "probs"


def test_parser_and_emitter_preserve_a_named_logits_argument():
    src = """
        object X : FinSet 3
        program p(logits) : X -> X
            sample y : X <- Categorical(logits=logits)
            return y
    """
    module = _parse(src)
    step = _first_sample(module)
    assert step.args is not None
    arg = step.args[0]
    assert isinstance(arg, DrawArgNamed)
    assert arg.parameter == "logits"
    assert isinstance(arg.value, DrawArgName)
    assert arg.value.text == "logits"
    assert "Categorical(logits=logits)" in module_to_source(module)


def test_parser_and_emitter_preserve_a_named_morphism_init_argument():
    src = """
        object X : FinSet 3
        morphism prior : X -> X ~ Categorical(logits=logits)
    """
    module = _parse(src)
    decl = module.statements[1]
    assert isinstance(decl, MorphismDecl)
    assert decl.init_family is not None
    assert len(decl.init_family.args) == 1
    arg = decl.init_family.args[0]
    assert isinstance(arg, DrawArgNamed)
    assert arg.parameter == "logits"
    assert isinstance(arg.value, DrawArgName)
    assert arg.value.text == "logits"
    assert "Categorical(logits=logits)" in module_to_source(module)


def test_logits_literal_is_not_validated_as_a_probability_simplex():
    src = """
        object X : FinSet 3
        program p : X -> X
            sample y : X <- Categorical(logits=[-2.0, 0.5, 1.25])
            return y
    """
    module = _parse(src)
    assert validate_family_arg_shapes(module) == []


def test_positional_categorical_argument_is_rejected_as_ambiguous():
    src = """
        object X : FinSet 3
        program p(probs) : X -> X
            sample y : X <- Categorical(probs)
            return y
    """
    diags = validate_family_arg_shapes(_parse(src))
    target = [d for d in diags if d.code == "family-arg-parameterization"]
    assert len(target) == 1
    assert target[0].severity == "error"


def test_categorical_rejects_probs_and_logits_together():
    src = """
        object X : FinSet 3
        program p : X -> X
            sample probs : X <- HalfNormal(scale=1.0)
            sample logits : X <- Normal(loc=0.0, scale=1.0)
            sample y : X <- Categorical(probs=probs, logits=logits)
            return y
    """
    diags = validate_family_arg_shapes(_parse(src))
    target = [d for d in diags if d.code == "family-arg-parameterization"]
    assert len(target) == 1
    assert target[0].severity == "error"


def test_positional_categorical_morphism_init_is_rejected_as_ambiguous():
    src = """
        object X : FinSet 3
        morphism prior : X -> X ~ Categorical(probs)
    """
    diags = validate_family_arg_shapes(_parse(src))
    target = [d for d in diags if d.code == "family-arg-parameterization"]
    assert len(target) == 1
    assert target[0].severity == "error"


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


def test_family_arg_parameterization_error_for_incomplete_schema():
    src = """
        object X : Real 1
        program p : X -> X
            sample x : X <- Normal(loc=1)
            return x
    """
    module = _parse(src)
    diags = validate_family_arg_shapes(module)
    target = [d for d in diags if d.code == "family-arg-parameterization"]
    assert target, f"expected family-arg-parameterization diagnostic, got {diags!r}"
    assert any(d.severity == "error" for d in target)


def test_simplex_literal_sum_warning():
    src = """
        object X : FinSet 3
        program p : X -> X
            sample y : X <- Categorical(probs=[0.1, 0.2, 0.3])
            return y
    """
    module = _parse(src)
    diags = validate_family_arg_shapes(module)
    sim = [d for d in diags if d.code == "family-arg-shape" and d.severity == "warning"]
    assert sim, f"expected simplex-sum warning, got {diags!r}"


def test_simplex_literal_valid_no_warning():
    src = """
        object X : FinSet 3
        program p : X -> X
            sample y : X <- Categorical(probs=[0.1, 0.2, 0.7])
            return y
    """
    module = _parse(src)
    diags = validate_family_arg_shapes(module)
    assert all(
        d.code != "family-arg-shape" or d.severity != "warning" for d in diags
    ), f"unexpected warning: {diags!r}"


def test_param_source_model_does_not_abort_source_shape_validation():
    """A transpiler refusal is not a source-language validation failure."""
    src = """
        object Feature : Real 2
        object Target : Real 1
        object Resp : FinSet 6
        morphism net : Feature -> Target [param_source=mlp(5, 3)] ~ Normal
        program prog : Resp -> Target
            observe y : Resp <- net(x)
            return y
        export prog
    """
    module = _parse(src)
    assert validate_family_arg_shapes(module) == []
