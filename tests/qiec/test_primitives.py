"""Pure primitives, tuples, and projections in the kernel."""

from __future__ import annotations

import pytest

from quivers.qiec import (
    BOOL,
    INT,
    REAL,
    STRING,
    CheckContext,
    Evaluator,
    KernelError,
    KernelRegistry,
    LiteralValue,
    PrimitiveApplication,
    PrimitiveId,
    Projection,
    Return,
    TupleValue,
    Var,
    infer_value,
    loads,
    dumps,
    primitive,
    PRIMITIVES,
    IMPLEMENTATIONS,
)
from quivers.qiec.evaluator import EvaluationError
from quivers.qiec.identifiers import SourceOrigin
from quivers.qiec.substitution import StaticSubstitution, substitute_value
from quivers.qiec.terms import Local, LiteralData
from quivers.qiec.types import TypeVariable, product_type

ORIGIN = SourceOrigin("primitives", ("test",), "primitive", "test")


def _apply(name: str, *arguments: LiteralData) -> PrimitiveApplication:
    """Apply a primitive to literal arguments at its own signature.

    Parameters
    ----------
    name : str
        The primitive's name.
    *arguments : LiteralData
        Literal values, typed by the signature's parameters.

    Returns
    -------
    PrimitiveApplication
        The well-typed application.
    """
    signature = primitive(name)
    return PrimitiveApplication(
        signature.id,
        name,
        tuple(
            LiteralValue(argument, parameter)
            for argument, parameter in zip(arguments, signature.parameters, strict=True)
        ),
        signature.result,
        ORIGIN,
    )


def test_every_primitive_has_an_implementation_and_a_stable_identity() -> None:
    """The registry is closed and every entry is runnable."""
    assert set(PRIMITIVES) == set(IMPLEMENTATIONS)
    identities = {signature.id for signature in PRIMITIVES.values()}
    assert len(identities) == len(PRIMITIVES)
    assert primitive("add_int").id == PrimitiveId.derive("builtin", "add_int")
    with pytest.raises(KeyError):
        primitive("add_complex")


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("add_int", (2, 3), 5),
        ("div_int", (7, 2), 3),
        ("div_int", (-7, 2), -3),
        ("mod_int", (-7, 2), -1),
        ("mod_int", (7, -2), 1),
        ("div_real", (1.0, 4.0), 0.25),
        ("pow_real", (2.0, 3.0), 8.0),
        ("lt_int", (1, 2), True),
        ("eq_string", ("a", "a"), True),
        ("and", (True, False), False),
        ("not", (False,), True),
        ("concat", ("ab", "cd"), "abcd"),
        ("int_to_real", (3,), 3.0),
        ("real_to_int", (-2.7,), -2),
    ],
)
def test_primitives_check_and_evaluate(
    name: str, arguments: tuple[LiteralData, ...], expected: object
) -> None:
    """A well-typed application infers the registry's result type and runs."""
    application = _apply(name, *arguments)
    assert infer_value(application, KernelRegistry()) == primitive(name).result
    assert Evaluator().evaluate(Return(application)) == expected


def test_integer_division_by_zero_is_an_evaluation_error() -> None:
    with pytest.raises(EvaluationError, match="div_int"):
        Evaluator().evaluate(Return(_apply("div_int", 1, 0)))


def test_primitive_applications_are_checked_against_the_registry() -> None:
    """Arity, argument types, the claimed result, and the identity all matter."""
    registry = KernelRegistry()
    signature = primitive("add_int")
    with pytest.raises(KernelError, match="takes 2 argument"):
        infer_value(
            PrimitiveApplication(
                signature.id, "add_int", (LiteralValue(1, INT),), INT, ORIGIN
            ),
            registry,
        )
    with pytest.raises(KernelError, match="argument 1 of primitive"):
        infer_value(
            PrimitiveApplication(
                signature.id,
                "add_int",
                (LiteralValue(1, INT), LiteralValue(1.0, REAL)),
                INT,
                ORIGIN,
            ),
            registry,
        )
    with pytest.raises(KernelError, match="produces"):
        infer_value(
            PrimitiveApplication(
                signature.id,
                "add_int",
                (LiteralValue(1, INT), LiteralValue(2, INT)),
                REAL,
                ORIGIN,
            ),
            registry,
        )
    with pytest.raises(KernelError, match="identity of another"):
        infer_value(
            PrimitiveApplication(
                primitive("sub_int").id,
                "add_int",
                (LiteralValue(1, INT), LiteralValue(2, INT)),
                INT,
                ORIGIN,
            ),
            registry,
        )
    with pytest.raises(KernelError, match="unknown primitive"):
        infer_value(
            PrimitiveApplication(signature.id, "add_complex", (), INT, ORIGIN),
            registry,
        )


def test_tuples_and_projections_check_evaluate_and_round_trip() -> None:
    registry = KernelRegistry()
    pair = TupleValue(
        (LiteralValue(1, INT), LiteralValue("x", STRING)), product_type(INT, STRING)
    )
    assert infer_value(pair, registry) == product_type(INT, STRING)
    second = Projection(pair, 1, STRING)
    assert infer_value(second, registry) == STRING
    assert Evaluator().evaluate(Return(pair)) == (1, "x")
    assert Evaluator().evaluate(Return(second)) == "x"
    assert loads(dumps(Return(second))) == Return(second)
    assert loads(dumps(Return(_apply("add_int", 1, 2)))) == Return(
        _apply("add_int", 1, 2)
    )

    with pytest.raises(KernelError, match="not the claimed"):
        infer_value(TupleValue(pair.items, product_type(INT, INT)), registry)
    with pytest.raises(KernelError, match="outside a product"):
        infer_value(Projection(pair, 2, INT), registry)
    with pytest.raises(KernelError, match="non-product"):
        infer_value(Projection(LiteralValue(1, INT), 0, INT), registry)
    with pytest.raises(KernelError, match="not the claimed"):
        infer_value(Projection(pair, 0, BOOL), registry)


def test_substitution_reaches_tuple_and_primitive_argument_types() -> None:
    """Static substitution rewrites the types inside the new value forms."""
    variable = TypeVariable("A")
    local = Local("x", variable)
    pair = TupleValue((Var(local), LiteralValue(1, INT)), product_type(variable, INT))
    substitution = StaticSubstitution(types=(("A", REAL),))
    rewritten = substitute_value(pair, substitution)
    assert rewritten == TupleValue(
        (Var(Local("x", REAL)), LiteralValue(1, INT)), product_type(REAL, INT)
    )
    projected = substitute_value(Projection(pair, 0, variable), substitution)
    assert projected == Projection(rewritten, 0, REAL)
    context = CheckContext().extend(Local("x", REAL))
    assert infer_value(projected, KernelRegistry(), context) == REAL
