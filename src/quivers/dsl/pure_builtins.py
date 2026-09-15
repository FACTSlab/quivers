"""The pure-expression builtins the QIEC lowering resolves by type.

Operators and builtin applications in ``program`` let steps and in QIEC
values resolve to primitives of the closed registry by their operand
types; reductions and last-axis operations resolve to their own terms.
The tables here are what the lowering and the program elaboration both
consult, so a name is a builtin in one place exactly when it is in the
other.
"""

from __future__ import annotations

from collections.abc import Mapping

from quivers.dsl.ast_nodes.let_expressions import LetBinaryOperator
from quivers.qiec.canonical import LOG_WEIGHT
from quivers.qiec.terms import ReductionOperator, RowwiseOperator
from quivers.qiec.types import BOOL, INT, REAL, STRING, TypeExpr

#: Operator and operand type to the primitive that implements it.
_BINARY_PRIMITIVES: Mapping[LetBinaryOperator, Mapping[TypeExpr, str]] = {
    "+": {INT: "add_int", REAL: "add_real", STRING: "concat"},
    "-": {INT: "sub_int", REAL: "sub_real"},
    "*": {INT: "mul_int", REAL: "mul_real"},
    "/": {INT: "div_int", REAL: "div_real"},
    "%": {INT: "mod_int"},
    "==": {INT: "eq_int", REAL: "eq_real", BOOL: "eq_bool", STRING: "eq_string"},
    "!=": {INT: "ne_int", REAL: "ne_real", BOOL: "ne_bool", STRING: "ne_string"},
    "<": {INT: "lt_int", REAL: "lt_real"},
    "<=": {INT: "le_int", REAL: "le_real"},
    ">": {INT: "gt_int", REAL: "gt_real"},
    ">=": {INT: "ge_int", REAL: "ge_real"},
    "&&": {BOOL: "and"},
    "||": {BOOL: "or"},
}

#: Builtin name and argument types to the primitive that implements it.
_BUILTIN_PRIMITIVES: Mapping[str, Mapping[tuple[TypeExpr, ...], str]] = {
    "real": {(INT,): "int_to_real"},
    "int": {(REAL,): "real_to_int"},
    "exp": {(REAL,): "exp"},
    "log": {(REAL,): "log"},
    "sqrt": {(REAL,): "sqrt"},
    "pow": {(REAL, REAL): "pow_real"},
    "abs": {(INT,): "abs_int", (REAL,): "abs_real"},
    "min": {(INT, INT): "min_int", (REAL, REAL): "min_real"},
    "max": {(INT, INT): "max_int", (REAL, REAL): "max_real"},
    "weight": {(REAL,): "as_weight"},
    "weight_value": {(LOG_WEIGHT,): "weight_value"},
    **{
        name: {(REAL,): name}
        for name in (
            "expm1",
            "log1p",
            "log2",
            "log10",
            "rsqrt",
            "square",
            "sign",
            "reciprocal",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "floor",
            "ceil",
            "round",
            "trunc",
            "erf",
            "erfc",
            "erfinv",
            "lgamma",
            "digamma",
            "sigmoid",
            "relu",
            "relu6",
            "elu",
            "selu",
            "gelu",
            "silu",
            "mish",
            "softplus",
            "logsigmoid",
            "softsign",
        )
    },
}

#: Builtin names that reduce a whole tensor to one number.
_REDUCTIONS: Mapping[str, ReductionOperator] = {
    "sum": "sum",
    "mean": "mean",
    "max": "max",
    "min": "min",
    "logsumexp": "logsumexp",
    "prod": "prod",
}

#: Builtin names that act along a tensor's last axis.
_ROWWISE: Mapping[str, RowwiseOperator] = {
    "softmax": "softmax",
    "log_softmax": "log_softmax",
    "cumsum": "cumsum",
    "sort": "sort",
    "normalize": "normalize",
}


#: Every name a let expression may apply as a builtin.
PURE_BUILTINS: frozenset[str] = frozenset(
    (*_BUILTIN_PRIMITIVES, *_REDUCTIONS, *_ROWWISE)
)

#: The operator each binary primitive reads back as.
PRIMITIVE_OPERATORS: Mapping[str, LetBinaryOperator] = {
    primitive: operator
    for operator, by_type in _BINARY_PRIMITIVES.items()
    for primitive in by_type.values()
}

#: The builtin each call primitive reads back as.
PRIMITIVE_BUILTINS: Mapping[str, str] = {
    primitive: builtin
    for builtin, by_types in _BUILTIN_PRIMITIVES.items()
    for primitive in by_types.values()
}


__all__ = ["PRIMITIVE_BUILTINS", "PRIMITIVE_OPERATORS", "PURE_BUILTINS"]
