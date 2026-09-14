"""The closed registry of pure primitives QIEC values may apply.

A primitive is nominal: it has a stable identity derived from its name, a
fixed monomorphic signature, and a capability tag a backend either
supports or refuses. Overloading is a surface concern: a source ``+`` on
two integers lowers to ``add_int`` and on two reals to ``add_real``, so
the kernel never has to choose an implementation by inspecting a runtime
value. The reference implementations here are what the evaluator runs;
each host runtime carries the same table in its own language.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Literal

from quivers.qiec.identifiers import PrimitiveId
from quivers.qiec.types import BOOL, INT, REAL, STRING, TypeExpr

type PrimitiveCapability = Literal[
    "arithmetic",
    "comparison",
    "boolean",
    "string",
    "conversion",
    "math",
]


@dataclass(frozen=True, slots=True)
class PrimitiveSignature:
    """One primitive's name, types, and the capability a backend needs for it.

    Parameters
    ----------
    name
        The stable nominal name, such as ``"add_int"``.
    parameters
        The argument types, in order.
    result
        The result type.
    capability
        The feature a target must support to render the primitive.
    """

    name: str
    parameters: tuple[TypeExpr, ...]
    result: TypeExpr
    capability: PrimitiveCapability

    @property
    def id(self) -> PrimitiveId:
        """The primitive's stable identity.

        Returns
        -------
        PrimitiveId
            An identity derived from the ``builtin`` namespace and the
            name, so the same primitive has the same identity in every
            module.
        """
        return PrimitiveId.derive("builtin", self.name)


def _truncating_division(dividend: int, divisor: int) -> int:
    """Integer division rounding toward zero, as every host target does.

    Parameters
    ----------
    dividend : int
        The number divided.
    divisor : int
        The number divided by.

    Returns
    -------
    int
        The quotient truncated toward zero. Python's ``//`` floors, which
        differs for a negative operand, so the host semantics are spelled
        out rather than inherited.

    Raises
    ------
    ZeroDivisionError
        If the divisor is zero.
    """
    quotient = abs(dividend) // abs(divisor)
    return quotient if (dividend < 0) == (divisor < 0) else -quotient


def _truncating_remainder(dividend: int, divisor: int) -> int:
    """The remainder of truncating division, carrying the dividend's sign.

    Parameters
    ----------
    dividend : int
        The number divided.
    divisor : int
        The number divided by.

    Returns
    -------
    int
        ``dividend - divisor * trunc(dividend / divisor)``.

    Raises
    ------
    ZeroDivisionError
        If the divisor is zero.
    """
    return dividend - divisor * _truncating_division(dividend, divisor)


def _signature(
    name: str,
    parameters: tuple[TypeExpr, ...],
    result: TypeExpr,
    capability: PrimitiveCapability,
    implementation: Callable[..., object],
) -> tuple[PrimitiveSignature, Callable[..., object]]:
    """Pair a signature with its reference implementation.

    Parameters
    ----------
    name : str
        The primitive's name.
    parameters : tuple[TypeExpr, ...]
        The argument types.
    result : TypeExpr
        The result type.
    capability : PrimitiveCapability
        The feature tag.
    implementation : Callable[..., object]
        The reference implementation over host values.

    Returns
    -------
    tuple[PrimitiveSignature, Callable[..., object]]
        The signature and implementation, ready for the tables below.
    """
    return PrimitiveSignature(name, parameters, result, capability), implementation


_ENTRIES: tuple[tuple[PrimitiveSignature, Callable[..., object]], ...] = (
    _signature("add_int", (INT, INT), INT, "arithmetic", lambda a, b: a + b),
    _signature("sub_int", (INT, INT), INT, "arithmetic", lambda a, b: a - b),
    _signature("mul_int", (INT, INT), INT, "arithmetic", lambda a, b: a * b),
    _signature("div_int", (INT, INT), INT, "arithmetic", _truncating_division),
    _signature("mod_int", (INT, INT), INT, "arithmetic", _truncating_remainder),
    _signature("neg_int", (INT,), INT, "arithmetic", lambda a: -a),
    _signature("abs_int", (INT,), INT, "arithmetic", abs),
    _signature("min_int", (INT, INT), INT, "arithmetic", min),
    _signature("max_int", (INT, INT), INT, "arithmetic", max),
    _signature("add_real", (REAL, REAL), REAL, "arithmetic", lambda a, b: a + b),
    _signature("sub_real", (REAL, REAL), REAL, "arithmetic", lambda a, b: a - b),
    _signature("mul_real", (REAL, REAL), REAL, "arithmetic", lambda a, b: a * b),
    _signature("div_real", (REAL, REAL), REAL, "arithmetic", lambda a, b: a / b),
    _signature("neg_real", (REAL,), REAL, "arithmetic", lambda a: -a),
    _signature("abs_real", (REAL,), REAL, "arithmetic", abs),
    _signature("min_real", (REAL, REAL), REAL, "arithmetic", min),
    _signature("max_real", (REAL, REAL), REAL, "arithmetic", max),
    _signature("pow_real", (REAL, REAL), REAL, "math", math.pow),
    _signature("exp", (REAL,), REAL, "math", math.exp),
    _signature("log", (REAL,), REAL, "math", math.log),
    _signature("sqrt", (REAL,), REAL, "math", math.sqrt),
    _signature("eq_int", (INT, INT), BOOL, "comparison", lambda a, b: a == b),
    _signature("ne_int", (INT, INT), BOOL, "comparison", lambda a, b: a != b),
    _signature("lt_int", (INT, INT), BOOL, "comparison", lambda a, b: a < b),
    _signature("le_int", (INT, INT), BOOL, "comparison", lambda a, b: a <= b),
    _signature("gt_int", (INT, INT), BOOL, "comparison", lambda a, b: a > b),
    _signature("ge_int", (INT, INT), BOOL, "comparison", lambda a, b: a >= b),
    _signature("eq_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a == b),
    _signature("ne_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a != b),
    _signature("lt_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a < b),
    _signature("le_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a <= b),
    _signature("gt_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a > b),
    _signature("ge_real", (REAL, REAL), BOOL, "comparison", lambda a, b: a >= b),
    _signature("eq_bool", (BOOL, BOOL), BOOL, "comparison", lambda a, b: a == b),
    _signature("ne_bool", (BOOL, BOOL), BOOL, "comparison", lambda a, b: a != b),
    _signature("eq_string", (STRING, STRING), BOOL, "comparison", lambda a, b: a == b),
    _signature("ne_string", (STRING, STRING), BOOL, "comparison", lambda a, b: a != b),
    _signature("and", (BOOL, BOOL), BOOL, "boolean", lambda a, b: a and b),
    _signature("or", (BOOL, BOOL), BOOL, "boolean", lambda a, b: a or b),
    _signature("not", (BOOL,), BOOL, "boolean", lambda a: not a),
    _signature("concat", (STRING, STRING), STRING, "string", lambda a, b: a + b),
    _signature("int_to_real", (INT,), REAL, "conversion", float),
    _signature("real_to_int", (REAL,), INT, "conversion", math.trunc),
)

#: Every primitive by name. The mapping is closed: a name absent here is
#: not a primitive, and no backend may resolve one dynamically.
PRIMITIVES: Mapping[str, PrimitiveSignature] = MappingProxyType(
    {signature.name: signature for signature, _ in _ENTRIES}
)

#: The reference evaluator's implementation of every primitive, by name.
IMPLEMENTATIONS: Mapping[str, Callable[..., object]] = MappingProxyType(
    {signature.name: implementation for signature, implementation in _ENTRIES}
)


def primitive(name: str) -> PrimitiveSignature:
    """Look up a primitive by name.

    Parameters
    ----------
    name : str
        The nominal name.

    Returns
    -------
    PrimitiveSignature
        The primitive's signature.

    Raises
    ------
    KeyError
        If no primitive has the name. Callers turn this into a
        source-located diagnostic; the registry itself has no location to
        blame.
    """
    try:
        return PRIMITIVES[name]
    except KeyError as error:
        raise KeyError(f"unknown QIEC primitive {name!r}") from error


__all__ = [
    "IMPLEMENTATIONS",
    "PRIMITIVES",
    "PrimitiveCapability",
    "PrimitiveSignature",
    "primitive",
]
