"""Type-expression AST nodes (categorical objects).

The discrete-type and continuous-space families are unified
into one ``TypeExpr`` family. Operators (``*`` product, ``+``
coproduct, ``/`` ``\\`` slash, ``T(X)`` effect-apply) cross the
discrete/continuous boundary syntactically; the compiler enforces
categorical validity downstream (e.g. slash is residuated-only).

Constructors split into two AST kinds so downstream walkers dispatch
without re-parsing the constructor name:

* :class:`DiscreteConstructor` (currently only ``FinSet(N)``)
* :class:`ContinuousConstructor` (``Real``, ``Simplex``, ``Sphere``,
  ``Ball``, ``CholeskyFactor``, ``Covariance``, ``Correlation``,
  ``Orthogonal``, ``Stiefel``, ``LowerTriangular``, ``Diagonal``)
"""

from typing import Literal

import didactic.api as dx

class TypeExpr(dx.TaggedUnion, discriminator="kind"):
    """Sum of type-expression node kinds."""

class TypeName(TypeExpr):
    """A named type reference (identifier or integer literal)."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["type_name"] = "type_name"

class TypeProduct(TypeExpr):
    """Product type: ``A * B``."""

    components: tuple[TypeExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["type_product"] = "type_product"

class TypeCoproduct(TypeExpr):
    """Coproduct type: ``A + B``."""

    components: tuple[TypeExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["type_coproduct"] = "type_coproduct"

class TypeSlash(TypeExpr):
    """Residuated slash type: ``result / argument`` or ``result \\ argument``.

    Legal only when both operands inhabit a residuated universe
    (typically a ``FreeResiduated`` object). The compiler enforces
    this at use-site; the grammar accepts the slash on any pair of
    type expressions.
    """

    result: TypeExpr
    argument: TypeExpr
    direction: Literal["/", "\\"]
    line: int = 0
    col: int = 0
    kind: Literal["type_slash"] = "type_slash"

class TypeEffectApply(TypeExpr):
    """Effect-typed type-application: ``T(X)``.

    The ``effect`` field names the effect (a previously-declared
    effect or stdlib effect); ``args`` are its applied arguments.
    Legal only inside a ``FreeResiduated`` whose ``effects`` list
    mentions the named effect.
    """

    effect: str
    args: tuple[TypeExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["type_effect_apply"] = "type_effect_apply"

class DiscreteConstructor(TypeExpr):
    """A discrete-type constructor call: currently ``FinSet(N)``."""

    constructor: Literal["FinSet"]
    args: tuple[str, ...] = ()
    kwargs: dict[str, str] = dx.field(default_factory=dict)
    line: int = 0
    col: int = 0
    kind: Literal["discrete_constructor"] = "discrete_constructor"

class ContinuousConstructor(TypeExpr):
    """A continuous-space constructor call.

    The eleven continuous constructors are: ``Real``, ``Simplex``,
    ``Sphere``, ``Ball``, ``CholeskyFactor``, ``Covariance``,
    ``Correlation``, ``Orthogonal``, ``Stiefel``, ``LowerTriangular``,
    ``Diagonal``. The compiler dispatches on the ``constructor``
    field to pick the appropriate
    :class:`quivers.continuous.spaces.ContinuousSpace` subclass.
    """

    constructor: Literal[
        "Real",
        "Simplex",
        "Sphere",
        "Ball",
        "CholeskyFactor",
        "Covariance",
        "Correlation",
        "Orthogonal",
        "Stiefel",
        "LowerTriangular",
        "Diagonal",
    ]
    args: tuple[str, ...] = ()
    kwargs: dict[str, str] = dx.field(default_factory=dict)
    line: int = 0
    col: int = 0
    kind: Literal["continuous_constructor"] = "continuous_constructor"

__all__ = [
    "ContinuousConstructor",
    "DiscreteConstructor",
    "TypeCoproduct",
    "TypeEffectApply",
    "TypeExpr",
    "TypeName",
    "TypeProduct",
    "TypeSlash",
]
