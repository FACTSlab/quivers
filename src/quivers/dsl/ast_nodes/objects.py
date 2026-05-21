"""Type-expression AST nodes (categorical objects).

The discrete-type and continuous-space families are unified
into one ``ObjectExpr`` family. Operators (``*`` product, ``+``
coproduct, ``/`` ``\\`` slash, ``T(X)`` effect-apply) cross the
discrete/continuous boundary syntactically; the compiler enforces
categorical validity downstream (e.g. slash is residuated-only).

Constructors split into two AST kinds so downstream walkers dispatch
without re-parsing the constructor name:

* `DiscreteConstructor` (currently only ``FinSet(N)``)
* `ContinuousConstructor` (``Real``, ``Simplex``, ``Sphere``,
  ``Ball``, ``CholeskyFactor``, ``Covariance``, ``Correlation``,
  ``Orthogonal``, ``Stiefel``, ``LowerTriangular``, ``Diagonal``)
"""

from typing import Literal

import didactic.api as dx

class ObjectExpr(dx.TaggedUnion, discriminator="kind"):
    """Sum of type-expression node kinds."""

class TypeName(ObjectExpr):
    """A named type reference (identifier or integer literal)."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["type_name"] = "type_name"

class ObjectProduct(ObjectExpr):
    """Product type: ``A * B``."""

    components: tuple[ObjectExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["object_product"] = "object_product"

class ObjectCoproduct(ObjectExpr):
    """Coproduct type: ``A + B``."""

    components: tuple[ObjectExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["object_coproduct"] = "object_coproduct"

class ObjectSlash(ObjectExpr):
    """Residuated slash type: ``result / argument`` or ``result \\ argument``.

    Legal only when both operands inhabit a residuated universe
    (typically a ``FreeResiduated`` object). The compiler enforces
    this at use-site; the grammar accepts the slash on any pair of
    type expressions.
    """

    result: ObjectExpr
    argument: ObjectExpr
    direction: Literal["/", "\\"]
    line: int = 0
    col: int = 0
    kind: Literal["object_slash"] = "object_slash"

class ObjectEffectApply(ObjectExpr):
    """Effect-typed type-application: ``T(X)``.

    The ``effect`` field names the effect (a previously-declared
    effect or stdlib effect); ``args`` are its applied arguments.
    Legal only inside a ``FreeResiduated`` whose ``effects`` list
    mentions the named effect.
    """

    effect: str
    args: tuple[ObjectExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["object_effect_apply"] = "object_effect_apply"

class DiscreteConstructor(ObjectExpr):
    """A discrete-type constructor call: currently ``FinSet(N)``."""

    constructor: Literal["FinSet"]
    args: tuple[str, ...] = ()
    kwargs: dict[str, str] = dx.field(default_factory=dict)
    line: int = 0
    col: int = 0
    kind: Literal["discrete_constructor"] = "discrete_constructor"

class ContinuousConstructor(ObjectExpr):
    """A continuous-space constructor call.

    The eleven continuous constructors are: ``Real``, ``Simplex``,
    ``Sphere``, ``Ball``, ``CholeskyFactor``, ``Covariance``,
    ``Correlation``, ``Orthogonal``, ``Stiefel``, ``LowerTriangular``,
    ``Diagonal``. The compiler dispatches on the ``constructor``
    field to pick the appropriate
    [`quivers.continuous.spaces.ContinuousSpace`][quivers.continuous.spaces.ContinuousSpace] subclass.
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
    "ObjectCoproduct",
    "ObjectEffectApply",
    "ObjectExpr",
    "TypeName",
    "ObjectProduct",
    "ObjectSlash",
]
