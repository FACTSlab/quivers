"""Space-expression AST nodes (continuous spaces)."""

from typing import Literal

import didactic.api as dx


class SpaceExpr(dx.TaggedUnion, discriminator="kind"):
    """Sum of space-expression node kinds."""


class SpaceName(SpaceExpr):
    """A bare identifier referencing a previously declared space."""

    name: str
    line: int = 0
    col: int = 0
    kind: Literal["space_name"] = "space_name"


class SpaceConstructor(SpaceExpr):
    """Space constructor call: ``Euclidean(3)`` or ``Euclidean(2, low=0.0, high=1.0)``."""

    constructor: str
    args: tuple[str, ...] = ()
    kwargs: dict[str, str] = dx.field(default_factory=dict)
    line: int = 0
    col: int = 0
    kind: Literal["space_constructor"] = "space_constructor"


class SpaceProduct(SpaceExpr):
    """Product space: ``A * B``."""

    components: tuple[SpaceExpr, ...]
    line: int = 0
    col: int = 0
    kind: Literal["space_product"] = "space_product"


__all__ = ["SpaceExpr", "SpaceName", "SpaceConstructor", "SpaceProduct"]
