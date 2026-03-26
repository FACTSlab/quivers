"""Monadic structures: monads, comonads, algebras, and distributive laws."""

from quivers.monadic.monads import (
    Monad,
    KleisliCategory,
    FuzzyPowersetMonad,
    FreeMonoidMonad,
)
from quivers.monadic.comonads import (
    Comonad,
    CoKleisliCategory,
    DiagonalComonad,
    CofreeComonad,
)
from quivers.monadic.algebras import (
    Algebra,
    FreeAlgebra,
    ObservedAlgebra,
    Coalgebra,
    CofreeCoalgebra,
    ObservedCoalgebra,
    EilenbergMooreCategory,
)
from quivers.monadic.distributive_laws import (
    DistributiveLaw,
    FreeMonoidPowersetLaw,
)

__all__ = [
    # Monads
    "Monad",
    "KleisliCategory",
    "FuzzyPowersetMonad",
    "FreeMonoidMonad",
    # Comonads
    "Comonad",
    "CoKleisliCategory",
    "DiagonalComonad",
    "CofreeComonad",
    # Algebras
    "Algebra",
    "FreeAlgebra",
    "ObservedAlgebra",
    "Coalgebra",
    "CofreeCoalgebra",
    "ObservedCoalgebra",
    "EilenbergMooreCategory",
    # Distributive laws
    "DistributiveLaw",
    "FreeMonoidPowersetLaw",
]
