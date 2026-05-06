"""Monadic structures: typeclass hierarchy, instances, transformers,
algebraic effects, comonads, algebras, and distributive laws.

Two layers coexist:

- The legacy interface (:class:`monads.Monad` ABC and friends) remains
  in place for back-compat.
- The new typeclass tower (:mod:`typeclasses`, :mod:`instances`,
  :mod:`transformers`, :mod:`algebraic`, :mod:`bridges`,
  :mod:`theories`) is the Phase 6 surface used by the categorial-
  effects integration in the DSL.

The two layers will converge once every legacy ``Monad`` subclass is
re-cast as a typeclass instance and the legacy ABC is retired.
"""

# Legacy interface (back-compat with pre-Phase-6 callers).
from quivers.monadic.monads import (
    Monad as LegacyMonad,
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

# New typeclass spine.
from quivers.monadic.typeclasses import (
    Functor,
    Applicative,
    Monad,
    Alternative,
    MonadPlus,
    Foldable,
    Traversable,
    MonadTrans,
)
from quivers.monadic.instances import (
    Identity,
    Maybe,
    Alternative_,
    Continuation,
    State,
    Reader,
    Writer,
    List as ListMonad,
)
from quivers.monadic.transformers import (
    StateT,
    ReaderT,
    MaybeT,
    ContT,
    WriterT,
)
from quivers.monadic.algebraic import (
    Operation,
    EffectSignature,
    Handler,
    FreeMonad,
)
from quivers.monadic.bridges import (
    Kleisli,
    ArrowMonad,
    kleisli,
    arrow_monad,
)


__all__ = [
    # Legacy
    "LegacyMonad",
    "KleisliCategory",
    "FuzzyPowersetMonad",
    "FreeMonoidMonad",
    "Comonad",
    "CoKleisliCategory",
    "DiagonalComonad",
    "CofreeComonad",
    "Algebra",
    "FreeAlgebra",
    "ObservedAlgebra",
    "Coalgebra",
    "CofreeCoalgebra",
    "ObservedCoalgebra",
    "EilenbergMooreCategory",
    "DistributiveLaw",
    "FreeMonoidPowersetLaw",
    # Typeclasses
    "Functor",
    "Applicative",
    "Monad",
    "Alternative",
    "MonadPlus",
    "Foldable",
    "Traversable",
    "MonadTrans",
    # Instances
    "Identity",
    "Maybe",
    "Alternative_",
    "Continuation",
    "State",
    "Reader",
    "Writer",
    "ListMonad",
    # Transformers
    "StateT",
    "ReaderT",
    "MaybeT",
    "ContT",
    "WriterT",
    # Algebraic effects
    "Operation",
    "EffectSignature",
    "Handler",
    "FreeMonad",
    # Bridges
    "Kleisli",
    "ArrowMonad",
    "kleisli",
    "arrow_monad",
]
