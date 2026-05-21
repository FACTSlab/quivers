"""Monadic structures: typeclass hierarchy, instances, transformers,
algebraic effects, comonads, algebras, and distributive laws.

The `Monad` ABC lives in `typeclasses`. Concrete monad
implementations (``FuzzyPowersetMonad``, ``FreeMonoidMonad``,
``GiryMonad``, etc.) subclass it directly and provide the
``fmap_obj`` / ``fmap`` / ``pure`` / ``apply`` / ``join`` operations.
``unit`` and ``multiply`` are exposed as aliases on the concrete
classes for the Eilenberg–Moore vocabulary.
"""

# Typeclass tower.
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

# Concrete monad instances of the tower.
from quivers.monadic.monads import (
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

# Stdlib effect instances and adjacent infrastructure.
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
    # Typeclasses
    "Functor",
    "Applicative",
    "Monad",
    "Alternative",
    "MonadPlus",
    "Foldable",
    "Traversable",
    "MonadTrans",
    # Concrete monads
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
    # Stdlib effect instances
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
