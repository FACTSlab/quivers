"""Monadic structures: typeclass hierarchy, instances, transformers,
algebraic effects, comonads, algebras, and distributive laws.

The `Monad` ABC lives in `typeclasses`. Concrete monad
implementations (``FuzzyPowersetMonad``, ``FreeMonoidMonad``,
``GiryMonad``, etc.) subclass it directly and provide the
``fmap_obj`` / ``fmap`` / ``pure`` / ``apply`` / ``join`` operations.
``unit`` and ``multiply`` are exposed as aliases on the concrete
classes for the Eilenberg–Moore vocabulary.
"""

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
from quivers.monadic.monads import (
    FuzzyPowersetMonad,
    FreeMonoidMonad,
    KleisliCategory,
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
from quivers.monadic.instances import (
    Identity,
    Maybe,
    Alternative_,
    Continuation,
    State,
    Reader,
    Writer,
    ListMonad,
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
    CoKleisli,
    kleisli,
    arrow_monad,
    cokleisli,
)
from quivers.monadic.laws import (
    check_functor_laws,
    check_monad_laws,
)
from quivers.monadic.theories import (
    TypeclassTheory,
    ThFunctor,
    ThApplicative,
    ThMonad,
    ThAlternative,
    ThMonadPlus,
    ThMonadTrans,
    ThFoldable,
    ThTraversable,
)

__all__ = [
    # typeclasses
    "Functor",
    "Applicative",
    "Monad",
    "Alternative",
    "MonadPlus",
    "Foldable",
    "Traversable",
    "MonadTrans",
    # monads
    "FuzzyPowersetMonad",
    "FreeMonoidMonad",
    "KleisliCategory",
    # comonads
    "Comonad",
    "CoKleisliCategory",
    "DiagonalComonad",
    "CofreeComonad",
    # algebras
    "Algebra",
    "FreeAlgebra",
    "ObservedAlgebra",
    "Coalgebra",
    "CofreeCoalgebra",
    "ObservedCoalgebra",
    "EilenbergMooreCategory",
    # distributive_laws
    "DistributiveLaw",
    "FreeMonoidPowersetLaw",
    # instances
    "Identity",
    "Maybe",
    "Alternative_",
    "Continuation",
    "State",
    "Reader",
    "Writer",
    "ListMonad",
    # transformers
    "StateT",
    "ReaderT",
    "MaybeT",
    "ContT",
    "WriterT",
    # algebraic
    "Operation",
    "EffectSignature",
    "Handler",
    "FreeMonad",
    # bridges
    "Kleisli",
    "ArrowMonad",
    "CoKleisli",
    "kleisli",
    "arrow_monad",
    "cokleisli",
    # laws
    "check_functor_laws",
    "check_monad_laws",
    # theories
    "TypeclassTheory",
    "ThFunctor",
    "ThApplicative",
    "ThMonad",
    "ThAlternative",
    "ThMonadPlus",
    "ThMonadTrans",
    "ThFoldable",
    "ThTraversable",
]
