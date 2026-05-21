"""Hughes-style arrow hierarchy for compositional computation.

Arrows present effectful computation as a *category of computations*
rather than as a sequence of binds. Following Hughes 2000, the
hierarchy is:

    Category_
    ├── Arrow
    │   ├── ArrowChoice
    │   ├── ArrowApply       — equivalent power to Monad via ArrowMonad
    │   ├── ArrowLoop        — feedback / fixed points
    │   └── ArrowZero
    │       └── ArrowPlus

Each typeclass is an ABC declaring the operations an arrow instance
must provide; default derivations are supplied where Hughes admits
them. Concrete instances (``Function``, ``VRel``, ``Stochastic``,
``Kleisli(M)``, ``Cokleisli(W)``, ``LinearMap``) live alongside in
`quivers.arrows.instances`.

Bridges between the monad and arrow hierarchies live in
[`quivers.monadic.bridges`][quivers.monadic.bridges]: every Monad gives a Kleisli arrow,
every ArrowApply gives a Monad. The two presentations of effectful
computation are interchangeable through these conversions.

References
----------
- Hughes, J. (2000). *Generalising Monads to Arrows*. Science of
  Computer Programming, 37(1–3), 67–111.
  [doi:10.1016/S0167-6423(99)00023-4](https://doi.org/10.1016/S0167-6423(99)00023-4)
"""

from quivers.arrows.typeclasses import (
    Arrow,
    ArrowApply,
    ArrowChoice,
    ArrowLoop,
    ArrowPlus,
    ArrowZero,
    Category_,
)


__all__ = [
    "Category_",
    "Arrow",
    "ArrowChoice",
    "ArrowApply",
    "ArrowLoop",
    "ArrowZero",
    "ArrowPlus",
]
