"""Stdlib effect instances for the typeclass hierarchy.

Each effect is a :class:`dx.Model` carrying any effect parameters
(e.g. ``Continuation(answer)``) plus a typeclass-instance interface
that implements the relevant operations. Effects compose with
arbitrary user-defined effects through the class-driven lifting
machinery in :mod:`quivers.stochastic.effect_lifts` — the lifts
dispatch on which classes an effect inhabits, never on the effect's
identity.

Currently shipped:

- :class:`Identity` — trivial monad / traversable; the no-effect case.

Planned (Phase 7 layering):

- :class:`Continuation(answer)` — Cont monad for scope-taking.
- :class:`Alternative_` — Hamblin powerset / alternatives.
- :class:`State(state_type)` — anaphora / discourse referents.
- :class:`Reader(env_type)` — assignment functions.
- :class:`Writer(monoid)` — supplements / nonrestrictive content.
- :class:`Maybe` — partiality / presupposition failure.
- :class:`List` — bag-of-readings parsers.

The closed-form instances above are equivalent presentations of the
same effects derivable as ``FreeMonad(sig)`` constructions in
:mod:`quivers.monadic.algebraic`.
"""

from __future__ import annotations

import didactic.api as dx

from quivers.core.morphisms import Morphism, identity as id_morph
from quivers.core.objects import SetObject
from quivers.monadic.typeclasses import Monad


class Identity(dx.Model):
    """The trivial monad: ``Id(A) = A``.

    All operations are identity functions; the laws hold trivially.
    Useful as a base case for monad-transformer stacks (``StateT(Identity)``
    is the standard non-stacked State monad) and as a ground for
    sanity checks of the typeclass-instance interface.
    """

    name: str = "Identity"

    # Functor
    def fmap_obj(self, A: SetObject) -> SetObject:
        return A

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f

    # Applicative
    def pure(self, A: SetObject) -> Morphism:
        return id_morph(A)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        # apply : Id(A → B) ⊗ Id(A) → Id(B). Since Id(X) = X, this
        # is the evaluation morphism of an internal-hom representation
        # over the V-Rel universe; concrete construction depends on
        # the closure of the underlying category. Identity_apply is
        # just function application, which we expose as identity on
        # the underlying V-relation when called in a context that
        # already has the function-value resolved.
        raise NotImplementedError(
            "Identity.apply requires evaluation of an internal-hom "
            "value; use the bridge to Kleisli(Identity) for the "
            "function-call semantics"
        )

    # Monad
    def join(self, A: SetObject) -> Morphism:
        # join : Id(Id(A)) → Id(A). Since Id(Id(A)) = Id(A) = A,
        # this is the identity on A.
        return id_morph(A)


# Register Identity as an instance of Monad at the abstract base level.
# Python ABCs accept registration via Monad.register(...), which makes
# isinstance(Identity(), Monad) True without forcing Identity to inherit
# from Monad directly. The class-driven lifting machinery uses
# isinstance() checks against the abstract bases, so registration is
# sufficient.
Monad.register(Identity)


__all__ = ["Identity"]
