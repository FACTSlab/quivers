"""Monad transformers: stack monadic effects.

A `MonadTrans` instance ``T`` provides a `lift`
embedding an inner-monad Kleisli arrow into the transformer.
Stacking ``T(m)`` for a base monad ``m`` produces a new monad
whose effects combine ``T``'s and ``m``'s.

Currently shipped transformers:

- `StateT(state)` — adds state-threading to any inner monad.
- `ReaderT(env)` — adds reader-effect to any inner monad.
- `MaybeT` — adds presupposition-failure to any inner monad.
- `ContT(answer)` — adds continuation-passing to any inner monad.
- `WriterT(monoid)` — adds accumulator side-channel to any
  inner monad.

Each transformer is itself a `Monad` instance once applied to
a base monad; the transformer's `lift` is a `MonadTrans`
operation.
"""

from __future__ import annotations

import didactic.api as dx

from quivers.core.morphisms import Morphism
from quivers.core.objects import ProductSet, SetObject
from quivers.monadic.typeclasses import Monad, MonadTrans


class _TransBase(dx.Model):
    """Common base for transformers; lift signature placeholder.

    Concrete transformer subclasses provide `lift` and the
    lifted monadic operations on the stacked monad.
    """

    name: str = "Transformer"

    def lift(self, m: Monad, A: SetObject) -> Morphism:
        raise NotImplementedError(
            "Transformer.lift requires a concrete inner monad's Kleisli "
            "embedding; override on the concrete transformer instance"
        )


class StateT(_TransBase):
    """State transformer: ``StateT(σ)(m)(A) = σ → m(A × σ)``.

    Stacks a state-threading effect on top of an arbitrary inner
    monad. The standard non-stacked State monad is recovered as
    ``StateT(σ)(Identity)``.

    Attributes
    ----------
    state : SetObject
        The state type ``σ``.
    """

    state: SetObject

    def fmap_obj(self, A: SetObject) -> SetObject:
        return ProductSet(components=(A, self.state))


MonadTrans.register(StateT)


class ReaderT(_TransBase):
    """Reader transformer: ``ReaderT(ρ)(m)(A) = ρ → m(A)``.

    Stacks a reader effect on top of an arbitrary inner monad.

    Attributes
    ----------
    env : SetObject
        The environment type ``ρ``.
    """

    env: SetObject

    def fmap_obj(self, A: SetObject) -> SetObject:
        return A


MonadTrans.register(ReaderT)


class MaybeT(_TransBase):
    """Maybe transformer: ``MaybeT(m)(A) = m(A + 1)``.

    Stacks a partiality / presupposition-failure effect on top of
    an arbitrary inner monad.
    """

    def fmap_obj(self, A: SetObject) -> SetObject:
        from quivers.core.objects import CoproductSet, FinSet

        nothing = FinSet(name=f"_nothing_{A!s}", cardinality=1)
        return CoproductSet(components=(A, nothing))


MonadTrans.register(MaybeT)


class ContT(_TransBase):
    """Continuation transformer: ``ContT(ρ)(m)(A) = (A → m(ρ)) → m(ρ)``.

    Stacks a continuation effect on top of an arbitrary inner monad.

    Attributes
    ----------
    answer : SetObject
        The answer type ``ρ``.
    """

    answer: SetObject

    def fmap_obj(self, A: SetObject) -> SetObject:
        return self.answer


MonadTrans.register(ContT)


class WriterT(_TransBase):
    """Writer transformer: ``WriterT(M)(m)(A) = m(A × M)``.

    Stacks an accumulator effect on top of an arbitrary inner monad.

    Attributes
    ----------
    monoid : SetObject
        The accumulator type ``M``.
    """

    monoid: SetObject

    def fmap_obj(self, A: SetObject) -> SetObject:
        return ProductSet(components=(A, self.monoid))


MonadTrans.register(WriterT)


__all__ = [
    "StateT",
    "ReaderT",
    "MaybeT",
    "ContT",
    "WriterT",
]
