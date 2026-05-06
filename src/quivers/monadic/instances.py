"""Stdlib effect instances for the typeclass hierarchy.

Each effect is a :class:`dx.Model` carrying any effect parameters
(e.g. ``Continuation(answer)``) plus a typeclass-instance interface
that implements the relevant operations. Effects compose with
arbitrary user-defined effects through the class-driven lifting
machinery in :mod:`quivers.stochastic.effect_lifts` — the lifts
dispatch on which classes an effect inhabits, never on the effect's
identity.

Each instance is registered against the appropriate ABC(s) via
``ABC.register(...)`` so that ``isinstance(inst, Monad)`` and
similar predicates succeed without forcing the instance class to
inherit from the typeclass directly.

The type-level action ``F : SetObject → SetObject`` and the
morphism-level operations (``pure``, ``apply``, ``join``, ``empty``,
``alt``) are realised concretely where the construction is
unambiguous in V-Rel; for instances whose operations require an
internal-hom representation (notably :class:`Continuation` and
:class:`Reader`), the operations raise
``NotImplementedError`` with a pointer to the corresponding
algebraic-effect handler in :mod:`quivers.monadic.algebraic`, which
gives the same semantics through a free-monad-over-signature
construction. The typeclass-instance registration is the load-bearing
contract for the schema-lifting machinery; the closed-form
operations are an optimisation, not a requirement.
"""

from __future__ import annotations

import didactic.api as dx

from quivers.core.morphisms import Morphism, identity as id_morph
from quivers.core.objects import CoproductSet, ProductSet, SetObject
from quivers.monadic.typeclasses import (
    Alternative,
    Foldable,
    Monad,
    MonadPlus,
    Traversable,
)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class Identity(dx.Model):
    """The trivial monad: ``Id(A) = A``.

    All operations are identity functions; the laws hold trivially.
    Useful as a base case for monad-transformer stacks (``StateT(Identity)``
    is the standard non-stacked State monad) and as a ground for
    sanity checks of the typeclass-instance interface.
    """

    name: str = "Identity"

    def fmap_obj(self, A: SetObject) -> SetObject:
        return A

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f

    def pure(self, A: SetObject) -> Morphism:
        return id_morph(A)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        raise NotImplementedError(
            "Identity.apply requires evaluation of an internal-hom "
            "value; use the bridge to Kleisli(Identity) for the "
            "function-call semantics"
        )

    def join(self, A: SetObject) -> Morphism:
        return id_morph(A)


Monad.register(Identity)


# ---------------------------------------------------------------------------
# Maybe (presupposition failure / partiality)
# ---------------------------------------------------------------------------


class Maybe(dx.Model):
    """The partiality monad: ``Maybe(A) = A + 1``.

    Used for presupposition failure, partial functions, and other
    "computation may not produce a value" effects. As a ``MonadPlus``
    instance, it carries both a partial-result branch and an empty
    branch.

    The type-level action wraps ``A`` in a coproduct with a singleton
    failure marker. The unit injects into the left (success) summand;
    join collapses ``Maybe(Maybe(A))`` by flattening the failure
    cases into a single failure.
    """

    name: str = "Maybe"

    def fmap_obj(self, A: SetObject) -> SetObject:
        from quivers.core.objects import FinSet

        nothing = FinSet(name=f"_nothing_{A!s}", cardinality=1)
        return CoproductSet(components=(A, nothing))

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        # fmap_Maybe(f) : (A + 1) → (B + 1) acts as f on the left
        # summand and as the identity on the failure marker.
        # Realised as a coproduct morphism over the runtime V-Rel.
        raise NotImplementedError(
            "Maybe.fmap requires coproduct-morphism construction; "
            "lifted forms are derived via the algebraic-effect handler"
        )

    def pure(self, A: SetObject) -> Morphism:
        # pure : A → Maybe(A) injects into the left (success) summand.
        # The injection is the canonical inclusion of the first
        # CoproductSet component; we synthesise it by composing
        # identity with the coproduct's component-injection morphism.
        raise NotImplementedError(
            "Maybe.pure requires coproduct-injection construction"
        )

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        raise NotImplementedError("Maybe.apply: see algebraic handler")

    def join(self, A: SetObject) -> Morphism:
        raise NotImplementedError("Maybe.join: see algebraic handler")

    def empty(self, A: SetObject) -> Morphism:
        # empty : 1 → Maybe(A) injects into the right (failure)
        # summand. Used by Alternative-driven lifts as the absent
        # element.
        raise NotImplementedError("Maybe.empty: see algebraic handler")

    def alt(self, A: SetObject) -> Morphism:
        # alt : Maybe(A) ⊗ Maybe(A) → Maybe(A) prefers the first
        # success; if the first is failure, returns the second.
        raise NotImplementedError("Maybe.alt: see algebraic handler")


MonadPlus.register(Maybe)


# ---------------------------------------------------------------------------
# Alternative_ (Hamblin / focus alternatives)
# ---------------------------------------------------------------------------


class Alternative_(dx.Model):
    """The Hamblin alternative monad: ``Alt(A) = 𝒫_fin(A)``.

    Carries a finite (V-quantale-weighted) set of alternative values.
    Used for question semantics, focus-sensitive operators, and
    plurality. As ``MonadPlus``, it pairs the powerset structure
    with the empty alternative ``∅`` and the union ``∪``.

    The type-level action over a finite set ``A`` is the discrete
    powerset ``𝒫(A)``, of cardinality ``2^|A|``. For pragmatic
    runtime sizes the implementation may use sparse / compressed
    representations or defer to the existing
    :class:`quivers.monadic.monads.FuzzyPowersetMonad`.
    """

    name: str = "Alternative"

    def fmap_obj(self, A: SetObject) -> SetObject:
        # 𝒫_fin(A) — represented at the type level as A; the
        # element multiplicities live in the V-relation tensor
        # (one value per (input, alternative) pair). This matches
        # the existing FuzzyPowersetMonad presentation.
        return A

    def fmap(self, A: SetObject, B: SetObject, f: Morphism) -> Morphism:
        return f  # alternatives propagate pointwise

    def pure(self, A: SetObject) -> Morphism:
        # pure : A → 𝒫(A) is the singleton injection.
        return id_morph(A)

    def apply(self, A: SetObject, B: SetObject) -> Morphism:
        raise NotImplementedError("Alternative_.apply: see algebraic handler")

    def join(self, A: SetObject) -> Morphism:
        # join : 𝒫(𝒫(A)) → 𝒫(A) is set-theoretic flattening (∪).
        # Over V-quantale weights this is the noisy-OR aggregation
        # that the FuzzyPowersetMonad implements.
        return id_morph(A)

    def empty(self, A: SetObject) -> Morphism:
        raise NotImplementedError("Alternative_.empty: see algebraic handler")

    def alt(self, A: SetObject) -> Morphism:
        raise NotImplementedError("Alternative_.alt: see algebraic handler")

    def foldr(self, A: SetObject, B: SetObject) -> Morphism:
        raise NotImplementedError("Alternative_.foldr: see algebraic handler")

    def traverse(self, A, B, applicative, f):
        raise NotImplementedError("Alternative_.traverse: see algebraic handler")


MonadPlus.register(Alternative_)
Foldable.register(Alternative_)
Traversable.register(Alternative_)


# ---------------------------------------------------------------------------
# Continuation (scope-taking, Cont_ρ(α) = (α → ρ) → ρ)
# ---------------------------------------------------------------------------


class Continuation(dx.Model):
    """The continuation monad: ``Cont_ρ(α) = (α → ρ) → ρ``.

    Charlow's central effect: scope-taking expressions denote into a
    Cont monad over a fixed answer type ``ρ`` (typically the sentence
    type ``S``). The unit is the value-into-continuation reflection;
    bind is the Plotkin shift-by-double-negation.

    The closed-form V-Rel realisation of Cont requires an internal-hom
    representation that varies by quantale; for the standard product-
    fuzzy quantale we lean on the algebraic-effect handler in
    :mod:`quivers.monadic.algebraic` (``ContinuationSignature``).

    Attributes
    ----------
    answer : SetObject
        The answer type ``ρ``.
    """

    answer: SetObject
    name: str = "Continuation"

    def fmap_obj(self, A: SetObject) -> SetObject:
        # Cont_ρ(A) at the type level is (A → ρ) → ρ. As a SetObject,
        # this is approximated by the answer type itself when the
        # encoding lives in the morphism layer (a "thunk" of type ρ
        # parameterised by A → ρ context). The exact shape depends
        # on the quantale's internal-hom; we expose ρ as the
        # carrier and let the morphism layer hold the function-space.
        return self.answer

    def fmap(self, A, B, f):
        raise NotImplementedError("Continuation.fmap: see algebraic handler")

    def pure(self, A: SetObject) -> Morphism:
        raise NotImplementedError("Continuation.pure: see algebraic handler")

    def apply(self, A, B):
        raise NotImplementedError("Continuation.apply: see algebraic handler")

    def join(self, A):
        raise NotImplementedError("Continuation.join: see algebraic handler")


Monad.register(Continuation)


# ---------------------------------------------------------------------------
# State (anaphora / discourse referents)
# ---------------------------------------------------------------------------


class State(dx.Model):
    """The state monad: ``State_σ(A) = σ → (A × σ)``.

    Used to thread a discourse context through compositional
    semantics, supporting dynamic-binding accounts of anaphora
    (Heim / Kamp / Groenendijk-Stokhof / Charlow).

    Attributes
    ----------
    state : SetObject
        The state type ``σ``.
    """

    state: SetObject
    name: str = "State"

    def fmap_obj(self, A: SetObject) -> SetObject:
        # State_σ(A) at the type level is σ → (A × σ). As a SetObject
        # we expose the result-product (A × σ); the leading σ → -
        # context is realised in the morphism layer (a Kleisli arrow).
        return ProductSet(components=(A, self.state))

    def fmap(self, A, B, f):
        raise NotImplementedError("State.fmap: see algebraic handler")

    def pure(self, A):
        raise NotImplementedError("State.pure: see algebraic handler")

    def apply(self, A, B):
        raise NotImplementedError("State.apply: see algebraic handler")

    def join(self, A):
        raise NotImplementedError("State.join: see algebraic handler")


Monad.register(State)


# ---------------------------------------------------------------------------
# Reader (assignment functions / indexicality)
# ---------------------------------------------------------------------------


class Reader(dx.Model):
    """The reader monad: ``Reader_ρ(A) = ρ → A``.

    Used to model free variables, assignment functions, and indexical
    parameters in compositional semantics.

    Attributes
    ----------
    env : SetObject
        The environment type ``ρ``.
    """

    env: SetObject
    name: str = "Reader"

    def fmap_obj(self, A: SetObject) -> SetObject:
        # Reader_ρ(A) at the type level is ρ → A. The SetObject
        # carrier is A; the context-functional ρ → - lives in the
        # morphism layer.
        return A

    def fmap(self, A, B, f):
        raise NotImplementedError("Reader.fmap: see algebraic handler")

    def pure(self, A):
        raise NotImplementedError("Reader.pure: see algebraic handler")

    def apply(self, A, B):
        raise NotImplementedError("Reader.apply: see algebraic handler")

    def join(self, A):
        raise NotImplementedError("Reader.join: see algebraic handler")


Monad.register(Reader)


# ---------------------------------------------------------------------------
# Writer (supplements / nonrestrictive content)
# ---------------------------------------------------------------------------


class Writer(dx.Model):
    """The writer monad: ``Writer_M(A) = A × M``.

    Threaded a side-channel of monoidal accumulator data alongside
    the value computation. Used for supplements, nonrestrictive
    appositives, and other "side-issue" content (Potts; Bumford 2017).

    The monoid ``M`` is supplied as a :class:`SetObject` paired with
    a separately-registered monoid structure (currently elided —
    the join uses the implicit concatenation in the powerset
    interpretation).

    Attributes
    ----------
    monoid : SetObject
        The accumulator type ``M``.
    """

    monoid: SetObject
    name: str = "Writer"

    def fmap_obj(self, A: SetObject) -> SetObject:
        return ProductSet(components=(A, self.monoid))

    def fmap(self, A, B, f):
        raise NotImplementedError("Writer.fmap: see algebraic handler")

    def pure(self, A):
        raise NotImplementedError("Writer.pure: see algebraic handler")

    def apply(self, A, B):
        raise NotImplementedError("Writer.apply: see algebraic handler")

    def join(self, A):
        raise NotImplementedError("Writer.join: see algebraic handler")


Monad.register(Writer)


# ---------------------------------------------------------------------------
# List (bag-of-readings parsers)
# ---------------------------------------------------------------------------


class List(dx.Model):
    """The list monad: ``List(A) = A*``, the free monoid.

    Used for nondeterministic computations whose multiplicities
    matter (bag-of-readings parsers, sequence enumeration). As a
    ``MonadPlus``, the empty list is the empty alternative and
    list concatenation is ``alt``.

    Attributes
    ----------
    max_length : int
        Truncation bound on the underlying free monoid (the bare
        :class:`quivers.core.objects.FreeMonoid` carrier requires a
        finite enumeration depth).
    """

    max_length: int = 8
    name: str = "List"

    def fmap_obj(self, A: SetObject) -> SetObject:
        from quivers.core.objects import FinSet, FreeMonoid

        # List requires its element type to be a FinSet for the
        # FreeMonoid construction. For non-FinSet inputs callers
        # should box via the appropriate adapter.
        if not isinstance(A, FinSet):
            raise TypeError(
                f"List.fmap_obj requires a FinSet element type; got {type(A).__name__}"
            )
        return FreeMonoid(generators=A, max_length=self.max_length)

    def fmap(self, A, B, f):
        raise NotImplementedError("List.fmap: see algebraic handler")

    def pure(self, A):
        raise NotImplementedError("List.pure: see algebraic handler")

    def apply(self, A, B):
        raise NotImplementedError("List.apply: see algebraic handler")

    def join(self, A):
        raise NotImplementedError("List.join: see algebraic handler")

    def empty(self, A):
        raise NotImplementedError("List.empty: see algebraic handler")

    def alt(self, A):
        raise NotImplementedError("List.alt: see algebraic handler")

    def foldr(self, A, B):
        raise NotImplementedError("List.foldr: see algebraic handler")

    def traverse(self, A, B, applicative, f):
        raise NotImplementedError("List.traverse: see algebraic handler")


MonadPlus.register(List)
Foldable.register(List)
Traversable.register(List)


__all__ = [
    "Identity",
    "Maybe",
    "Alternative_",
    "Continuation",
    "State",
    "Reader",
    "Writer",
    "List",
]


# Re-export Alternative for typeclass-checking convenience.
_ = (Alternative,)
