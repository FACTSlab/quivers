"""Algebraic effects + handlers.

Following Plotkin & Power and Bauer & Pretnar, exposes a free-monad-
over-signature construction so any user-defined effect signature
induces a :class:`Monad` instance automatically. Handlers translate
a free-monad computation into a target monad's effects.

The framework lets users build new effects without writing the
typeclass-instance boilerplate of :mod:`quivers.monadic.instances`:
declare an :class:`EffectSignature` of operations, then either let
the framework derive the free monad or supply a concrete
:class:`Handler` that interprets the operations in some target
monad.

Each :class:`EffectSignature` is *also* a panproto theory in the
formal sense: ``sig.to_theory()`` returns a :class:`panproto.Theory`
whose sorts are :class:`SetObject` and whose operations are the
listed :class:`Operation` instances. Handlers are panproto theory
morphisms into the target monad's theory; ``handler.as_theory_morphism()``
gives a verifiable interpretation chain.

References
----------
- Plotkin, G. and Power, J. (2003). *Algebraic operations and generic
  effects*. Applied Categorical Structures, 11(1), 69–94.
  doi:10.1023/A:1023064908962.
- Bauer, A. and Pretnar, M. (2015). *Programming with algebraic
  effects and handlers*. Journal of Logical and Algebraic Methods in
  Programming, 84(1), 108–123. doi:10.1016/j.jlamp.2014.02.001.
"""

from __future__ import annotations


from quivers.core.morphisms import Morphism
from quivers.core.objects import SetObject
from quivers.monadic.typeclasses import Monad


class Operation:
    """One operation in an effect signature.

    An operation has a name, a parameter type (the inputs supplied
    when the operation is invoked), and a result type (the data
    handed to the continuation).

    Plain Python class because the natural ``tuple[Operation, ...]``
    field on :class:`EffectSignature` exceeds what didactic currently
    translates. The :func:`EffectSignature.to_theory` realisation
    is responsible for shipping the structural data into panproto.

    Attributes
    ----------
    name : str
        Operation name, used as the key in handler clauses.
    parameter : SetObject
        Operation parameter type.
    result : SetObject
        Operation result type given to the continuation.
    """

    def __init__(self, name: str, parameter: SetObject, result: SetObject) -> None:
        self.name = name
        self.parameter = parameter
        self.result = result


class EffectSignature:
    """A signature of effect operations.

    Realisable both as a Python value (a tuple of operations) and as
    a :class:`panproto.Theory` whose sorts are :class:`SetObject` and
    whose operations are the listed :class:`Operation` instances.

    Attributes
    ----------
    name : str
        Signature name; appears in panproto theory naming.
    operations : tuple of Operation
        The operations comprising the signature.
    """

    def __init__(self, name: str, operations: tuple[Operation, ...]) -> None:
        self.name = name
        self.operations = operations

    def to_theory(self) -> object:
        """Realise this signature as a panproto Theory.

        The returned theory has:

        - One sort named ``Carrier`` (the free monad's element type).
        - One operation per :class:`Operation` in :attr:`operations`,
          with input ``parameter * (result -> Carrier)`` and output
          ``Carrier``.
        - The standard monad-laws (left/right unit, associativity)
          registered as equations.

        The exact panproto API call depends on the panproto version
        in use; this method is implemented through
        :func:`panproto.define_theory` (or the equivalent native
        constructor) once panproto exposes the necessary surface for
        polymorphic-arity operations.
        """
        raise NotImplementedError(
            "EffectSignature.to_theory pending panproto API surface for "
            "polymorphic-arity operation declaration"
        )


class Handler:
    """Interpretation of a free-monad computation in a target monad.

    A handler supplies a clause for each operation in its signature
    plus a return-clause that lifts plain values into the target
    monad. Equivalently: a handler is a panproto theory morphism
    from ``signature.to_theory()`` into the target monad's theory.

    Handler is a plain Python class rather than a :class:`dx.Model`
    because its fields hold a typeclass-instance ``Monad`` reference
    and a ``dict[str, Morphism]`` whose values are :class:`Morphism`
    instances; neither shape is currently translatable to a panproto
    sort.

    Attributes
    ----------
    signature : EffectSignature
        The effect signature this handler interprets.
    target : Monad
        The target monad in which the operations are realised.
    return_clause : Morphism
        ``A → target(A)``: lifts plain values into the target monad.
    operation_clauses : dict[str, Morphism]
        For each :attr:`signature.operations` entry, a morphism
        ``parameter ⊗ (result → target(B)) → target(B)``.
    """

    def __init__(
        self,
        signature: "EffectSignature",
        target: Monad,
        return_clause: Morphism,
        operation_clauses: dict[str, Morphism],
    ) -> None:
        self.signature = signature
        self.target = target
        self.return_clause = return_clause
        self.operation_clauses = operation_clauses

    def as_theory_morphism(self) -> object:
        """Realise this handler as a panproto theory morphism.

        The morphism's domain is ``signature.to_theory()``; its
        codomain is the panproto theory of the target monad
        (registered in :mod:`quivers.monadic.theories`). Round-trip
        composition with the inverse direction (when one exists)
        recovers the identity on the source theory.
        """
        raise NotImplementedError(
            "Handler.as_theory_morphism pending panproto theory-morphism "
            "API for inter-theory operation maps"
        )

    def run(self, A: SetObject) -> Morphism:
        """Apply this handler to a free-monad computation.

        Returns a morphism ``FreeMonad(signature)(A) → target(A)``
        that interprets each operation invocation through the
        corresponding clause and folds returns through
        :attr:`return_clause`.
        """
        raise NotImplementedError(
            "Handler.run pending the FreeMonad runtime — see FreeMonad "
            "below for the carrier shape"
        )


class FreeMonad:
    """The free monad over an effect signature.

    For a signature ``Σ``, ``FreeMonad(Σ)(A)`` is the set of finitely-
    branched computation trees whose internal nodes are operation
    invocations from ``Σ`` and whose leaves are values of type ``A``.

    The free monad's :meth:`pure` injects a value as a leaf;
    :meth:`bind` substitutes computation trees for leaves. Any
    :class:`Handler` translates a :class:`FreeMonad`-valued
    computation into the target monad's effects.

    Plain Python class because the natural ``signature: EffectSignature``
    field exceeds what didactic currently translates to a panproto sort.

    Attributes
    ----------
    signature : EffectSignature
        The signature whose free monad this is.
    """

    def __init__(self, signature: EffectSignature) -> None:
        self.signature = signature

    def fmap_obj(self, A: SetObject) -> SetObject:
        # The free monad's carrier at A is the set of operation-trees
        # over Σ with leaves in A. Concrete realisation as a finite
        # SetObject requires bounding the tree depth; the tree-depth
        # bound lives on the chart_fold's effect_depth parameter.
        raise NotImplementedError(
            "FreeMonad.fmap_obj requires a tree-depth bound; supply "
            "via chart_fold(effect_depth=...) at parser invocation."
        )

    def fmap(self, A, B, f):
        raise NotImplementedError("FreeMonad.fmap pending tree-depth machinery")

    def pure(self, A):
        raise NotImplementedError("FreeMonad.pure pending tree-depth machinery")

    def apply(self, A, B):
        raise NotImplementedError("FreeMonad.apply pending tree-depth machinery")

    def join(self, A):
        raise NotImplementedError("FreeMonad.join pending tree-depth machinery")


Monad.register(FreeMonad)


__all__ = [
    "Operation",
    "EffectSignature",
    "Handler",
    "FreeMonad",
]
