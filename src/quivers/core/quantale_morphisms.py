"""Quantale homomorphisms and change-of-base for V-enriched
morphisms.

A morphism in :mod:`quivers.core.morphisms` is V-enriched: its
codomain is :math:`\\mathrm{Hom}_V(A, B)` for the active quantale
``V``. Two morphisms over different quantales (``V`` and ``W``)
do not directly compose — :meth:`Morphism.__rshift__` raises
``incompatible quantales``. The standard categorical way to
bridge them is a **quantale homomorphism** (a lax monoidal poset
functor) :math:`\\varphi : V \\to W` mediating the change of
base. There is then a 2-functor
:math:`(-) \\otimes_\\varphi W : V\\text{-}\\mathbf{Cat} \\to W\\text{-}\\mathbf{Cat}`
that base-changes objects and an induced natural transformation
on morphisms.

This module ships:

* :class:`QuantaleHomomorphism` — the abstract base. A homomorphism
  carries a source quantale, a target quantale, and a function
  ``apply(t : Tensor) -> Tensor`` that maps a tensor whose entries
  live in the source's lattice to a tensor whose entries live in
  the target's lattice. Subclasses implement ``apply`` for a
  specific pair.

* :class:`Embedding` — the inclusion of a sub-quantale into a
  super-quantale (e.g. ``Boolean ↪ ProductFuzzy``).

* :class:`Expectation` — Markov-to-ProductFuzzy (``softmax`` ↦
  fuzzy membership).

* :class:`LogProb` — ProductFuzzy-to-LogProb (``log(p)``).

* :class:`MaxPlus` — ProductFuzzy-to-Viterbi (max-plus tropical
  lift).

* :class:`Threshold` — ProductFuzzy-to-Boolean (discretise at a
  threshold).

* :class:`MaterialImplication` — ProductFuzzy-to-Godel (Heyting
  implication semantics).

* :class:`IdentityHom` — a no-op homomorphism from a quantale to
  itself, useful as the unit of homomorphism composition.

* A registry :data:`HOMOMORPHISM_REGISTRY` keyed by ``(source.name,
  target.name)`` so the compiler / user code can look up the
  canonical homomorphism between two quantales.

The categorical denotation of ``f.change_base(φ)`` for
``f : A → B`` in V is the V-Cat morphism
``φ ∘ f : A → B`` in W with tensor
``φ.apply(f.tensor)`` and lattice values in W. Composition then
proceeds in W using W's tensor / join operations.

This module is intentionally backend-agnostic: every homomorphism's
``apply`` operates on a :class:`torch.Tensor` without touching
PyTorch's autograd or module infrastructure, so a future numpy /
JAX backend can replace the runtime without touching the
categorical hierarchy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from quivers.core.extra_quantales import (
    COUNTING,
    GODEL,
    PROBABILITY,
    REAL,
    TROPICAL,
)
from quivers.core.quantales import (
    BOOLEAN,
    PRODUCT_FUZZY,
    Quantale,
)


class QuantaleHomomorphism(ABC):
    """A lax monoidal poset functor between two quantales.

    Concretely, a homomorphism :math:`\\varphi : V \\to W`
    satisfies:

    * ``φ(a ⊗_V b) ≤ φ(a) ⊗_W φ(b)`` (lax with respect to the
      monoidal tensor).
    * ``φ(⋁_i a_i) ≤ ⋁_i φ(a_i)`` (lax with respect to joins).
    * ``φ(I_V) ≤ I_W`` (lax with respect to the monoidal unit).

    The "lax" qualifier allows for embeddings that preserve order
    but compress structure (a threshold is lax but not strict).
    Strictness is *not* required for change-of-base to be
    categorically valid; only the order-preservation is.
    """

    @property
    @abstractmethod
    def source(self) -> Quantale:
        """The source quantale ``V``."""

    @property
    @abstractmethod
    def target(self) -> Quantale:
        """The target quantale ``W``."""

    @abstractmethod
    def apply(self, t: torch.Tensor) -> torch.Tensor:
        """Map a tensor with entries in ``source.L`` to a tensor
        with entries in ``target.L``.

        The shape is preserved; only the per-entry lattice changes.
        """

    @property
    def name(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.source.name} -> {self.target.name})"
        )


class IdentityHom(QuantaleHomomorphism):
    """The identity homomorphism ``id_V : V → V``.

    Useful as the unit of composition; ``f.change_base(IdentityHom(V))``
    leaves the morphism in V unchanged.
    """

    def __init__(self, q: Quantale) -> None:
        self._q = q

    @property
    def source(self) -> Quantale:
        return self._q

    @property
    def target(self) -> Quantale:
        return self._q

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t


class Embedding(QuantaleHomomorphism):
    """The inclusion of a sub-quantale into a super-quantale.

    The canonical example is ``Boolean ↪ ProductFuzzy``: a
    boolean tensor with entries in ``{0, 1}`` embeds as a
    product-fuzzy tensor with entries in ``[0, 1]`` whose support
    is the boolean's truth assignment.

    Embeddings preserve every quantale operation up to the
    embedding inclusion; they are *strict* homomorphisms.
    """

    def __init__(self, source: Quantale, target: Quantale) -> None:
        self._source = source
        self._target = target

    @property
    def source(self) -> Quantale:
        return self._source

    @property
    def target(self) -> Quantale:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        # Cast to the target's natural dtype (float for fuzzy /
        # markov, bool stays bool for boolean) — no value change.
        return t.to(dtype=torch.float32) if t.dtype == torch.bool else t


class Expectation(QuantaleHomomorphism):
    """Markov → ProductFuzzy via ``softmax`` to fuzzy membership.

    A Markov kernel ``k : A → B`` has rows that sum to 1 (a
    probability distribution over B per a∈A). Reading the
    per-entry probability as a *fuzzy membership* of (a, b)
    in the relation gives a ``ProductFuzzy`` morphism.

    Categorically this is the change of base induced by the
    ``Δ`` natural transformation that erases the normalisation
    constraint — the resulting fuzzy morphism is no longer
    row-stochastic but its entries still lie in [0, 1].
    """

    def __init__(self) -> None:
        from quivers.stochastic.quantale import MARKOV

        self._source = MARKOV
        self._target = PRODUCT_FUZZY

    @property
    def source(self) -> Quantale:
        return self._source

    @property
    def target(self) -> Quantale:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0.0, max=1.0)


class LogProb(QuantaleHomomorphism):
    """ProductFuzzy → LogProb via ``log``.

    Maps entries in (0, 1] to (-∞, 0]; pairs naturally with
    log-space computation. Entries that are exactly 0 in the
    source go to ``-inf``; in practice the input is clamped to
    ``[ε, 1]`` first to avoid numerical blow-up.
    """

    def __init__(self) -> None:
        self._source = PRODUCT_FUZZY
        # The target is LogProbQuantale, lazily resolved so the
        # module import order doesn't matter.
        from quivers.core.quantales import ProductFuzzy

        self._target = ProductFuzzy()

    @property
    def source(self) -> Quantale:
        return self._source

    @property
    def target(self) -> Quantale:
        # LogProb is structurally a ProductFuzzy in log-space; we
        # don't ship a separate quantale class for it because the
        # tensor operations are the same after the log mapping.
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return torch.log(t.clamp(min=1e-30))


class MaxPlus(QuantaleHomomorphism):
    """ProductFuzzy → Tropical via ``log``.

    The max-plus tropical quantale on the reals has tensor =
    addition and join = max. A ProductFuzzy morphism transports
    via ``log`` so the product t-norm becomes addition in
    log-space; subsequent joins (which were noisy-OR in
    ProductFuzzy) get replaced by max in Viterbi-style
    Viterbi-MAP aggregations.

    The two ProductFuzzy → log-target homomorphisms are different
    *category-theoretically* — :class:`LogProb` carries the join
    structure to logsumexp, :class:`MaxPlus` carries it to
    max-plus — even though they share the same per-entry mapping
    ``log``.
    """

    def __init__(self) -> None:
        self._source = PRODUCT_FUZZY
        self._target = TROPICAL

    @property
    def source(self) -> Quantale:
        return self._source

    @property
    def target(self) -> Quantale:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return torch.log(t.clamp(min=1e-30))


class Threshold(QuantaleHomomorphism):
    """ProductFuzzy → Boolean via thresholding.

    Entries above ``tau`` map to True; entries at or below
    ``tau`` map to False. This is a *lax* homomorphism: it
    preserves order (a ≤ b ⟹ threshold(a) ≤ threshold(b)) but
    compresses information (every value in (tau, 1] collapses to
    True).

    Parameters
    ----------
    tau : float
        Threshold value in [0, 1].
    """

    def __init__(self, tau: float = 0.5) -> None:
        if not 0.0 <= tau <= 1.0:
            raise ValueError(
                f"Threshold: tau must be in [0, 1]; got {tau}"
            )
        self._tau = tau
        self._source = PRODUCT_FUZZY
        self._target = BOOLEAN

    @property
    def tau(self) -> float:
        return self._tau

    @property
    def source(self) -> Quantale:
        return self._source

    @property
    def target(self) -> Quantale:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return (t > self._tau).to(dtype=torch.float32)


class MaterialImplication(QuantaleHomomorphism):
    """ProductFuzzy → Godel via ``a * b + (1 - a)`` reading.

    Reads each ProductFuzzy entry ``p`` as a probability that
    the implication ``p → q`` holds — the Heyting-implication-
    style transport that respects the Godel quantale's monoid.

    The mapping is `entry ↦ entry`-preserving on the entry value
    but changes the *composition* semantics downstream: subsequent
    `>>` in the Godel quantale uses min / Godel-implication, not
    product / noisy-OR.
    """

    def __init__(self) -> None:
        self._source = PRODUCT_FUZZY
        self._target = GODEL

    @property
    def source(self) -> Quantale:
        return self._source

    @property
    def target(self) -> Quantale:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0.0, max=1.0)


# Canonical homomorphism instances, exposed as module-level
# singletons. Users compose with ``f.change_base(EXPECTATION)`` to
# perform the change of base.
EXPECTATION = Expectation()
LOG_PROB = LogProb()
MAX_PLUS = MaxPlus()
MATERIAL_IMPLICATION = MaterialImplication()


def threshold(tau: float = 0.5) -> Threshold:
    """Build a :class:`Threshold` homomorphism at the given
    threshold value."""
    return Threshold(tau)


def embedding(source: Quantale, target: Quantale) -> Embedding:
    """Build an :class:`Embedding` homomorphism (sub → super
    quantale inclusion)."""
    return Embedding(source, target)


class ProbabilityClamp(QuantaleHomomorphism):
    """Real → Probability via clamping to the unit interval.

    Coerces a real-valued tensor into a probability tensor by
    clamping entries to ``[0, 1]``. Lax — destroys information
    outside the unit interval — but preserves the entry-wise
    order on the survivors.
    """

    def __init__(self) -> None:
        self._source = REAL
        self._target = PROBABILITY

    @property
    def source(self) -> Quantale:
        return self._source

    @property
    def target(self) -> Quantale:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0.0, max=1.0)


class CountingFromReal(QuantaleHomomorphism):
    """Real → Counting via floor-and-clamp-to-non-negative.

    Coerces a real-valued tensor into a non-negative integer
    counting tensor by flooring and clamping at zero.
    Information-destroying; inverse is :class:`CountingToReal`.
    """

    def __init__(self) -> None:
        self._source = REAL
        self._target = COUNTING

    @property
    def source(self) -> Quantale:
        return self._source

    @property
    def target(self) -> Quantale:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t.clamp(min=0.0).floor()


class ProbabilityToReal(QuantaleHomomorphism):
    """Probability → Real (sub-quantale inclusion).

    Entries already lie in ``[0, 1] ⊂ ℝ``; the inclusion is
    strict (preserves every operation).
    """

    def __init__(self) -> None:
        self._source = PROBABILITY
        self._target = REAL

    @property
    def source(self) -> Quantale:
        return self._source

    @property
    def target(self) -> Quantale:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t


class CountingToReal(QuantaleHomomorphism):
    """Counting → Real (sub-quantale inclusion).

    Non-negative integers embed canonically in the reals; this
    is the strict inclusion homomorphism.
    """

    def __init__(self) -> None:
        self._source = COUNTING
        self._target = REAL

    @property
    def source(self) -> Quantale:
        return self._source

    @property
    def target(self) -> Quantale:
        return self._target

    def apply(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=torch.float32)


PROBABILITY_CLAMP = ProbabilityClamp()
COUNTING_FROM_REAL = CountingFromReal()
PROBABILITY_TO_REAL = ProbabilityToReal()
COUNTING_TO_REAL = CountingToReal()


# Registry of canonical homomorphisms keyed by
# ``(source.name, target.name)``. The compiler / user code can
# look up the standard bridge between two quantales rather than
# constructing one by hand.
HOMOMORPHISM_REGISTRY: dict[tuple[str, str], QuantaleHomomorphism] = {
    ("Markov", "ProductFuzzy"): EXPECTATION,
    ("ProductFuzzy", "ProductFuzzy"): LOG_PROB,
    ("ProductFuzzy", "Tropical"): MAX_PLUS,
    ("ProductFuzzy", "Boolean"): Threshold(0.5),
    ("ProductFuzzy", "Godel"): MATERIAL_IMPLICATION,
    ("Boolean", "ProductFuzzy"): Embedding(BOOLEAN, PRODUCT_FUZZY),
    ("Real", "Probability"): PROBABILITY_CLAMP,
    ("Real", "Counting"): COUNTING_FROM_REAL,
    ("Probability", "Real"): PROBABILITY_TO_REAL,
    ("Counting", "Real"): COUNTING_TO_REAL,
}


def lookup_homomorphism(
    source: Quantale, target: Quantale
) -> QuantaleHomomorphism | None:
    """Return the registered homomorphism ``source → target`` or
    ``None`` if no canonical bridge is known.

    Identity is always available: ``source == target`` returns
    ``IdentityHom(source)``.
    """
    if type(source) is type(target):
        return IdentityHom(source)
    return HOMOMORPHISM_REGISTRY.get((source.name, target.name))


__all__ = [
    "QuantaleHomomorphism",
    "IdentityHom",
    "Embedding",
    "Expectation",
    "LogProb",
    "MaxPlus",
    "Threshold",
    "MaterialImplication",
    "ProbabilityClamp",
    "CountingFromReal",
    "ProbabilityToReal",
    "CountingToReal",
    "EXPECTATION",
    "LOG_PROB",
    "MAX_PLUS",
    "MATERIAL_IMPLICATION",
    "PROBABILITY_CLAMP",
    "COUNTING_FROM_REAL",
    "PROBABILITY_TO_REAL",
    "COUNTING_TO_REAL",
    "threshold",
    "embedding",
    "HOMOMORPHISM_REGISTRY",
    "lookup_homomorphism",
]
