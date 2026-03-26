"""Semiring abstractions for parameterizing chart parsing.

A semiring (S, ⊕, ⊗, 0, 1) provides the algebraic structure for
chart parsing (Goodman, 1999). Different semirings yield different
parsing algorithms from the same CKY skeleton:

- **LogProbSemiring** — marginal log-probability (logsumexp / +)
- **ViterbiSemiring** — best-derivation log-probability (max / +)
- **BooleanSemiring** — recognition (or / and)
- **CountingSemiring** — derivation counting (+ / ×)

Each semiring defines:

- ``times(a, b)`` — multiplicative combination (⊗)
- ``plus(scores, dim)`` — additive aggregation (⊕) over a dimension
- ``zero`` — additive identity (0): score for impossible derivations
- ``one`` — multiplicative identity (1): score for trivial derivations

Categorical perspective
-----------------------
Semirings are the decategorification of rig categories (bimonoidal
categories). The ``times`` operation corresponds to the tensor
product, ``plus`` to the coproduct. Chart parsing is a functor
from the free monoidal category (grammar) to the semiring.

The connection to quantales: a quantale is a complete lattice with
an associative binary operation distributing over arbitrary joins.
The log-probability semiring is the quantale ([-∞, 0], logsumexp, +).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ChartSemiring(ABC):
    """Abstract semiring for chart parsing.

    Subclasses implement the four semiring operations. All operations
    must be differentiable (where applicable) to support gradient-based
    learning.
    """

    @abstractmethod
    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiplicative combination (⊗).

        Combines scores from two chart cells being joined by a rule.

        Parameters
        ----------
        a : torch.Tensor
            Left operand scores.
        b : torch.Tensor
            Right operand scores.

        Returns
        -------
        torch.Tensor
            Combined scores.
        """

    @abstractmethod
    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        """Additive aggregation (⊕) over a dimension.

        Marginalizes over multiple derivations of the same span.

        Parameters
        ----------
        scores : torch.Tensor
            Scores to aggregate.
        dim : int
            Dimension to aggregate over.

        Returns
        -------
        torch.Tensor
            Aggregated scores.
        """

    @abstractmethod
    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Binary additive combination (⊕) of two tensors.

        Used for combining a cell with unary rule contributions.

        Parameters
        ----------
        a : torch.Tensor
            First operand.
        b : torch.Tensor
            Second operand.

        Returns
        -------
        torch.Tensor
            Combined scores.
        """

    @property
    @abstractmethod
    def zero(self) -> float:
        """Additive identity: the score of an impossible derivation."""

    @property
    @abstractmethod
    def one(self) -> float:
        """Multiplicative identity: the score of a trivial derivation."""


class LogProbSemiring(ChartSemiring):
    """Log-probability semiring: (logsumexp, +, -∞, 0).

    This is the default semiring for marginal parsing. ``times`` is
    addition in log-space (= multiplication of probabilities) and
    ``plus`` is logsumexp (= addition of probabilities in log-space).
    """

    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(scores, dim=dim)

    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.logaddexp(a, b)

    @property
    def zero(self) -> float:
        return float("-inf")

    @property
    def one(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "LogProbSemiring()"


class ViterbiSemiring(ChartSemiring):
    """Viterbi semiring: (max, +, -∞, 0).

    Finds the single best derivation. ``plus`` takes the max over
    derivations rather than summing. Useful for Viterbi parsing
    and best-first decoding.
    """

    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        return scores.max(dim=dim).values

    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.maximum(a, b)

    @property
    def zero(self) -> float:
        return float("-inf")

    @property
    def one(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        return "ViterbiSemiring()"


class BooleanSemiring(ChartSemiring):
    """Boolean semiring: (or, and, 0, 1).

    For recognition only: determines whether a parse exists without
    computing scores. Values are 0.0 (false) and 1.0 (true).
    """

    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # and in {0, 1}: min or product
        return a * b

    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        # or in {0, 1}: clamp of sum or max
        return scores.max(dim=dim).values

    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.maximum(a, b)

    @property
    def zero(self) -> float:
        return 0.0

    @property
    def one(self) -> float:
        return 1.0

    def __repr__(self) -> str:
        return "BooleanSemiring()"


class CountingSemiring(ChartSemiring):
    """Counting semiring: (+, ×, 0, 1).

    Counts the number of derivations. ``plus`` sums counts and
    ``times`` multiplies them (product rule for independent choices).

    Note: Not differentiable in a useful sense for gradient-based
    learning, but useful for analysis.
    """

    def times(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def plus(self, scores: torch.Tensor, dim: int) -> torch.Tensor:
        return scores.sum(dim=dim)

    def plus_pair(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    @property
    def zero(self) -> float:
        return 0.0

    @property
    def one(self) -> float:
        return 1.0

    def __repr__(self) -> str:
        return "CountingSemiring()"


# default semiring instance
LOG_PROB = LogProbSemiring()
VITERBI = ViterbiSemiring()
BOOLEAN = BooleanSemiring()
COUNTING = CountingSemiring()
