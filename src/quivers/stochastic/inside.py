"""CKY inside algorithm for probabilistic context-free grammars.

Implements the inside algorithm as a differentiable PyTorch module,
enabling end-to-end gradient-based learning of PCFG parameters.

A PCFG is specified by two stochastic morphisms:

- ``binary : N -> N * N`` — binary production probabilities.
  For each nonterminal A, ``binary[A, B, C]`` is the probability
  of the rule A -> B C.
- ``lexical : N -> T`` — terminal production probabilities.
  For each nonterminal A, ``lexical[A, t]`` is the probability
  of the rule A -> t.

The inside algorithm computes, for each nonterminal A and span
(i, j) of the input sentence:

    beta(A, i, j) = P(w_i ... w_{j-1} | A)

The sentence log-probability is ``log beta(start, 0, L)`` where
``start`` is the start symbol index (default 0).

All computation is done in log-space for numerical stability,
using logsumexp for marginalization. This preserves gradient
flow for learning rule probabilities end-to-end.

Categorical perspective
-----------------------
The inside algorithm implements a morphism

    inside(binary, lexical) : FreeMonoid(T) -> 1

that maps variable-length terminal strings to their probability
under the grammar. This is the counit of the adjunction between
the free monad on the polynomial functor induced by the grammar
and the forgetful functor to strings.

Examples
--------
>>> from quivers.core.objects import FinSet, ProductSet
>>> from quivers.stochastic.morphisms import StochasticMorphism
>>> N = FinSet("N", 5)
>>> T = FinSet("T", 10)
>>> binary = StochasticMorphism(N, ProductSet(N, N))
>>> lexical = StochasticMorphism(N, T)
>>> cky = InsideAlgorithm(binary, lexical, start=0)
>>> tokens = torch.randint(0, 10, (4, 6))  # batch=4, length=6
>>> log_probs = cky(tokens)  # (4,)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quivers.core.morphisms import Morphism
from quivers.core.objects import ProductSet


class InsideAlgorithm(nn.Module):
    """CKY inside algorithm for differentiable PCFG parsing.

    Computes sentence log-probabilities under a PCFG defined by
    binary and lexical production rules, both expressed as
    stochastic morphisms.

    Parameters
    ----------
    binary : Morphism
        Binary production rules. Must be a morphism ``N -> N * N``
        where N is a finite set of nonterminals. The tensor has
        shape ``(|N|, |N|, |N|)`` with ``binary[A, B, C]`` =
        P(A -> B C).
    lexical : Morphism
        Lexical (terminal) production rules. Must be a morphism
        ``N -> T`` where T is a finite set of terminals. The
        tensor has shape ``(|N|, |T|)`` with ``lexical[A, t]`` =
        P(A -> t).
    start : int
        Index of the start symbol in N (default 0).

    Raises
    ------
    TypeError
        If the morphisms have incompatible types.
    """

    def __init__(
        self,
        binary: Morphism,
        lexical: Morphism,
        start: int = 0,
    ) -> None:
        super().__init__()

        # validate types
        if not isinstance(binary.codomain, ProductSet):
            raise TypeError(
                f"binary morphism codomain must be a ProductSet, "
                f"got {binary.codomain!r}"
            )

        if binary.domain != lexical.domain:
            raise TypeError(
                f"binary and lexical must share the same domain "
                f"(nonterminals), got {binary.domain!r} and "
                f"{lexical.domain!r}"
            )

        self._binary = binary
        self._lexical = lexical
        self._start = start
        self._n_nonterm = binary.domain.size
        self._n_term = lexical.codomain.size

        # register the parameter modules
        self._binary_mod = binary.module()
        self._lexical_mod = lexical.module()

    @property
    def n_nonterminals(self) -> int:
        """Number of nonterminal symbols."""
        return self._n_nonterm

    @property
    def n_terminals(self) -> int:
        """Number of terminal symbols."""
        return self._n_term

    @property
    def start(self) -> int:
        """Index of the start symbol."""
        return self._start

    def _fill_chart(self, tokens: torch.Tensor) -> torch.Tensor:
        """Fill the inside chart without in-place tensor modifications.

        Uses a cell-list approach to preserve autograd graph.

        Parameters
        ----------
        tokens : torch.Tensor
            Shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Chart of shape ``(batch, N, seq_len, seq_len+1)``.
        """
        batch, seq_len = tokens.shape

        # get rule probabilities in log-space
        log_binary = torch.log(self._binary.tensor.clamp(min=1e-30))
        log_lexical = torch.log(self._lexical.tensor.clamp(min=1e-30))

        N = self._n_nonterm

        # use cell list to avoid in-place writes
        # cells[i][j] is a (batch, N) tensor
        cells: list[list[torch.Tensor | None]] = [
            [None for _ in range(seq_len + 1)]
            for _ in range(seq_len)
        ]

        # lexical step
        for i in range(seq_len):
            tok_i = tokens[:, i]
            cells[i][i + 1] = log_lexical[:, tok_i].T  # (batch, N)

        # binary step
        for span_len in range(2, seq_len + 1):
            for i in range(seq_len - span_len + 1):
                j = i + span_len
                parts = []

                for k in range(i + 1, j):
                    left = cells[i][k]    # (batch, N)
                    right = cells[k][j]   # (batch, N)

                    # combine: (batch, N_A, N_B, N_C)
                    combined = (
                        log_binary.unsqueeze(0)
                        + left.unsqueeze(1).unsqueeze(3)
                        + right.unsqueeze(1).unsqueeze(2)
                    )

                    # marginalize over B and C
                    split_score = torch.logsumexp(
                        combined.reshape(batch, N, -1), dim=-1,
                    )
                    parts.append(split_score)

                stacked = torch.stack(parts, dim=0)
                cells[i][j] = torch.logsumexp(stacked, dim=0)

        # reassemble into chart tensor
        chart = torch.full(
            (batch, N, seq_len, seq_len + 1),
            float("-inf"),
            device=tokens.device,
        )

        for i in range(seq_len):
            for j in range(i + 1, seq_len + 1):
                if cells[i][j] is not None:
                    chart[:, :, i, j] = cells[i][j]

        return chart

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute sentence log-probabilities via the inside algorithm.

        Parameters
        ----------
        tokens : torch.Tensor
            Integer tensor of terminal indices. Shape
            ``(batch, seq_len)`` or ``(seq_len,)`` for a single
            sentence.

        Returns
        -------
        torch.Tensor
            Log-probability of each sentence under the grammar.
            Shape ``(batch,)`` or scalar for a single sentence.
        """
        squeeze = False

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze = True

        if tokens.shape[1] == 0:
            raise ValueError("cannot parse empty sentences")

        chart = self._fill_chart(tokens)
        result = chart[:, self._start, 0, tokens.shape[1]]

        if squeeze:
            return result.squeeze(0)

        return result

    def inside_chart(self, tokens: torch.Tensor) -> torch.Tensor:
        """Compute the full inside chart (for analysis/debugging).

        Parameters
        ----------
        tokens : torch.Tensor
            Integer tensor of terminal indices. Shape
            ``(batch, seq_len)`` or ``(seq_len,)``.

        Returns
        -------
        torch.Tensor
            The full inside chart in log-space. Shape
            ``(batch, N, seq_len, seq_len+1)`` where entry
            ``[b, A, i, j]`` is ``log P(w_i..w_{j-1} | A)``.
        """
        squeeze = False

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze = True

        if tokens.shape[1] == 0:
            raise ValueError("cannot parse empty sentences")

        chart = self._fill_chart(tokens)

        if squeeze:
            return chart.squeeze(0)

        return chart

    def __repr__(self) -> str:
        return (
            f"InsideAlgorithm("
            f"N={self._n_nonterm}, T={self._n_term}, "
            f"start={self._start})"
        )
