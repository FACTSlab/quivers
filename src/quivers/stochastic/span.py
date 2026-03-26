"""Span-based CKY components for the deductive system framework.

This module instantiates the abstract deduction primitives
(``Axiom``, ``Deduction``, ``Goal``, ``Schedule``) for span-indexed
chart parsing (CKY). Items are indexed by ``(i, j, A)`` where
``[i, j)`` is a string span and ``A`` is a category index.

The chart tensor has shape ``(batch, n_categories, seq_len, seq_len+1)``
(categories-first for efficient indexing of the start symbol).

Components
----------
LexicalAxiom
    Populates length-1 spans from a learnable lexicon.
BinarySpanDeduction
    Applies binary rules: chart[i,k,L] ⊗ chart[k,j,R] → chart[i,j,A].
UnarySpanDeduction
    Applies unary rules iteratively to convergence.
SpanGoal
    Extracts chart[start_cat, 0, seq_len].
CKYSchedule
    Bottom-up evaluation by span length.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from quivers.stochastic.deduction import Axiom, Deduction, Goal, Schedule
from quivers.stochastic.semiring import ChartSemiring
from quivers.stochastic._rule_system import RuleSystem


# ================================================================
# axiom: lexical entries
# ================================================================


class LexicalAxiom(Axiom):
    """Populate length-1 spans from a learnable lexicon.

    Creates the chart tensor and fills
    ``chart[b, :, i, i+1] = log_softmax(lexicon[tokens[b, i]])``.

    Parameters
    ----------
    n_terminals : int
        Vocabulary size.
    n_categories : int
        Number of category types.
    """

    def __init__(self, n_terminals: int, n_categories: int) -> None:
        super().__init__()
        self._n_cat = n_categories
        self.lexicon_logits = nn.Parameter(torch.randn(n_terminals, n_categories) * 0.1)

    @property
    def log_lexicon(self) -> torch.Tensor:
        """Log-probability lexicon, shape (n_terminals, n_categories)."""
        return torch.log_softmax(self.lexicon_logits, dim=-1)

    def forward(self, tokens, semiring):
        batch, seq_len = tokens.shape
        log_lex = self.log_lexicon[tokens]  # (batch, seq_len, C)

        # build chart as list-of-lists (functional, no in-place writes)
        # stored as flat dict keyed by (i, j)
        chart_cells: dict[tuple[int, int], torch.Tensor] = {}

        for i in range(seq_len):
            chart_cells[(i, i + 1)] = log_lex[:, i, :]  # (batch, C)

        # pack into metadata for the schedule
        return _SpanChart(
            cells=chart_cells,
            batch=batch,
            seq_len=seq_len,
            n_categories=self._n_cat,
            device=tokens.device,
            semiring=semiring,
        )


class _SpanChart:
    """Internal chart representation using functional (non-in-place) cells.

    This wraps a dict of ``(i, j) -> (batch, C)`` tensors. The
    schedule fills cells bottom-up, and ``to_tensor()`` assembles
    them into a dense ``(batch, C, seq_len, seq_len+1)`` tensor.
    """

    __slots__ = ("cells", "batch", "seq_len", "n_categories", "device", "semiring")

    def __init__(self, cells, batch, seq_len, n_categories, device, semiring):
        self.cells = cells
        self.batch = batch
        self.seq_len = seq_len
        self.n_categories = n_categories
        self.device = device
        self.semiring = semiring

    def to_tensor(self) -> torch.Tensor:
        """Assemble cells into a dense chart tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_categories, seq_len, seq_len+1)``.
        """
        chart = torch.full(
            (self.batch, self.n_categories, self.seq_len, self.seq_len + 1),
            self.semiring.zero,
            device=self.device,
        )

        for (i, j), cell in self.cells.items():
            chart[:, :, i, j] = cell

        return chart


# ================================================================
# deductions
# ================================================================


def _scatter_semiring(
    scores: torch.Tensor,
    indices: torch.Tensor,
    n_categories: int,
    batch: int,
    device: torch.device,
    semiring: ChartSemiring,
) -> torch.Tensor:
    """Scatter rule scores into category bins via semiring plus."""
    result = torch.full(
        (batch, n_categories),
        semiring.zero,
        device=device,
    )

    for cat_idx in range(n_categories):
        mask = indices == cat_idx

        if mask.any():
            relevant = scores[:, mask]
            aggregated = semiring.plus(relevant, dim=-1)
            result = torch.cat(
                [
                    result[:, :cat_idx],
                    aggregated.unsqueeze(-1),
                    result[:, cat_idx + 1 :],
                ],
                dim=-1,
            )

    return result


class BinarySpanDeduction(Deduction):
    """Binary rule application over span items.

    For each split point k in [i+1, j), combines
    ``chart[i, k, lefts]`` and ``chart[k, j, rights]``
    to produce ``chart[i, j, results]``.

    Parameters
    ----------
    rule_system : RuleSystem
        The structural rules.
    learnable : bool
        Whether rule weights are learnable.
    """

    def __init__(
        self,
        rule_system: RuleSystem,
        learnable: bool = True,
    ) -> None:
        super().__init__()

        results, lefts, rights = rule_system.binary_tensors()
        self.register_buffer("_results", results)
        self.register_buffer("_lefts", lefts)
        self.register_buffer("_rights", rights)
        self._n_rules = rule_system.n_binary

        weights = rule_system.binary_weight_tensor()

        if learnable and self._n_rules > 0:
            self.weights = nn.Parameter(weights)

        else:
            self.register_buffer("weights", weights)

    def forward(self, chart, semiring, **context):
        """Apply binary rules for one span cell (i, j).

        Expected context keys: ``i``, ``j``, ``chart_cells``.
        Returns the (batch, C) tensor for cell (i, j).
        """
        i = context["i"]
        j = context["j"]
        chart_cells = context["chart_cells"]
        n_categories = context["n_categories"]

        if self._n_rules == 0:
            return torch.full(
                (chart_cells[(i, i + 1)].shape[0], n_categories),
                semiring.zero,
                device=self._results.device,
            )

        parts = []

        for k in range(i + 1, j):
            left_cell = chart_cells[(i, k)]
            right_cell = chart_cells[(k, j)]

            left_scores = left_cell[:, self._lefts]
            right_scores = right_cell[:, self._rights]
            combined = semiring.times(left_scores, right_scores)

            if self._n_rules > 0:
                combined = semiring.times(
                    combined,
                    self.weights.unsqueeze(0),
                )

            parts.append(combined)

        stacked = torch.stack(parts, dim=0)
        split_scores = semiring.plus(stacked, dim=0)

        return _scatter_semiring(
            split_scores,
            self._results,
            n_categories,
            split_scores.shape[0],
            device=self._results.device,
            semiring=semiring,
        )


class UnarySpanDeduction(Deduction):
    """Unary rule application, iterated to approximate fixpoint.

    Parameters
    ----------
    rule_system : RuleSystem
        The structural rules.
    learnable : bool
        Whether rule weights are learnable.
    iterations : int
        Number of fixpoint iterations.
    """

    def __init__(
        self,
        rule_system: RuleSystem,
        learnable: bool = True,
        iterations: int = 3,
    ) -> None:
        super().__init__()

        unary = rule_system.unary_tensors()

        if unary is not None:
            self.register_buffer("_results", unary[0])
            self.register_buffer("_inputs", unary[1])

        else:
            self.register_buffer(
                "_results",
                torch.zeros(0, dtype=torch.long),
            )
            self.register_buffer(
                "_inputs",
                torch.zeros(0, dtype=torch.long),
            )

        self._n_rules = rule_system.n_unary
        self._iterations = iterations

        weights = rule_system.unary_weight_tensor()

        if learnable and self._n_rules > 0:
            self.weights = nn.Parameter(weights)

        else:
            self.register_buffer("weights", weights)

    @property
    def has_rules(self) -> bool:
        """Whether this deduction has any unary rules."""
        return self._n_rules > 0

    def forward(self, chart, semiring, **context):
        """Apply unary rules to a cell tensor.

        Expects ``chart`` to be a (batch, C) cell tensor.
        Returns updated (batch, C) tensor.
        """
        cell = chart

        if not self.has_rules:
            return cell

        n_categories = cell.shape[1]

        for _ in range(self._iterations):
            input_scores = cell[:, self._inputs]

            if self._n_rules > 0:
                input_scores = semiring.times(
                    input_scores,
                    self.weights.unsqueeze(0),
                )

            additions = _scatter_semiring(
                input_scores,
                self._results,
                n_categories,
                cell.shape[0],
                device=cell.device,
                semiring=semiring,
            )

            cell = semiring.plus_pair(cell, additions)

        return cell


# ================================================================
# goal
# ================================================================


class SpanGoal(Goal):
    """Extract chart[start_cat, 0, seq_len].

    Parameters
    ----------
    start_idx : int
        Index of the start category.
    """

    def __init__(self, start_idx: int) -> None:
        super().__init__()
        self._start_idx = start_idx

    def forward(self, chart):
        # chart is (batch, C, seq_len, seq_len+1)
        seq_len = chart.shape[2]
        return chart[:, self._start_idx, 0, seq_len]


# ================================================================
# schedule
# ================================================================


class CKYSchedule(Schedule):
    """Bottom-up CKY: process spans in increasing length order.

    For each span length 2..n:
      1. Apply binary deductions for each cell (i, i+span_len).
      2. Apply unary deductions to the resulting cell.
    """

    def run(self, chart, deductions, semiring):
        # chart is a _SpanChart
        span_chart = chart
        cells = span_chart.cells
        seq_len = span_chart.seq_len
        n_categories = span_chart.n_categories

        # separate binary and unary deductions
        binary_deds = [d for d in deductions if isinstance(d, BinarySpanDeduction)]
        unary_deds = [d for d in deductions if isinstance(d, UnarySpanDeduction)]

        # apply unary rules to lexical cells first
        for i in range(seq_len):
            cell = cells[(i, i + 1)]

            for ud in unary_deds:
                if ud.has_rules:
                    cell = ud(cell, semiring)

            cells[(i, i + 1)] = cell

        # fill chart bottom-up
        for span_len in range(2, seq_len + 1):
            for i in range(seq_len - span_len + 1):
                j = i + span_len

                # binary step: accumulate across all binary deductions
                cell = None

                for bd in binary_deds:
                    contrib = bd(
                        None,
                        semiring,
                        i=i,
                        j=j,
                        chart_cells=cells,
                        n_categories=n_categories,
                    )

                    if cell is None:
                        cell = contrib

                    else:
                        cell = semiring.plus_pair(cell, contrib)

                if cell is None:
                    cell = torch.full(
                        (span_chart.batch, n_categories),
                        semiring.zero,
                        device=span_chart.device,
                    )

                # unary step
                for ud in unary_deds:
                    if ud.has_rules:
                        cell = ud(cell, semiring)

                cells[(i, j)] = cell

        # assemble into dense tensor
        return span_chart.to_tensor()
