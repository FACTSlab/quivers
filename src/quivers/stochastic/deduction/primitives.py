"""Abstract weighted deductive system framework.

A weighted deductive system (Shieber, Schabes & Pereira 1995;
Goodman 1999; Nederhof 2003) provides a categorical abstraction
for parsing algorithms. The core structure is a V-enriched
hypergraph where:

- Items are nodes, indexed into a chart tensor.
- Deduction rules are weighted hyperedges.
- The semiring V determines the scoring algebra.
- A schedule determines the evaluation order.

Different parsing algorithms (CKY, Earley, agenda-based) are
different schedules over the same deductive system. Different
semirings (log-prob, Viterbi, boolean, counting) yield different
computations from the same rules.

The five abstract primitives are:

- ``Axiom`` — creates and populates the initial chart.
- ``Deduction`` — a single weighted inference step (chart -> chart).
- ``Goal`` — extracts the result from a completed chart.
- ``Schedule`` — evaluation strategy (ordering of deductions).
- ``DeductiveSystem`` — ties axiom, deductions, goal, schedule,
  and semiring into a single nn.Module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from quivers.stochastic.semiring import ChartSemiring, LOG_PROB


class Axiom(nn.Module):
    """Initial items in a weighted deductive system.

    An axiom creates the chart tensor and populates it with
    initial weights derived from the input. For span-based
    chart parsing, this is the lexical step.
    """

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        semiring: ChartSemiring,
    ) -> torch.Tensor:
        """Create and populate the initial chart.

        Parameters
        ----------
        input : torch.Tensor
            Raw input (e.g. token indices).
        semiring : ChartSemiring
            The scoring semiring.

        Returns
        -------
        torch.Tensor
            The chart tensor with initial items filled in.
        """
        ...


class Deduction(nn.Module):
    """A weighted inference step in a deductive system.

    A deduction reads from the chart, applies inference rules,
    and writes updated weights. This is a morphism in the
    V-enriched category of chart states.
    """

    @abstractmethod
    def forward(
        self,
        chart: torch.Tensor,
        semiring: ChartSemiring,
        **context,
    ) -> torch.Tensor:
        """Apply this deduction step.

        Parameters
        ----------
        chart : torch.Tensor
            Current chart state.
        semiring : ChartSemiring
            The scoring semiring.
        **context
            Schedule-provided context (e.g. ``span_length``,
            ``span_start``, ``split`` for CKY).

        Returns
        -------
        torch.Tensor
            Updated chart.
        """
        ...


class Goal(nn.Module):
    """Extract the result from a completed chart.

    A goal identifies which chart items constitute the answer
    and extracts their weights.
    """

    @abstractmethod
    def forward(self, chart: torch.Tensor) -> torch.Tensor:
        """Extract goal items from the chart.

        Parameters
        ----------
        chart : torch.Tensor
            Completed chart.

        Returns
        -------
        torch.Tensor
            Goal item weights.
        """
        ...


class Schedule(ABC):
    """Evaluation strategy for a deductive system.

    Different schedules compute the same fixpoint in different
    orders. CKY processes spans bottom-up; an agenda schedule
    uses a priority queue. The schedule is independent of the
    deduction rules.
    """

    @abstractmethod
    def run(
        self,
        chart: torch.Tensor,
        deductions: nn.ModuleList,
        semiring: ChartSemiring,
    ) -> torch.Tensor:
        """Execute deductions on chart to fixpoint.

        Parameters
        ----------
        chart : torch.Tensor
            Chart with axioms filled in.
        deductions : nn.ModuleList
            The deduction steps.
        semiring : ChartSemiring
            The scoring semiring.

        Returns
        -------
        torch.Tensor
            Completed chart.
        """
        ...


class DeductiveSystem(nn.Module):
    """A weighted deductive system evaluated to fixpoint.

    This is the abstract core of all parsing algorithms::

        input -> axiom -> schedule(deductions) -> goal -> output

    The system is an nn.Module whose learnable parameters come
    from the axiom (e.g. lexical weights) and deductions (e.g.
    rule weights).

    Parameters
    ----------
    axiom : Axiom
        Initial item population.
    deductions : list[Deduction]
        Inference rules.
    goal : Goal
        Result extraction.
    schedule : Schedule
        Evaluation strategy.
    semiring : ChartSemiring, optional
        Scoring algebra (default: LOG_PROB).
    """

    def __init__(
        self,
        axiom: Axiom,
        deductions: list[Deduction],
        goal: Goal,
        schedule: Schedule,
        semiring: ChartSemiring | None = None,
    ) -> None:
        super().__init__()
        self.axiom = axiom
        self.deductions = nn.ModuleList(deductions)
        self.goal = goal
        self._schedule = schedule
        self._semiring = semiring or LOG_PROB

    @property
    def semiring(self) -> ChartSemiring:
        """The scoring algebra."""
        return self._semiring

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run the deductive system on input.

        Parameters
        ----------
        input : torch.Tensor
            Raw input (e.g. token indices).

        Returns
        -------
        torch.Tensor
            Goal item weights.
        """
        chart = self.axiom(input, self._semiring)
        chart = self._schedule.run(chart, self.deductions, self._semiring)
        return self.goal(chart)
