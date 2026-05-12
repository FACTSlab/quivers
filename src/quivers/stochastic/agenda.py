"""Agenda-based weighted-deduction engine.

A general framework for evaluating weighted deductive systems by
agenda-driven semi-naïve enumeration. Subsumes CKY, Earley,
Viterbi, A* parsing, Knuth's algorithm, semi-naïve Datalog
evaluation, and bidirectional MLTT type-checking under one
runtime, parameterised by an :class:`Item` algebra, a list of
arity-n :class:`InferenceRule` hyperedges, a :class:`ChartSemiring`,
an :class:`Agenda` data structure, a priority function, and a
goal predicate.

Categorical denotation: the chart is the least pre-fixed point of
the rule-system functor

.. math::

    F : \\mathbf{Set}^{I^{\\mathrm{op}}}_{K} \\to \\mathbf{Set}^{I^{\\mathrm{op}}}_{K}

in the :math:`K`-enriched lattice of charts (Tarski-Knaster). The
chart is the :math:`K`-presheaf :math:`I^{\\mathrm{op}} \\to K`
assigning the aggregate inside weight to each item. The agenda is
an operational realisation of the fixed-point computation; the
strategy-independence theorem (Goodman 1999 §3) guarantees that
the chart's final value is independent of the agenda discipline
when the semiring is idempotent.

References
----------
- Shieber, Schabes & Pereira (1995) "Principles and Implementation
  of Deductive Parsing." Journal of Logic Programming 24(1-2):3-36.
  doi:10.1016/0743-1066(95)00035-I
- Pereira & Warren (1983) "Parsing as Deduction." Proceedings of
  the 21st Annual Meeting of the ACL, pp. 137-144.
  doi:10.3115/981311.981338
- Knuth (1977) "A Generalization of Dijkstra's Algorithm."
  Information Processing Letters 6(1):1-5.
  doi:10.1016/0020-0190(77)90002-3
- Goodman (1999) "Semiring Parsing." Computational Linguistics
  25(4):573-605. https://aclanthology.org/J99-4004/
- Klein & Manning (2001) "Parsing and Hypergraphs." Proceedings
  of IWPT, pp. 123-134. doi:10.1007/1-4020-2295-6_18
- McAllester (2002) "On the Complexity Analysis of Static
  Analyses." Journal of the ACM 49(4):512-537.
  doi:10.1145/581771.581774
- Nederhof (2003) "Weighted Deductive Parsing and Knuth's
  Algorithm." Computational Linguistics 29(1):135-143.
  doi:10.1162/089120103321337467
- Eisner, Goldlust & Smith (2005) "Compiling Comp Ling: Practical
  Weighted Dynamic Programming and the Dyna Language."
  Proceedings of HLT-EMNLP, pp. 281-290.
  https://aclanthology.org/H05-1036/
- Eisner & Blatz (2007) "Program Transformations for Optimization
  of Parsing Algorithms and other Weighted Logic Programs."
  Proceedings of the 11th Conference on Formal Grammar.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from quivers.structural.encoder import Encoder

from quivers.stochastic.semiring import ChartSemiring, LOG_PROB


# ---------------------------------------------------------------------------
# Items, patterns, and rule structure
# ---------------------------------------------------------------------------


Item = tuple[Any, ...]
"""An item is a tuple of (constructor_name, *arguments).

The constructor name disambiguates item shapes; the remaining
slots are the item's payload. Arguments may be other items
(structural), strings (atoms), integers (positions / cardinals),
or :class:`Wildcard` placeholders in patterns.
"""


@dataclass(frozen=True)
class Wildcard:
    """Pattern-position wildcard.

    A wildcard matches any value in the corresponding slot and
    binds it to ``name`` for use in the conclusion. Two wildcards
    with the same ``name`` in a single rule's premise list must
    bind to equal values (variable-sharing).
    """

    name: str

    def __repr__(self) -> str:
        return f"_{self.name}"


WILDCARD_SENTINEL = "__WC__"


def make_wildcard(name: str) -> Wildcard:
    """Build a fresh wildcard with the given variable name."""
    return Wildcard(name)


Pattern = tuple[Any, ...]
"""A pattern is an item-shaped tuple that may contain :class:`Wildcard`
positions. A pattern matches an item if (a) the constructor name
matches, (b) the arities match, (c) each non-wildcard slot is
structurally equal to the corresponding item slot, and (d) all
wildcards with the same name bind to equal values."""


Bindings = dict[str, Any]
"""A bindings environment maps wildcard variable names to the
values they captured during pattern-matching."""


@dataclass(frozen=True)
class InferenceRule:
    """A weighted inference rule — an arity-n hyperedge.

    Parameters
    ----------
    name : str
        Rule identifier (used for diagnostics and provenance).
    premises : tuple of Pattern
        Patterns the rule's antecedents must match.
    conclusion : Pattern
        Pattern of the item the rule produces. Wildcards must
        appear in the premises (so they are bound by the time
        the conclusion is constructed).
    weight_fn : Callable, optional
        Function ``(bindings, premise_weights, semiring) -> Weight``
        producing the conclusion's weight from the matched
        premises. Defaults to the semiring product of premise
        weights (the standard semiring-parsing aggregation).
    side_condition : Callable, optional
        Predicate ``bindings -> bool``. The rule fires only when
        this returns ``True``. Used for guards like adjacency
        ``i < k < j`` in CKY-style rules.
    """

    name: str
    premises: tuple[Pattern, ...]
    conclusion: Pattern
    weight_fn: Callable[[Bindings, tuple, ChartSemiring], Any] | None = None
    side_condition: Callable[[Bindings], bool] | None = None


def instantiate(pattern: Pattern, bindings: Bindings) -> Item:
    """Substitute wildcards in a pattern with their bound values.

    Returns a concrete item with no wildcards. Raises
    :class:`KeyError` if a wildcard in the pattern has no binding.

    The pattern may be a bare :class:`Wildcard` (treated as "the
    entire item is the wildcard"), a structural tuple, or a leaf
    value (returned unchanged). Recursion runs over tuple
    children.
    """
    if isinstance(pattern, Wildcard):
        return bindings[pattern.name]
    if not isinstance(pattern, tuple):
        return pattern
    out: list[Any] = []
    for p in pattern:
        if isinstance(p, Wildcard):
            out.append(bindings[p.name])
        elif isinstance(p, tuple):
            out.append(instantiate(p, bindings))
        else:
            out.append(p)
    return tuple(out)


def match(
    pattern: Pattern, item: Item, bindings: Bindings | None = None
) -> Bindings | None:
    """Match an item against a pattern."""
    if bindings is None:
        bindings = {}
    if isinstance(pattern, Wildcard):
        existing = bindings.get(pattern.name)
        if existing is None:
            return {**bindings, pattern.name: item}
        if existing == item:
            return bindings
        return None
    if not isinstance(pattern, tuple) or not isinstance(item, tuple):
        if pattern == item:
            return bindings
        return None
    if len(pattern) != len(item):
        return None
    out = dict(bindings)
    for p, v in zip(pattern, item):
        if isinstance(p, Wildcard):
            existing = out.get(p.name)
            if existing is None:
                out[p.name] = v
            elif existing != v:
                return None
        elif isinstance(p, tuple) and isinstance(v, tuple):
            sub = match(p, v, out)
            if sub is None:
                return None
            out = sub
        elif p != v:
            return None
    return out


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


class Chart(ABC):
    """A K-valued presheaf on the item algebra.

    Categorically, a chart is the K-presheaf
    :math:`C : I^{\\mathrm{op}} \\to K` produced by the deduction
    system on an input. The chart is the least pre-fixed point of
    the rule-system functor; operationally, the agenda engine
    populates it bottom-up.

    Concrete subclasses choose a storage strategy: dense
    tensor-indexed (for CKY-style spans), sparse hash-indexed
    (for Datalog atoms), or term-structural (for MLTT
    judgments).
    """

    @abstractmethod
    def lookup(self, pattern: Pattern) -> Iterable[tuple[Item, torch.Tensor]]:
        """Enumerate (item, weight) pairs matching a pattern."""
        ...

    @abstractmethod
    def insert_or_aggregate(
        self,
        item: Item,
        weight: torch.Tensor,
        semiring: ChartSemiring,
    ) -> bool:
        """Insert or aggregate an item's weight.

        Returns ``True`` if the item's stored weight changed
        (so it should be re-enqueued for downstream firings);
        ``False`` if the weight was unchanged (the inference was
        redundant under the semiring's idempotent join).
        """
        ...

    @abstractmethod
    def get(self, item: Item) -> torch.Tensor | None:
        """Return the chart's weight at an item, or ``None`` if absent."""
        ...

    @abstractmethod
    def items(self) -> Iterable[tuple[Item, torch.Tensor]]:
        """Enumerate all (item, weight) pairs in the chart."""
        ...


class HashChart(Chart):
    """Dictionary-backed chart for arbitrary item algebras.

    Stores items as keys in a Python dict; supports linear-scan
    pattern matching. Suitable for Datalog atoms, MLTT
    judgments, and any deduction whose items don't have a dense
    integer encoding. Falls back to O(|chart| · |pattern|) match
    cost per lookup; faster specialisations should override
    :meth:`lookup` with an indexed structure.

    Weights are stored as :class:`torch.Tensor` values; autograd
    flows through ``insert_or_aggregate``'s
    ``semiring.plus(torch.stack([…]))`` reduction, so the chart's
    final values carry gradients with respect to any
    ``requires_grad=True`` rule-weight parameters that fed into
    the agenda.
    """

    def __init__(self) -> None:
        self._store: dict[Item, torch.Tensor] = {}

    def lookup(self, pattern: Pattern) -> Iterable[tuple[Item, torch.Tensor]]:
        for item, w in self._store.items():
            b = match(pattern, item)
            if b is not None:
                yield item, w

    def insert_or_aggregate(
        self,
        item: Item,
        weight: torch.Tensor,
        semiring: ChartSemiring,
    ) -> bool:
        if item not in self._store:
            self._store[item] = weight
            return True
        existing = self._store[item]
        # Aggregate via the semiring's plus (logsumexp for log-prob,
        # max for Viterbi, OR for boolean, +/- for inside).
        # We stack and reduce along the new leading dim.
        stacked = torch.stack([existing, weight])
        merged = semiring.plus(stacked, dim=0)
        changed = not torch.equal(merged, existing)
        self._store[item] = merged
        return changed

    def get(self, item: Item) -> torch.Tensor | None:
        return self._store.get(item)

    def items(self) -> Iterable[tuple[Item, torch.Tensor]]:
        return self._store.items()


# ---------------------------------------------------------------------------
# Chart as first-class differentiable value
# ---------------------------------------------------------------------------


class ChartView:
    """A first-class, differentiable view onto a chart.

    Wraps a :class:`Chart` produced by an agenda run and exposes
    the user-facing presheaf-evaluation operations:

    * :meth:`weight` — query the aggregated weight at a single
      ground item; returns a differentiable :class:`torch.Tensor`.
    * :meth:`enumerate` — enumerate items matching a pattern with
      their weights.
    * :meth:`derivations` — extract the derivation forest at an
      item (under the derivation semiring; for the basic
      Boolean / LOG_PROB / Viterbi semirings, returns the
      flat list of matched derivations as a placeholder).
    * :meth:`goal_weight` — return the weight at the goal items
      identified at run time.

    Categorically, the chart is the K-presheaf
    :math:`I^{\\mathrm{op}} \\to K` produced by the deduction
    system; ``ChartView``'s methods are presheaf evaluations.
    Gradients flow through these operations because the
    underlying tensors carry ``requires_grad`` from the rule
    weights.
    """

    def __init__(self, result: "AgendaResult") -> None:
        self._result = result

    @property
    def chart(self) -> Chart:
        """The underlying K-presheaf."""
        return self._result.chart

    @property
    def attached_loss(self) -> torch.Tensor | None:
        """The sum of rule-attached and chart-attached loss values
        fired during this deduction's run, or ``None`` if no losses
        were declared at those sites."""
        return self._result.attached_loss

    @property
    def semiring(self) -> ChartSemiring:
        """The semiring the chart was computed over."""
        return self._result.semiring

    def weight(self, item: Item) -> torch.Tensor:
        """Return the aggregated weight at a concrete ``item``.

        Raises :class:`KeyError` if the item was never derived.
        Returns a differentiable :class:`torch.Tensor`.
        """
        w = self._result.chart.get(item)
        if w is None:
            raise KeyError(f"item {item!r} not in chart")
        return w

    def try_weight(
        self,
        item: Item,
        default: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the weight at ``item`` or ``default`` (or the
        semiring's zero) if the item is absent."""
        w = self._result.chart.get(item)
        if w is not None:
            return w
        if default is not None:
            return default
        return torch.tensor(
            float(self.semiring.zero)
            if hasattr(self.semiring.zero, "__float__")
            else 0.0,
            dtype=torch.get_default_dtype(),
        )

    def enumerate(self, pattern: Pattern) -> list[tuple[Item, torch.Tensor]]:
        """Enumerate (item, weight) pairs matching ``pattern``.

        Wildcards in ``pattern`` (instances of :class:`Wildcard`)
        match any slot value. The returned weights are
        differentiable.
        """
        return list(self._result.chart.lookup(pattern))

    def embedding(self, item: Item) -> torch.Tensor:
        """Return the vector embedding of ``item`` under the deduction's
        attached encoder.

        Requires the originating :class:`DeductionSystem` to carry an
        ``_item_encoder`` attribute (set by the DSL compiler when
        the deduction's body declares ``encoder C``). Items are
        converted to :class:`quivers.structural.Term` form on the fly.
        """
        encoder = getattr(self._result, "encoder", None)
        if encoder is None:
            raise RuntimeError(
                "chart has no attached encoder; declare "
                "`signature ... encoder ...` in the deduction block"
            )
        return encoder(item)

    def derivations(self, item: Item) -> list[Item]:
        """Return the set of items derived under ``item`` in the
        chart's derivation forest.

        For the current implementation (which records only the
        aggregate weight at each item), this returns the chart's
        item set as a flat enumeration. The full derivation
        forest — a tree-structured object whose leaves are
        axioms and whose internal nodes are rule firings — is
        recoverable by re-running the agenda with the derivation
        semiring (a follow-up upgrade); the public surface here
        keeps the entry point in place for that upgrade.
        """
        # In the current implementation we expose the set of
        # items that contributed to the item's weight; a
        # derivation-semiring run is required for the full tree.
        # For the common case where the user wants "every item
        # in the chart," returning the item list is the closest
        # correct answer.
        if item not in (it for it, _ in self._result.chart.items()):
            return []
        return [it for it, _ in self._result.chart.items()]

    def goal_weight(self) -> torch.Tensor:
        """Aggregate weight over all goal items.

        If a single goal item was identified, returns its weight
        directly; otherwise aggregates via the semiring's
        ``plus`` reduction (the standard parse-as-marginal
        computation).
        """
        if not self._result.goal_items:
            return torch.tensor(
                float(self.semiring.zero)
                if hasattr(self.semiring.zero, "__float__")
                else 0.0,
                dtype=torch.get_default_dtype(),
            )
        if len(self._result.goal_items) == 1:
            return self._result.goal_items[0][1]
        weights = torch.stack([w for _, w in self._result.goal_items])
        return self.semiring.plus(weights, dim=0)

    @property
    def goal_items(self) -> list[tuple[Item, torch.Tensor]]:
        """List of (item, weight) pairs matching the goal predicate."""
        return self._result.goal_items

    def __repr__(self) -> str:
        return (
            f"ChartView(items={sum(1 for _ in self._result.chart.items())},"
            f" goal={len(self._result.goal_items)},"
            f" semiring={self._result.semiring.__class__.__name__})"
        )


# ---------------------------------------------------------------------------
# Agendas
# ---------------------------------------------------------------------------


class Agenda(ABC):
    """A queue of items pending further inference.

    The agenda discipline (FIFO / LIFO / priority) determines the
    evaluation order; correctness is independent of discipline
    for idempotent semirings (Goodman 1999 §3) but efficiency
    depends on it (McAllester 2002).
    """

    @abstractmethod
    def push(self, item: Item, weight: torch.Tensor) -> None: ...

    @abstractmethod
    def pop(self) -> tuple[Item, torch.Tensor]: ...

    @abstractmethod
    def empty(self) -> bool: ...


class FIFOAgenda(Agenda):
    """First-in-first-out agenda (semi-naïve Datalog discipline).

    The agenda processes items in the order they were derived.
    Correct for any monotone semiring; the canonical choice for
    Datalog and Earley parsing.
    """

    def __init__(self) -> None:
        self._queue: list[tuple[Item, torch.Tensor]] = []

    def push(self, item: Item, weight: torch.Tensor) -> None:
        self._queue.append((item, weight))

    def pop(self) -> tuple[Item, torch.Tensor]:
        return self._queue.pop(0)

    def empty(self) -> bool:
        return not self._queue


class LIFOAgenda(Agenda):
    """Last-in-first-out agenda (depth-first proof-search discipline).

    The agenda processes items in reverse-derivation order.
    Suitable for goal-directed search like Agda elaboration and
    Twelf-style MLTT type-checking, where depth-first proof
    construction is the natural strategy.
    """

    def __init__(self) -> None:
        self._stack: list[tuple[Item, torch.Tensor]] = []

    def push(self, item: Item, weight: torch.Tensor) -> None:
        self._stack.append((item, weight))

    def pop(self) -> tuple[Item, torch.Tensor]:
        return self._stack.pop()

    def empty(self) -> bool:
        return not self._stack


class PriorityQueueAgenda(Agenda):
    """Priority-queue agenda parameterised by a priority function.

    Items are processed in decreasing priority order. Used by
    Knuth's algorithm (priority = current best score),
    A* parsing (priority = g + h with admissible h), and any
    best-first chart algorithm.

    Parameters
    ----------
    priority_fn : Callable
        ``(item, weight) -> float``. Higher priorities are
        processed first. The chart's monotone-priority property
        (Nederhof 2003) ensures correctness when the semiring is
        superior.
    """

    def __init__(self, priority_fn: Callable[[Item, torch.Tensor], float]) -> None:
        self._priority_fn = priority_fn
        self._heap: list[tuple[float, int, Item, torch.Tensor]] = []
        self._counter = 0  # tie-breaker for deterministic ordering

    def push(self, item: Item, weight: torch.Tensor) -> None:
        # heapq is a min-heap; negate priority for max-heap behaviour.
        priority = -float(self._priority_fn(item, weight))
        self._counter += 1
        heapq.heappush(self._heap, (priority, self._counter, item, weight))

    def pop(self) -> tuple[Item, torch.Tensor]:
        _, _, item, weight = heapq.heappop(self._heap)
        return item, weight

    def empty(self) -> bool:
        return not self._heap


# ---------------------------------------------------------------------------
# The agenda engine
# ---------------------------------------------------------------------------


@dataclass
class AgendaResult:
    """The chart produced by an agenda-driven deduction.

    Attributes
    ----------
    chart : Chart
        The final K-presheaf on the item algebra.
    semiring : ChartSemiring
        The semiring used for weight aggregation.
    goal_items : list of (Item, torch.Tensor)
        Items in the chart matching the goal predicate, with
        their final aggregated weights.
    iterations : int
        Number of agenda steps consumed (diagnostic).
    """

    chart: Chart
    semiring: ChartSemiring
    goal_items: list[tuple[Item, torch.Tensor]] = field(default_factory=list)
    iterations: int = 0
    encoder: Encoder | None = None
    # Sum of all rule-attached + chart-attached loss values fired
    # during this run, populated by `DeductionSystem.run` when a
    # loss registry is attached. `None` if no losses fired.
    attached_loss: torch.Tensor | None = None


def run_agenda(
    axioms: Iterable[tuple[Item, torch.Tensor]],
    rules: Iterable[InferenceRule],
    semiring: ChartSemiring | None = None,
    agenda: Agenda | None = None,
    goal: Callable[[Item], bool] | None = None,
    max_iterations: int = 100_000,
    chart: Chart | None = None,
    rule_callback: (
        Callable[
            [str, list[tuple[Item, torch.Tensor]], Item, torch.Tensor],
            None,
        ]
        | None
    ) = None,
) -> AgendaResult:
    """Run the agenda-driven deduction engine to fixed point.

    Categorical denotation: returns the least pre-fixed point of
    the rule-system functor on the input axioms. The
    strategy-independence theorem (Goodman 1999) guarantees that
    the chart's final value is independent of the ``agenda``
    discipline when ``semiring`` is idempotent.

    Algorithm (Shieber-Schabes-Pereira 1995, Fig. 2 with
    semiring-weighted aggregation per Goodman 1999):

    1. Push every axiom onto the agenda.
    2. While the agenda is non-empty:
       a. Pop an item.
       b. Insert-or-aggregate into the chart.
       c. If the chart's weight changed, fire every rule for which
          the popped item is a premise: scan the chart for matching
          siblings, compute the conclusion's weight via the rule's
          ``weight_fn``, and push the conclusion onto the agenda.
    3. Return the chart with all items that match ``goal``.

    Parameters
    ----------
    axioms : iterable of (Item, weight)
        Initial items with their input-derived weights.
    rules : iterable of InferenceRule
        The deduction system's hyperedges.
    semiring : ChartSemiring, optional
        Weight aggregation. Defaults to log-prob.
    agenda : Agenda, optional
        Evaluation strategy. Defaults to FIFO (semi-naïve).
    goal : callable, optional
        Predicate selecting result items. Defaults to "every item".
    max_iterations : int, optional
        Safety bound on agenda steps; raises if exceeded.
    chart : Chart, optional
        Initial chart (defaults to empty :class:`HashChart`).
    """
    semiring = semiring or LOG_PROB
    agenda = agenda or FIFOAgenda()
    chart = chart or HashChart()
    rules = tuple(rules)

    for item, w in axioms:
        agenda.push(item, w)

    iterations = 0
    while not agenda.empty():
        if iterations >= max_iterations:
            raise RuntimeError(
                f"agenda exceeded max_iterations={max_iterations}; "
                f"likely a non-terminating deduction"
            )
        iterations += 1
        item, weight = agenda.pop()
        changed = chart.insert_or_aggregate(item, weight, semiring)
        if not changed:
            continue
        # Fire every rule for which this item could be a premise.
        for rule in rules:
            for premise_idx, premise_pattern in enumerate(rule.premises):
                bindings = match(premise_pattern, item)
                if bindings is None:
                    continue
                # Collect all sibling premises by matching the
                # remaining premise patterns against the chart.
                _fire_rule_with_premise(
                    rule,
                    premise_idx,
                    item,
                    weight,
                    bindings,
                    chart,
                    semiring,
                    agenda,
                    rule_callback,
                )

    goal_items: list[tuple[Item, torch.Tensor]] = []
    if goal is not None:
        for item, w in chart.items():
            if goal(item):
                goal_items.append((item, w))
    else:
        goal_items = list(chart.items())

    return AgendaResult(
        chart=chart,
        semiring=semiring,
        goal_items=goal_items,
        iterations=iterations,
    )


def _fire_rule_with_premise(
    rule: InferenceRule,
    fixed_idx: int,
    fixed_item: Item,
    fixed_weight: torch.Tensor,
    bindings: Bindings,
    chart: Chart,
    semiring: ChartSemiring,
    agenda: Agenda,
    rule_callback: (
        Callable[
            [str, list[tuple[Item, torch.Tensor]], Item, torch.Tensor],
            None,
        ]
        | None
    ) = None,
) -> None:
    """Fire a rule where one premise is the popped item.

    Recursively scans the chart for matches to the remaining
    premise patterns under the running bindings. When all
    premises are matched, instantiates the conclusion, computes
    its weight via the rule's ``weight_fn`` (or the semiring
    product of premise weights if no ``weight_fn`` is given),
    checks the side condition, and pushes onto the agenda.
    """
    _fire_remaining_premises(
        rule=rule,
        accumulated=[(fixed_item, fixed_weight)],
        bindings=bindings,
        chart=chart,
        semiring=semiring,
        agenda=agenda,
        remaining_indices=[i for i in range(len(rule.premises)) if i != fixed_idx],
        fixed_idx=fixed_idx,
        fixed_pair=(fixed_item, fixed_weight),
        rule_callback=rule_callback,
    )


def _fire_remaining_premises(
    *,
    rule: InferenceRule,
    accumulated: list,
    bindings: Bindings,
    chart: Chart,
    semiring: ChartSemiring,
    agenda: Agenda,
    remaining_indices: list[int],
    fixed_idx: int,
    fixed_pair,
    rule_callback: (
        Callable[
            [str, list[tuple[Item, torch.Tensor]], Item, torch.Tensor],
            None,
        ]
        | None
    ) = None,
) -> None:
    """Recursively match remaining premise patterns.

    On each call, picks the next remaining premise and scans the
    chart for matching items consistent with the running
    bindings. Recurses until all premises are matched, then
    fires.
    """
    if not remaining_indices:
        _fire(
            rule,
            bindings,
            fixed_idx,
            fixed_pair,
            chart,
            semiring,
            agenda,
            rule_callback,
        )
        return
    next_idx = remaining_indices[0]
    next_pattern = rule.premises[next_idx]
    for sibling, sib_w in chart.lookup(next_pattern):
        sub_bindings = match(next_pattern, sibling, bindings)
        if sub_bindings is None:
            continue
        _fire_remaining_premises(
            rule=rule,
            accumulated=accumulated + [(sibling, sib_w)],
            bindings=sub_bindings,
            chart=chart,
            semiring=semiring,
            agenda=agenda,
            remaining_indices=remaining_indices[1:],
            fixed_idx=fixed_idx,
            fixed_pair=fixed_pair,
            rule_callback=rule_callback,
        )


def _fire(
    rule: InferenceRule,
    bindings: Bindings,
    fixed_idx: int,
    fixed_pair,
    chart: Chart,
    semiring: ChartSemiring,
    agenda: Agenda,
    rule_callback: (
        Callable[
            [str, list[tuple[Item, torch.Tensor]], Item, torch.Tensor],
            None,
        ]
        | None
    ) = None,
) -> None:
    """All premises matched — instantiate and push the conclusion.

    Collects every premise's weight from the chart at the matched
    items, computes the conclusion's weight via the rule's
    ``weight_fn`` (defaulting to the semiring product), and
    pushes the new item onto the agenda. The pushed item will
    only update the chart if its aggregated weight increases the
    chart's value (the idempotent-merge property).
    """
    if rule.side_condition is not None and not rule.side_condition(bindings):
        return
    # Gather premise (item, weight) pairs in declaration order.
    antecedents: list[tuple[Item, torch.Tensor]] = []
    for i, premise_pattern in enumerate(rule.premises):
        if i == fixed_idx:
            antecedents.append(fixed_pair)
            continue
        try:
            premise_item = instantiate(premise_pattern, bindings)
        except KeyError:
            return  # premise has unresolved wildcards — can't fire
        w = chart.get(premise_item)
        if w is None:
            return
        antecedents.append((premise_item, w))

    premise_weights = tuple(w for _, w in antecedents)
    # Compute the conclusion's weight.
    if rule.weight_fn is not None:
        conclusion_weight = rule.weight_fn(
            bindings,
            premise_weights,
            semiring,
        )
    elif not premise_weights:
        conclusion_weight = torch.tensor(
            float(semiring.one) if hasattr(semiring.one, "__float__") else 0.0,
            dtype=torch.get_default_dtype(),
        )
    else:
        acc = premise_weights[0]
        for w in premise_weights[1:]:
            acc = semiring.times(acc, w)
        conclusion_weight = acc
    try:
        conclusion_item = instantiate(rule.conclusion, bindings)
    except KeyError:
        return
    agenda.push(conclusion_item, conclusion_weight)
    if rule_callback is not None:
        rule_callback(
            rule.name,
            antecedents,
            conclusion_item,
            conclusion_weight,
        )


# ---------------------------------------------------------------------------
# Public deduction-system entry
# ---------------------------------------------------------------------------


@dataclass
class DeductionSystem:
    """A weighted deductive system parameterised over its components.

    The system is parameterised by:

    - An item algebra (implicit in the patterns the rules use).
    - A list of arity-n :class:`InferenceRule` hyperedges.
    - A :class:`ChartSemiring` for weight aggregation.
    - An axiom injector ``In -> [(Item, Weight)]`` producing the
      input's lexical / boundary items.
    - A goal predicate ``Item -> bool`` selecting the result items.
    - (Optional) a chart constructor and an agenda strategy.

    The same data structure subsumes CKY (FIFO agenda, span items,
    Boolean / inside semiring), Viterbi (priority agenda with the
    current weight as priority, max-times semiring), A* parsing
    (priority agenda with an admissible heuristic, tropical
    semiring), MLTT type-checking (LIFO agenda, judgment items,
    Boolean semiring), and weighted Datalog (FIFO, atoms, any
    naturally-ordered semiring).
    """

    rules: tuple[InferenceRule, ...]
    semiring: ChartSemiring
    axiom_injector: Callable[[Any], list[tuple[Item, torch.Tensor]]]
    goal: Callable[[Item], bool]
    agenda_factory: Callable[[], Agenda] = FIFOAgenda
    chart_factory: Callable[[], Chart] = HashChart
    max_iterations: int = 100_000

    def run(self, input_value: Any) -> AgendaResult:
        """Run the deduction system on an input value."""
        axioms = self.axiom_injector(input_value)
        registry = getattr(self, "_loss_registry", None)
        deduction_name = getattr(self, "_deduction_name", None)
        rule_loss_acc: list[torch.Tensor] = []

        def _rule_callback(
            rule_name: str,
            antecedents: list[tuple[Item, torch.Tensor]],
            conclusion: Item,
            conclusion_w: torch.Tensor,
        ) -> None:
            if registry is None or deduction_name is None:
                return
            env = {
                "rule": rule_name,
                "deduction": deduction_name,
                "antecedents": list(antecedents),
                "conclusion": conclusion,
                "weight": conclusion_w,
            }
            val = registry.evaluate_on(
                "rule",
                target=rule_name,
                env=env,
                rule_deduction=deduction_name,
            )
            rule_loss_acc.append(val)

        result = run_agenda(
            axioms=axioms,
            rules=self.rules,
            semiring=self.semiring,
            agenda=self.agenda_factory(),
            goal=self.goal,
            max_iterations=self.max_iterations,
            chart=self.chart_factory(),
            rule_callback=(_rule_callback if registry is not None else None),
        )
        # Propagate any attached item-encoder to the result.
        comp = getattr(self, "_item_encoder", None)
        if comp is not None:
            result.encoder = comp
        # Evaluate chart-attached losses on the completed chart.
        if registry is not None and deduction_name is not None:
            chart_env = {
                "deduction": deduction_name,
                "chart": result.chart,
                "goal_items": result.goal_items,
            }
            chart_loss = registry.evaluate_on(
                "chart",
                target=deduction_name,
                env=chart_env,
            )
            losses = rule_loss_acc + [chart_loss]
        else:
            losses = rule_loss_acc
        if losses:
            total = losses[0]
            for v in losses[1:]:
                total = total + v
            result.attached_loss = total
        return result

    def __call__(self, input_value: Any) -> ChartView:
        """Run the deduction and return a :class:`ChartView`.

        Convenience for the user-facing presheaf-evaluation API:
        the chart's weights are differentiable tensors, and the
        view exposes ``weight``, ``enumerate``, ``derivations``,
        and ``goal_weight`` methods for downstream programs.
        """
        return ChartView(self.run(input_value))


# ---------------------------------------------------------------------------
# Strategy factories — concrete parsers as agenda specialisations
# ---------------------------------------------------------------------------


def cky_agenda() -> Agenda:
    """The CKY (bottom-up sweep) agenda — semi-naïve FIFO.

    For context-free + Boolean / inside semirings, FIFO order
    suffices to reach the chart's fixed point in
    :math:`O(n^3 \\cdot |R|)` time (McAllester 2002).
    """
    return FIFOAgenda()


def earley_agenda() -> Agenda:
    """The Earley (predict / scan / complete) agenda — FIFO.

    Earley's algorithm is identical to FIFO agenda-driven
    deduction over the predict / scan / complete items
    (Pereira & Warren 1983).
    """
    return FIFOAgenda()


def viterbi_agenda(priority_fn: Callable[[Item, torch.Tensor], float]) -> Agenda:
    """Viterbi-style best-first agenda.

    Priorities are the current best weight in the
    :math:`(\\max, \\times)` semiring; the agenda is a priority
    queue. Equivalent to Knuth's algorithm when the semiring is
    superior (Nederhof 2003).
    """
    return PriorityQueueAgenda(priority_fn)


def astar_agenda(
    g_plus_h: Callable[[Item, torch.Tensor], float],
) -> Agenda:
    """A* parsing agenda (Klein & Manning 2003).

    Priority is :math:`g + h` where :math:`g` is the current
    accumulated cost and :math:`h` is an admissible heuristic on
    the remaining cost. With an admissible :math:`h`, the agenda
    enumerates items in optimal order (Knuth 1977).
    """
    return PriorityQueueAgenda(g_plus_h)


def knuth_agenda() -> Agenda:
    """Knuth's best-first hyperpath search.

    Priority is the current chart weight (the item's best-so-far
    score). For a superior semiring, this is Dijkstra's algorithm
    on AND-OR hypergraphs (Knuth 1977; Nederhof 2003).
    """

    def _priority(_item: Item, weight: torch.Tensor) -> float:
        return float(weight) if weight.numel() == 1 else float(weight.sum())

    return PriorityQueueAgenda(_priority)


def depth_first_agenda() -> Agenda:
    """LIFO agenda — depth-first proof search (Agda / Twelf style)."""
    return LIFOAgenda()


def semi_naive_agenda() -> Agenda:
    """Semi-naïve Datalog evaluation (McAllester 2002) — FIFO."""
    return FIFOAgenda()
