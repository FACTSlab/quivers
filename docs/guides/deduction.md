# Weighted Deduction Systems

QVR's `deduction` block declares a weighted deductive system, a hyperedge of inference rules over an item algebra, paired with a semiring for weight aggregation. A single agenda-driven runtime evaluates the system; concrete parsing algorithms (CKY, Earley, Viterbi, A*, Knuth's, semi-naive Datalog, MLTT proof search) fall out as parameter settings.

## The framework

Categorically, a weighted deduction system is a tuple

$$\mathcal{D} = (I, R, K, \mathrm{ax}, \mathrm{goal})$$

with $I$ an *item algebra* (an abstract data type of judgments / facts / sequents), $R$ a *hypergraph* of inference rules (each a premise multiset, a conclusion, and a weight in $K$, universally quantified over pattern variables), $K$ a *algebra* / semiring for weight aggregation, $\mathrm{ax}$ an *axiom injector* producing the input's lexical / boundary items, and $\mathrm{goal} \subseteq I$ a *termination predicate*.

The chart is the $K$-presheaf $C : I^{\mathrm{op}} \to K$ that is the least pre-fixed point of the rule-system functor in the $K$-enriched lattice (Tarski-Knaster). The chart is *strategy-independent*; the agenda is the operational realization of the fixed-point computation.

See [References](#references) below.

## Surface

```text
object Atom : FinSet 4
deduction CG : Atom -> Atom [semiring=LogProb, start=S, depth=4]
    atoms NP, S, VP

    rule fwd_app   : X/Y, Y     |- X
    rule bwd_app   : Y, Y\X     |- X
    rule harm_comp : X/Y, Y/Z   |- X/Z
```

The unified option block carries the deduction-level configuration (semiring, start symbol, depth bound, optional `tolerance` for convergent-cycle LogProb, optional `max_iterations` cap). The indented body declares the atom set and rules; single-uppercase-letter identifiers in rule patterns (`X`, `Y`, `Z`) bind as wildcards, and non-wildcard names match literally. The declaration compiles to a `DeductionSystem` registered under the declared name (here `CG`).

## Charts as first-class differentiable values

Running a deduction yields a `ChartView`, a $K$-presheaf value with surface methods for querying:

```python
view = CG([
    (("span", "NP", 0, 1), torch.tensor(-0.3, requires_grad=True)),
    (("span", "VP", 1, 3), torch.tensor(-0.4, requires_grad=True)),
])

log_weight = view.weight(("span", "S", 0, 3))
all_S = view.enumerate(("span", "S", Wildcard("i"), Wildcard("j")))
total = view.goal_weight()

loss = -log_weight
loss.backward()  # gradients flow back to the axiom weights
```

The weights are `torch.Tensor` values; autograd flows through the agenda's semiring operations (`semiring.times`, `semiring.plus`) so any `requires_grad=True` rule weight or axiom weight feeding into the deduction is differentiable. This enables grammatical-inference / weighted-deduction loss objectives end-to-end.

## Strategy independence

For idempotent semirings (Boolean, LogProb, Viterbi), the chart's value at any item is independent of the agenda strategy (Goodman 1999, §3). The `test_strategy_independence_idempotent` test runs the same deduction under CKY-sweep, semi-naive FIFO, depth-first LIFO, and Earley-FIFO agendas and asserts the chart's goal-item weights agree to within `1e-5`.

## Pre-registered stdlib

The `quivers.stochastic.stdlib` module ships ready-to-use `DeductionSystem` instances:

| System         | Item algebra              | Semiring   | Strategy             |
|----------------|---------------------------|------------|----------------------|
| `CCG`          | `("span", cat, i, j)`     | LogProb    | CKY sweep            |
| `Lambek`       | sequents over patterns    | LogProb    | CKY sweep            |
| `STLC`         | check / synth judgments   | Boolean    | depth-first          |
| `MLTT`         | typing judgments          | Boolean    | depth-first (bounded depth) |
| `Datalog`      | ground atoms              | Boolean    | semi-naive FIFO      |
| `Dijkstra`     | `("dist", node)`          | Viterbi    | Knuth's              |
| `HMM`          | `("alpha", t, state)`     | LogProb    | CKY sweep            |
| `ViterbiHMM`   | same                      | Viterbi    | CKY sweep            |
| `EditDistance` | `("dist", i, j)`          | Viterbi    | CKY sweep            |

Each is a callable: `view = Datalog(axiom_list)` returns a `ChartView`.

## Strategy factory functions

The agenda module exposes canonical strategy factories that may be passed to the `agenda_factory=` parameter of a `DeductionSystem`:

- `cky_agenda()`, `earley_agenda()`, FIFO (semi-naive).
- `viterbi_agenda(priority_fn)`: priority-queue.
- `astar_agenda(g_plus_h)`: priority-queue with an admissible heuristic.
- `knuth_agenda()`: best-first hyperpath search (Knuth 1977; Nederhof 2003).
- `depth_first_agenda()`: LIFO (proof search).
- `semi_naive_agenda()`: FIFO for Datalog-shaped systems (McAllester 2002).

## panproto integration

A `DeductionDecl` compiles to a panproto schema via `QVR_DEDUCTION_PROTOCOL`. Vertex kinds: `deduction_system`, `deduction_rule`, `deduction_atom`, `deduction_premise`, `deduction_conclusion`. Schema morphisms over the protocol correspond to deduction-system specializations.

```python
from quivers.dsl import Compiler
from quivers.dsl.parser import parse_file
from quivers.dsl.program_theory import extract_deduction_schema

compiler = Compiler(parse_file("docs/examples/source/ccg.qvr"))
compiler.compile()
schema = extract_deduction_schema(compiler)
# schema is a panproto.Schema, usable with diff / lens-generation tooling.
```

## References

- Shieber, Schabes & Pereira (1995). [*Principles and implementation of deductive parsing*](https://doi.org/10.1016/0743-1066(95)00035-I). Journal of Logic Programming 24(1–2):3–36.
- Pereira & Warren (1983). [*Parsing as deduction*](https://doi.org/10.3115/981311.981338). In Proceedings of the 21st Annual Meeting of the ACL, pp. 137–144.
- Knuth (1977). [*A generalization of Dijkstra's algorithm*](https://doi.org/10.1016/0020-0190(77)90002-3). Information Processing Letters 6(1):1–5.
- Goodman (1999). [*Semiring parsing*](https://aclanthology.org/J99-4004/). Computational Linguistics 25(4):573–605.
- Klein & Manning (2001). [*Parsing and hypergraphs*](https://doi.org/10.1007/1-4020-2295-6_18). In Proceedings of IWPT, pp. 123–134.
- McAllester (2002). [*On the complexity analysis of static analyses*](https://doi.org/10.1145/581771.581774). Journal of the ACM 49(4):512–537.
- Nederhof (2003). [*Weighted deductive parsing and Knuth's algorithm*](https://doi.org/10.1162/089120103321337467). Computational Linguistics 29(1):135–143.
- Eisner, Goldlust & Smith (2005). [*Compiling Comp Ling: Practical weighted dynamic programming and the Dyna language*](https://aclanthology.org/H05-1036/). In Proceedings of HLT-EMNLP, pp. 281–290.
- Eisner & Blatz (2007). [*Program transformations for optimization of parsing algorithms and other weighted logic programs*](https://www.cs.jhu.edu/~jason/papers/eisner+blatz.fg06.pdf). In Proceedings of the 11th Conference on Formal Grammar.
