# Probabilistic Context-Free Grammar

## QVR Source

```qvr
object Term : 16

deduction PCFG : Term -> Term {
    atoms {
        S, NP, VP,
        Det, N, V,
        the, a, cat, dog, sleeps, runs,
        span, leaf
    }

    rule branch
        : span(I, K, B), span(K, J, C)
        |- span(I, J, A)

    rule anchor
        : leaf(I, T)
        |- span(I, J, A)

    lexicon {
        "the"     : Det = the     @ learnable
        "a"       : Det = a       @ learnable
        "cat"     : N   = cat     @ learnable
        "dog"     : N   = dog     @ learnable
        "sleeps"  : V   = sleeps  @ learnable
        "runs"    : V   = runs    @ learnable
    }

    semiring  LogProb
    start     S
    depth     6
}
```

## Overview

A PCFG is an agenda-based weighted deduction over CKY chart items `span(I, J, N)` (token range `[I, J)` carrying nonterminal `N`) under the `LogProb` semiring. Two rule families drive the parse:

- `branch`: binary branching: adjacent spans of categories `B` and `C` combine into a span of some category `A`. The wildcard `A` is bound at firing time, weighted by the `(A → B C)` production probability the lexicon supplies.
- `anchor`: lexical anchoring: a `leaf(I, T)` axiom for a token at position `I` carrying preterminal `T` lifts to a span of width 1.

Production probabilities are learnable per lexicon entry; the `@ learnable` marker allocates a per-entry `nn.Parameter` log-weight that the optimizer can adjust during training.

## Walkthrough

`atoms { … }` enumerates the constructor universe: nonterminals (`S`, `NP`, `VP`), preterminals / POS tags (`Det`, `N`, `V`), the closed-class terminal vocabulary (`the`, `a`, `cat`, `dog`, …), and the chart-item / leaf constructors (`span`, `leaf`).

The `branch` rule's pattern variables `A`, `B`, `C` range freely over nonterminal atoms; concrete branchings are restricted by which combinations the surrounding training data exercises (the lexicon controls only the leaf side here, leaving branching weights to a richer setting where each `(A, B, C)` triple is also `@ learnable`). The `anchor` rule grounds a leaf at position `I` into a unit-width span, with the preterminal-to-token mapping supplied by the `lexicon` block.

The `lexicon { … }` block ships one `(word, category, lf)` entry per closed-class terminal; each entry is a learnable weight on a `(category → token)` emission. The semiring is `LogProb`, so inside probabilities accumulate in log space; gradients flow back through the agenda's semiring operations to the learnable weights.

## DSL Features

- **Single deduction block** declares the whole grammar: rule set, lexicon, semiring, start symbol, depth bound, no separate parser combinator.
- **`lexicon { … }`** is the axiom-injection sugar for label-indexed lookups: every `"word" : Cat = lf @ learnable` line becomes a `(leaf(I, Cat), weight)` axiom whenever the input token at position `I` equals `"word"`.
- **Pattern-polymorphic rules**: the `branch` rule is one sequent that fires for *any* nonterminal triple `(A, B, C)` consistent with the chart. There is no separate production declaration per triple.
- **Chart as first-class differentiable value**: at runtime the deduction's `chart.weight(item)`, `chart.enumerate(pattern)`, and `chart.goal_weight()` return `torch.Tensor` values whose gradients flow back through the agenda's semiring operations.

## Try it

```python
from quivers.dsl import load

prog = load("docs/examples/source/pcfg.qvr")
```

To fit per-rule probabilities, wrap the deduction in a [`program`](../guides/dsl.md) that draws lexicon weights from a [Dirichlet](https://doi.org/10.1093/biomet/74.2.237) prior per preterminal and observes the chart's `chart.goal_weight()` against a corpus of sentences. The `@ learnable` markers on each lexicon entry expose `nn.Parameter`s the optimizer adjusts; an [`AutoNormalGuide`](../api/inference/guide.md) plus [`SVI`](../api/inference/svi.md) over an [`ELBO`](../api/inference/elbo.md) objective drives the fit. For Viterbi or counting semirings the chart returns a single max-derivation score or a derivation count, which can be conditioned in the same way.

## Categorical Perspective

A PCFG is a weighted deduction in the `LogProb` semiring whose chart is a `K`-valued presheaf over the item algebra `I` of `span(I, J, N)` triples. Branching rules are binary hyperedges in the rule-system hypergraph; anchor rules are unary hyperedges. The least pre-fixed point of the rule-system functor in the `K`-enriched lattice of charts is the inside table; the strategy-independence theorem (Goodman 1999) says CKY-sweep, A\*, and Knuth's algorithm all compute the same chart value. The compiler picks a default strategy from the rule arities and the semiring's algebraic properties.

## CKY and Inside-Outside

For deductions whose semiring is commutative + idempotent + supports inverses (`LogProb`, `Viterbi`), the runtime emits an analytic outside pass via Eisner & Goldlust 2005 in addition to the inside fixpoint; this yields closed-form gradients of any chart value with respect to rule weights and is faster and more numerically stable than backpropagating through the unrolled agenda. For other semirings the runtime falls back to autodiff through the chart-fill operations.

## Connections to Language Modeling

Summing `chart.goal_weight()` over derivations gives the sentence's marginal log-probability under the grammar; that scalar is a perfectly ordinary log-prob and can be used as a language model in a downstream `program`. EM-style training reduces to maximising `chart.goal_weight()` on observed sentences; the analytic outside pass handles the gradient. Changing the `semiring` field (e.g., from `LogProb` to `Viterbi`) selects a different aggregation strategy without touching the rules.
