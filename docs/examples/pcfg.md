# Probabilistic Context-Free Grammar

## Overview

A [probabilistic context-free grammar](https://en.wikipedia.org/wiki/Probabilistic_context-free_grammar) as an agenda-based weighted deduction over [CKY](https://en.wikipedia.org/wiki/CYK_algorithm) chart items `span(I, J, N)` (token range `[I, J)` carrying nonterminal `N`) under the `LogProb` semiring. Each production is a concrete `rule` with no free wildcards on the conclusion: the chart fires only the productions the grammar licenses, the weights are bindings-keyed under `#[learnable]`, and the chart's goal weight at `span(0, n, S)` is the corpus inside log-probability.

## QVR Source

```qvr
object Term : FinSet 16

deduction PCFG : Term -> Term [semiring=LogProb, start=S, depth=4]
    atoms S, NP, VP, Det, N, V, span, the, a, cat, dog, sleeps, runs
    rule s_np_vp : span(I, K, NP), span(K, J, VP) |- span(I, J, S) #[learnable]
    rule np_det_n : span(I, K, Det), span(K, J, N) |- span(I, J, NP) #[learnable]
    rule vp_v : span(I, J, V) |- span(I, J, VP) #[learnable]
    lexicon
        "the"    : Det = the     #[learnable]
        "a"      : Det = a       #[learnable]
        "cat"    : N   = cat     #[learnable]
        "dog"    : N   = dog     #[learnable]
        "sleeps" : V   = sleeps  #[learnable]
        "runs"   : V   = runs    #[learnable]
```

## Walkthrough

The `atoms` block enumerates the constructor universe: nonterminals (`S`, `NP`, `VP`), preterminals (`Det`, `N`, `V`), the chart-item constructor `span`, and the closed-class terminal vocabulary (`the`, `a`, `cat`, `dog`, `sleeps`, `runs`).

The three concrete branching rules declare the grammar's productions one by one. `s_np_vp` combines an `NP` span and an adjacent `VP` span into an `S` span; `np_det_n` combines a `Det` and an `N` into an `NP`; `vp_v` lifts a `V` span to a `VP`. Because each conclusion has no free wildcard, `#[learnable]` allocates one log-weight per rule firing, giving exactly one production probability per rule. Pattern variables `I`, `J`, `K` are single-uppercase identifiers ranging over token positions.

The `lexicon` block ships one entry per closed-class terminal; each `"word" : Cat = lf #[learnable]` line becomes a unit-width `span(I, I+1, Cat)` axiom with the lexical entry's logical form whenever the input token at position `I` matches `"word"`. The semiring is `LogProb`, so inside probabilities accumulate in log space; gradients flow back through the agenda's semiring operations to the learnable weights.

## Try it

```python
from quivers.dsl import load

prog = load("docs/examples/source/pcfg.qvr")
```

To fit per-rule probabilities, wrap the deduction in a [`program`](../guides/dsl-overview.md) that draws lexicon weights from a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior per preterminal and observes the chart's `chart.goal_weight()` against a corpus of sentences. The `#[learnable]` markers on each lexicon entry expose `nn.Parameter`s the optimizer adjusts; an [`AutoNormalGuide`](../api/inference/guide.md) plus [`SVI`](../api/inference/svi.md) over an [`ELBO`](../api/inference/elbo.md) objective drives the fit. For Viterbi or counting semirings the chart returns a single max-derivation score or a derivation count, which can be conditioned in the same way.

## Categorical Perspective

A PCFG is a weighted deduction in the `LogProb` semiring whose chart is a `K`-valued presheaf over the item algebra `I` of `span(I, J, N)` triples. Branching rules are binary hyperedges in the rule-system hypergraph; anchor rules are unary hyperedges. The least pre-fixed point of the rule-system functor in the `K`-enriched lattice of charts is the inside table; the strategy-independence theorem (Goodman 1999) says CKY-sweep, A\*, and Knuth's algorithm all compute the same chart value. The compiler picks a default strategy from the rule arities and the semiring's algebraic properties.

## CKY and Inside-Outside

For deductions whose semiring is commutative + idempotent + supports inverses (`LogProb`, `Viterbi`), the runtime emits an analytic outside pass via Eisner & Goldlust 2005 in addition to the inside fixpoint; this yields closed-form gradients of any chart value with respect to rule weights and is faster and more numerically stable than backpropagating through the unrolled agenda. For other semirings the runtime falls back to autodiff through the chart-fill operations.

## Connections to Language Modeling

Summing `chart.goal_weight()` over derivations gives the sentence's marginal log-probability under the grammar; that scalar is a perfectly ordinary log-prob and can be used as a language model in a downstream `program`. EM-style training reduces to maximizing `chart.goal_weight()` on observed sentences; the analytic outside pass handles the gradient. Changing the `semiring` field (e.g., from `LogProb` to `Viterbi`) selects a different aggregation strategy without touching the rules.
