# Montague NLI

## Overview

A two-stage natural-language-inference architecture composed entirely out of QVR's weighted-deduction surface: a Montague-style grammar that derives logical forms from token spans, then an entailment prover that closes the LFs under modus ponens and functorial substitution. Everything is declared in `atoms { ... }` plus `rule` sequents; no grammar formalism or proof system is baked into the language.

## QVR Source

```qvr
object Term : 8

deduction Montague : Term -> Term {
    atoms {
        NP, S, N,
        Fwd, Bwd,
        span,
        App,
        every_lf, some_lf, pred_dog, pred_cat, pred_anim, pred_bark
    }

    rule fwd_app
        : span(I, K, Fwd(X, Y), F), span(K, J, X, A)
        |- span(I, J, Y, App(F, A))

    rule bwd_app
        : span(I, K, X, A), span(K, J, Bwd(X, Y), F)
        |- span(I, J, Y, App(F, A))

    lexicon {
        "every"  : Fwd(Fwd(Bwd(NP, S), S), NP) = every_lf  @ learnable
        "some"   : Fwd(Fwd(Bwd(NP, S), S), NP) = some_lf   @ learnable
        "dog"    : NP                          = pred_dog  @ learnable
        "cat"    : NP                          = pred_cat  @ learnable
        "animal" : NP                          = pred_anim @ learnable
        "barks"  : Bwd(NP, S)                  = pred_bark @ learnable
    }

    semiring  LogProb
    start     S
    depth     6
}

deduction Prover : Term -> Term {
    atoms { Claim, Implies, App }

    rule modus_ponens
        : Claim(P), Claim(Implies(P, Q))
        |- Claim(Q)

    rule app_subst
        : Claim(App(F, X)), Claim(Implies(X, Y))
        |- Claim(App(F, Y))

    semiring  LogProb
}
```

## Walkthrough

The grammar half of the module declares atomic category constructors (`NP`, `S`, `N`), slash constructors (`Fwd`, `Bwd`), a chart-item constructor `span(i, j, cat, lf)` that packages a derivation covering tokens `[i, j)` of category `cat` with logical form `lf`, and a function-application LF combinator `App`. Forward and backward application are sequents that combine a slash-typed span with its complement. The lexicon ships learnable log-weights per entry; the `LogProb` semiring carries differentiable inside scores. See [Montague (1973)](https://doi.org/10.1007/978-94-010-2506-5_10) for the type-driven semantic compositionality this fragment instantiates.

The prover half closes the resulting Claims under modus ponens and a functorial-substitution rule: from `Claim(App(F, X))` and `Claim(Implies(X, Y))`, conclude `Claim(App(F, Y))`. The two deductions chain by feeding the grammar's logical forms into the prover's `Claim` constructor at fit time.

## Try it

```python
from quivers.dsl import load

prog = load("docs/examples/source/montague_nli.qvr")
```

A worked training script that pairs the deductions with an NLI corpus lives at `docs/examples/source/train_montague_nli.py`. The script parses a premise/hypothesis pair into Claims via the Montague deduction, runs the Prover to depth `6`, and observes a Bernoulli over the goal-weight of `Claim(hypothesis_lf)` to compute the NLI loss.

## Categorical Perspective

Each `deduction` block denotes a weighted relation in the [agenda-based deduction semiring](../semantics/composition-rules.md): an arrow $\mathrm{Term} \to \mathrm{Term}$ in the [LogProb quantale](../semantics/quantales.md) whose underlying tensor is the chart of derivable items keyed by their derivation log-weights. Composing grammar with prover is composition in the same enriched category, so the gradient of the prover's goal weight flows back through the grammar's lexicon entries during training.
