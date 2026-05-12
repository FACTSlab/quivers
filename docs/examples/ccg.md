# Weighted Combinatory Categorial Grammar

## QVR Source

```qvr
object Term : 16

deduction CCG : Term -> Term {
    atoms {
        NP, S, N, VP, PP,
        Fwd, Bwd,
        span
    }

    rule fwd_app
        : span(I, K, Fwd(X, Y)), span(K, J, Y)
        |- span(I, J, X)

    rule bwd_app
        : span(I, K, Y), span(K, J, Bwd(X, Y))
        |- span(I, J, X)

    rule fwd_comp
        : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, Z))
        |- span(I, J, Fwd(X, Z))

    rule bwd_comp
        : span(I, K, Bwd(Y, Z)), span(K, J, Bwd(X, Y))
        |- span(I, J, Bwd(X, Z))

    rule fwd_xcomp
        : span(I, K, Fwd(X, Y)), span(K, J, Bwd(Y, Z))
        |- span(I, J, Bwd(X, Z))

    rule bwd_xcomp
        : span(I, K, Fwd(Y, Z)), span(K, J, Bwd(X, Y))
        |- span(I, J, Fwd(X, Z))

    semiring  LogProb
    start     S
    depth     6
}
```

## Overview

Combinatory Categorial Grammar (CCG) is expressed as an agenda-based weighted deduction whose items are chart spans `span(I, J, X)` (token range `[I, J)` carrying category `X`). The structural combinators of CCG — forward and backward application, harmonic composition, and crossed composition — each become one sequent rule. The semiring is `LogProb`, so inside scores flow as differentiable tensors back to whatever axiom / rule weights the user marks `learnable`.

## Walkthrough

`object Term : 16` declares a finite carrier for chart items; the concrete cardinality is irrelevant because the deduction reasons symbolically over constructor-tagged tuples, not over enumerated elements of `Term`.

`atoms { … }` lists every identifier the rules may match literally — category atoms (`NP`, `S`, `N`, `VP`, `PP`), slash constructors (`Fwd`, `Bwd`), and the chart-item constructor (`span`). Identifiers not listed here that appear in a rule pattern are bound as wildcards; the convention is single uppercase letters (`X`, `Y`, `Z`, `I`, `J`, `K`).

Each `rule` is a sequent: premises on the left of `|-`, conclusion on the right. `Fwd(X, Y)` constructs the forward-slash category `X/Y`; `Bwd(X, Y)` constructs the backward-slash category `X\Y`. Adjacent spans whose end / start indices agree fire whichever rule's pattern matches their categories.

`semiring LogProb` selects log-space inside scores. `start S` declares the goal category for a successful parse. `depth 6` bounds derivation depth to keep the agenda finite.

## DSL Features

- **`deduction { … }` block**: declares the agenda-based weighted deduction in a single record. The block's seven irreducible parameters — item algebra (via `atoms`), rule set, semiring, axiom source, goal predicate, start symbol, depth bound — are field-by-field.
- **`atoms { … }`**: closes the constructor universe. Every identifier appearing in a rule pattern must be either an atom or a single-uppercase wildcard variable.
- **Sequent rules**: arbitrary-arity premises on the left of `|-`, single conclusion on the right; rules with one premise are unary chart rules, with two are binary, and so on.
- **Slash constructors**: `Fwd(X, Y)` and `Bwd(X, Y)` are user-declared atoms, not built-in syntax. The combinators are theorems in this presentation.

## Categorical Perspective

CCG is the internal language of a closed monoidal category. The forward slash `X/Y` and backward slash `X\Y` are internal hom-objects (exponentials); the application rule is the counit of the hom-tensor adjunction, `[Y, X] ⊗ Y → X`. Composition corresponds to chaining adjunctions: given `X/Y` and `Y/Z`, transitivity yields `X/Z`. Crossed composition relies on a braiding isomorphism to swap argument order. The type of an expression completely determines what it can combine with, because the closed structure forces all combination to go through the adjunction.

## Semiring Selection

The choice of semiring affects the parser's behaviour: `LogProb` accumulates inside log-probabilities (numerically stable, differentiable); `Viterbi` returns the highest-weight derivation; `Counting` counts distinct derivations; `Boolean` checks membership without weights. The same deduction block serves all four objectives via the `semiring` field.
