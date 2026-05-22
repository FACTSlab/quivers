# Quantifier Scope

## Overview

Quantifier-scope ambiguity, treated as the choice of which scope-taking lift to apply during composition. Generalized quantifiers are typed as continuations `Cont(X) = S/(S/X)`; surface and inverse scope arise from different sequent-rule applications of an applicative lift and a scope-extruding bind. The deduction itself is a `LogProb`-weighted relation; a downstream `program` block over the chart's per-derivation weights can be conditioned to learn or marginalize over which reading the parser commits to. Canonical reference: [Bumford & Charlow (2026)](https://www.cambridge.org/core/elements/abs/effectdriven-interpretation/56671E539160AAA1DACF8555B82A2FE4), *Effect-Driven Interpretation*.

## QVR Source

```qvr
# Quantifier Scope via the Continuation Monad
#
# Generalised quantifiers as Cont-typed denotations whose lifts
# through the residuation calculus give surface vs. inverse
# scope readings. Categories carry slash (Fwd, Bwd) constructors
# and a unary Cont(X) constructor for the continuation-typed
# lift S/(S/X); chart items are span(I, J, X) triples.
#
# Deduction:
#
#   fwd_app       : X/Y, Y                 |- X            base application
#   bwd_app       : Y,   X\Y               |- X            base application
#   fwd_app_cont  : Cont(A/B), Cont(B)     |- Cont(A)      applicative lift
#   pure_cont     : A                      |- Cont(A)      pure
#   scope_take    : Cont(A), B\A           |- Cont(B)      scope bind
#
# Concrete answer types are baked into the rules (Cont(_) over
# arbitrary S); a richer fragment with multiple answer types
# would parameterise the rules over S.

object Term : FinSet 16

deduction QScope : Term -> Term [semiring=LogProb, start=S, depth=4, tolerance=1e-5]
    atoms S, NP, N, Fwd, Bwd, Cont, span, every, dog, barks
    rule fwd_app : span(I, K, Fwd(A, B)), span(K, J, B) |- span(I, J, A) #[learnable]
    rule bwd_app : span(I, K, B), span(K, J, Bwd(A, B)) |- span(I, J, A) #[learnable]
    rule fwd_app_cont : span(I, K, Cont(Fwd(A, B))), span(K, J, Cont(B)) |- span(I, J, Cont(A)) #[learnable]
    rule pure_cont : span(I, J, A) |- span(I, J, Cont(A)) #[learnable, bounded]
    rule scope_take : span(I, K, Cont(A)), span(K, J, Bwd(B, A)) |- span(I, J, Cont(B)) #[learnable]
    rule cont_elim : span(I, J, Cont(S)) |- span(I, J, S) #[learnable]
    lexicon
        "every" : Cont(Fwd(NP, N)) = every #[learnable]
        "dog"   : N                = dog   #[learnable]
        "barks" : Bwd(S, NP)       = barks #[learnable]
```

## Walkthrough

The `atoms` block declares category atoms (`S`, `NP`, `N`), slash constructors (`Fwd`, `Bwd`), the unary continuation constructor (`Cont`), the chart-item constructor (`span`), and the closed-class terminal vocabulary (`every`, `dog`, `barks`). Pattern variables (single-uppercase identifiers) are bound at firing time.

Six sequent rules realize the scope fragment:

- **`fwd_app` / `bwd_app`**: base forward and backward application from the underlying [Lambek calculus](type-logical.md).
- **`fwd_app_cont`**: the applicative lift of forward application under `Cont`. Two scope-taking expressions compose by pulling both continuations to the outside.
- **`pure_cont`**: the unit `A |- Cont(A)` of the continuation monad. Promotes any non-scope-taking constituent to a trivial scope-taker. Marked `#[bounded]` so the agenda's `depth=4` bound terminates the otherwise unbounded `Cont`-tower.
- **`scope_take`**: the scope-extruding bind. A continuation-typed expression of type `Cont(A)` adjacent to a `Bwd(B, A)` (a functor expecting an `A` argument) absorbs the surrounding context and yields a `Cont(B)`. Surface vs inverse scope corresponds to the order in which two scope-takers apply `scope_take` against the surrounding sentence-internal functors.
- **`cont_elim`**: the lower-at-answer-type closing step `Cont(S) |- S`, which collapses a saturated continuation reading to a flat `S` so the grammar's `start=S` goal applies.

A determiner is a generalised quantifier in continuation form: `"every" : Cont(Fwd(NP, N)) = every` lifts the determiner type `NP/N` into the continuation monad, so applying it to a common-noun-typed argument lifted into `Cont(N)` via `pure_cont` yields `Cont(NP)`, which then binds an inhomogeneous-typed VP under `scope_take` to give a saturated `Cont(S)`; the final `cont_elim` lowers to a flat `S`. The deduction's semiring is `LogProb`, so every distinct derivation of the start category `S` contributes a differentiable inside score; summing over derivations gives the marginal log-probability under the grammar, while taking max gives the most-likely single reading.

## Try it

```python
from quivers.dsl import load

prog = load("docs/examples/source/quantifier_scope.qvr")
```

To use the deduction inside a probabilistic program, wrap the chart's per-derivation weights in an outer [`program`](../guides/dsl-overview.md) that draws a categorical reading variable and observes a downstream comprehension judgment; conditioning on the judgment and marginalizing over the scoping yields a posterior over which reading speakers commit to. The [`MonadicProgram.marginalize`](../api/program.md) step is the categorical handle for the scoping marginal: the projection $\pi : \Phi \times R \to \Phi$ integrates out the reading coordinate $R$ via log-sum-exp on the joint log-likelihood.

## Categorical Perspective

`Cont` is the continuation monad on the category of formulas: $\mathrm{Cont}(X) = S/(S/X)$ realizes double negation $\neg\neg X$ in the answer-type $S$. The `pure_cont` rule is its unit $\eta : X \to \mathrm{Cont}(X)$; `fwd_app_cont` realizes its applicative-functor structure; `scope_take` realizes the [monadic bind](https://doi.org/10.1016/0890-5401(91)90052-4) $\mu \circ \mathrm{Cont}(f)$ in the form licensed by the [Lambek calculus](https://doi.org/10.2307/2310058) residuation laws. Surface vs inverse scope is then a choice of derivation in the rule-system multicategory; the agenda's `LogProb`-enriched chart records every choice, and downstream `marginalize` integrates them away into a single scope-marginal likelihood.

## Connections

The fragment composes directly with [type-logical](type-logical.md) and [multimodal](multimodal-tlg.md) grammars: replacing `Cont` with a controlled modality `Dia` yields a modal scope fragment in which structural rules are licensed only inside the scope-taker.
