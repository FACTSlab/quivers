# Type-Logical Grammar (Lambek Calculus)

## QVR Source

```qvr
object Term : 16

deduction Lambek : Term -> Term {
    atoms {
        S, NP, N, VP, PP,
        Fwd, Bwd, Tns,
        span
    }

    rule right_app
        : span(I, K, Fwd(A, B)), span(K, J, B)
        |- span(I, J, A)

    rule left_app
        : span(I, K, B), span(K, J, Bwd(A, B))
        |- span(I, J, A)

    rule tensor_intro
        : span(I, K, A), span(K, J, B)
        |- span(I, J, Tns(A, B))

    rule tensor_left
        : span(I, J, Tns(A, B))
        |- span(I, J, A)

    rule tensor_right
        : span(I, J, Tns(A, B))
        |- span(I, J, B)

    semiring  LogProb
    start     S
    depth     6
}
```

## Overview

Type-logical grammar, grounded in the non-commutative Lambek calculus, is a resource-conscious approach to syntax: every hypothesis is used exactly once and argument order is preserved. The deduction above lists slash (`Fwd`, `Bwd`) and tensor (`Tns`) constructors over chart-spans `span(I, J, X)`, and licenses right / left application, product introduction, and product elimination as sequent rules.

## Walkthrough

`atoms { … }` declares the constructor vocabulary. Category atoms are `S`, `NP`, `N`, `VP`, `PP`; structural constructors are `Fwd(A, B) ≡ A/B`, `Bwd(A, B) ≡ A\B`, `Tns(A, B) ≡ A⊗B`. The chart-item constructor `span(I, J, X)` packages a derivation covering tokens `[I, J)` carrying category `X`. Single-uppercase identifiers (`A`, `B`, `I`, `J`, `K`) appearing in rule patterns bind as wildcards.

The rules realize the four logical core operations of the Lambek calculus:

- **`right_app`**: modus ponens for forward slash: `A/B, B ⊢ A`.
- **`left_app`**: modus ponens for backward slash: `B, A\B ⊢ A`.
- **`tensor_intro`**: product introduction: adjacent derivations of `A` and `B` combine into a derivation of `A⊗B`.
- **`tensor_left` / `tensor_right`**: product elimination: a derivation of `A⊗B` projects to derivations of either component over the same span.

Together these rules yield the equational theory of the residuated monoid. The agenda runs to depth 6 by default; the `LogProb` semiring accumulates inside log-probabilities that flow back as gradients to learnable axiom weights.

## DSL Features

- **Sequent rules with arbitrary arity**: rule bodies declare premises on the left of `|-` and a single conclusion on the right; the compiler routes unary patterns to unary chart cells and binary patterns to binary chart cells.
- **Resource sensitivity is structural**: there is no contraction or weakening rule, so every premise in a sequent must match a distinct chart cell.
- **Order preservation**: pattern variables appear in textual order; the parser enforces left-to-right span composition.
- **Tensor and slash as user atoms**: there is no special syntax, `Tns`, `Fwd`, `Bwd` are atoms declared in the `atoms { … }` block and may be replaced or extended by the user.

## Try it

```python
from quivers.dsl import load

prog = load("docs/examples/source/type_logical.qvr")
```

Pair the deduction with a `lexicon { ... }` block of `"word" : Cat = lf @ learnable` axioms and fit per-entry log-weights with an [`AutoNormalGuide`](../api/inference/guide.md) plus [`SVI`](../api/inference/svi.md) over an [`ELBO`](../api/inference/elbo.md) objective, conditioning on the chart's goal weight per observed sentence. Switching `semiring` from `LogProb` to `Viterbi` returns the most-likely derivation under the current weights without changing any rule.

## Categorical Perspective

The Lambek calculus is the internal language of a residuated monoidal category (biclosed monoidal category). The tensor `⊗` is the monoidal product; the two slashes are its left and right adjoints. The residuation laws

```
A ⊗ B  ⊢  C   iff   A  ⊢  C/B   iff   B  ⊢  A\C
```

are the statement of that adjunction. Because there is no contraction (copying) or weakening (discarding), every derivation consumes its input span exactly once; the agenda's span indexing enforces this by attaching each item to a single token range.

## Connections to Other Formalisms

The Lambek calculus is strictly more expressive than context-free grammar (handling extraction, gapping) but remains decidable and efficiently parseable. Compared to CCG it is more restricted: CCG implicitly permits structural rules (weakening, contraction) that the Lambek calculus does not. The multimodal extensions (see [multimodal-tlg](multimodal-tlg.md)) introduce controlled structural operators that license specific deviations from strict linearity.
