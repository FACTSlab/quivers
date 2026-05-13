# Custom Sequent Rules

## QVR Source

```qvr
object Term : 16

deduction AB : Term -> Term {
    atoms {
        S, NP, N, VP, PP,
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

    semiring  LogProb
    start     S
    depth     6
}
```

## Overview

Every rule in a `deduction { … }` block is a sequent declared in the DSL itself; there are no built-in named rule schemas to import. This example defines an AB grammar (Ajdukiewicz-Bar-Hillel) plus harmonic composition over chart-spans `span(I, J, X)`. `Fwd(X, Y)` is the forward-slash constructor `X/Y`; `Bwd(X, Y)` is the backward-slash constructor `X\Y`.

## Walkthrough

`atoms { … }` lists every identifier the rules may match literally. Category atoms (`S`, `NP`, `N`, `VP`, `PP`), slash constructors (`Fwd`, `Bwd`), and the chart-item constructor (`span`) are atoms. Identifiers that appear in a rule pattern but are *not* listed in `atoms` are pattern variables; the convention is single uppercase letters (`X`, `Y`, `Z`, `I`, `J`, `K`).

Each rule's body is a sequent: comma-separated premises on the left of `|-`, a single conclusion on the right. The premise multiplicity determines whether the rule fires on a single chart cell (unary) or on a pair of adjacent cells (binary).

A variable appearing multiple times in the same rule unifies across occurrences: in `fwd_app`, the `Y` in the first premise must match the `Y` in the second premise. Different rules instantiate independently.

## DSL Features

- **`rule NAME : premises |- conclusion`**: a sequent rule. Arbitrary-arity premise lists are supported; the compiler dispatches to the appropriate chart-cell shape.
- **Pattern variables vs atoms**: single-uppercase identifiers bind as wildcards; every other identifier in a rule pattern must appear in the surrounding `atoms { … }` block.
- **Constructor applications in patterns**: `Fwd(X, Y)`, `Bwd(X, Y)`, `span(I, J, X)` are patterns whose head is an atom and whose arguments are nested patterns. Matching is structural and unification-based.

## Pattern Matching and Unification

The compiler uses first-order pattern matching with occurs-checking-free unification to fire rules:

- **Variables** (`X`, `Y`, `Z`): metavariables that match any concrete subterm. Repeated occurrences within a rule must unify.
- **Constructor patterns** (`Fwd(X, Y)`): match constructor-application items whose head is the same atom and whose children unify.

The agenda fires each rule at every chart cell (unary) or every pair of adjacent cells (binary) whose categories unify with the rule's premises; the conclusion item is then inserted into the chart under the rule's weight.

## Extending the Rule Set

Additional combinators are spelled out as more sequent rules in the same block:

- **Type-raising** (unary): `rule type_raise : span(I, J, X) |- span(I, J, Fwd(Y, Bwd(Y, X)))`, note that introducing a fresh wildcard like `Y` in the conclusion requires either an `axioms = …` source or a downstream rule that pins it down.
- **Restricted composition**: `rule restricted_comp : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, NP)) |- span(I, J, Fwd(X, NP))`, by replacing the third wildcard with the literal `NP` atom we constrain when the rule fires.

Rule premise multiplicity is unary or binary; combinators that need three or more premises are expressed as a chain of binary rules sharing intermediate categories.

## Categorical Perspective

Sequent rules are hyperedges in the rule-system multicategory. A binary rule is a 2-input / 1-output hyperedge whose endpoints are pattern templates; firing the rule against the chart is the substitution along a pattern morphism into the category of concrete chart items. Variable unification across premises is exactly the pullback in the category of variable assignments: two premise patterns sharing a variable `Y` constrain the two assignments to agree on `Y`'s value. The agenda's least-pre-fixed-point computation in the `LogProb`-enriched lattice of charts is independent of firing order (Goodman 1999 §3); the runtime picks a default strategy from rule arities and semiring properties.
