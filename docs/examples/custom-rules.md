# Custom Rules of Inference

## QVR Source

```qvr
category S, NP, N, VP, PP
object Token : 256

rule forward_app(X, Y) : X/Y, Y => X
rule backward_app(X, Y) : Y, X\Y => X
rule forward_comp(X, Y, Z) : X/Y, Y/Z => X/Z
rule backward_comp(X, Y, Z) : Y\Z, X\Y => X\Z

let grammar = parser(
    rules=[forward_app, backward_app, forward_comp, backward_comp],
    terminal=Token,
    start=S
)

output grammar
```

## Overview

The `rule` declaration lets you define inference rules using sequent-style notation instead of relying on pre-defined schemas. This example manually defines the four standard CCG rules (forward/backward application and composition), demonstrating the custom rule syntax, universally quantified variables, and compile-time instantiation.

## Walkthrough

The category and token declarations are identical to previous examples.

`rule forward_app(X, Y) : X/Y, Y => X` introduces a rule with two universally quantified variables. The left side of `=>` lists premises (input categories in linear order); the right side is the conclusion. Here: a functor $X/Y$ adjacent to a $Y$ yields $X$. The compiler instantiates this for every pair of categories, generating grounded rules like `S/NP, NP => S`.

`rule backward_app(X, Y) : Y, X\Y => X` is the mirror image: $Y$ on the left and $X \backslash Y$ on the right yields $X$. The sequent notation makes linear order explicit.

`rule forward_comp(X, Y, Z) : X/Y, Y/Z => X/Z` has three variables. Two functors compose: $X/Y$ and $Y/Z$ produce $X/Z$ by threading the inner argument through.

`rule backward_comp(X, Y, Z) : Y\Z, X\Y => X\Z` is the backward version: $Y \backslash Z$ and $X \backslash Y$ produce $X \backslash Z$.

The parser collects these four rules and instantiates each for all valid category combinations at compile time, producing an explicit rule table.

## DSL Features

- **`rule` keyword**: Syntax is `rule Name(vars) : Premises => Conclusion`. Premises are comma-separated and ordered left-to-right matching input order.
- **Universally quantified variables** (X, Y, Z): Match any category. A variable appearing multiple times must unify to the same category in all occurrences.
- **Slash patterns** ($X/Y$, $X \backslash Y$): Recognized as composite category patterns by the compiler, not atomic categories.
- **Compile-time instantiation**: For $n$ atomic categories and $k$ variables, the compiler generates up to $n^k$ grounded rule instances.

## Pattern Matching and Unification

The compiler uses pattern matching and unification to instantiate custom rules:

- **Variable patterns** ($X$, $Y$, $Z$): Metavariables that match any concrete category. Multiple occurrences of the same variable must unify.
- **Slash patterns** ($X/Y$): Match functor categories whose components unify with $X$ and $Y$. Will not match atomic or product categories.
- **Product patterns** ($X \otimes Y$): Match product categories whose components unify with $X$ and $Y$.

The compiler exhaustively iterates over all valid assignments subject to unification constraints.

## Equivalence to Pre-defined Schemas

These four rules are equivalent to the pre-defined `evaluation` (forward + backward application) and `harmonic_composition` (forward + backward composition) schemas. The reason to use custom rules is transparency: you can see and modify the exact rule structure, omit rules you don't want (e.g., drop `backward_comp`), or add constraints (e.g., restrict which categories participate in composition).

## Advanced Custom Rules

The custom rule syntax supports more than standard CCG rules:

- **Type-raising**: `rule type_raise(X, Y, Z) : X => (Z/X)/Y`
- **Null insertion**: `rule null_insertion(X) : => X` (derives a category from an empty span)
- **Ternary combination**: `rule ternary_combo(X, Y, Z, W) : X, Y, Z => W`
- **Restricted composition**: `rule restricted_comp(X, Y) : X/Y, Y/NP => X/NP` (composition only when the right functor's argument is NP)

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

Custom rules are schema functors from category patterns to rule instances. Given a rule like `forward_comp(X, Y, Z) : X/Y, Y/Z => X/Z`, the compiler treats the pattern as a functor from the category of variable assignments (all triples of concrete categories) to the set of grounded rules. Unification of variables across multiple occurrences corresponds to a pullback: the compiler requires that the diagram of pattern substitutions commutes. Different presentations of the same logical structure (custom rules vs. pre-defined schemas) generate the same set of grounded rules.

## Mixing Custom and Pre-defined Rules

You can freely mix custom rules with pre-defined schemas:

```qvr
category S, NP, N, VP, PP
object Token : 256

rule coord_and(X, Y) : X, and, X => X  # Custom coordination rule

let grammar = parser(
    rules=[forward_app, harmonic_composition, coord_and],
    terminal=Token,
    start=S
)

output grammar
```

The compiler instantiates all rules (custom and pre-defined) into the same rule table, and the parser applies them uniformly.
