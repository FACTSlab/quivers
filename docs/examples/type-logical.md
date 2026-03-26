# Type-Logical Grammar (Lambek Calculus)

## QVR Source

```qvr
category S, NP, N, VP, PP
object Token : 256

let grammar = parser(
    rules=[evaluation, adjunction_units, tensor_introduction, tensor_projection],
    terminal=Token,
    start=S
)

output grammar
```

## Overview

Type-logical grammar, grounded in the Lambek calculus, is a resource-conscious approach to syntax that requires each linguistic resource to be used exactly once in a specified order. This example demonstrates the four rule schemas that implement the Lambek calculus: `evaluation`, `adjunction_units`, `tensor_introduction`, and `tensor_projection`.

## Walkthrough

`category S, NP, N, VP, PP` introduces five atomic formulas. In the Lambek calculus, formulas are built inductively: if $X$ and $Y$ are formulas, then $X/Y$ (right division) and $X \backslash Y$ (left division) are also formulas. The grammar can express categories like $S/(\mathrm{NP} \backslash \mathrm{VP})$ or $\mathrm{VP}/\mathrm{PP}$.

`object Token : 256` introduces the terminal vocabulary. Tokens are assigned base categories via a lexicon.

The parser uses four rule schemas:

- **`evaluation`** encodes the beta rule: $X/Y, Y \vdash X$ and $Y, X \backslash Y \vdash X$. Given a functor and its argument in the correct linear order, derive the result.
- **`adjunction_units`** implements identity laws ($A \vdash A$), ensuring every atomic formula can be trivially derived as a goal.
- **`tensor_introduction`** combines two adjacent derivations using disjoint resources: if $X \vdash A$ and $Y \vdash B$ (disjoint spans), then $X, Y \vdash A \otimes B$. The disjointness constraint enforces resource sensitivity.
- **`tensor_projection`** decomposes a product: if $X \vdash A \otimes B$, split into derivations of $A$ from the left part and $B$ from the right part. This is needed for bottom-up parsing to identify constituent structure.

Together these four schemas yield a complete implementation of the Lambek calculus. The parser uses a chart-based algorithm (similar to CKY) guaranteed to terminate on finite input.

## DSL Features

- **`evaluation` schema**: Generates the elimination rules for $X/Y$ and $X \backslash Y$ across all category pairs.
- **`adjunction_units` schema**: Provides structural units (identity/base cases) for the logical system.
- **`tensor_introduction` and `tensor_projection`**: Manage the tensor product (concatenation), ensuring the linear structure of the input is preserved throughout derivation.
- **Residuation laws**: The three connectives ($\otimes$, $/$, $\backslash$) satisfy $A \otimes B \vdash C \iff A \vdash C/B \iff B \vdash A \backslash C$, enforcing resource sensitivity.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The Lambek calculus adds linearity to the closed monoidal category picture: each resource (atomic variable) appears exactly once in any proof term. This corresponds to the free symmetric monoidal closed category on a set of generators. The tensor product $\otimes$ is concatenation, and the divisions $/$ and $\backslash$ are its left and right adjoints, respectively. The residuation laws $A \otimes B \vdash C \iff A \vdash C/B \iff B \vdash A \backslash C$ are exactly the statement of this adjunction. Because there is no contraction (copying) or weakening (discarding), every derivation consumes its input span exactly once, and the chart-based parser enforces this by restricting each cell to its designated span.

## Connections to Other Formalisms

The Lambek calculus is more expressive than context-free grammar (handling extraction, gapping) but less expressive than unrestricted phrase-structure grammars, remaining decidable and efficiently parseable.

Compared to CCG, the Lambek calculus is more restricted: it enforces strict linearity and resource sensitivity, while CCG implicitly permits structural rules (weakening, contraction). The multimodal extensions (see multimodal-tlg) introduce controlled structural operators that can license specific deviations from strict linearity.
