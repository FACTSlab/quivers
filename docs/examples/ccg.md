# Weighted Combinatory Categorial Grammar

## QVR Source

```qvr
category S, NP, N, VP, PP
object Token : 256

let grammar = parser(
    rules=[evaluation, harmonic_composition, crossed_composition],
    terminal=Token,
    start=S
)

output grammar
```

## Overview

Combinatory Categorial Grammar (CCG) assigns syntactic categories to words and applies combinatory rules to derive sentence structures. This example demonstrates symbolic category declarations, pre-defined rule schema combinators (`evaluation`, `harmonic_composition`, `crossed_composition`), and the schema-strategy separation in the QVR language.

## Walkthrough

`category S, NP, N, VP, PP` introduces five symbolic categories. Unlike numeric indices, these are readable names that the compiler maps to indices internally. Categories can appear in slash expressions like `S / NP`.

`object Token : 256` introduces a terminal category with 256 indices representing possible tokens.

The parser declaration specifies three rule schemas. The `rules` parameter lists abstract schema functors, not grounded rules. The `evaluation` schema generates forward application ($X/Y, Y \Rightarrow X$) and backward application ($Y, X \backslash Y \Rightarrow X$) for every pair of categories. `harmonic_composition` generates forward composition ($X/Y, Y/Z \Rightarrow X/Z$) and backward composition ($Y \backslash Z, X \backslash Y \Rightarrow X \backslash Z$) for all compatible triples. `crossed_composition` generates rules that permute argument order, handling phenomena like gapping and coordination.

`terminal=Token` tells the parser which category represents terminals. `start=S` specifies that derivations must produce a sentence.

## DSL Features

- **`category` declaration**: Introduces symbolic labels (mapped to indices internally) that can be used in slash expressions, rule specifications, and parser configurations.
- **Rule schemas** (`evaluation`, `harmonic_composition`, `crossed_composition`): Parameterized rules that generate grounded instances for every relevant combination of categories at compile time.
- **`terminal` parameter**: Declares which category represents terminal symbols, structuring the parsing algorithm accordingly.
- **Slash notation** ($X/Y$, $X \backslash Y$): Forward slash expects argument $Y$ on the right; backward slash expects $Y$ on the left.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

CCG is the internal language of a closed monoidal category. The forward slash $X/Y$ and backward slash $X \backslash Y$ are internal hom-objects (exponentials), and the application rule is the counit of the hom-tensor adjunction: $[Y, X] \otimes Y \to X$. Composition corresponds to chaining adjunctions; given $X/Y$ and $Y/Z$, transitivity yields $X/Z$. Crossed composition relies on a braiding isomorphism to swap argument order. The type of an expression completely determines what it can combine with, because the closed structure forces all combination to go through the adjunction.

## Grammar Expressiveness and Linguistic Coverage

CCG is more expressive than context-free grammar while remaining efficiently parseable. Functor categories let modifiers like `VP \ VP` attach flexibly without extra rules. The schemas provided here handle a wide variety of linguistic phenomena, though more exotic rule sets can be added via custom rules (see the custom-rules example).

## Semiring Selection and Parsing Strategies

The choice of semiring affects the parser's behavior: `ViterbiSemiring` finds the highest-weight parse, `LogSemiring` uses log-probabilities for numerical stability, `CountSemiring` counts distinct parses, and `BooleanSemiring` checks membership without weights. Parametrizing the parser over the semiring lets the same grammar serve different objectives.
