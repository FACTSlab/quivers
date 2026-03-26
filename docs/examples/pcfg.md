# Probabilistic Context-Free Grammar

## QVR Source

```qvr
quantale product_fuzzy

object N : 10
object T : 64

stochastic binary_rules : N -> N * N
stochastic lexical_rules : N -> T

let pcfg = parser(rules=[binary_rules, lexical_rules], start=0)

output pcfg
```

## Overview

A PCFG associates probabilities with context-free grammar productions. This example demonstrates stochastic morphisms in a product semiring, the `parser` combinator, type-driven dispatch between binary and lexical rules, and quantale-based weight aggregation.

## Walkthrough

The declaration `quantale product_fuzzy` selects a quantale where weights live in [0, 1], ordered by the usual inequality, with multiplication as the monoidal operation. All stochastic morphisms in the program carry weights from this quantale, and composing rules during parsing multiplies their weights.

`object N : 10` introduces a nonterminal category with ten indices (0 through 9). `object T : 64` introduces the terminal category with 64 indices.

`stochastic binary_rules : N -> N * N` declares a stochastic morphism from nonterminals to pairs of nonterminals in the Kleisli category of the stochastic monad. Each instance carries a probability weight: for example, nonterminal 0 might expand to the pair (0, 1) with probability 0.3 and to (1, 2) with probability 0.7.

`stochastic lexical_rules : N -> T` declares a stochastic morphism from nonterminals to terminals. Nonterminal 3 might expand to terminal 42 with probability 0.8 and to terminal 15 with probability 0.2.

The `parser` combinator accepts the two morphisms and a start index. The compiler inspects the target types to determine which rules are binary (N -> N * N) versus lexical (N -> T). The resulting `pcfg` object performs CKY parsing to compute all parse trees with their probabilities.

`output pcfg` exports the parser for use in subsequent computations.

## DSL Features

- **`stochastic` keyword**: Declares morphisms with multiple weighted outputs per input, treated as first-class values that can be passed to combinators like `parser`.
- **Product type `N * N`**: The compiler recognizes a morphism targeting a product type as a binary rule (type-driven dispatch).
- **`parser` combinator**: Higher-order function that takes a list of morphisms and a start index, builds a parse table, and provides an interface for computing derivations.
- **`quantale product_fuzzy`**: Sets the global semiring for weight aggregation: multiplication for combining probabilities along a derivation, addition (or max) for combining alternative derivations.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

Branching rules like `binary_rules : N -> N * N` are morphisms in the Kleisli category of the stochastic monad, where a morphism from $X$ to $Y$ is a function $X \to \mathcal{D}(Y)$ mapping each element to a probability distribution. Kleisli composition of these morphisms multiplies the weights along derivation paths, which is exactly what the CKY algorithm does when it combines subtrees. The product quantale provides the algebraic scaffolding: associativity of Kleisli composition means the parser handles nested derivation structure without manual weight management, and quantale distributivity ensures that aggregation over alternative derivations is well-defined. The type system enforces the binary/lexical distinction at compile time, since $N \to N \otimes N$ and $N \to T$ are different morphism types.

## CKY Algorithm and Derivation

The CKY algorithm is a bottom-up dynamic programming approach. Given n terminals, it constructs a triangular table where entry (i, j) stores the nonterminals that can derive the subsequence from position i to j, along with derivation probabilities.

The algorithm has three phases. First, lexical rules populate the base cases: for each position and terminal, it computes which nonterminals can produce that terminal. Second, it iterates over increasing span lengths, considering all split points and querying binary_rules to find nonterminals that expand into compatible pairs. The weight of each new derivation is the product of the left-span weight, right-span weight, and rule weight. Third, it returns entries for the full span [0, n) corresponding to the start nonterminal.

The stochastic morphism interface abstracts storage details; whether rules live in a dense table, sparse hash map, or are computed on the fly, the parser works the same way.

## Nonterminal Indices and Start Symbols

When you use `start=0`, you identify a numeric index into the nonterminal category N. If the QVR program instead used a `category` declaration with symbolic names, you could write `start=S` to refer by name.

In this example, `object N : 10` creates indices 0 through 9. Index 0 often represents the sentence category (S), index 1 noun phrases (NP), and so on, though the mapping is user-defined.

## Connections to Language Modeling

PCFGs assign a sentence probability by summing over all parse tree probabilities. This sum can serve as a language model for tasks like machine translation or speech recognition.

The CKY algorithm extends to inside-outside probabilities, enabling EM learning of rule probabilities from unparsed sentences. Changing the `quantale` declaration (e.g., to log-probabilities or a different semiring) changes the aggregation strategy without modifying the grammar.
