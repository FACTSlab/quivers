# Bayesian Bidirectional RNN

## Overview

The bidirectional RNN processes a sequence in both forward and backward directions using independent recurrent paths, then combines their outputs. This example demonstrates the tensor product operator (`@`) for parallel composition of morphisms, showing how it differs from sequential composition with `>>`.

## QVR Source

```qvr
object Token : 256
type Embedded = Euclidean 64
type FwdHidden = Euclidean 64
type BwdHidden = Euclidean 64
type Combined = Euclidean 128
type Output = Euclidean 32

embed tok_embed : Token -> Embedded

continuous fwd_cell : Embedded * FwdHidden -> FwdHidden ~ Normal [scale=0.1]

let forward_path = tok_embed >> scan(fwd_cell)

continuous bwd_cell : Embedded * BwdHidden -> BwdHidden ~ Normal [scale=0.1]

let backward_path = tok_embed >> scan(bwd_cell)

continuous combine : Combined -> Output ~ Normal [scale=0.1]

let birnn = (forward_path @ backward_path) >> combine

output birnn
```

## Walkthrough

### Type Declarations for Bidirectional Processing

`FwdHidden` and `BwdHidden` are both 64-dimensional but are declared as distinct types so forward and backward hidden states cannot be accidentally mixed. `Combined = Euclidean 128` is the concatenation of both directions (64 + 64).

### Shared Token Embedding

`embed tok_embed : Token -> Embedded` is shared by both forward and backward paths, giving tokens consistent initial representations in both directions while keeping the directional computations independent.

### Forward and Backward Paths

`fwd_cell` and `bwd_cell` have identical type signatures but independent parameters. Each is composed with the shared embedding and scanned: `let forward_path = tok_embed >> scan(fwd_cell)` processes tokens left-to-right, `let backward_path = tok_embed >> scan(bwd_cell)` processes tokens right-to-left (sequence reversal is handled at the data level, not in the DSL).

### Tensor Product Composition

`let birnn = (forward_path @ backward_path) >> combine` uses the tensor product `@` to run both paths in parallel on the same input. The `@` operator applies both morphisms independently and pairs their outputs into a product type. The paired final hidden states (64 + 64 = 128 dimensions) are then passed to `combine`, which projects to 32-dimensional output.

The tensor product differs from `>>`: where `>>` threads data sequentially, `@` applies morphisms in parallel without data dependency between them.

## DSL Features

- **Tensor product operator**: `f @ g` applies `f` and `g` independently to the same input and pairs their outputs into a product. It is associative and has an identity element, forming a monoidal structure.
- **Shared components**: Using `tok_embed` in both paths refers to the same morphism with the same parameters. This ensures consistent embeddings while keeping directional computations independent.
- **Independent path morphisms**: `fwd_cell` and `bwd_cell` have identical signatures but separate parameters, so each direction learns its own transformations.
- **Combining parallel paths**: `(f @ g) >> combine` first runs parallel composition, then sequential composition on the paired result. The combine morphism learns to integrate information from both directions.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The tensor product `@` is the monoidal product on morphisms. For morphisms $f : A \to B$ and $g : A \to C$ sharing a source, $f \otimes g : A \to B \times C$ applies both independently and pairs the results. This captures computational parallelism: the two paths never interact until their outputs are combined, and each maintains its own state space and parameters.

The combine morphism then acts as a projection from the product space $\mathrm{FwdHidden} \times \mathrm{BwdHidden}$ into the output space. The monoidal structure is associative, so $(f \otimes g) \otimes h = f \otimes (g \otimes h)$, and more than two paths can be composed in parallel. The bidirectional architecture addresses the limitation that a unidirectional RNN at position $t$ has no access to context from positions after $t$; the tensor product makes the independence and parallelism of the two directional passes explicit in the categorical structure.
