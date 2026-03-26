# Bayesian Elman Network

## Overview

The Elman network is a simple recurrent architecture that decomposes its cell into separate transformation and context-persistence morphisms composed with `>>`. This example demonstrates compositional morphism decomposition, where each morphism has a distinct role, and shows how scale parameters control initialization behavior.

## QVR Source

```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 128
type Output = Euclidean 64

embed tok_embed : Token -> Embedded

continuous transition : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
continuous context_copy : Hidden -> Hidden ~ Normal [scale=0.01]
continuous output_proj : Hidden -> Output ~ Normal [scale=0.1]

let cell = transition >> context_copy
let elman = tok_embed >> scan(cell) >> output_proj

output elman
```

## Walkthrough

### Type Structure for Simple Recurrence

The Elman network uses the same type structure as the vanilla RNN: discrete tokens, a 64-dimensional embedding, a 128-dimensional hidden state, and a 64-dimensional output. No composite state vectors are needed.

### Transition Morphism

`continuous transition : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]` is the primary transformation. It takes the product of current embedding and previous hidden state and produces the next hidden state. All state evolution flows through this single morphism.

### Context Copy Morphism

`continuous context_copy : Hidden -> Hidden ~ Normal [scale=0.01]` is an endomorphism (same source and target type) that refines the state after transition. Its small scale (0.01 vs. 0.1) initializes it near identity, so it initially passes most state information through with minimal modification. This separates the roles of transformation (transition) and state persistence (context_copy).

### Cell Composition

`let cell = transition >> context_copy` composes the two into a single cell suitable for scanning. The cell is not a primitive morphism but a composite, making the information flow explicit: transition handles input-state transformation, context_copy handles state refinement.

### Full Elman Network

`let elman = tok_embed >> scan(cell) >> output_proj` constructs the complete network. The scan combinator treats the composed cell as atomic at each time step, applying transition then context_copy internally.

## DSL Features

- **Compositional morphism decomposition**: Rather than a single monolithic cell, the Elman architecture splits transformation and persistence into two morphisms composed with `>>`. Each has a single responsibility, making the architecture modular and interpretable.
- **Scale parameters**: Different scales for different roles. `transition` uses 0.1 (moderate initialization) while `context_copy` uses 0.01 (near-identity initialization). Scale specifies the standard deviation of the prior over weights.
- **Scan on composite morphisms**: `scan(cell)` works on any morphism with the right type signature, whether primitive or composed. The internal composition is opaque to scan.
- **Endomorphisms**: `context_copy : Hidden -> Hidden` has matching source and target types. Endomorphisms form a monoid under composition and can be applied repeatedly without type errors.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The Elman cell's explicit decomposition into $\mathrm{transition} >> \mathrm{context\_copy}$ is a factorization of the cell morphism in the Kleisli category. The transition morphism maps $\mathrm{Embedded} \times \mathrm{Hidden} \to \mathrm{Hidden}$, and context_copy is an endomorphism on $\mathrm{Hidden}$. Endomorphisms on an object form a monoid under composition, meaning context_copy could be composed with itself or with other Hidden-to-Hidden morphisms without type errors.

The different scale parameters implement different priors: the small scale on context_copy concentrates probability mass near zero, so the morphism initially acts close to an identity. This makes the design intent explicit in the categorical structure rather than hiding it inside a single opaque cell. The tradeoff is that without gating, all information must survive repeated application of the same transition morphism, leading to the same gradient flow issues as the vanilla RNN.
