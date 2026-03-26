# Multi-Layer Bayesian Transformer

## Overview

A multi-layer transformer with four stacked blocks, each containing multi-head attention and feed-forward networks. This example demonstrates type aliases, replicated morphisms with array notation, the `fan` combinator for parallel heads, and the `stack` combinator for independent layer replication.

## QVR Source

```qvr
object Token : 256
type Latent = Euclidean 64
type HeadOut = Euclidean 16
type FFHidden = Euclidean 128

embed tok_embed : Token -> Latent

continuous head[4] : Latent -> HeadOut ~ Normal [scale=0.1]
continuous attn_proj : Latent -> Latent ~ Normal [scale=0.1]
continuous ff_up : Latent -> FFHidden ~ Normal
continuous ff_down : FFHidden -> Latent ~ Normal [scale=0.1]
continuous residual_attn : Latent -> Latent ~ Normal [scale=0.01]
continuous residual_ff : Latent -> Latent ~ Normal [scale=0.01]

let layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
let transformer = tok_embed >> stack(layer, 4)

output transformer
```

## Walkthrough

### Type Declarations

The transformer operates on five types. `object Token : 256` establishes a finite discrete domain of 256 tokens, not a Euclidean type but a discrete domain over which the embedding is defined.

Three type aliases follow: `type Latent = Euclidean 64` is the 64-dimensional representation space that flows through most of the architecture. `type HeadOut = Euclidean 16` is the output of each attention head; with four heads outputting 16 dimensions each, the fan combinator concatenates them into 64 dimensions (4 x 16 = 64), matching the latent dimension. `type FFHidden = Euclidean 128` is the intermediate dimension of the feed-forward network, expanding from 64 to 128 before projecting back down.

These type declarations shape the entire data flow. The type system ensures that compositions are dimensionally consistent.

### Embedding Layer

`embed tok_embed : Token -> Latent` lifts discrete tokens into the continuous Latent space. This is a lookup table that retrieves fixed embedding vectors, not a learned weight-matrix transformation like continuous morphisms.

### Attention Head Components

`continuous head[4] : Latent -> HeadOut ~ Normal [scale=0.1]` uses array notation `[4]` to create four independent morphisms sharing the same type signature. Each has weights drawn from a Normal distribution with scale 0.1. The array notation is syntactic sugar for four separate declarations with independent parameters.

### Attention Projection

`continuous attn_proj : Latent -> Latent ~ Normal [scale=0.1]` recombines the concatenated outputs of the four attention heads (64 dimensions) back into the 64-dimensional latent space, mixing information across heads.

### Feed-Forward Network Layers

`continuous ff_up : Latent -> FFHidden ~ Normal` projects from 64 to 128 dimensions (using default Normal initialization). `continuous ff_down : FFHidden -> Latent ~ Normal [scale=0.1]` projects back to 64 dimensions.

### Residual Connections

`continuous residual_attn : Latent -> Latent ~ Normal [scale=0.01]` and `continuous residual_ff : Latent -> Latent ~ Normal [scale=0.01]` handle residual connections for the attention and feed-forward blocks respectively. Their smaller scales (0.01 vs. 0.1) encourage the network to learn small residual updates rather than dramatic transformations, following standard deep learning initialization practice.

### Layer Composition

`let layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff` composes a single transformer layer. Reading left to right: `fan(head)` replicates the input four times and applies each head in parallel, producing four 16-dimensional outputs that are concatenated into 64 dimensions. The result flows through attention projection, residual connection, feed-forward expansion, feed-forward contraction, and final residual connection. The `>>` operator is Kleisli composition, threading the stochastic output of one morphism as input to the next.

### Stacking Independent Layers

`let transformer = tok_embed >> stack(layer, 4)` creates four independent deep copies of the layer morphism, each with separate parameters. Unlike weight tying, stacking produces four distinct instances. Input tokens are first embedded, then passed through each of the four transformer layers in sequence, with each layer learning different features.

### Output Declaration

`output transformer` marks the composed morphism as the target of inference and learning.

## DSL Features

- **Object types**: `object Token : 256` creates a discrete finite domain, distinct from Euclidean types. Used for categorical inputs like tokens or class labels.
- **Type aliases**: `type Latent = Euclidean 64` names a dimension for readability. Transparent to the type system.
- **Embed morphisms**: `embed` declares lookup-table morphisms from discrete objects to continuous spaces, unlike `continuous` morphisms that learn weight-matrix transformations.
- **Continuous morphisms**: `continuous name : Source -> Target ~ Distribution [options]` declares learnable linear transformations with stochastic weights drawn from the specified prior.
- **Array notation**: `head[4]` creates four independent morphisms sharing a type signature and prior, avoiding four separate declarations.
- **Composition operator**: `>>` is Kleisli composition. `f >> g` threads the probabilistic output of `f` as input to `g`.
- **Fan combinator**: `fan(head)` replicates input across the four head morphisms in parallel and concatenates their outputs.
- **Stack combinator**: `stack(layer, 4)` creates four independent sequential copies of `layer`, each with separate parameters. Output of one feeds into the next.
- **Let bindings**: `let` introduces named morphism definitions for readability and reuse. Purely definitional.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The transformer operates in the Kleisli category $\mathrm{Kl}(V)$ where $V$ is the probability distribution monad: objects are types and morphisms are probabilistic functions that sample parameters from priors. Each morphism $f : A \to B$ represents a distribution over deterministic linear maps rather than a single fixed map. Composing two morphisms $f >> g$ chains their weight distributions, so the composite $A \to C$ samples parameters for both $f$ and $g$ independently.

The fan combinator is the categorical diagonal $\Delta : X \to X^n$, replicating an element for independent parallel processing. The stack combinator is repeated sequential composition with independent parameters: $\mathrm{stack}(\mathrm{layer}, 4)$ forms $\mathrm{layer}_1 >> \mathrm{layer}_2 >> \mathrm{layer}_3 >> \mathrm{layer}_4$ where each $\mathrm{layer}_i$ has its own weight distribution. This is what makes each layer free to learn different features at different depths.
