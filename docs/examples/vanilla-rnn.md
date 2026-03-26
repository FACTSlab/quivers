# Bayesian Vanilla RNN

## Overview

The vanilla RNN is the simplest recurrent architecture: a single morphism processes input and previous hidden state to produce the next hidden state. This example demonstrates the `scan` combinator for threading state across sequences and product domains (`A * B`) for combining input with state.

## QVR Source

```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 128
type Output = Euclidean 64

embed tok_embed : Token -> Embedded

continuous cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
continuous output_proj : Hidden -> Output ~ Normal [scale=0.1]

let rnn = tok_embed >> scan(cell) >> output_proj

output rnn
```

## Walkthrough

### Type System for RNNs

Four types establish the dimensional flow. `object Token : 256` is a discrete 256-token vocabulary. `type Embedded = Euclidean 64` is the continuous space tokens are projected into. `type Hidden = Euclidean 128` is the hidden state, larger than the embedding to provide additional representational capacity for recurrent transformations. `type Output = Euclidean 64` is the output space.

### Token Embedding

`embed tok_embed : Token -> Embedded` performs a lookup into a table of embedding vectors, one per token. Unlike continuous morphisms that learn weight-matrix transformations of continuous inputs, embedding morphisms produce specific output vectors for each discrete input symbol. This is the interface between the discrete token domain and the continuous computational domain.

### Recurrent Cell Morphism

`continuous cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]` is the core of the RNN. The product type `Embedded * Hidden` represents an ordered pair: the current input embedding and the previous hidden state. The cell learns to blend these two inputs into the next hidden state. The type system enforces that only morphisms declared with product domains can receive paired inputs, and dimensions must match the signature.

When the scan combinator uses this cell, it automatically constructs the product at each time step by pairing the current input embedding with the previous hidden state.

### Output Projection

`continuous output_proj : Hidden -> Output ~ Normal [scale=0.1]` projects the final hidden state into the output space. By keeping the output projection as a separate morphism, the architecture stays modular: different downstream tasks can use different projections applied to the same learned hidden representations.

### RNN Composition

`let rnn = tok_embed >> scan(cell) >> output_proj` composes three stages. Tokens are embedded into 64-dimensional vectors. The scan combinator applies the cell morphism across the input sequence, threading 128-dimensional hidden state from step to step. The final hidden state is projected to 64-dimensional output. The actual mechanics of scan (state initialization, iteration, output collection) are handled by the DSL runtime.

### Output Declaration

`output rnn` designates the composed morphism as the target of learning and inference.

## DSL Features

- **Product domains**: `A * B` represents ordered pairs. In `cell : Embedded * Hidden -> Hidden`, the cell receives both current input and previous state. The type system prevents accidental dimension mismatches.
- **Scan combinator**: `scan(cell)` takes a morphism `f : A * S -> S` and produces a new morphism that iterates `f` across a sequence of `A` values, threading state `S` from step to step. This is a categorical catamorphism (left fold) over the sequence.
- **Continuous morphisms in RNNs**: Weights are drawn from the specified prior (here Normal with scale 0.1). Each execution samples new weights, providing a basis for uncertainty quantification through multiple forward passes.
- **Let bindings**: Named definitions like `let rnn = ...` improve readability and enable reuse. They define composite morphisms from simpler ones.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The vanilla RNN is a fold (catamorphism) over sequential data. The cell morphism $\mathrm{cell} : \mathrm{Embedded} \times \mathrm{Hidden} \to \mathrm{Hidden}$ is a binary operation combining an input with an accumulator to produce a new accumulator. The scan combinator applies this operation iteratively, threading the accumulator from step to step and collecting intermediate results.

The product domain $\mathrm{Embedded} \times \mathrm{Hidden}$ is the categorical product: morphisms into $A \times B$ correspond to pairs of morphisms into $A$ and into $B$. The scan combinator constructs these products automatically at each time step. The full composition lives in the Kleisli category $\mathrm{Kl}(V)$ over the probability monad, so composition is associative and enriched with stochasticity from the weight priors.

The vanilla RNN's limitation is that all temporal information is collapsed into a single hidden representation. Information from early in the sequence must survive composition through many cell applications, leading to vanishing or exploding gradients when the sequence is long.
