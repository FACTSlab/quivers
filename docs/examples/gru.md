# Bayesian GRU

## Overview

The GRU is a simplified alternative to the LSTM that uses two gates instead of three and maintains a single hidden state instead of separate cell and hidden states. This example demonstrates inline distribution sampling (`<-` syntax) as an alternative to separate morphism declarations, and shows how architectural simplification maps to fewer morphisms in the DSL.

## QVR Source

```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 128
type Output = Euclidean 64

embed tok_embed : Token -> Embedded

continuous gate_z : Embedded * Hidden -> Hidden ~ LogitNormal
continuous gate_r : Embedded * Hidden -> Hidden ~ LogitNormal
continuous candidate : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]

program gru_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
    draw z ~ gate_z(x_t, h_prev)
    draw r ~ gate_r(x_t, h_prev)

    let reset_hidden = r * h_prev

    h_cand <- Normal(reset_hidden, 0.5)

    let z_complement = 1.0 - z
    let h_new = z_complement * h_prev + z * h_cand

    return h_new

continuous output_proj : Hidden -> Output ~ Normal [scale=0.1]

let gru = tok_embed >> scan(gru_cell) >> output_proj

output gru
```

## Walkthrough

### Type Declarations for Simplified State

The GRU does not need a separate State type because it maintains only a single hidden state. `type Hidden = Euclidean 128` carries all temporal information. This is a significant simplification over the LSTM, which concatenates cell and hidden states into a 128-dimensional State vector.

### Gate Morphisms with LogitNormal Prior

`continuous gate_z : Embedded * Hidden -> Hidden ~ LogitNormal` defines the update gate, controlling how much new state replaces old state. `continuous gate_r : Embedded * Hidden -> Hidden ~ LogitNormal` defines the reset gate, controlling how much previous state influences candidate generation. Both use LogitNormal priors so gate values fall naturally in [0, 1].

### Monadic GRU Cell Program

The program applies both gates via `draw` statements, then computes the reset-gated previous state: `let reset_hidden = r * h_prev`. When the reset gate is near 0, the previous state is effectively forgotten before candidate generation; when near 1, it passes through freely.

The inline distribution `h_cand <- Normal(reset_hidden, 0.5)` samples the candidate hidden state from a Normal distribution centered at the reset-gated state with standard deviation 0.5. The `<-` syntax samples directly from a specified distribution rather than applying a learned morphism; the mean depends on a computed value while the standard deviation is fixed.

The final interpolation `h_new = z_complement * h_prev + z * h_cand` is a convex combination controlled by the update gate. When $z \approx 1$, the state updates to the candidate; when $z \approx 0$, the previous state is retained.

### Output Projection and Full Architecture

`let gru = tok_embed >> scan(gru_cell) >> output_proj` combines embedding, scanned GRU program, and output projection.

## DSL Features

- **Inline distribution sampling**: `name <- Distribution(params)` samples from a distribution with arbitrary computed parameters, without a separately declared morphism. The `<-` operator is distinct from `draw ... ~`, which applies a declared morphism with learned parameters.
- **Two-gate architecture**: Two LogitNormal-prior morphisms (update and reset) instead of the LSTM's three, reducing parameter count by roughly 30%.
- **Convex interpolation**: The update equation `(1-z) * h_prev + z * h_cand` blends previous and candidate states, with the gate controlling the mix.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The GRU operates on a single state space $\mathrm{Hidden}$ rather than the LSTM's product $\mathrm{Hidden} \times \mathrm{Hidden}$, eliminating the explicit memory/output factorization. The reset and update gates act at different points in the computation: the reset gate modulates candidate generation (controlling how much history the candidate sees), while the update gate modulates the final state interpolation (controlling how much the state changes). This sequential application of gates produces a different compositional structure than the LSTM's parallel gates.

The interpolation $h_{\mathrm{new}} = (1 - z) \cdot h_{\mathrm{prev}} + z \cdot h_{\mathrm{cand}}$ preserves gradient flow similarly to the LSTM's additive cell update. Both terms are weighted and summed, so gradients pass through without the repeated multiplicative scaling that causes vanishing gradients in vanilla RNNs. The LogitNormal priors keep gate values in [0, 1], preventing extreme gradient magnification.
