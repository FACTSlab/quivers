# Bayesian LSTM

## Overview

The LSTM addresses the vanishing/exploding gradient problems of vanilla RNNs by introducing separate memory cells and gating mechanisms. This example demonstrates `program` blocks, which are monadic computations that sequence multiple stochastic draws and deterministic operations within a single cell, going beyond simple linear morphism composition.

## QVR Source

```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 64
type State = Euclidean 128
type Output = Euclidean 32

embed tok_embed : Token -> Embedded

continuous gate_i : Embedded * State -> Hidden ~ LogitNormal
continuous gate_f : Embedded * State -> Hidden ~ LogitNormal
continuous gate_o : Embedded * State -> Hidden ~ LogitNormal
continuous cell_cand : Embedded * State -> Hidden ~ Normal [scale=0.5]

program lstm_cell(x_t, state_prev) : Embedded * State -> State
    draw i_gate ~ gate_i(x_t, state_prev)
    draw f_gate ~ gate_f(x_t, state_prev)
    draw o_gate ~ gate_o(x_t, state_prev)
    draw g_cand ~ cell_cand(x_t, state_prev)

    let c_new = f_gate * g_cand + i_gate * g_cand
    let two_c = 2.0 * c_new
    let sig_2c = sigmoid(two_c)
    let tanh_c = 2.0 * sig_2c - 1.0
    let h_new = o_gate * tanh_c

    return (c_new, h_new)

continuous output_proj : State -> Output ~ Normal [scale=0.1]

let lstm = tok_embed >> scan(lstm_cell) >> output_proj

output lstm
```

## Walkthrough

### Type Declarations and State Representation

The LSTM uses a layered type hierarchy to support its dual-state architecture. `type State = Euclidean 128` is double the Hidden dimension of 64 because it concatenates the cell state (c) and hidden state (h) into a single vector for scanning. `type Output = Euclidean 32` is smaller than the state, compressing information at the output boundary.

### Gate Morphisms and LogitNormal Distributions

Three gate morphisms map from current input and previous state to gate activations:

- `gate_i` (input gate): controls how much new candidate information enters the cell state.
- `gate_f` (forget gate): controls how much previous cell state is retained.
- `gate_o` (output gate): controls how much cell state is revealed to the hidden state.

All three use `LogitNormal` priors. LogitNormal produces values in [0, 1] through a logistic transformation, matching the semantics of gates without requiring explicit sigmoid activations.

### Cell Candidate Morphism

`continuous cell_cand : Embedded * State -> Hidden ~ Normal [scale=0.5]` produces the candidate update. Unlike the gates, it uses a Normal prior so values are unbounded before gating. The scale of 0.5 keeps initial candidates relatively small.

### Monadic LSTM Cell Program

The `program lstm_cell(x_t, state_prev) : Embedded * State -> State` block combines multiple stochastic draws with deterministic arithmetic.

The four `draw` statements sample from the gate and candidate morphisms. Each draw applies the morphism and names the result for subsequent use. The `draw` operation both samples parameters from the prior and applies the morphism.

The `let` bindings implement the LSTM equations. `c_new = f_gate * g_cand + i_gate * g_cand` is a simplified cell state update. The tanh approximation uses the identity $\tanh(x) \approx 2 \cdot \sigma(2x) - 1$, composing sigmoid (a DSL built-in) with scaling and shifting. The hidden state update `h_new = o_gate * tanh_c` gates the transformed cell state.

`return (c_new, h_new)` packs the new cell state and hidden state back into the 128-dimensional State space.

### Output Projection and Composition

`let lstm = tok_embed >> scan(lstm_cell) >> output_proj` composes embedding, recurrent scanning of the monadic program, and output projection. The scan combinator applies the full program at each time step, threading the 128-dimensional state through the sequence.

## DSL Features

- **Programs**: `program name(args) : Type -> Type` defines a monadic computation that sequences multiple stochastic draws and deterministic operations, producing a morphism in the Kleisli category.
- **Draw statements**: `draw name ~ morphism(args)` applies a stochastic morphism, samples from its distribution, and binds the result. Multiple draws compose through the probability monad.
- **Deterministic let bindings in programs**: `let name = expr` performs arithmetic on previously drawn values without introducing new stochasticity. Supports element-wise `+`, `*`, scalar operations.
- **Built-in functions**: `sigmoid` maps real values to (0, 1). The LSTM uses it to approximate tanh.
- **LogitNormal distribution**: Produces values in (0, 1) by passing a Normal random variable through the logistic sigmoid. Suited to gate parameters.
- **Scan wrapping programs**: `scan(lstm_cell)` works the same as scan on simple morphisms. At each time step, it executes the full program, threading the state output to the next step.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The LSTM extends the vanilla RNN's fold structure by factoring the state space into a product $\mathrm{State} \cong \mathrm{Hidden} \times \mathrm{Hidden}$, separating memory (cell state) from output (hidden state). The program block is a computation in the Kleisli category that combines multiple morphism applications (draws) with deterministic transformations (let bindings) into a single composite morphism. This cannot be expressed as simple `>>` composition because the intermediate values interact through arithmetic rather than just threading.

The cell state update $c_{\mathrm{new}} = f \cdot c_{\mathrm{prev}} + i \cdot g$ is additive, which preserves gradient flow: gradients pass through addition without scaling, and the multiplicative gates (bounded in [0, 1] by LogitNormal priors) prevent extreme gradient magnification. This additive structure is what fixes the vanishing gradient problem of vanilla RNNs, where gradients must pass through repeated composed multiplications.
