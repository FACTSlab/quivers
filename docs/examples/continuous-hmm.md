# Continuous State-Space Model

## Overview

A continuous state-space model extends the HMM to continuous latent states and observations, with a state transition function and an observation function, both stochastic. This example demonstrates the `scan` combinator for threading state through a sequence, the `observe` statement for Bayesian filtering, and the separation of generative and inference programs over the same morphisms.

## QVR Source

```qvr
type State = Euclidean 16
type Obs = Euclidean 8

kernel transition : State -> State ~ Normal [scale=0.1]
kernel emission : State -> Obs ~ Normal [scale=0.1]

program generative_step : State -> State
    s_new <- transition
    observe o <- emission(s_new)
    return s_new

kernel inference_cell : Obs * State -> State ~ Normal [scale=0.1]

let filter = scan(inference_cell)

kernel decoder : State -> Obs ~ Normal [scale=0.1]

let filter_and_reconstruct = scan(inference_cell) >> decoder

export filter_and_reconstruct
```

## Walkthrough

The type declarations introduce two Euclidean spaces: `State` (16-d latent) and `Obs` (8-d observed).

`continuous transition : State -> State ~ Normal [scale=0.1]` evolves the latent state by one time step, and `continuous emission : State -> Obs ~ Normal [scale=0.1]` projects a state to an observation. Both are differentiable Normal-conditional morphisms.

`generative_step` is a one-step monadic program: bind a new state from `transition`, score the observation `o <- emission(s_new)` against the runtime observations dict, and return the new state. To unroll over time, this program is composed with itself via `repeat` or threaded through `scan`.

`inference_cell : Obs * State -> State ~ Normal [scale=0.1]` is a recurrent cell that incorporates a new observation into the running state estimate. `let filter = scan(inference_cell)` constructs a temporal-recurrence morphism that threads state across a sequence of observations: for an input of shape `(batch, seq_len, obs_dim)`, it returns the final state estimate of shape `(batch, state_dim)`.

`decoder : State -> Obs ~ Normal [scale=0.1]` decodes a state back to observation space; composing `scan(inference_cell) >> decoder` produces `filter_and_reconstruct`, the exported pipeline that filters then reconstructs.

## DSL Features

- **`scan(cell)`**: Threads hidden state across a sequence by repeatedly applying `cell : A * H -> H`. For an input of shape `(batch, seq_len, input_dim)`, returns the final hidden state of shape `(batch, hidden_dim)`.
- **Bind operator `<-`**: The unique sampling-step sigil; samples from the right-hand morphism and binds the result.
- **`observe v <- F(args)`**: Conditions the computation on an externally-supplied value via the runtime `observations` dict. Dual of sampling.
- **`continuous` keyword**: Marks morphisms as differentiable, enabling reparameterization for gradient-based learning.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The `scan` combinator implements Kleisli composition threaded through time. Given a step morphism $f : S \to S$ in the Kleisli category (where $S$ carries both state and noise), `scan` produces the $n$-fold composition $f^n$ while collecting all intermediate results. Because Kleisli composition is associative, the computation decomposes into local single-step updates, which is why online/streaming inference works: each filtering step depends only on the previous belief and the current observation, not on the full history.

The generative and filtering programs apply the same underlying morphisms (`state_transition`, `observation_function`) but differ in direction. The generative process composes $* \to S$ (initial state) with the step morphism to produce states and observations. The filtering process takes observations as input and uses `observe` to invert the observation morphism, recovering a posterior over states. This inversion is Bayes' rule expressed as conditioning in the Kleisli category, and the `scan` combinator threads it through the full sequence.
