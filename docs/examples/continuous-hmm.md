# Continuous State-Space Model

## Overview

A continuous state-space model extends the HMM to continuous latent states and observations, with a state transition function and an observation function, both stochastic. This example demonstrates the `scan` combinator for threading state through a sequence, the `observe` statement for Bayesian filtering, and the separation of generative and inference programs over the same morphisms.

## QVR Source

```qvr
quantale product_fuzzy

type State = Euclidean 4
type Obs = Euclidean 2
type Noise = Euclidean 4
type Measurement = Euclidean 2

program generative_step(state_t_minus_1) {

  noise <- continuous ~ Normal
  state_t <- state_transition(state_t_minus_1, noise)
  obs_noise <- continuous ~ Normal
  obs_t <- observation_function(state_t, obs_noise)
  return (state_t, obs_t)
}

continuous state_transition : (State, Noise) -> State ~ Normal
continuous observation_function : (State, Measurement) -> Obs ~ Normal
continuous initial_state : UnitSpace -> State ~ Normal

program generative {

  state_0 <- initial_state
  states_and_obs <- scan(generative_step, range(n_steps), state_0)
  return states_and_obs
}

program inference_step(belief_t_minus_1, obs_t) {

  predicted_state <- state_transition(belief_t_minus_1, draw Normal)
  observe obs_t ~ observation_function(predicted_state, draw Normal)
  return predicted_state
}

program filter {

  observations <- load_observations(n_steps)
  state_0 <- initial_state
  beliefs <- scan(inference_step, observations, state_0)
  return beliefs
}

program decoder(latent_sequence) {

  reconstructed <- map(observation_function, latent_sequence)
  return reconstructed
}

output generative
```

## Walkthrough

`quantale product_fuzzy` sets multiplicative probability composition, as in the discrete HMM.

The type declarations define four Euclidean spaces: `State` (4-d latent), `Obs` (2-d observed), `Noise` (4-d process noise), and `Measurement` (2-d observation noise). Separating noise types from state/observation types makes the roles explicit.

`state_transition : (State, Noise) -> State ~ Normal` evolves the latent state by one time step given process noise. In a linear system this would implement $s_t = A \cdot s_{t-1} + \text{noise}$. `observation_function : (State, Measurement) -> Obs ~ Normal` projects the state to the observation space. `initial_state : UnitSpace -> State ~ Normal` provides the prior over the starting state.

`generative_step` advances by one time step: draw process noise, apply `state_transition`, draw observation noise, apply `observation_function`, return both the new state and the observation. The `generative` program samples an initial state, then uses `scan(generative_step, range(n_steps), state_0)` to thread the state through the full sequence. `scan` repeatedly applies the step function, passing each step's output state as the next step's input, and collects all intermediate results.

`inference_step` performs one Bayesian filtering update: predict the next state via `state_transition`, then condition on the actual observation with `observe obs_t ~ observation_function(predicted_state, draw Normal)`. The `observe` statement multiplies the predicted distribution by the observation likelihood, implementing the Bayesian update. The `filter` program loads observations and scans `inference_step` through them, producing a sequence of posterior beliefs.

`decoder` applies `observation_function` via `map` to a latent sequence, generating observations without any state threading. This is useful for reconstructing observations from inferred latents.

## DSL Features

- **`scan(f, sequence, init)`**: Threads state through a sequence by repeatedly applying `f`. The step function receives the previous state and the current sequence element, and returns the next state. This is a probabilistic fold.
- **`draw Normal`**: Inline sampling shorthand, used when the sample is consumed immediately and does not need a name.
- **`observe`**: Conditions the computation on an observed value. Dual of sampling: instead of generating data, it incorporates evidence.
- **`map(f, sequence)`**: Applies `f` to each element independently (no state threading). Used in the decoder.
- **`continuous` keyword**: Marks morphisms as differentiable, enabling reparameterization for gradient-based learning.
- **Subprogram parameters**: `generative_step(state_t_minus_1)` and `inference_step(belief_t_minus_1, obs_t)` accept arguments, enabling `scan` to thread state through them.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The `scan` combinator implements Kleisli composition threaded through time. Given a step morphism $f : S \to S$ in the Kleisli category (where $S$ carries both state and noise), `scan` produces the $n$-fold composition $f^n$ while collecting all intermediate results. Because Kleisli composition is associative, the computation decomposes into local single-step updates, which is why online/streaming inference works: each filtering step depends only on the previous belief and the current observation, not on the full history.

The generative and filtering programs apply the same underlying morphisms (`state_transition`, `observation_function`) but differ in direction. The generative process composes $* \to S$ (initial state) with the step morphism to produce states and observations. The filtering process takes observations as input and uses `observe` to invert the observation morphism, recovering a posterior over states. This inversion is Bayes' rule expressed as conditioning in the Kleisli category, and the `scan` combinator threads it through the full sequence.
