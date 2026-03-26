# Hidden Markov Model (Discrete)

## Overview

A Hidden Markov Model tracks a discrete latent state that evolves via a transition kernel and emits observations at each step. This example demonstrates the `quantale` declaration, discrete `object` spaces, the `repeat` combinator for runtime-variable sequence length, and composition of stochastic morphisms.

## QVR Source

```qvr
quantale product_fuzzy

object State : 8
object Obs : 16

stochastic initial : State -> State
stochastic transition : State -> State
stochastic emission : State -> Obs

let n_step = repeat(transition) >> emission
let hmm = initial >> n_step

output hmm
```

## Walkthrough

`quantale product_fuzzy` sets the algebraic rule for combining probabilities during composition: the product quantale multiplies probabilities along a path, which is standard probability theory. A different quantale would change how path weights combine (see the categorical perspective below for the Viterbi example).

`object State : 8` and `object Obs : 16` declare finite discrete spaces. The `object` keyword (as opposed to `type ... = Euclidean`) means these are finite sets, not continuous manifolds. Stochastic morphisms between objects correspond to stochastic matrices.

`stochastic initial : State -> State` defines the initial state distribution. Despite having `State` as both domain and codomain, its role is to produce a distribution over starting states. `stochastic transition : State -> State` is the Markov transition kernel: given a current state, it returns a distribution over successor states. `stochastic emission : State -> Obs` maps a state to a distribution over observations.

`repeat(transition)` produces a morphism that applies `transition` a number of times determined at runtime (contrast with `stack`, which fixes the count at compile time). Composing with `emission` via `>>` gives `n_step`: apply transitions, then emit. `let hmm = initial >> n_step` prepends the initial distribution, yielding a complete generative model from no input to an observation sequence.

## DSL Features

- **`quantale`**: Declares how probabilities compose. `product_fuzzy` gives standard multiplicative composition.
- **`object` vs `type`**: `object` is a finite discrete space (stochastic matrices); `type ... = Euclidean` is continuous (density functions).
- **`stochastic` keyword**: Marks morphisms as probabilistic but not necessarily differentiable (contrast with `continuous`).
- **`repeat(f)`**: Runtime-variable iteration of a morphism. The repetition count is set when the model is invoked, not at compile time.
- **Kleisli composition (`>>`)**: Chains stochastic morphisms. `initial >> repeat(transition) >> emission` is itself a stochastic morphism.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

Swapping the quantale from `product_fuzzy` to a tropical (max-plus) semiring turns the forward algorithm into the Viterbi algorithm. Under the product quantale, composing transition matrices multiplies probabilities along paths, and summing over intermediate states computes total path probability. Under the tropical quantale, multiplication becomes addition of log-probabilities and summation becomes maximization, so the same composition structure finds the most-likely path instead. The categorical framework makes this explicit: the forward and Viterbi algorithms are the same morphism composition evaluated in different quantales.

The `repeat` combinator implements iterated Kleisli composition of the transition endomorphism. Because Kleisli composition is associative, the $n$-fold composition $\mathrm{transition}^n$ is well-defined regardless of grouping, and the `initial` morphism (a map from the terminal object to `State`) provides the entry point that turns the whole pipeline into a generative model producing observations with no external input.
