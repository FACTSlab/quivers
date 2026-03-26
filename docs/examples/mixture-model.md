# Bayesian Gaussian Mixture Model

## Overview

A Gaussian Mixture Model assigns data points to one of $k$ Gaussian components, each with its own mean and precision, weighted by mixing proportions. This example demonstrates parametric programs, the `bind` operator (`<-`), `observe` for conditioning, `map` for iteration over data, and compositional construction of priors (exponential + softmax to get a Dirichlet).

## QVR Source

```qvr
program gmm {

  n_data: int
  k_components: int = 4
  data_dim: int = 2
}

let n_data_range = range(n_data)

type Precision = Euclidean 1
type Mean = Euclidean data_dim
type MixtureWeights = Euclidean k_components
type Data = Euclidean data_dim

continuous shape_prior : UnitSpace -> Precision ~ Gamma(shape=2.0, rate=0.1)
continuous rate_prior : UnitSpace -> Precision ~ Exponential(rate=0.5)

let component_precision = shape_prior >> softplus

continuous mean_prior : UnitSpace -> Mean ~ Normal

let component_means = stack(mean_prior, k_components)

continuous weight_prior : UnitSpace -> MixtureWeights ~ Exponential(rate=1.0)

let mixture_weights = weight_prior >> softmax

continuous likelihood : (Mean, Precision) -> Data ~ Normal

program generative_step(i) {

  z <- categorical(mixture_weights)
  mu <- component_means[z]
  tau <- component_precision
  x <- likelihood(mu, tau)
  return x
}

program generative_process {

  precisions <- component_precision.sample()
  means <- component_means.sample()
  weights <- mixture_weights.sample()
  data <- map(generative_step, n_data_range)
  return data
}

program inference_step(x_i, i) {

  z <- categorical(mixture_weights)
  mu <- component_means[z]
  tau <- component_precision
  observe x_i ~ likelihood(mu, tau)
}

program inference {

  precisions <- component_precision.sample()
  means <- component_means.sample()
  weights <- mixture_weights.sample()
  observations <- load_data(n_data_range)
  map(inference_step, zip(observations, n_data_range))
  return (precisions, means, weights, observations)
}

output generative_process
```

## Walkthrough

The `program gmm { ... }` block declares runtime parameters: `n_data` (number of observations), `k_components` (default 4), and `data_dim` (default 2). These are not known at compile time.

Type declarations set up the spaces: `Precision` is 1-d (scalar), `Mean` and `Data` are `data_dim`-dimensional, and `MixtureWeights` is `k_components`-dimensional.

The precision prior is `shape_prior` (Gamma with shape=2.0, rate=0.1, giving a mean around 20) composed with `softplus` to guarantee positivity. An alternative exponential prior (`rate_prior`) is also declared.

The mean prior is a standard normal, and `stack(mean_prior, k_components)` produces `k_components` independent draws.

The mixture weight prior is constructed compositionally: `weight_prior` draws `k_components` values from Exponential(1.0), then `softmax` normalizes them to sum to one. Drawing $k$ independent Exponential(1) samples and normalizing produces a symmetric Dirichlet(1) distribution, so the Dirichlet is built from simpler pieces rather than declared as a primitive.

`likelihood : (Mean, Precision) -> Data ~ Normal` defines the per-component observation distribution.

`generative_step` generates one data point: sample a component index `z` from `categorical(mixture_weights)`, look up `component_means[z]` and `component_precision`, then draw from the likelihood. `generative_process` samples the global parameters once, then `map(generative_step, n_data_range)` generates all data points.

`inference_step` mirrors the generative step but replaces sampling with `observe x_i ~ likelihood(mu, tau)`, which conditions on the observed data point by multiplying the current density by the likelihood. `inference` loads real data and maps `inference_step` over it, returning posterior samples of all parameters.

## DSL Features

- **`program` block with parameters**: Parametric stochastic computations with runtime-variable inputs (`n_data`, `k_components`).
- **Bind operator (`<-`)**: Draws a sample from the right-hand distribution and binds it to the left-hand variable.
- **`observe`**: Conditions the computation on observed data by multiplying in the likelihood. Dual of sampling.
- **`softplus` / `softmax`**: Deterministic transformations composed with stochastic morphisms to enforce constraints (positivity, normalization).
- **`categorical(weights)`**: Discrete distribution over component indices, parameterized by mixture weights.
- **Indexing (`component_means[z]`)**: Selects from an array of parameters using a discrete random variable.
- **`map(f, sequence)`**: Applies a subprogram to each element of a sequence. Used for both generation and inference over data points.
- **`stack(f, k)`**: Produces `k` independent copies of morphism `f` (here, `k` independent mean priors).

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The composition `weight_prior >> softmax` constructs a symmetric Dirichlet distribution without naming Dirichlet as a primitive. The Exponential distribution is a morphism from the terminal object to a 1-d positive space; `stack` lifts it to $k$ independent copies; and `softmax` is a natural transformation from $\mathbb{R}^k$ to the $(k{-}1)$-simplex. Composing these yields a morphism from the terminal object to the simplex, which is exactly the symmetric Dirichlet(1). This illustrates the compositional principle: distributions usually treated as primitives in other frameworks can be decomposed into simpler morphisms and transformations.

The duality between `generative_process` and `inference` reflects Bayes' rule at the level of morphism composition. The generative process composes priors with the likelihood to produce a joint distribution over parameters and data ($* \to \Theta \times X$). The inference process reverses the direction by using `observe` to condition on data, producing a posterior over parameters ($X \to \Theta$). These are related by the factorization $p(\theta, x) = p(\theta)p(x \mid \theta) = p(x)p(\theta \mid x)$.
