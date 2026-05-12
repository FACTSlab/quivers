# Bayesian Gaussian Mixture Model

## Overview

A Gaussian Mixture Model assigns data points to one of $k$ Gaussian components, each with its own mean and precision, weighted by mixing proportions. This example demonstrates parametric programs, the `bind` operator (`<-`), `observe` for conditioning, `map` for iteration over data, and compositional construction of priors (exponential + softmax to get a Dirichlet).

## QVR Source

```qvr
object Unit : 1
object Obs : 1

program gmm : Unit -> Obs
    mu_1 <- Normal(0.0, 3.0)
    mu_2 <- Normal(0.0, 3.0)
    mu_3 <- Normal(0.0, 3.0)
    mu_4 <- Normal(0.0, 3.0)

    tau_1 <- Gamma(2.0, 1.0)
    tau_2 <- Gamma(2.0, 1.0)
    tau_3 <- Gamma(2.0, 1.0)
    tau_4 <- Gamma(2.0, 1.0)

    let sigma_1 = 1.0 / softplus(tau_1)
    let sigma_2 = 1.0 / softplus(tau_2)
    let sigma_3 = 1.0 / softplus(tau_3)
    let sigma_4 = 1.0 / softplus(tau_4)

    weight_1 <- Exponential(1.0)
    weight_2 <- Exponential(1.0)
    weight_3 <- Exponential(1.0)
    weight_4 <- Exponential(1.0)

    let total = weight_1 + weight_2 + weight_3 + weight_4
    let p1 = weight_1 / total
    let p2 = weight_2 / total
    let p3 = weight_3 / total

    let mix_mu = p1 * mu_1 + p2 * mu_2 + p3 * mu_3 + (1.0 - p1 - p2 - p3) * mu_4
    let mix_sigma = p1 * sigma_1 + p2 * sigma_2 + p3 * sigma_3 + (1.0 - p1 - p2 - p3) * sigma_4

    observe x <- Normal(mix_mu, mix_sigma)

    return x

export gmm
```

## Walkthrough

The model fixes a four-component Gaussian mixture with priors on per-component means, scales, and mixing weights.

Each `mu_k <- Normal(0.0, 3.0)` is a scalar bind of the $k$-th component mean from a wide Normal prior. Each `tau_k <- Gamma(2.0, 1.0)` draws a positive precision parameter; the `let sigma_k = 1.0 / softplus(tau_k)` deterministic step converts the precision to a standard deviation via `softplus` (ensuring positivity) and inversion.

The four `weight_k <- Exponential(1.0)` binds draw independent Exponential(1) values; the deterministic `let` steps normalise them to a length-four simplex `(p1, p2, p3, 1 - p1 - p2 - p3)`. This is the Gamma–Dirichlet construction of a symmetric Dirichlet(1) prior over mixing weights.

The let-bindings `mix_mu` and `mix_sigma` form a soft (weighted-mean) mixture of the component parameters. The single `observe x <- Normal(mix_mu, mix_sigma)` step scores the observation against this soft mixture, accumulating a sub-probability factor on the trace.

## DSL Features

- **Bind operator (`<-`)**: Samples from the right-hand distribution and binds the result to the left-hand variable.
- **`observe`**: Conditions on observed data by multiplying the trace by the likelihood. Dual of sampling.
- **`softplus`**: Deterministic positivity-preserving transformation used inside `let` steps.
- **Arithmetic in `let`**: `+`, `-`, `*`, `/` plus built-ins compose previously bound variables into derived random variables.
- **`export`**: Marks the program as a compiled output of the module.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The composition `weight_prior >> softmax` constructs a symmetric Dirichlet distribution without naming Dirichlet as a primitive. The Exponential distribution is a morphism from the terminal object to a 1-d positive space; `stack` lifts it to $k$ independent copies; and `softmax` is a natural transformation from $\mathbb{R}^k$ to the $(k{-}1)$-simplex. Composing these yields a morphism from the terminal object to the simplex, which is exactly the symmetric Dirichlet(1). This illustrates the compositional principle: distributions usually treated as primitives in other frameworks can be decomposed into simpler morphisms and transformations.

The bind/observe duality reflects Bayes' rule at the level of morphism composition. The sequence of `<-` binds composes priors into a joint distribution over the component parameters ($* \to \Theta$). The `observe x <- Normal(mix_mu, mix_sigma)` step conditions on data, contributing a sub-probability factor in $\mathcal{G}_{\le 1}$ whose total mass is the likelihood. Inference recovers the posterior over $\Theta$ via the factorization $p(\theta, x) = p(\theta)p(x \mid \theta) = p(x)p(\theta \mid x)$.
