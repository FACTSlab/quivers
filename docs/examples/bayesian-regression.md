# Bayesian Linear Regression

## QVR Source

```qvr
object Predictor : 1
object Response : 1

program bayesian_regression : Predictor -> Response
    sigma <- HalfCauchy(2.0)
    beta_0 <- Normal(0.0, 5.0)
    beta_1 <- Normal(0.0, 2.0)
    x <- Normal(0.0, 1.0)

    let mu = beta_0 + beta_1 * x

    observe y ~ Normal(mu, sigma)
    return y

output bayesian_regression
```

## Overview

Bayesian linear regression models $y = \beta_0 + \beta_1 x + \varepsilon$ with prior distributions on the parameters and conditions on observed data via the `observe` statement. This example demonstrates the `program` declaration, the `<-` bind operator (probabilistic sampling), deterministic `let` bindings, distribution constructors, and the `observe` conditioning statement.

## Walkthrough

`object Predictor : 1` and `object Response : 1` specify the input and output spaces, each 1-dimensional.

`program bayesian_regression : Predictor -> Response` declares a probabilistic program that takes a predictor value and produces a response value.

`sigma <- HalfCauchy(2.0)` samples the noise standard deviation from a Half-Cauchy prior with scale 2.0. The `<-` operator is the bind of the probability monad: it declares `sigma` as a random variable drawn from the given distribution. The Half-Cauchy is a heavy-tailed prior suitable for positive-valued scale parameters.

`beta_0 <- Normal(0.0, 5.0)` and `beta_1 <- Normal(0.0, 2.0)` sample the intercept and slope from Gaussian priors. The wider prior on `beta_0` reflects weaker assumptions about the intercept; the tighter prior on `beta_1` encodes a mild expectation of moderate slope.

`x <- Normal(0.0, 1.0)` samples a predictor value. In inference mode, observed predictor values would replace this; in generative mode, this produces synthetic data.

`let mu = beta_0 + beta_1 * x` computes the linear predictor deterministically. The `let` keyword signals a non-random computation; `mu` inherits its randomness from its inputs rather than being sampled independently.

`observe y ~ Normal(mu, sigma)` conditions the model on the observed response. During inference, this multiplies the posterior probability by the likelihood of the observed $y$ under $\mathrm{Normal}(\mu, \sigma)$, implementing Bayesian updating.

`return y` specifies the program's output. `output bayesian_regression` exports it.

## DSL Features

- **`program` keyword**: Declares a probabilistic program with a type signature (`InputType -> OutputType`) and a monadic body.
- **`<-` (bind)**: Samples a random variable from a distribution. Subsequent statements can depend on the sampled value.
- **Distribution constructors**: `Normal`, `HalfCauchy`, `Exponential`, `Beta`, `Categorical`, etc.
- **`let`**: Deterministic computation over random variables. The result is a derived random variable.
- **`observe`**: Conditions the model on data. Multiplies posterior probability by the observation's likelihood.
- **`return`**: Designates the program's output value.
- **Arithmetic on random variables**: `+`, `-`, `*`, `/` are supported directly.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

A probabilistic program is a Kleisli morphism $A \to TB$ in the probability monad, where $T$ maps a set to its space of distributions. The `<-` bind operator is Kleisli composition: running $f : A \to TB$ and then $g : B \to TC$ yields $(g \circ_K f) : A \to TC$. In this example, the prior (sampling $\beta_0$, $\beta_1$, $\sigma$) composes with the likelihood ($\mathrm{Normal}(\mu, \sigma)$) via monadic bind, producing a joint distribution over parameters and observations. The `observe` statement then conditions this joint distribution, computing the posterior $P(\theta \mid y) \propto P(y \mid \theta) \cdot P(\theta)$ by Bayes' rule. Inference algorithms (VI, MCMC) are computational methods for evaluating the resulting integrals.

## Connections to Graphical Models

A probabilistic program is a procedural encoding of a graphical model. Each `<-` statement is a node; each `observe` statement is an observed variable. In this example: $\beta_0$, $\beta_1$, and $\sigma$ are root nodes (no parents), $\mu$ is computed from $\beta_0$, $\beta_1$, and $x$, and $y$ depends on $\mu$ and $\sigma$.

Quivers abstracts over inference algorithms, so the same model specification works with VI, MCMC, or other methods.

## Extensions and Advanced Usage

For multi-dimensional regression, add predictor variables and coefficients. For hierarchical models, nest probabilistic programs (samples from one become parameters of another). For Bayesian nonparametrics, use distributions like the Dirichlet process. A Bayesian linear regression program can serve as a component in a larger hierarchical model or be extended with non-linear transformations and richer likelihood models.
