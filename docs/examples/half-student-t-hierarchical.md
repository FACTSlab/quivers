# Hierarchical Regression with HalfStudentT Scale Prior

## Overview

A Bayesian linear regression with a heavy-tailed [half-Student-t](https://en.wikipedia.org/wiki/Folded-t_and_half-t_distributions) prior on the noise scale $\sigma$. `HalfStudentT(df, scale)` is the canonical weakly-informative prior for hierarchical variance parameters: degrees of freedom $\nu$ controls how heavy the tail is (small $\nu$ favours heavy-tailed shrinkage; $\nu \to \infty$ recovers the half-Normal).

Generative structure:

$$
\begin{aligned}
\sigma &\sim \mathrm{HalfStudentT}(3, 1),\\
\beta_0 &\sim \mathrm{Normal}(0, 5),\\
\beta_1 &\sim \mathrm{Normal}(0, 2),\\
y_n \mid \beta_0, \beta_1, \sigma &\sim \mathrm{Normal}(\beta_0 + \beta_1 x_n, \sigma).
\end{aligned}
$$

## QVR Source

```qvr
object Resp : FinSet 1

program hierarchical_regression : Resp -> Resp
    sample sigma <- HalfStudentT(3.0, 1.0)
    sample beta_0 <- Normal(0.0, 5.0)
    sample beta_1 <- Normal(0.0, 2.0)

    let mu = beta_0 + beta_1 * x

    observe y : Resp <- Normal(mu, sigma)
    return sigma

export hierarchical_regression
```

## Walkthrough

`sample sigma <- HalfStudentT(3.0, 1.0)` draws the noise scale under a half-Student-t with $\nu = 3$ degrees of freedom and unit scale; the small $\nu$ admits occasional large scale draws that absorb data outliers without distorting the posterior on the regression coefficients. `beta_0` and `beta_1` carry weakly-informative Normal priors; the `let mu = ...` line binds the linear predictor; the `observe` clause scores the response under a Normal likelihood. `return sigma` projects onto the noise-scale posterior, the diagnostic the analyst most commonly inspects.

## Use Cases

Standard hierarchical-modeling pattern: any model with a learnable noise / spread parameter benefits from the half-Student-t prior over the half-Normal when the analyst expects occasional outliers or scale-shifts across subgroups. Drop-in replacement for `HalfCauchy` (which is the $\nu = 1$ limit) in models where the heavy Cauchy tail is too informative.

## References

- Andrew Gelman. 2006. Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis*, 1(3):515-534.
