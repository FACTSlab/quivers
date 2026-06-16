# Linear Regression with Logistic Noise

## Overview

A linear regression whose additive observation noise follows the [Logistic distribution](https://en.wikipedia.org/wiki/Logistic_distribution) rather than the conventional Normal. The Logistic distribution is symmetric and unimodal with sub-Gaussian decay rate but heavier shoulders, making the model a smooth choice when the residual distribution is approximately bell-shaped but with slightly more mass at moderate distances than Gaussian errors would imply.

Generative structure:

$$
\begin{aligned}
s &\sim \mathrm{HalfNormal}(2),\\
\beta_0 &\sim \mathrm{Normal}(0, 5),\\
\beta_1 &\sim \mathrm{Normal}(0, 2),\\
y_n \mid \beta_0, \beta_1, s &\sim \mathrm{Logistic}(\beta_0 + \beta_1 x_n,\, s).
\end{aligned}
$$

The Logistic density is $f(y; \mu, s) = \frac{e^{-(y - \mu)/s}}{s \bigl(1 + e^{-(y - \mu)/s}\bigr)^2}$ with mean $\mu$ and variance $s^2 \pi^2 / 3$.

## QVR Source

```qvr
object Resp : FinSet 1

program logistic_regression : Resp -> Resp
    sample scale <- HalfNormal(2.0)
    sample beta_0 <- Normal(0.0, 5.0)
    sample beta_1 <- Normal(0.0, 2.0)

    let mu = beta_0 + beta_1 * x

    observe y : Resp <- Logistic(mu, scale)
    return scale

export logistic_regression
```

## Walkthrough

Two scalar regression coefficients carry Normal priors; the noise scale $s$ uses `HalfNormal(2)`. The linear predictor `mu` is bound by a `let` so the renderer materialises a deterministic relation in the transformed-parameters block (Stan, BUGS, JAGS) or as a `mu = ...` line in the trace-style backends (NumPyro, Pyro, PyMC). The observation noise is `Logistic(mu, scale)`, with the QVR family name `Logistic` translating to each backend's spelling through `FAMILY_META.target_names`.

## Use Cases

Approximation to Normal regression when the analyst suspects modest excess mass at moderate residuals: the Logistic is roughly $1.6\times$ heavier at the shoulders than Normal for the same variance. Common surrogate for the logistic-noise discrete-choice family ($Y_n = \mathbb{1}[\mu_n + \varepsilon_n > 0]$ with $\varepsilon_n$ Logistic recovers binary logistic regression).

## References

- N. Balakrishnan. 1991. *Handbook of the Logistic Distribution*. CRC Press.
