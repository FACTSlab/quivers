# Linear Regression with Logistic Noise

## Overview

A linear regression whose additive observation noise follows the [Logistic distribution](https://en.wikipedia.org/wiki/Logistic_distribution) rather than the conventional Normal. The Logistic distribution is symmetric and unimodal with exponential tails, which are heavier than Gaussian tails.

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

## QVR source

```qvr
# Linear Regression with Logistic Noise
#
# A linear regression whose additive observation noise follows the
# Logistic distribution rather than the conventional Normal. The
# Logistic distribution has sub-Gaussian tails (kurtosis 1.2 vs
# Normal's 0) but heavier shoulders, making the model a smoother
# alternative when the response distribution has a touch more
# excess mass than Gaussian errors would imply.
#
# Generative structure:
#
#   scale   ~ HalfNormal(2)                 Logistic scale, half-Normal prior
#   beta_0  ~ Normal(0, 5)                  intercept
#   beta_1  ~ Normal(0, 2)                  slope
#   y_n     ~ Logistic(beta_0 + beta_1 * x_n, scale)
#
# Reference: [Balakrishnan 1991](https://doi.org/10.1201/9781482277098).

object Resp : FinSet 64
object Val : Real 1

program logistic_noise_regression : Resp -> Val
    sample scale <- HalfNormal(2.0)
    sample beta_0 <- Normal(0.0, 5.0)
    sample beta_1 <- Normal(0.0, 2.0)

    let mu = beta_0 + beta_1 * x

    observe y : Resp <- Logistic(mu, scale)
    return scale

export logistic_noise_regression
```

## Walkthrough

Two scalar regression coefficients carry Normal priors; the noise scale $s$ uses `HalfNormal(2)`. The linear predictor `mu` is bound by a `let` so the renderer materialises a deterministic relation in the transformed-parameters block (Stan, BUGS, JAGS) or as a `mu = ...` line in the trace-style backends (NumPyro, Pyro, PyMC). The observation noise is `Logistic(mu, scale)`, with the QVR family name `Logistic` translating to each backend's spelling through `FAMILY_META.target_names`.

The program returns `scale`, a scalar real, so the declared codomain is `Val : Real 1`: the value space of what comes back. `Resp` names the plate extent of the response, sized at 64 rows, and appears in the signature only as the domain.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.

### Generating synthetic data

Draw a predictor, form the linear predictor at known coefficients, and add Logistic noise through the inverse CDF $\mu + s\,\log\bigl(u / (1 - u)\bigr)$ with $u \sim \mathrm{Uniform}(0, 1)$. The `Resp` plate is sized at 64, so the snippet generates 64 rows.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/logistic_noise_regression.qvr")
model = prog.morphism

N = 64
true_scale = 0.4
true_beta_0 = 1.0
true_beta_1 = -1.5

x = torch.randn(N)
mu_true = true_beta_0 + true_beta_1 * x
u = torch.rand(N)
y = mu_true + true_scale * (torch.log(u) - torch.log1p(-u))
observations = {"x": x, "y": y}
x_in = torch.zeros(N, 1)
```

### SVI fit

The ELBO loss carries the prior and entropy terms alongside the likelihood, so it sits above the oracle negative log-likelihood even at convergence; what the fit demonstrates is the descent, not a match to the oracle.

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

z = (y - mu_true) / true_scale
oracle_nll = float(
    (torch.log(torch.tensor(true_scale))
     + z
     + 2.0 * torch.nn.functional.softplus(-z)).sum()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = []
for _ in range(300):
    losses.append(svi.step(x_in, observations))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
print(f"oracle -log p(y): {oracle_nll:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

N_mcmc = 32
x_mcmc = x[:N_mcmc]
y_mcmc = y[:N_mcmc]
obs_mcmc = {"x": x_mcmc, "y": y_mcmc}
x_in_mcmc = torch.zeros(N_mcmc, 1)

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in_mcmc, obs_mcmc)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

## Use cases

Approximation to Normal regression when the analyst suspects modest excess mass at moderate residuals: the Logistic is roughly $1.6\times$ heavier at the shoulders than Normal for the same variance. Common surrogate for the logistic-noise discrete-choice family ($Y_n = \mathbb{1}[\mu_n + \varepsilon_n > 0]$ with $\varepsilon_n$ Logistic recovers binary logistic regression).

## References

- N. Balakrishnan. 1991. *Handbook of the Logistic Distribution*. CRC Press.
