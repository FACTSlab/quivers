# Stochastic Volatility

## Overview

The canonical log-volatility return model of [Kim, Shephard, and Chib (1998)](https://doi.org/10.1111/1467-937X.00050). The latent log-volatility follows an [AR(1)](https://en.wikipedia.org/wiki/Autoregressive_model) chain centered on a mean `mu` with autoregressive coefficient `phi`, and the observed return is mean-zero Normal with time-varying scale `exp(h_t / 2)`. The exponential link makes the volatility positive by construction.

## QVR Source

```qvr
object Step : FinSet 200

program stochastic_volatility : Step -> Step
    sample mu <- Normal(0.0, 10.0)
    sample phi <- Uniform(-1.0, 1.0)
    sample sigma_h <- HalfCauchy(2.5)

    let h_mean = mu + phi * (h_prev - mu)

    sample h : Step <- Normal(h_mean, sigma_h)

    let scale = exp(0.5 * h)

    observe r : Step <- Normal(0.0, scale)
    return phi

export stochastic_volatility
```

## Walkthrough

`mu`, `phi`, and `sigma_h` are the AR(1) hyperparameters of the log-volatility chain; `phi` is constrained to the [stationarity interval](https://en.wikipedia.org/wiki/Stationary_process) `(-1, 1)`. The identifier `h_prev` is exogenous host-data: it is never declared inside the program, so the runtime resolves it from the observations dict at trace time, where the caller supplies the lagged latent log-volatility. The current-step `h` is a latent draw with mean `mu + phi * (h_prev - mu)` realising the AR(1) recursion, and `exp(0.5 * h)` is the standard SV link to the per-step return scale. The observed returns are mean-zero Normal scaled by the time-varying volatility.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

Pick ground-truth log-volatility dynamics, simulate the AR(1) chain `h_t`, then draw mean-zero Normal returns whose scale is `exp(h_t / 2)`. The lagged log-volatility `h_prev` is the host-data the program reads from the observations dict.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/stochastic_volatility.qvr")
model = prog.morphism

T = 200
mu_true, phi_true, sigma_h_true = 0.0, 0.95, 0.25
h = torch.zeros(T)
h[0] = mu_true
for t in range(1, T):
    h[t] = mu_true + phi_true * (h[t - 1] - mu_true) + sigma_h_true * torch.randn(())
h_prev = torch.cat([torch.zeros(1), h[:-1]])
returns = torch.exp(0.5 * h) * torch.randn(T)
x_in = torch.zeros(T, 1)
observations = {"r": returns, "h_prev": h_prev}
```

### SVI fit

Re-initialise the program and fit the AR(1) hyperparameters by maximising the ELBO. The negative ELBO drops over the run; the oracle log-likelihood at the true `(mu, phi, sigma_h, h)` is reported for reference.

```python
import torch.distributions as D
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/stochastic_volatility.qvr")
model = prog.morphism

guide = AutoNormalGuide(model, observed_names={"r"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO())

loss0 = svi.step(x_in, observations)
losses = [svi.step(x_in, observations) for _ in range(400)]
loss_final = sum(losses[-20:]) / 20.0
oracle_ll = D.Normal(0.0, torch.exp(0.5 * h)).log_prob(returns).sum().item()
print(f"initial ELBO loss: {loss0:.1f}")
print(f"final ELBO loss:   {loss_final:.1f}")
print(f"oracle -log p(r):  {-oracle_ll:.1f}")
```

### NUTS posterior

The program declares explicit `sample` sites for `mu`, `phi`, `sigma_h`, and the latent log-volatility plate `h`, so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them directly under small warmup and sample budgets.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(0)
prog = load("docs/examples/source/stochastic_volatility.qvr")
model = prog.morphism

kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
print(f"mu posterior mean:      {result.samples['mu'].mean().item():.3f}")
print(f"phi posterior mean:     {result.samples['phi'].mean().item():.3f}")
print(f"sigma_h posterior mean: {result.samples['sigma_h'].mean().item():.3f}")
```


## Categorical Perspective

The model is a Kleisli morphism over the latent log-volatility plate, composed with a per-step Normal observation kernel whose scale depends on the latent. In the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category, the chain `h_prev -> h -> r` is associative Kleisli composition; the SVI guide approximates the joint posterior $p(\mu, \phi, \sigma_h, h \mid r)$.
