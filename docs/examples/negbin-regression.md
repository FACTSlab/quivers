# Multi-Output Negative Binomial Regression

## Overview

A multi-output [negative-binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution) regression for overdispersed count data, using the mean / dispersion parameterization that follows the log link convention shared with [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression). Each output dimension carries its own coefficients and dispersion; the response is the standard [NB2](https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations) form with per-cell variance `mu + mu^2 / dispersion`, recovering Poisson in the limit of infinite dispersion.

## QVR Source

```qvr
object Item : FinSet 200
object Out : FinSet 3
object Resp : FinSet 600

program negbin_regression : Resp -> Resp
    sample beta_0 : Out <- Normal(0.0, 5.0)
    sample beta_1 : Out <- Normal(0.0, 5.0)
    sample dispersion : Out <- Gamma(2.0, 0.5)

    let b0 = beta_0[out_idx]
    let b1 = beta_1[out_idx]
    let disp = dispersion[out_idx]
    let eta = b0 + b1 * x
    let mu = exp(eta)
    let probs = disp / (disp + mu)

    observe y : Resp <- NegativeBinomial(disp, probs)
    return beta_1

export negbin_regression
```

## Walkthrough

Per-output coefficient and dispersion plates broadcast through `out_idx` gathers. The per-cell linear predictor `eta = b0 + b1 * x` is mapped through the log link `exp` to give the conditional mean `mu`. The NB2 parameterization uses `probs = dispersion / (dispersion + mu)` so the resulting `NegativeBinomial(dispersion, probs)` has mean `mu` and variance `mu * (1 + mu / dispersion)`. The Gamma prior on dispersion encodes a soft preference for finite overdispersion; per-output dispersion permits heterogeneous count regimes across the response axis.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/negbin_regression.qvr")
model = prog.morphism

N, D = 21, 3
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_beta_0 = torch.tensor([1.0, 0.5, 2.0])
true_beta_1 = torch.tensor([0.5, -0.3, 0.8])
true_dispersion = torch.tensor([5.0, 10.0, 3.0])
mu_true = torch.exp(true_beta_0[out_idx] + true_beta_1[out_idx] * x)
probs_true = true_dispersion[out_idx] / (true_dispersion[out_idx] + mu_true)
y = torch.distributions.NegativeBinomial(true_dispersion[out_idx], probs_true).sample()

observations = {"x": x, "y": y, "out_idx": out_idx}
x_in = torch.zeros(ND, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.NegativeBinomial(true_dispersion[out_idx], probs_true)
    .log_prob(y)
    .mean()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "y", "out_idx"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = []
for _ in range(300):
    losses.append(svi.step(x_in, observations))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
print(f"oracle NLL:   {oracle_nll:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

N_mcmc, D_mcmc = 11, 3
ND_mcmc = N_mcmc * D_mcmc
out_idx_mcmc = torch.arange(D_mcmc).repeat(N_mcmc)
x_mcmc = x[:ND_mcmc]
y_mcmc = y[:ND_mcmc]
obs_mcmc = {"x": x_mcmc, "y": y_mcmc, "out_idx": out_idx_mcmc}
x_in_mcmc = torch.zeros(ND_mcmc, 1)

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in_mcmc, obs_mcmc)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical Perspective

The negative binomial is the Gamma-Poisson [mixture](https://en.wikipedia.org/wiki/Compound_probability_distribution): a `Poisson(rate)` kernel with `rate ~ Gamma(dispersion, dispersion / mu)` marginalizes to `NegativeBinomial(dispersion, mu / (mu + dispersion))`. The model factors through this mixture by sampling per-cell from the closed-form negative binomial; categorically the family is the pushforward of the Gamma-Poisson joint kernel along the rate-projection.
