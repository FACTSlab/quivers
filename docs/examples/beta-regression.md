# Multi-Output Beta Regression

## Overview

A multi-output beta regression ([Ferrari and Cribari-Neto 2004](https://doi.org/10.1080/0266476042000214501)) for response variables on the open unit interval. Each output dimension carries its own logit-link coefficients and precision; the per-cell mean is mapped through the [sigmoid](https://en.wikipedia.org/wiki/Logistic_function) link to the unit interval, and the [Beta](https://en.wikipedia.org/wiki/Beta_distribution) likelihood is reparameterised in mean / precision form.

## QVR Source

```qvr
object Item : FinSet 200
object Out : FinSet 3
object Resp : FinSet 600

program beta_regression : Resp -> Resp
    sample beta_0 : Out <- Normal(0.0, 5.0)
    sample beta_1 : Out <- Normal(0.0, 5.0)
    sample phi : Out <- HalfCauchy(2.5)

    let b0 = beta_0[out_idx]
    let b1 = beta_1[out_idx]
    let p = phi[out_idx]
    let eta = b0 + b1 * x
    let mu = sigmoid(eta)
    let alpha = mu * p
    let beta_p = (1.0 - mu) * p

    observe y : Resp <- Beta(alpha, beta_p)
    return beta_1

export beta_regression
```

## Walkthrough

Per-output coefficient plates `beta_0 : Out` and `beta_1 : Out` carry one coefficient per response dimension; per-output precision `phi : Out` permits heterogeneous dispersion across the response axis. The per-cell linear predictor `eta = b0 + b1 * x` is mapped to the unit interval via `mu = sigmoid(eta)`. The mean / precision form of the Beta is `Beta(mu * phi, (1 - mu) * phi)`, giving mean `mu` and variance `mu * (1 - mu) / (1 + phi)`. Plate-gathers `beta_0[out_idx]`, `beta_1[out_idx]`, `phi[out_idx]` broadcast each output's coefficients across its share of the flat `Resp` plate.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/beta_regression.qvr")
model = prog.morphism

N, D = 21, 3
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_beta_0 = torch.tensor([0.5, -0.3, 0.0])
true_beta_1 = torch.tensor([1.0, -1.5, 0.8])
true_phi = torch.tensor([10.0, 20.0, 15.0])
mu_true = torch.sigmoid(true_beta_0[out_idx] + true_beta_1[out_idx] * x)
alpha_true = mu_true * true_phi[out_idx]
beta_true = (1.0 - mu_true) * true_phi[out_idx]
y = torch.distributions.Beta(alpha_true, beta_true).sample()

observations = {"x": x, "y": y, "out_idx": out_idx}
x_in = torch.zeros(ND, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Beta(alpha_true, beta_true).log_prob(y).mean()
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

The model is a [Kleisli morphism](https://ncatlab.org/nlab/show/Kleisli+category) in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category whose codomain factors through the [unit interval](https://en.wikipedia.org/wiki/Logistic_function); the per-cell Beta likelihood factors through the link `sigmoid` as a `1 -> G((0, 1))` kernel. The plate-gather `beta_1[out_idx]` is the Kleisli pullback of the `Out`-indexed plate along the fibration `Resp -> Out` carried by the runtime index.


## References

- Silvia Ferrari and Francisco Cribari-Neto. 2004. Beta regression for modelling rates and proportions. *Journal of Applied Statistics*, 31(7):799–815.
