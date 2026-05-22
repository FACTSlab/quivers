# Weibull Survival Regression

## Overview

A parametric [survival regression](https://en.wikipedia.org/wiki/Survival_analysis) with a [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) baseline ([Klein and Moeschberger 2003](https://doi.org/10.1007/b97377)). Each item carries a covariate `x_i` whose linear contribution scales the Weibull rate; the shape parameter `k` governs whether the hazard is decreasing (k < 1), constant (k = 1, exponential), or increasing (k > 1).

## QVR Source

```qvr
object Item : FinSet 200

program survival_weibull : Item -> Item
    sample alpha <- Normal(0.0, 5.0)
    sample beta <- Normal(0.0, 5.0)
    sample k <- Gamma(2.0, 1.0)

    let eta = alpha + beta * x
    let scale = exp(-eta / k)

    observe t : Item <- Weibull(scale, k)
    return beta

export survival_weibull
```

## Walkthrough

The identifier `x` is the exogenous covariate: it is never declared inside the program, so the runtime resolves it from the observations dict at trace time, where the caller supplies the per-item predictor. The reparameterisation `scale = exp(-eta / k)` is the Weibull [proportional-hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model) convention: positive shifts in the linear predictor `eta = alpha + beta * x` increase the hazard and shorten survival times, matching the canonical direction. The shape `k` has a Gamma prior centered at 2. The observed event times `t` are uncensored Weibull draws; right-censoring is handled at the inference layer by substituting the [Weibull survival function](https://en.wikipedia.org/wiki/Weibull_distribution#Survival_function) for the density on censored rows.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/survival_weibull.qvr")
model = prog.morphism

N = 64
true_alpha = 0.5
true_beta = 1.0
true_k = 2.0

x = torch.randn(N)
eta_true = true_alpha + true_beta * x
scale_true = torch.exp(-eta_true / true_k)
t = torch.distributions.Weibull(scale_true, true_k).sample()

observations = {"x": x, "t": t}
x_in = torch.zeros(N, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Weibull(scale_true, true_k).log_prob(t).mean()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "t"})
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

N_mcmc = 32
x_mcmc = x[:N_mcmc]
t_mcmc = t[:N_mcmc]
obs_mcmc = {"x": x_mcmc, "t": t_mcmc}
x_in_mcmc = torch.zeros(N_mcmc, 1)

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in_mcmc, obs_mcmc)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical Perspective

The model denotes a Kleisli morphism into the positive reals in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category. The Weibull is the [exponential](https://en.wikipedia.org/wiki/Exponential_distribution) family generalization with shape; the proportional-hazards link makes the model canonical in the [exponential family](https://en.wikipedia.org/wiki/Exponential_family) representation.


## References

- John P. Klein and Melvin L. Moeschberger. 2003. *Survival Analysis: Techniques for Censored and Truncated Data*, 2nd edition. Springer.
