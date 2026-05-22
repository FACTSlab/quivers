# Gamma Regression

## Overview

A Bayesian regression for non-negative component totals, fit with a [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) likelihood and a per-category log-rate predictor. The model uses K = 3 categories: each category's log-shape is a linear function of the predictor, and the response is the raw per-cell Gamma tally rather than its simplex projection. This is the unnormalized layer of a Dirichlet regression ([Maier 2014](https://doi.org/10.32614/CRAN.package.DirichletReg)): independent Gamma draws with shapes `(alpha_1, ..., alpha_K)` and unit rate normalize to a sample on the [K-simplex](https://en.wikipedia.org/wiki/Simplex) via the [Gamma / Dirichlet relationship](https://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution), but here the program observes the Gamma layer directly and leaves the deterministic normalization to downstream code.

## QVR Source

```qvr
object Item : FinSet 200
object Cat : FinSet 3
object Resp : FinSet 600

program gamma_regression : Resp -> Resp
    sample beta_0 : Cat <- Normal(0.0, 2.0)
    sample beta_1 : Cat <- Normal(0.0, 2.0)

    let b0 = beta_0[cat_idx]
    let b1 = beta_1[cat_idx]
    let alpha = exp(b0 + b1 * x)

    observe tally : Resp <- Gamma(alpha, 1.0)
    return beta_1

export gamma_regression
```

## Walkthrough

Per-category coefficient plates `beta_0 : Cat` and `beta_1 : Cat` carry one coefficient per category. The per-cell log-shape `b0 + b1 * x` is mapped through the exponential to give the positive Gamma shape `alpha`. The observed `tally : Resp <- Gamma(alpha, 1.0)` lives on the per-cell positive reals, indexed by the flattened `(Item, Cat)` axis. Plate-gather `beta_0[cat_idx]` selects each cell's category coefficient from the per-category plate.

To turn this into a Dirichlet regression on the K-simplex, normalize the tallies per Item: `y_{n, k} = tally_{n, k} / sum_j tally_{n, j}`. That projection is a deterministic post-composition on the Gamma response, applied outside the program because the QVR `Resp` axis is the flattened `(Item, Cat)` index and does not expose the K-axis as a named dimension to a `let`-level reducer.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/gamma_regression.qvr")
model = prog.morphism

N, K = 21, 3
NK = N * K
cat_idx = torch.arange(K).repeat(N)
x = torch.randn(NK)

true_b0 = torch.tensor([1.0, 0.5, -0.5])
true_b1 = torch.tensor([0.8, -1.0, 0.3])
alpha_true = torch.exp(true_b0[cat_idx] + true_b1[cat_idx] * x)
tally = torch.distributions.Gamma(alpha_true, 1.0).sample()

observations = {"x": x, "tally": tally, "cat_idx": cat_idx}
x_in = torch.zeros(NK, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

oracle_nll = float(
    -torch.distributions.Gamma(alpha_true, 1.0).log_prob(tally).mean()
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"x", "tally", "cat_idx"})
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

N_mcmc, K_mcmc = 11, 3
NK_mcmc = N_mcmc * K_mcmc
cat_idx_mcmc = torch.arange(K_mcmc).repeat(N_mcmc)
x_mcmc = x[:NK_mcmc]
tally_mcmc = tally[:NK_mcmc]
obs_mcmc = {"x": x_mcmc, "tally": tally_mcmc, "cat_idx": cat_idx_mcmc}
x_in_mcmc = torch.zeros(NK_mcmc, 1)

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1)
result = mc.run(model, x_in_mcmc, obs_mcmc)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical Perspective

The model factors as a K-fold tensor product of [`Gamma(alpha_k, 1)`](https://en.wikipedia.org/wiki/Gamma_distribution) kernels, one per category, with shape parameters driven by a per-category linear predictor through the exponential link. Pushing the joint Gamma kernel through the deterministic normalizing map `t |-> t / sum t` lands on the K-simplex as the [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) kernel, recovering Dirichlet regression as a post-composition. Working in the unnormalized Gamma layer keeps the per-category log-shapes linear in the predictor; the simplex projection is left as a deterministic adjunct that the runtime applies on demand.


## References

- [Maier 2014](https://doi.org/10.32614/CRAN.package.DirichletReg).
