# AR(1) Time Series

## Overview

The canonical first-order [autoregressive](https://en.wikipedia.org/wiki/Autoregressive_model) model: each observation is a Normal draw centered on a linear function of the previous observation, with three free parameters (intercept, autoregressive coefficient, and noise scale). The model is the reference point for the more elaborate state-space and stochastic-volatility examples in the gallery; it shows the minimum DSL surface needed to fit a stationary linear time series.

## QVR Source

```qvr
object Step : FinSet 200

program ar1 : Step -> Step
    sample alpha <- Normal(0.0, 5.0)
    sample phi <- Uniform(-1.0, 1.0)
    sample sigma <- HalfCauchy(1.0)

    let mu = alpha + phi * y_prev

    observe y : Step <- Normal(mu, sigma)
    return phi

export ar1
```

## Walkthrough

`alpha` is the intercept of the AR(1) recurrence, `phi` is the autoregressive coefficient constrained to the [stationarity interval](https://en.wikipedia.org/wiki/Stationary_process) `(-1, 1)` via a `Uniform` prior, and `sigma` is the per-step Normal scale with a [half-Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution#Related_distributions) prior. The identifier `y_prev` is exogenous host-data: it is never declared inside the program, so the runtime resolves it from the observations dict at trace time, where the caller supplies the lagged response series. The mean `mu = alpha + phi * y_prev` is then the per-step recurrence and the observed series `y` is Normal noise around it.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

Pick ground-truth dynamics and forward-sample a single AR(1) trajectory of length `T`. The lagged-response vector `y_prev` is the host-data the program reads at trace time; the first entry is anchored at the stationary mean.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/ar1.qvr")
model = prog.morphism

T = 64
alpha_true, phi_true, sigma_true = 0.5, 0.7, 0.3
y = torch.zeros(T)
y[0] = alpha_true / (1.0 - phi_true)
for t in range(1, T):
    y[t] = alpha_true + phi_true * y[t - 1] + sigma_true * torch.randn(())
y_prev = torch.cat([torch.zeros(1), y[:-1]])
x_in = torch.zeros(T, 1)
observations = {"y": y, "y_prev": y_prev}
```

### SVI fit

Re-initialise the model so the optimiser starts from the prior, then fit `alpha`, `phi`, and `sigma` by maximising the ELBO. The negative ELBO falls from its starting value down toward the oracle log-likelihood at the true parameters.

```python
import torch.distributions as D
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/ar1.qvr")
model = prog.morphism

guide = AutoNormalGuide(model, observed_names={"y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optim, ELBO())

loss0 = svi.step(x_in, observations)
losses = [svi.step(x_in, observations) for _ in range(800)]
loss_final = sum(losses[-20:]) / 20.0
oracle_ll = D.Normal(
    alpha_true + phi_true * y_prev, sigma_true,
).log_prob(y).sum().item()
print(f"initial ELBO loss: {loss0:.1f}")
print(f"final ELBO loss:   {loss_final:.1f}")
print(f"oracle -log p(y):  {-oracle_ll:.1f}")
```

### NUTS posterior

The model declares explicit `sample` sites for `alpha`, `phi`, and `sigma`, so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them directly. Small budgets keep the snippet inside tens of seconds while still producing a usable posterior summary.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(0)
prog = load("docs/examples/source/ar1.qvr")
model = prog.morphism

kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
for name, samples in result.samples.items():
    print(f"{name}: posterior mean = {samples.mean().item():.3f}")
```


## Categorical Perspective

The model is a Kleisli morphism in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) whose denotation is the standard AR(1) joint $p(\alpha, \phi, \sigma) \prod_t \mathcal{N}(y_t \mid \alpha + \phi y_{t-1}, \sigma)$. The `Step`-indexed plate is the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) of the per-step Normal kernel along the trivial projection $\mathrm{Step} \to \mathbf{1}$.


## References

- Michèle Giry. 1982. A categorical approach to probability theory. In Bernhard Banaschewski, editor, *Categorical Aspects of Topology and Analysis*, volume 915 of *Lecture Notes in Mathematics*, pages 68–85. Springer, Berlin, Heidelberg.
