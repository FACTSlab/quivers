# AR(1) Time Series

## Overview

The canonical first-order [autoregressive](https://en.wikipedia.org/wiki/Autoregressive_model) model: each observation is a Normal draw centered on a linear function of the previous observation, with three free parameters (intercept, autoregressive coefficient, and noise scale). The model is the reference point for the more elaborate state-space and stochastic-volatility examples in the gallery; it shows the minimum DSL surface needed to fit a stationary linear time series.

## QVR Source

```qvr
object Step : 200

program ar1 : Step -> Step
    alpha <- Normal(0.0, 5.0)
    phi <- Uniform(-1.0, 1.0)
    sigma <- HalfCauchy(1.0)

    let mu = alpha + phi * y_prev
    observe y : Step <- Normal(mu, sigma)
    return phi

export ar1
```

## Walkthrough

`alpha` is the intercept of the AR(1) recurrence, `phi` is the autoregressive coefficient constrained to the [stationarity interval](https://en.wikipedia.org/wiki/Stationary_process) `(-1, 1)` via a `Uniform` prior, and `sigma` is the per-step Normal scale with a [half-Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution#Related_distributions) prior. The identifier `y_prev` is exogenous host-data: it is never declared inside the program, so the runtime resolves it from the observations dict at trace time, where the caller supplies the lagged response series. The mean `mu = alpha + phi * y_prev` is then the per-step recurrence and the observed series `y` is Normal noise around it.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/ar1.qvr")
model = prog.morphism

y_obs = torch.randn(200)
y_lag = torch.cat([torch.zeros(1), y_obs[:-1]])

guide = AutoNormalGuide(model, observed_names={"y"})
optim = torch.optim.Adam(list(model.parameters()) + list(guide.parameters()), lr=1e-2)
svi = SVI(model, guide, optim, ELBO())
for _ in range(2000):
    svi.step(torch.zeros(200, 1), {"y": y_obs, "y_prev": y_lag})
```

## Categorical Perspective

The model is a Kleisli morphism in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category whose denotation is the standard AR(1) joint $p(\alpha, \phi, \sigma) \prod_t \mathcal{N}(y_t \mid \alpha + \phi y_{t-1}, \sigma)$. The `Step`-indexed plate is the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) of the per-step Normal kernel along the trivial projection $\mathrm{Step} \to \mathbf{1}$.
