# Stochastic Volatility

## Overview

The canonical log-volatility return model of [Kim, Shephard, and Chib (1998)](https://doi.org/10.1111/1467-937X.00050). The latent log-volatility follows an [AR(1)](https://en.wikipedia.org/wiki/Autoregressive_model) chain centred on a mean `mu` with autoregressive coefficient `phi`, and the observed return is mean-zero Normal with time-varying scale `exp(h_t / 2)`. The exponential link makes the volatility positive by construction.

## QVR Source

```qvr
object Step : 200

program stochastic_volatility : Step -> Step
    mu <- Normal(0.0, 10.0)
    phi <- Uniform(-1.0, 1.0)
    sigma_h <- HalfCauchy(2.5)

    h_prev : Step <- Normal(mu, 1.0)
    let h_mean = mu + phi * (h_prev - mu)
    h : Step <- Normal(h_mean, sigma_h)

    let scale = exp(0.5 * h)
    observe r : Step <- Normal(0.0, scale)
    return phi

export stochastic_volatility
```

## Walkthrough

`mu`, `phi`, and `sigma_h` are the AR(1) hyperparameters of the log-volatility chain; `phi` is constrained to the [stationarity interval](https://en.wikipedia.org/wiki/Stationary_process) `(-1, 1)`. The `h_prev` plate carries the lagged log-volatility, the `h` plate carries the current latent log-volatility, and `exp(0.5 * h)` is the standard SV link to the per-step return scale. The observed returns are mean-zero Normal scaled by the time-varying volatility.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/stochastic_volatility.qvr")
model = prog.morphism

returns = torch.randn(200) * 0.5
guide = AutoNormalGuide(model, observed_names={"r"})
optim = torch.optim.Adam(list(model.parameters()) + list(guide.parameters()), lr=5e-3)
svi = SVI(model, guide, optim, ELBO())
for _ in range(3000):
    svi.step(torch.zeros(200, 1), {"r": returns})
```

## Categorical Perspective

The model is a Kleisli morphism over the latent log-volatility plate, composed with a per-step Normal observation kernel whose scale depends on the latent. In the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category, the chain `h_prev -> h -> r` is associative Kleisli composition; the SVI guide approximates the joint posterior $p(\mu, \phi, \sigma_h, h \mid r)$.
