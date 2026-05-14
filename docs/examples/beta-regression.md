# Multi-Output Beta Regression

## Overview

A multi-output beta regression ([Ferrari and Cribari-Neto 2004](https://doi.org/10.1080/0266476042000214501)) for response variables on the open unit interval. Each output dimension carries its own logit-link coefficients and precision; the per-cell mean is mapped through the [sigmoid](https://en.wikipedia.org/wiki/Logistic_function) link to the unit interval, and the [Beta](https://en.wikipedia.org/wiki/Beta_distribution) likelihood is reparameterised in mean / precision form.

## QVR Source

```qvr
object Item : 200
object Out : 3
object Resp : 600

program beta_regression : Resp -> Resp
    beta_0 : Out <- Normal(0.0, 5.0)
    beta_1 : Out <- Normal(0.0, 5.0)
    phi : Out <- HalfCauchy(2.5)

    x : Resp <- Normal(0.0, 1.0)
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

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/beta_regression.qvr")
model = prog.morphism

torch.manual_seed(0)
N, D = 200, 3
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_b0 = torch.tensor([0.5, -0.3, 0.0])
true_b1 = torch.tensor([1.0, -1.5, 0.8])
true_phi = torch.tensor([10.0, 20.0, 15.0])
mu = torch.sigmoid(true_b0[out_idx] + true_b1[out_idx] * x)
a = mu * true_phi[out_idx]
b = (1.0 - mu) * true_phi[out_idx]
y = torch.distributions.Beta(a, b).sample()

guide = AutoNormalGuide(model, observed_names={"x", "y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2
)
svi = SVI(model, guide, optim, ELBO())
ctx = {"x": x, "y": y, "out_idx": out_idx}
for _ in range(1500):
    svi.step(torch.zeros(ND, 1), ctx)

print("post beta_1:", guide._loc("beta_1").detach().squeeze())  # ~ [1.00, -1.53, 0.80]
```

## Categorical Perspective

The model is a [Kleisli morphism](https://ncatlab.org/nlab/show/Kleisli+category) in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category whose codomain factors through the [unit interval](https://en.wikipedia.org/wiki/Logistic_function); the per-cell Beta likelihood factors through the link `sigmoid` as a `1 -> G((0, 1))` kernel. The plate-gather `beta_1[out_idx]` is the Kleisli pullback of the `Out`-indexed plate along the fibration `Resp -> Out` carried by the runtime index.
