# Multi-Output Negative Binomial Regression

## Overview

A multi-output [negative-binomial](https://en.wikipedia.org/wiki/Negative_binomial_distribution) regression for overdispersed count data, using the mean / dispersion parameterization that follows the log link convention shared with [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression). Each output dimension carries its own coefficients and dispersion; the response is the standard [NB2](https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations) form with per-cell variance `mu + mu^2 / dispersion`, recovering Poisson in the limit of infinite dispersion.

## QVR Source

```qvr
object Item : 200
object Out : 3
object Resp : 600

program negbin_regression : Resp -> Resp
    beta_0 : Out <- Normal(0.0, 5.0)
    beta_1 : Out <- Normal(0.0, 5.0)
    dispersion : Out <- Gamma(2.0, 0.5)

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

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/negbin_regression.qvr")
model = prog.morphism

torch.manual_seed(0)
N, D = 200, 3
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_b0 = torch.tensor([1.0, 0.5, 2.0])
true_b1 = torch.tensor([0.5, -0.3, 0.8])
true_disp = torch.tensor([5.0, 10.0, 3.0])
mu = torch.exp(true_b0[out_idx] + true_b1[out_idx] * x)
probs = true_disp[out_idx] / (true_disp[out_idx] + mu)
y = torch.distributions.NegativeBinomial(true_disp[out_idx], probs).sample()

guide = AutoNormalGuide(model, observed_names={"x", "y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=2e-2
)
svi = SVI(model, guide, optim, ELBO())
ctx = {"x": x, "y": y, "out_idx": out_idx}
for _ in range(2000):
    svi.step(torch.zeros(ND, 1), ctx)

print("post beta_1:", guide._loc("beta_1").detach().squeeze())  # ~ [0.54, -0.28, 0.82]
```

## Categorical Perspective

The negative binomial is the Gamma-Poisson [mixture](https://en.wikipedia.org/wiki/Compound_probability_distribution): a `Poisson(rate)` kernel with `rate ~ Gamma(dispersion, dispersion / mu)` marginalizes to `NegativeBinomial(dispersion, mu / (mu + dispersion))`. The model factors through this mixture by sampling per-cell from the closed-form negative binomial; categorically the family is the pushforward of the Gamma-Poisson joint kernel along the rate-projection.
