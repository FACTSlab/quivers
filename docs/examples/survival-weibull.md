# Weibull Survival Regression

## Overview

A parametric [survival regression](https://en.wikipedia.org/wiki/Survival_analysis) with a [Weibull](https://en.wikipedia.org/wiki/Weibull_distribution) baseline ([Klein and Moeschberger 2003](https://doi.org/10.1007/b97377)). Each item carries a covariate `x_i` whose linear contribution scales the Weibull rate; the shape parameter `k` governs whether the hazard is decreasing (k < 1), constant (k = 1, exponential), or increasing (k > 1).

## QVR Source

```qvr
object Item : 200

program survival_weibull : Item -> Item
    alpha <- Normal(0.0, 5.0)
    beta <- Normal(0.0, 5.0)
    k <- Gamma(2.0, 1.0)

    x : Item <- Normal(0.0, 1.0)
    let eta = alpha + beta * x
    let scale = exp(-eta / k)

    observe t : Item <- Weibull(scale, k)
    return beta

export survival_weibull
```

## Walkthrough

The reparameterisation `scale = exp(-eta / k)` is the Weibull [proportional-hazards](https://en.wikipedia.org/wiki/Proportional_hazards_model) convention: positive shifts in the linear predictor `eta = alpha + beta * x` increase the hazard and shorten survival times, matching the canonical direction. The shape `k` has a Gamma prior centred at 2. The observed event times `t` are uncensored Weibull draws; right-censoring is handled at the inference layer by substituting the [Weibull survival function](https://en.wikipedia.org/wiki/Weibull_distribution#Survival_function) for the density on censored rows.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/survival_weibull.qvr")
model = prog.morphism

x = torch.randn(200)
t = torch.distributions.Weibull(1.0, 1.5).sample((200,))
guide = AutoNormalGuide(model, observed_names={"x", "t"})
optim = torch.optim.Adam(list(model.parameters()) + list(guide.parameters()), lr=1e-2)
svi = SVI(model, guide, optim, ELBO())
for _ in range(1500):
    svi.step(torch.zeros(200, 1), {"x": x, "t": t})
```

## Categorical Perspective

The model denotes a Kleisli morphism into the positive reals in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category. The Weibull is the [exponential](https://en.wikipedia.org/wiki/Exponential_distribution) family generalisation with shape; the proportional-hazards link makes the model canonical in the [exponential family](https://en.wikipedia.org/wiki/Exponential_family) representation.
