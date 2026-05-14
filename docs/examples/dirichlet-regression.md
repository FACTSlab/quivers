# Dirichlet Regression

## Overview

Dirichlet regression ([Maier 2014](https://doi.org/10.32614/CRAN.package.DirichletReg)) for compositional response variables on the K-dimensional [probability simplex](https://en.wikipedia.org/wiki/Simplex). The model uses K = 3 categories and exploits the [Gamma / Dirichlet relationship](https://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution): independent Gamma draws with shapes `(alpha_1, ..., alpha_K)` and unit rate normalise to a Dirichlet sample on the K-simplex.

## QVR Source

```qvr
object Item : 200
object Cat : 3
object Resp : 600

program dirichlet_regression : Resp -> Resp
    beta_0 : Cat <- Normal(0.0, 2.0)
    beta_1 : Cat <- Normal(0.0, 2.0)

    x : Resp <- Normal(0.0, 1.0)
    let b0 = beta_0[cat_idx]
    let b1 = beta_1[cat_idx]
    let alpha = exp(b0 + b1 * x)

    observe tally : Resp <- Gamma(alpha, 1.0)
    return beta_1

export dirichlet_regression
```

## Walkthrough

Per-category coefficient plates `beta_0 : Cat` and `beta_1 : Cat` carry one coefficient per simplex axis. The per-cell log-shape `b0 + b1 * x` is mapped through the exponential to give the positive Gamma shape `alpha`; the observed Gamma tallies normalise downstream to a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) sample on the K-simplex. Plate-gather `beta_0[cat_idx]` selects each cell's category coefficient from the per-category plate.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/dirichlet_regression.qvr")
model = prog.morphism

torch.manual_seed(0)
N, K = 200, 3
NK = N * K
cat_idx = torch.arange(K).repeat(N)
x = torch.randn(NK)

true_b0 = torch.tensor([1.0, 0.5, -0.5])
true_b1 = torch.tensor([0.8, -1.0, 0.3])
alpha = torch.exp(true_b0[cat_idx] + true_b1[cat_idx] * x)
tally = torch.distributions.Gamma(alpha, 1.0).sample()

guide = AutoNormalGuide(model, observed_names={"x", "tally"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2
)
svi = SVI(model, guide, optim, ELBO())
ctx = {"x": x, "tally": tally, "cat_idx": cat_idx}
for _ in range(1500):
    svi.step(torch.zeros(NK, 1), ctx)

print("post beta_1:", guide._loc("beta_1").detach().squeeze())  # ~ [0.75, -1.06, 0.32]
```

## Categorical Perspective

The model factors through the [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) / [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) representation: a K-fold tensor product of `Gamma(alpha_k, 1)` kernels, pushforward through the normalising map `t |-> t / sum t` lands on the K-simplex as the Dirichlet kernel. Working in the unnormalised Gamma layer keeps the per-category log-shapes linear in the predictor; the simplex projection is a deterministic post-composition that the runtime applies on demand.
