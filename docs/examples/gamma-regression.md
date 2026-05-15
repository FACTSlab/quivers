# Gamma Regression

## Overview

A Bayesian regression for non-negative component totals, fit with a [Gamma](https://en.wikipedia.org/wiki/Gamma_distribution) likelihood and a per-category log-rate predictor. The model uses K = 3 categories: each category's log-shape is a linear function of the predictor, and the response is the raw per-cell Gamma tally rather than its simplex projection. This is the unnormalized layer of a Dirichlet regression ([Maier 2014](https://doi.org/10.32614/CRAN.package.DirichletReg)): independent Gamma draws with shapes `(alpha_1, ..., alpha_K)` and unit rate normalize to a sample on the [K-simplex](https://en.wikipedia.org/wiki/Simplex) via the [Gamma / Dirichlet relationship](https://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution), but here the program observes the Gamma layer directly and leaves the deterministic normalization to downstream code.

## QVR Source

```qvr
object Item : 200
object Cat : 3
object Resp : 600

program gamma_regression : Resp -> Resp
    beta_0 : Cat <- Normal(0.0, 2.0)
    beta_1 : Cat <- Normal(0.0, 2.0)

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

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/gamma_regression.qvr")
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

The model factors as a K-fold tensor product of [`Gamma(alpha_k, 1)`](https://en.wikipedia.org/wiki/Gamma_distribution) kernels, one per category, with shape parameters driven by a per-category linear predictor through the exponential link. Pushing the joint Gamma kernel through the deterministic normalizing map `t |-> t / sum t` lands on the K-simplex as the [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) kernel, recovering Dirichlet regression as a post-composition. Working in the unnormalized Gamma layer keeps the per-category log-shapes linear in the predictor; the simplex projection is left as a deterministic adjunct that the runtime applies on demand.
