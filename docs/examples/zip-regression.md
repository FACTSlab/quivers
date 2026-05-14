# Multi-Output Zero-Inflated Poisson Regression

## Overview

The zero-inflated Poisson regression ([Lambert 1992](https://doi.org/10.2307/1269547)) is a two-component mixture of a point mass at zero and a [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) rate. The model fits count data with an excess of structural zeros relative to a plain Poisson likelihood. Each output dimension carries its own zero-inflation logits and rate coefficients, and the per-cell zero-inflation indicator is integrated out by a scoped `marginalize` block.

## QVR Source

```qvr
object Item : 200
object Out : 2
object Resp : 400

program zip_regression : Resp -> Resp
    alpha_zero : Out <- Normal(0.0, 5.0)
    beta_zero : Out <- Normal(0.0, 5.0)
    alpha_rate : Out <- Normal(0.0, 5.0)
    beta_rate : Out <- Normal(0.0, 5.0)

    x : Resp <- Normal(0.0, 1.0)
    let az = alpha_zero[out_idx]
    let bz = beta_zero[out_idx]
    let ar = alpha_rate[out_idx]
    let br = beta_rate[out_idx]

    let pi_z = sigmoid(az + bz * x)
    let rate = exp(ar + br * x)

    marginalize z : Resp <- ContinuousBernoulli(pi_z) in {
        observe y : Resp <- Poisson(rate)
    }
    return beta_rate

export zip_regression
```

## Walkthrough

Per-output coefficient plates `alpha_zero`, `beta_zero` carry the [logit](https://en.wikipedia.org/wiki/Logit)-link zero-inflation probability `pi_{n, d}`, and `alpha_rate`, `beta_rate` carry the log-link Poisson rate `rate_{n, d}`. The zero-inflation indicator `z` is sampled per cell from a [`ContinuousBernoulli`](https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution) relaxation of the underlying Bernoulli, then integrated out by the enclosing `marginalize z` block: the coordinate is pushed forward through the projection on the trace's `z` axis, integrating out the indicator via reparameterised sampling. The continuous-Bernoulli relaxation gives a closed-form tractable density on `(0, 1)` and lets SVI integrate the coordinate via reparameterised sampling; the canonical logsumexp marginalization over the two integer states is the limiting case as the relaxation temperature tightens.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/zip_regression.qvr")
model = prog.morphism

torch.manual_seed(0)
N, D = 200, 2
ND = N * D
out_idx = torch.arange(D).repeat(N)
x = torch.randn(ND)

true_ar = torch.tensor([0.5, 1.0])
true_br = torch.tensor([1.0, -0.5])
rate = torch.exp(true_ar[out_idx] + true_br[out_idx] * x)
y = torch.poisson(rate)
zi_mask = torch.rand(ND) < 0.25
y[zi_mask] = 0

guide = AutoNormalGuide(model, observed_names={"x", "y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=2e-2
)
svi = SVI(model, guide, optim, ELBO())
ctx = {"x": x, "y": y, "out_idx": out_idx}
for _ in range(1500):
    svi.step(torch.zeros(ND, 1), ctx)

print("post beta_rate:", guide._loc("beta_rate").detach().squeeze())  # ~ [1.15, -0.43]
```

## Categorical Perspective

The model factors as a Kleisli composite of two kernels: a per-cell `ContinuousBernoulli(pi)` kernel on the unit interval and a `Poisson(rate)` kernel on the non-negative integers. The scoped `marginalize` step pushes forward the joint measure on the trace's `z` axis through projection, integrating out the indicator and leaving the marginal Poisson likelihood reweighted by the per-cell mixing weight. Categorically the construction is a coproduct fibration over the binary indicator axis, followed by [logsumexp](https://en.wikipedia.org/wiki/LogSumExp) on the accumulated log-likelihood in the discrete-limit case and reparameterised integration in the relaxed case.
