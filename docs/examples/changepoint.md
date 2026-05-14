# Bayesian Change-Point Detection

## Overview

A canonical Bayesian [change-point model](https://doi.org/10.2307/2347570) in the rate of a [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) series, with a single switch at an unknown time `tau`. The classical application is the British coal-mining disaster series: the rate of accidents per year drops at an estimated change-point in the late nineteenth century. Two rates, one before and one after the switch, are tied together by a soft indicator parameterized as $s(t) = \sigma(k \cdot (t - \tau))$, which is the standard differentiable relaxation of the indicator $\mathbf{1}\{t > \tau\}$ and lets gradient-based variational inference fit the change-point location directly.

## QVR Source

```qvr
object Step : 100

program changepoint : Step -> Step
    tau <- Uniform(0.0, 100.0)
    rate_before <- Gamma(2.0, 1.0)
    rate_after <- Gamma(2.0, 1.0)

    t : Step <- Normal(0.0, 1.0)
    let s = sigmoid(20.0 * (t - tau))
    let rate = (1.0 - s) * rate_before + s * rate_after

    observe y : Step <- Poisson(rate)
    return tau
```

## Walkthrough

`tau` is the change-point time with a uniform prior on the observation window. `rate_before` and `rate_after` are Gamma-prior intensities, each centered on a prior mean of 2 events per period. The per-step plate `t : Step` carries the time stamps; at fit time the runtime supplies the actual time vector via the `t` observation. The soft indicator $s = \sigma(20 (t - \tau))$ with steepness 20 gives a sharp but differentiable switch; the per-step rate is the convex combination of the two regime rates, and the observation `y : Step <- Poisson(rate)` is the per-step Poisson likelihood.

For a strict hard change-point with integer-valued `tau`, replace the soft indicator with a `marginalize tau_idx : Tau <- Categorical(uniform) in { ... }` block: the runtime then log-sum-exps the per-step likelihood over the `Tau` axis, realizing the exact Bayesian change-point posterior at the cost of a $|Tau|$-fold widening of the inner loop.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/changepoint.qvr")
model = prog.morphism

tau_true = 60
T = 100
rate_before, rate_after = 2.0, 5.0
y = torch.cat([
    torch.poisson(torch.ones(tau_true) * rate_before),
    torch.poisson(torch.ones(T - tau_true) * rate_after),
])
t_vec = torch.arange(T, dtype=torch.float32)

guide = AutoNormalGuide(model, observed_names={"y", "t"})
optim = torch.optim.Adam(list(model.parameters()) + list(guide.parameters()), lr=1e-2)
svi = SVI(model, guide, optim, ELBO())
for _ in range(1500):
    loss = svi.step(torch.zeros(T, 1), {"y": y, "t": t_vec})

# Unconstrained loc for tau passes through a sigmoid to (0, 100).
import math
tau_loc = guide.loc_tau.item()
tau_post = 100.0 * (1.0 / (1.0 + math.exp(-tau_loc)))
print(f"posterior tau mean: {tau_post:.1f}, true: {tau_true}")
```

## Categorical Perspective

The model is a Kleisli morphism in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category whose denotation is the joint $p(\tau, r_1, r_2) \prod_t \mathrm{Poisson}(y_t \mid r(t; \tau, r_1, r_2))$. The soft indicator is the [reparameterisation gradient](https://doi.org/10.48550/arXiv.1312.6114)-friendly relaxation of an indicator function; in the hard-change-point variant, the discrete `tau_idx` is integrated out by `marginalize`, realizing the right Kan extension along the trivial projection from the time axis to a singleton.
