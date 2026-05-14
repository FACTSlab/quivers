# Multi-Coefficient Horseshoe Regression

## Overview

A multiple regression with the horseshoe prior ([Carvalho, Polson, and Scott 2010](https://doi.org/10.1093/biomet/asq017)). The horseshoe is a [global-local scale mixture of Normals](https://en.wikipedia.org/wiki/Sparsity-inducing_prior): a single global scale `tau` and per-coordinate local scales `lambda_p` jointly define the coefficient prior, inducing a spike-near-zero / heavy-tail mixture that adaptively shrinks small effects toward zero while leaving large effects nearly unbiased.

## QVR Source

```qvr
object Item : 200
object Coef : 4
object Resp : 800

program horseshoe_regression : Resp -> Resp
    tau <- HalfCauchy(1.0)
    lambda_local : Coef <- HalfCauchy(1.0)
    z_raw : Coef <- Normal(0.0, 1.0)

    alpha <- Normal(0.0, 5.0)
    sigma <- HalfCauchy(2.0)

    x : Resp <- Normal(0.0, 1.0)
    let lam = lambda_local[coef_idx]
    let z = z_raw[coef_idx]
    let beta = tau * lam * z
    let mu = alpha + beta * x

    observe y : Resp <- Normal(mu, sigma)
    return tau

export horseshoe_regression
```

## Walkthrough

The horseshoe has no closed-form marginal density: the right idiom in quivers is the explicit `tau * lambda * z` decomposition built from existing primitives. `tau ~ HalfCauchy(1)` is the global scale; `lambda_local : Coef <- HalfCauchy(1)` is a per-coordinate local scale plate; `z_raw : Coef <- Normal(0, 1)` is the standard-Normal raw draw. The deterministic combination `beta = tau * lam * z` gives the implied coefficient with heavy-tailed marginal density.

The per-coefficient plate-draws scale automatically to any P without rewriting the body. Each observation cell carries a `coef_idx` indicating which coefficient it loads on; the per-cell mean is `alpha + beta[coef_idx] * x`.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/horseshoe_regression.qvr")
model = prog.morphism

torch.manual_seed(0)
N, P = 200, 4
NP = N * P
coef_idx = torch.arange(P).repeat(N)
x = torch.randn(NP)

true_beta = torch.tensor([2.0, 0.0, -1.5, 0.0])
true_alpha = 0.3
true_sigma = 0.5
y = true_alpha + true_beta[coef_idx] * x + true_sigma * torch.randn(NP)

guide = AutoNormalGuide(model, observed_names={"x", "y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2
)
svi = SVI(model, guide, optim, ELBO())
ctx = {"x": x, "y": y, "coef_idx": coef_idx}
for _ in range(2000):
    svi.step(torch.zeros(NP, 1), ctx)

betas = []
with torch.no_grad():
    for _ in range(200):
        s = guide.rsample(torch.zeros(1, 1))
        betas.append(s["tau"].item() * s["lambda_local"].squeeze() * s["z_raw"].squeeze())
print("posterior mean beta:", torch.stack(betas).mean(0))  # ~ [2.00, 0.07, -1.52, 0.02]
```

The non-zero coefficients are recovered accurately and the zero coefficients are shrunk toward zero, matching the horseshoe's [spike-and-tail](https://doi.org/10.1093/biomet/asq017) behavior.

## Categorical Perspective

The model factors as the Kleisli composite of a global-local hyperprior kernel, a deterministic Hadamard product `tau * lambda * z` lifted into the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category as a Dirac kernel, and the per-row Normal likelihood. The plate-draws `lambda_local` and `z_raw` are Kleisli sections of the `Coef`-indexed plate, and `lambda_local[coef_idx]` is the [Kleisli pullback](https://ncatlab.org/nlab/show/Kleisli+category) along the fibration `Resp -> Coef` carried by the runtime index.
