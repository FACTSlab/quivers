# Bayesian Gaussian Mixture Model

## Overview

A finite [Gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model) assigns each observation to one of $K$ [Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) components, with the per-row component drawn from a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)-distributed mixing prior. This example demonstrates the canonical quivers idiom for finite mixtures: a scoped `marginalize` block that integrates out the discrete per-row component assignment by [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) over the $K$ classes at every observation.

## QVR Source

```qvr
object Component : 4
object Item : 1
object Resp : 200

program gmm : Resp -> Resp
    probs : Component <- HalfNormal(1.0)
    idx : Resp <- HalfNormal(1.0)
    mu_shift <- Normal(0.0, 1.0)

    marginalize cls : Component <- Dirichlet(probs)
        over Item
        in {
            observe r : Resp via idx <- Normal(mu_shift, 1.0)
        }
    return probs

export gmm
```

## Walkthrough

The plate-bound `probs : Component` carries the Dirichlet concentration vector for the per-row component assignment. The per-row fibration `idx : Resp` names the per-observe fibration into the singleton `Item` grouping plate. The scalar `mu_shift` is a continuous latent shared across components, drawn from a Normal prior.

The scoped marginalize block

<!-- compile: false -->
```qvr
marginalize cls : Component <- Dirichlet(probs)
    over Item
    in {
        observe r : Resp via idx <- Normal(mu_shift, 1.0)
    }
```

introduces the per-row component latent `cls : Component` with a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior on the mixing simplex. Inside the body, `observe r : Resp via idx <- Normal(mu_shift, 1.0)` scores each response under the shared Normal likelihood; the runtime accumulates one per-class log-likelihood at every row, scatter-sums into the `Item`-indexed accumulator, and [log-sum-exps](https://en.wikipedia.org/wiki/LogSumExp) over `Component`. At the end of the scope `cls` falls out of scope; the integrated marginal is the pushforward measure on $\Phi$ alone.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)

prog = load("docs/examples/source/mixture_model.qvr")
model = prog.morphism

N, K = 200, 4
true_probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
mus = torch.tensor([-3.0, -1.0, 1.0, 3.0])
z_true = torch.multinomial(true_probs, N, replacement=True)
x = mus[z_true] + 0.3 * torch.randn(N)

# Per-row per-class log-likelihood under the K Normal components.
ll = torch.distributions.Normal(mus.view(1, K), 1.0).log_prob(x.view(N, 1))

obs = {
    "probs": true_probs,
    "idx": torch.zeros(N, dtype=torch.long),
    "_grouped_ll_cls_0": ll,
}
guide = AutoNormalGuide(model, observed_names=set(obs.keys()))
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=2e-2,
)
svi = SVI(model, guide, optim, ELBO())
for _ in range(500):
    loss = svi.step(torch.zeros(1, 1), obs)
print("GMM loss:", loss)
```

The grouped marginalize block exposes the per-row per-class log-likelihood slot `_grouped_ll_cls_0` directly, so a synthetic experiment supplies that tensor under known component means.

## Categorical Perspective

The discrete latent `cls : Component` is integrated out by pushforward along the projection $\Phi \times \mathsf{Component} \to \Phi$. The grouped marginalize block is the right Kan extension of the per-class log-likelihood along the per-row fibration $\mathsf{Resp} \to \mathsf{Item}$ in $\mathbf{Kern}$, followed by a [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) reduction along the `Component` axis weighted by the categorical prior implied by the Dirichlet.

## See Also

- [Latent Dirichlet Allocation](lda.md), the topic-model generalization.
