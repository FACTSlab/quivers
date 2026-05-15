# Bayesian Gaussian Mixture Model

## Overview

A finite [Gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model) assigns each observation to one of $K$ [Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) components, with the per-row component drawn from a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)-distributed mixing prior. This example demonstrates the canonical quivers idiom for finite mixtures: per-component means and scales are continuous latents on the `Component` plate, and the discrete per-row component assignment is integrated out by a scoped `marginalize` block whose body genuinely depends on the marginalized variable, yielding the canonical [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) over $K$ classes at every observation:

$$
p(r_n) \;=\; \sum_{k=1}^{K} \mathrm{probs}[k] \; \mathcal{N}\!\bigl(r_n;\, \mu[k],\, \sigma[k]\bigr).
$$

## QVR Source

```qvr
object Component : 4
object Item : 8
object Resp : 200

program gmm : Resp -> Resp
    probs : Component <- HalfNormal(1.0)
    mu : Component <- Normal(0.0, 5.0)
    sigma : Component <- HalfNormal(1.0)
    idx : Resp <- HalfNormal(1.0)

    marginalize cls : Component <- Dirichlet(probs)
        over Item
        in {
            let mu_nk = factor n : Resp, k : Component in mu[k]
            let sigma_nk = factor n : Resp, k : Component in sigma[k]
            observe r : Resp via idx <- Normal(mu_nk, sigma_nk)
        }
    return probs

export gmm
```

## Walkthrough

The plate-bound `probs : Component` carries the mixing weights; the marginalize header takes a Dirichlet prior on the simplex with `probs` as the concentration. The per-component emission parameters `mu : Component` and `sigma : Component` are continuous latents on the `Component` plate: each `mu[k]` is the centre of the $k$-th Gaussian and each `sigma[k]` its scale. The per-row fibration `idx : Resp` names the per-observe fibration from `Resp` into the `Item` grouping plate.

The scoped marginalize block

<!-- compile: false -->
```qvr
marginalize cls : Component <- Dirichlet(probs)
    over Item
    in {
        let mu_nk = factor n : Resp, k : Component in mu[k]
        let sigma_nk = factor n : Resp, k : Component in sigma[k]
        observe r : Resp via idx <- Normal(mu_nk, sigma_nk)
    }
```

introduces the per-row component latent `cls : Component` under a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior on the mixing simplex. Inside the body, the two `factor` expressions build per-(`Resp`, `Component`) parameter tensors by gathering `mu[k]` and `sigma[k]` for every row, so the observation `Normal(mu_nk, sigma_nk)` genuinely depends on the marginalized variable: each row's per-class log-likelihood reads off the $k$-th component's mean and scale. The runtime scatter-sums each row's contribution into the `Item`-indexed accumulator and [log-sum-exps](https://en.wikipedia.org/wiki/LogSumExp) over `Component`, integrating `cls` out by pushforward along the projection $\Phi \times \mathsf{Component} \to \Phi$. At the end of the scope `cls` falls out of scope.

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

The grouped marginalize block exposes the per-row per-class log-likelihood slot `_grouped_ll_cls_0` directly, so a synthetic experiment can supply that tensor under known component means and recover the mixing weights through SVI.

## Categorical Perspective

The discrete latent `cls : Component` is integrated out by [pushforward](https://en.wikipedia.org/wiki/Pushforward_measure) along the projection $\Phi \times \mathsf{Component} \to \Phi$. The grouped marginalize block is the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) of the per-class log-likelihood along the per-row fibration $\mathsf{Resp} \to \mathsf{Item}$ in $\mathbf{Kern}$, followed by a [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) reduction along the `Component` axis weighted by the categorical prior implied by the Dirichlet.

## See Also

- [Latent Dirichlet Allocation](lda.md), the topic-model generalization.
