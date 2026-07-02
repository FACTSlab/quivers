# Bayesian Gaussian Mixture Model

## Overview

A finite [Gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model) assigns each observation to one of $K$ [Gaussian](https://en.wikipedia.org/wiki/Normal_distribution) components, with the per-row component drawn from a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution)-distributed mixing prior. This example demonstrates the canonical quivers idiom for finite mixtures: per-component means and scales are continuous latents on the `Component` plate, and the discrete per-row component assignment is integrated out by a scoped `marginalize` block whose body genuinely depends on the marginalized variable, yielding the canonical [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) over $K$ classes at every observation:

$$
p(r_n) \;=\; \sum_{k=1}^{K} \mathrm{probs}[k] \; \mathcal{N}\!\bigl(r_n;\, \mu[k],\, \sigma[k]\bigr).
$$

## QVR Source

```qvr
composition log_prob [level=algebra]

object Component : FinSet 3
object Item : FinSet 8
object Resp : FinSet 100

program gmm(alpha : Real) : Resp -> Resp
    sample probs <- Dirichlet(alpha) [over=Component]
    sample mu : Component <- Normal(0.0, 5.0)
    sample sigma : Component <- HalfNormal(1.0)
    sample idx : Resp <- HalfNormal(1.0)

    marginalize cls : Component <- Categorical(probs) [over=Item, reduction=logsumexp]
        observe r : Resp <- Normal(mu[cls], sigma[cls]) [via=idx]

    return probs

export gmm
```

## Walkthrough

`composition log_prob as algebra` selects the log-probability semiring so the program's `Score` effect accumulates log-densities additively. `object Component : FinSet 3`, `object Item : FinSet 8`, `object Resp : FinSet 100` declare the three discrete plates: $K = 3$ components, $I = 8$ item groups, $N = 100$ observed rows. `program gmm(alpha : Real) : Resp -> Resp` parameterises the program by the Dirichlet concentration.

`sample probs <- Dirichlet(alpha) [over=Component]` draws the mixing weights as a single point on the `Component` simplex; `over=Component` names the family's event axis. `sample mu : Component <- Normal(0.0, 5.0)` and `sample sigma : Component <- HalfNormal(1.0)` draw the per-component mean and scale as plate-bound continuous latents. `sample idx : Resp <- HalfNormal(1.0)` registers the per-row fibration site that names the runtime map from `Resp` into the `Item` grouping plate.

The scoped `marginalize` block

<!-- compile: false -->
```qvr
marginalize cls : Component <- Categorical(probs) [over=Item, reduction=logsumexp]
    observe r : Resp <- Normal(mu[cls], sigma[cls]) [via=idx]
```

introduces the per-row component latent `cls : Component` under a [Categorical](https://en.wikipedia.org/wiki/Categorical_distribution) prior parameterised by `probs`. The body's `Normal(mu[cls], sigma[cls])` looks up the chosen component's mean and scale and scores the observed row `r` against that per-class Gaussian. The `[over=Item]` grouping plate accumulates each observation into its `Item` bucket; `[via=idx]` names the runtime fibration from each row into its item group. The `[reduction=logsumexp]` reduction integrates `cls` out by pushforward along the projection $\Phi \times \mathsf{Component} \to \Phi$, recovering the closed-form per-row mixture marginal $\sum_k \mathrm{probs}[k]\,\mathcal{N}(r_n;\,\mu[k],\,\sigma[k])$. At the end of the scope `cls` falls out of scope.

`return probs` projects the program's joint kernel onto the mixing-weight site.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/mixture_model.qvr")
fit = prog.gmm(alpha=1.0)
model = fit.morphism

K, N = 3, 100
true_probs = torch.tensor([0.4, 0.35, 0.25])
true_mu    = torch.tensor([-3.0, 0.0, 3.0])
true_sigma = torch.tensor([0.5, 0.7, 0.4])

comps = torch.distributions.Categorical(true_probs).sample((N,))
r     = torch.distributions.Normal(true_mu[comps], true_sigma[comps]).sample()
idx   = torch.randint(0, 8, (N,))

observations = {"r": r, "idx": idx, "probs": true_probs}
x_in = torch.zeros(N, 1)
```

The simplex-supported `probs` site is supplied via `observations` so the grouped marginalize block sees a `(K,)` mixing prior at every batch position. The Gaussian per-component parameters `mu` and `sigma` remain unobserved and are recovered by SVI.

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(
    model, observed_names={"r", "idx", "probs"},
)
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(300)]
print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical Perspective

The discrete latent `cls : Component` is integrated out by [pushforward](https://en.wikipedia.org/wiki/Pushforward_measure) along the projection $\Phi \times \mathsf{Component} \to \Phi$. The grouped marginalize block is the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) of the per-class log-likelihood along the per-row fibration $\mathsf{Resp} \to \mathsf{Item}$ in $\mathbf{Kern}$, followed by a [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) reduction along the `Component` axis weighted by the categorical prior implied by the Dirichlet.

## See Also

- [Latent Dirichlet Allocation](lda.md), the topic-model generalization.
