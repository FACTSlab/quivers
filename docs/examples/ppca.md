# Probabilistic Principal Component Analysis

## Overview

[Probabilistic PCA](https://en.wikipedia.org/wiki/Probabilistic_principal_component_analysis) ([Tipping & Bishop 1999](https://doi.org/10.1111/1467-9868.00196)) factors a data matrix through a low-rank loading matrix $W$ acting on a per-item latent code $z$:

$$
z_i \sim \mathcal{N}(0, I_K), \quad y_i \mid z_i \sim \mathcal{N}(W z_i, \sigma^2 I_D).
$$

The model is identifiable up to a $K \times K$ orthogonal rotation of $W$; the maximum-likelihood $W$ recovers the leading-$K$ [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) scaled by $\sqrt{\lambda_k - \sigma^2}$, where $\lambda_k$ are the data covariance eigenvalues. PPCA differs from [factor analysis](factor-analysis.md) only in the observation noise: PPCA uses a single isotropic scalar $\sigma$, factor analysis a free diagonal $\psi$.

In quivers, the loading matrix is a [`LatentMorphism`](../api/core/morphisms.md) $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ carrying a [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) prior, and the per-item latent code is itself a learnable morphism $Z : \mathsf{Item} \to \mathsf{LatentDim}$. The model mean is the composition $Z \mathbin{>>} W$, evaluated under `composition real as algebra` as the canonical PPCA matmul.

## QVR Source

```qvr
composition real as algebra

object LatentDim : FinSet 2
object ObsDim : FinSet 5
object Item : FinSet 64

morphism Z : Item -> LatentDim [role=latent]

morphism W : LatentDim -> ObsDim [role=latent]

let ppca = Z >> W

export ppca
```

## Walkthrough

The two latent declarations introduce the per-item code and the loading matrix as first-class arrows. The composition `Z >> W` is real-algebra matmul: under `composition real as algebra` the `(i, d)` entry of the resulting `Item x ObsDim` tensor is exactly $\sum_k Z_{i,k} W_{k,d}$, the PPCA model mean.

The matrix-normal prior

<!-- compile: false -->
```qvr
morphism W : LatentDim -> ObsDim [role=latent] ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

places a [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on the loading matrix with the dom and cod axes bound positionally to the row and column covariance arguments. The [Kronecker structure](https://en.wikipedia.org/wiki/Kronecker_product) $V \otimes U$ expresses independent row and column correlation in the loadings.

The PPCA / factor analysis distinction lives in the choice of downstream observation kernel applied to the matmul mean: a single shared scalar `sigma` for PPCA, a free diagonal `psi_d` for factor analysis. The morphism surface itself (the `Z >> W` matmul) is shared.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
import torch.distributions as D
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/ppca.qvr")

N, K, Dn = 64, 2, 5
W_true     = torch.randn(K, Dn)
Z_true     = torch.randn(N, K)
sigma_true = 0.2
Y          = Z_true @ W_true + sigma_true * torch.randn(N, Dn)
```

### SVI fit

```python
from quivers.inference import (
    AutoNormalGuide, ELBO, SVI, lift_to_bayesian_program,
)

model, x_in, observations = lift_to_bayesian_program(
    prog,
    location_fn=lambda _: prog.morphism.tensor,
    parameter_prior_scale=1.0,
    observation_family=D.Normal,
    observation_kwargs={"scale": sigma_true},
    target_key="Y",
    x=torch.zeros(N, 1),
    observations={"Y": Y},
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"Y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(200)]
print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

The recovered factorisation `Z @ W` matches the data up to the $K \times K$ rotation invariance of PPCA.

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

PPCA is a pair of arrows in a real-algebra category: the per-item code $Z : \mathsf{Item} \to \mathsf{LatentDim}$ and the loading $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$. Their composition $Z \mathbin{>>} W$ is the [`LatentMorphism`](../api/core/morphisms.md) $\mathsf{Item} \to \mathsf{ObsDim}$ whose tensor is the model mean. Marginalising the latent factor under an isotropic noise kernel recovers the closed-form covariance $W^\top W + \sigma^2 I$ on the observation side.

The morphism-valued [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on $W$ is a measure on the hom-object $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{ObsDim})$, treating the loading as a first-class arrow rather than a flat vector of entries.

## See Also

- [Factor Analysis](factor-analysis.md), the free-diagonal generalisation.
- [DSL Guide](../guides/dsl-overview.md) for the morphism-valued prior surface.


## References

- [Tipping & Bishop 1999](https://doi.org/10.1111/1467-9868.00196).
