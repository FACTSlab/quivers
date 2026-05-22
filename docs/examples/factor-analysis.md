# Factor Analysis

## Overview

Classical [factor analysis](https://en.wikipedia.org/wiki/Factor_analysis) ([Spearman 1904](https://www.jstor.org/stable/1412107); [Bartholomew, Knott, and Moustaki 2011](https://doi.org/10.1002/9781119970583)) decomposes a $D$-dimensional observation as a linear-Gaussian transformation of a $K$-dimensional latent factor plus a free diagonal idiosyncratic noise:

$$
z_i \sim \mathcal{N}(0, I_K), \quad y_i \mid z_i \sim \mathcal{N}(W z_i, \mathrm{diag}(\psi)).
$$

The free diagonal $\psi$ distinguishes factor analysis from [probabilistic PCA](ppca.md), whose noise is isotropic. The loading matrix is the canonical example of a morphism-valued latent in quivers: declared as an arrow $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ with a [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) prior on its row and column covariances. The per-item latent factor is itself a learnable morphism $Z : \mathsf{Item} \to \mathsf{LatentDim}$, and the model mean is the composition $Z \mathbin{>>} W$.

## QVR Source

```qvr
composition real as algebra

object LatentDim : FinSet 2
object ObsDim : FinSet 5
object Item : FinSet 64

morphism Z : Item -> LatentDim [role=latent]

morphism W : LatentDim -> ObsDim [role=latent]

let factor_analysis = Z >> W

export factor_analysis
```

## Walkthrough

The two latent declarations introduce the per-item factor and the loading matrix as arrows. Under `composition real as algebra` the composition `Z >> W` is a real-valued matmul: the `(i, d)` entry of the `Item x ObsDim` model tensor is $\sum_k Z_{i,k} W_{k,d}$, the factor analysis model mean.

The top-level morphism prior

<!-- compile: false -->
```qvr
morphism W : LatentDim -> ObsDim [role=latent] ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

places a [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on the loading matrix. The two axes named under `over (dom, cod)` bind positionally to the family's event axes: `LatentDim` is the row axis carrying the row-covariance argument, `ObsDim` is the column axis carrying the column-covariance argument. The [Kronecker covariance](https://en.wikipedia.org/wiki/Kronecker_product) structure expresses independent row and column correlation in the loadings, the natural prior for a low-rank factor decomposition.

The distinction between factor analysis and PPCA appears in the observation noise kernel applied to the matmul mean: factor analysis uses a free diagonal $\psi_d$ per dimension; PPCA collapses that diagonal to a single isotropic scalar.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
import torch.distributions as D
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/factor_analysis.qvr")
fa   = prog.morphism

N, K, Dn = 64, 2, 5
W_true   = torch.randn(K, Dn)
Z_true   = torch.randn(N, K)
psi_true = 0.3
Y        = Z_true @ W_true + psi_true * torch.randn(N, Dn)
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
    observation_kwargs={"scale": 0.3},
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

Factor analysis is identifiable only up to a $K \times K$ orthogonal rotation of $W$, so the rotation-invariant covariance $W^\top W$ is the natural recovery target rather than $W$ itself.

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

The factor analysis mean is a composition $Z \mathbin{>>} W$ in the [Kleisli category](https://en.wikipedia.org/wiki/Kleisli_category) over the [Giry monad](https://doi.org/10.1007/BFb0092872) under `composition real as algebra`. The loading morphism $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ carries a [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior whose [tensor-product](https://en.wikipedia.org/wiki/Tensor_product) factorisation $\mathrm{vec}(W) \sim \mathcal{N}(0, V \otimes U)$ expresses the prior as the product of two univariate Gaussians on the row and column axes. The morphism-valued prior surface treats the matrix as a first-class arrow and its prior as a measure on the hom-object $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{ObsDim})$.

## See Also

- [Probabilistic PCA](ppca.md), the isotropic-noise special case.
- [DSL Guide: Hierarchical Bayesian Models](../guides/programs-hierarchical.md#hierarchical-models-with-parametric-templates) for the morphism-valued prior surface.
