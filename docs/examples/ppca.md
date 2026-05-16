# Probabilistic Principal Component Analysis

## Overview

[Probabilistic PCA](https://en.wikipedia.org/wiki/Probabilistic_principal_component_analysis) ([Tipping & Bishop 1999](https://doi.org/10.1111/1467-9868.00196)) factors a data matrix through a low-rank loading matrix $W$ acting on a per-item latent code $z$:

$$
z_i \sim \mathcal{N}(0, I_K), \quad y_i \mid z_i \sim \mathcal{N}(W z_i, \sigma^2 I_D).
$$

The model is identifiable up to a $K \times K$ orthogonal rotation of $W$; the maximum-likelihood $W$ recovers the leading-$K$ [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) scaled by $\sqrt{\lambda_k - \sigma^2}$, where $\lambda_k$ are the data covariance eigenvalues. PPCA differs from [factor analysis](factor-analysis.md) only in the observation noise: PPCA uses a single isotropic scalar $\sigma$, factor analysis a free diagonal $\psi$.

In quivers, the loading matrix is a [`LatentMorphism`](../api/core/morphisms.md) $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ carrying a [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) prior, and the per-item latent code is itself a learnable morphism $Z : \mathsf{Item} \to \mathsf{LatentDim}$. The model mean is the composition $Z \mathbin{>>} W$, evaluated under `algebra real` as the canonical PPCA matmul.

## QVR Source

```qvr
algebra real

object LatentDim : 2
object ObsDim : 3
object Item : 200

latent Z : Item -> LatentDim
latent W : LatentDim -> ObsDim ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)

let ppca = Z >> W

export ppca
```

## Walkthrough

The two latent declarations introduce the per-item code and the loading matrix as first-class arrows. The composition `Z >> W` is real-algebra matmul: under `algebra real` the `(i, d)` entry of the resulting `Item x ObsDim` tensor is exactly $\sum_k Z_{i,k} W_{k,d}$, the PPCA model mean.

The matrix-normal prior

<!-- compile: false -->
```qvr
latent W : LatentDim -> ObsDim ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

places a [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on the loading matrix with the dom and cod axes bound positionally to the row and column covariance arguments. The [Kronecker structure](https://en.wikipedia.org/wiki/Kronecker_product) $V \otimes U$ expresses independent row and column correlation in the loadings.

The PPCA / factor analysis distinction lives in the choice of downstream observation kernel applied to the matmul mean: a single shared scalar `sigma` for PPCA, a free diagonal `psi_d` for factor analysis. The morphism surface itself (the `Z >> W` matmul) is shared.

## Try it

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)

prog = load("docs/examples/source/ppca.qvr")
model = prog.morphism

# The model tensor materialises `Z @ W`: the per-item, per-dim
# PPCA mean. Train it as a deterministic low-rank fit to a
# data matrix Y by gradient descent.
N, D, K = 200, 3, 2
W_true = torch.randn(K, D)
Z_true = torch.randn(N, K)
Y = Z_true @ W_true + 0.1 * torch.randn(N, D)

opt = torch.optim.Adam(prog.parameters(), lr=5e-2)
for _ in range(500):
    opt.zero_grad()
    loss = (model.tensor - Y).pow(2).mean()
    loss.backward()
    opt.step()

print("residual MSE:", (model.tensor - Y).pow(2).mean().item())
```

The mean-squared residual approaches the irreducible noise floor; the recovered factorisation `Z @ W` matches the data up to the $K \times K$ rotation invariance.

## Categorical Perspective

PPCA is a pair of arrows in a real-algebra category: the per-item code $Z : \mathsf{Item} \to \mathsf{LatentDim}$ and the loading $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$. Their composition $Z \mathbin{>>} W$ is the [`LatentMorphism`](../api/core/morphisms.md) $\mathsf{Item} \to \mathsf{ObsDim}$ whose tensor is the model mean. Marginalising the latent factor under an isotropic noise kernel recovers the closed-form covariance $W^\top W + \sigma^2 I$ on the observation side.

The morphism-valued [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on $W$ is a measure on the hom-object $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{ObsDim})$, treating the loading as a first-class arrow rather than a flat vector of entries.

## See Also

- [Factor Analysis](factor-analysis.md), the free-diagonal generalisation.
- [DSL Guide](../guides/dsl-overview.md) for the morphism-valued prior surface.
