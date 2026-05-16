# Factor Analysis

## Overview

Classical [factor analysis](https://en.wikipedia.org/wiki/Factor_analysis) ([Spearman 1904](https://doi.org/10.2307/1412159); [Bartholomew, Knott, and Moustaki 2011](https://doi.org/10.1002/9781119970583)) decomposes a $D$-dimensional observation as a linear-Gaussian transformation of a $K$-dimensional latent factor plus a free diagonal idiosyncratic noise:

$$
z_i \sim \mathcal{N}(0, I_K), \quad y_i \mid z_i \sim \mathcal{N}(W z_i, \mathrm{diag}(\psi)).
$$

The free diagonal $\psi$ distinguishes factor analysis from [probabilistic PCA](ppca.md), whose noise is isotropic. The loading matrix is the canonical example of a morphism-valued latent in quivers: declared as an arrow $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ with a [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) prior on its row and column covariances. The per-item latent factor is itself a learnable morphism $Z : \mathsf{Item} \to \mathsf{LatentDim}$, and the model mean is the composition $Z \mathbin{>>} W$.

## QVR Source

```qvr
algebra real

object LatentDim : 2
object ObsDim : 3
object Item : 200

latent Z : Item -> LatentDim
latent W : LatentDim -> ObsDim ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)

let factor_analysis = Z >> W

export factor_analysis
```

## Walkthrough

The two latent declarations introduce the per-item factor and the loading matrix as arrows. Under `algebra real` the composition `Z >> W` is a real-valued matmul: the `(i, d)` entry of the `Item x ObsDim` model tensor is $\sum_k Z_{i,k} W_{k,d}$, the factor analysis model mean.

The top-level morphism prior

<!-- compile: false -->
```qvr
latent W : LatentDim -> ObsDim ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

places a [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on the loading matrix. The two axes named under `over (dom, cod)` bind positionally to the family's event axes: `LatentDim` is the row axis carrying the row-covariance argument, `ObsDim` is the column axis carrying the column-covariance argument. The [Kronecker covariance](https://en.wikipedia.org/wiki/Kronecker_product) structure expresses independent row and column correlation in the loadings, the natural prior for a low-rank factor decomposition.

The distinction between factor analysis and PPCA appears in the observation noise kernel applied to the matmul mean: factor analysis uses a free diagonal $\psi_d$ per dimension; PPCA collapses that diagonal to a single isotropic scalar.

## Try it

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)

prog = load("docs/examples/source/factor_analysis.qvr")
model = prog.morphism

# Fit the deterministic low-rank composition Z @ W to a data
# matrix Y. Recovery is up to the rotation invariance of factor
# analysis, so we check W Wt rather than W itself.
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

W_fit = model.right.tensor
print("W^T W (fit):", W_fit.T @ W_fit)
print("W^T W (true):", W_true.T @ W_true)
```

Factor analysis is identifiable only up to a $K \times K$ orthogonal rotation of $W$, so the rotation-invariant covariance $W^\top W$ is the correct recovery target rather than $W$ itself.

## Categorical Perspective

The factor analysis mean is a composition $Z \mathbin{>>} W$ in the [Kleisli category](https://en.wikipedia.org/wiki/Kleisli_category) over the [Giry monad](https://doi.org/10.1007/BFb0092872) under `algebra real`. The loading morphism $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ carries a [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior whose [tensor-product](https://en.wikipedia.org/wiki/Tensor_product) factorisation $\mathrm{vec}(W) \sim \mathcal{N}(0, V \otimes U)$ expresses the prior as the product of two univariate Gaussians on the row and column axes. The morphism-valued prior surface treats the matrix as a first-class arrow and its prior as a measure on the hom-object $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{ObsDim})$.

## See Also

- [Probabilistic PCA](ppca.md), the isotropic-noise special case.
- [DSL Guide: Hierarchical Bayesian Models](../guides/programs-hierarchical.md#hierarchical-models-with-parametric-templates) for the morphism-valued prior surface.
