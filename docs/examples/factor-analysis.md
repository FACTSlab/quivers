# Factor Analysis

## Overview

Classical [factor analysis](https://en.wikipedia.org/wiki/Factor_analysis) ([Spearman 1904](https://doi.org/10.2307/1412159), [Bartholomew, Knott, and Moustaki 2011](https://doi.org/10.1002/9781119970583)) decomposes a $D$-dimensional observation as a linear-Gaussian transformation of a $K$-dimensional latent factor plus a free diagonal idiosyncratic noise:

$$
z \sim \mathcal{N}(0, I_K), \quad y \mid z \sim \mathcal{N}(W z + b, \mathrm{diag}(\psi)).
$$

The free diagonal $\psi$ distinguishes factor analysis from [probabilistic PCA](ppca.md), whose covariance is isotropic. The loading matrix is the canonical example of a morphism-valued latent in quivers: it is declared as a morphism $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ carrying a [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) prior over its row and column covariances.

## QVR Source

```qvr
object LatentDim : 2
object ObsDim : 3
object Item : 200

latent W : LatentDim -> ObsDim ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)

program factor_analysis : Item -> Item
    psi_1 <- HalfCauchy(1.0)
    psi_2 <- HalfCauchy(1.0)
    psi_3 <- HalfCauchy(1.0)

    z_1 : Item <- Normal(0.0, 1.0)
    z_2 : Item <- Normal(0.0, 1.0)

    b_1 <- Normal(0.0, 1.0)
    b_2 <- Normal(0.0, 1.0)
    b_3 <- Normal(0.0, 1.0)

    w_1_1 <- Normal(0.0, 1.0)
    w_1_2 <- Normal(0.0, 1.0)
    w_2_1 <- Normal(0.0, 1.0)
    w_2_2 <- Normal(0.0, 1.0)
    w_3_1 <- Normal(0.0, 1.0)
    w_3_2 <- Normal(0.0, 1.0)

    let mu_1 = b_1 + w_1_1 * z_1 + w_1_2 * z_2
    let mu_2 = b_2 + w_2_1 * z_1 + w_2_2 * z_2
    let mu_3 = b_3 + w_3_1 * z_1 + w_3_2 * z_2

    observe y_1 : Item <- Normal(mu_1, psi_1)
    observe y_2 : Item <- Normal(mu_2, psi_2)
    observe y_3 : Item <- Normal(mu_3, psi_3)
    return psi_1

export factor_analysis
```

## Walkthrough

The top-level declaration

<!-- compile: false -->
```qvr
latent W : LatentDim -> ObsDim ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

introduces the loading matrix as a morphism with a [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) prior. The two axes named under `over (dom, cod)` bind positionally to the family's event axes: the singleton domain `LatentDim` is the row axis carrying the row-covariance argument, and the singleton codomain `ObsDim` is the column axis carrying the column-covariance argument. The [Kronecker covariance](https://en.wikipedia.org/wiki/Kronecker_product) structure expresses independent row and column correlation in the loadings, the natural prior for a low-rank factor decomposition.

Inside the program body, `z_1 : Item <- Normal(0.0, 1.0)` and `z_2 : Item <- Normal(0.0, 1.0)` declare two `Item`-indexed plates of per-item latent factor values on the $K = 2$ latent dimensions. The per-dimension intercepts `b_d` and free diagonal scales `psi_d` are scalar latents. The `let` steps assemble the linear predictor $\mu_d = b_d + \sum_k W_{d,k} z_{k}$ entrywise, and the three `observe y_d : Item <- Normal(mu_d, psi_d)` steps score the observed responses.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)

prog = load("docs/examples/source/factor_analysis.qvr")
model = prog.morphism

N, D, K = 200, 3, 2
W_true = torch.randn(D, K) * 0.7
z = torch.randn(N, K)
y_true = z @ W_true.T + 0.1 * torch.randn(N, D)
obs = {f"y_{d + 1}": y_true[:, d] for d in range(D)}

guide = AutoNormalGuide(model, observed_names=set(obs.keys()))
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=3e-2,
)
svi = SVI(model, guide, optim, ELBO())
for _ in range(1500):
    loss = svi.step(torch.zeros(N, 1), obs)

W_fit = torch.tensor([
    [guide._loc(f"w_{d + 1}_{k + 1}").item() for k in range(K)]
    for d in range(D)
])
print("fitted W Wt:", W_fit @ W_fit.T)
print("true   W Wt:", W_true @ W_true.T)
```

Factor analysis is identifiable only up to a $K \times K$ orthogonal rotation of $W$, so we check recovery in the [rotation-invariant covariance](https://en.wikipedia.org/wiki/Factor_analysis#Mathematical_model_of_the_same_factor_analysis) $W W^\top$ rather than $W$ itself.

## Categorical Perspective

The factor analysis model is a Kleisli morphism $\mathsf{Item} \to \mathcal{G}(\mathsf{Item})$ in the Kleisli category of the [Giry monad](https://doi.org/10.1007/BFb0092872). The loading-matrix morphism $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ carries a [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) prior whose Kronecker factorisation $\mathrm{vec}(W) \sim \mathcal{N}(0, V \otimes U)$ expresses the prior as the [tensor product](https://en.wikipedia.org/wiki/Tensor_product) of two univariate Gaussians on the row and column axes. The morphism-valued prior surface is the categorically natural way to express this: the matrix is treated as a first-class arrow, and its prior is a measure on the hom-object $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{ObsDim})$.

## See Also

- [Probabilistic PCA](ppca.md), the isotropic-noise special case.
- [DSL Guide: Hierarchical Bayesian Models](../guides/dsl.md#hierarchical-bayesian-models) for the plate-bind and morphism-valued prior surface.
