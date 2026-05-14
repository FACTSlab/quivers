# Probabilistic Principal Component Analysis

## Overview

[Probabilistic PCA](https://en.wikipedia.org/wiki/Probabilistic_principal_component_analysis) ([Tipping & Bishop 1999](https://doi.org/10.1111/1467-9868.00196)) is the isotropic-noise special case of [factor analysis](factor-analysis.md): a single scalar $\sigma$ controls the observation noise on every output dimension, in contrast to the free diagonal of factor analysis. The generative model is

$$
z \sim \mathcal{N}(0, I_K), \quad y \mid z \sim \mathcal{N}(W z + b, \sigma^2 I_D).
$$

The model is identifiable up to a $K \times K$ orthogonal rotation of $W$; the maximum-likelihood $W$ recovers the leading-$K$ [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) scaled by $\sqrt{\lambda_k - \sigma^2}$ where $\lambda_k$ are the data covariance eigenvalues. The loading matrix is again declared as a morphism-valued latent with a [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) prior.

## QVR Source

```qvr
object LatentDim : 2
object ObsDim : 3
object Item : 200

latent W : LatentDim -> ObsDim ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)

program ppca : Item -> Item
    sigma <- HalfCauchy(1.0)

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

    observe y_1 : Item <- Normal(mu_1, sigma)
    observe y_2 : Item <- Normal(mu_2, sigma)
    observe y_3 : Item <- Normal(mu_3, sigma)
    return sigma

export ppca
```

## Walkthrough

The PPCA constraint is encoded by reusing a single scalar `sigma <- HalfCauchy(1.0)` across all three observation kernels, in contrast to the per-dimension `psi_d` of factor analysis. Everything else mirrors the factor analysis surface: the morphism-valued declaration

<!-- compile: false -->
```qvr
latent W : LatentDim -> ObsDim ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

places a [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) prior on the loading matrix with the dom / cod axes bound positionally to the row and column covariance arguments. Per-item latent factors are declared as `Item`-indexed plates; the linear predictor is assembled entrywise in `let` steps; and the three `observe y_d : Item <- Normal(mu_d, sigma)` steps share the same isotropic scale.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)

prog = load("docs/examples/source/ppca.qvr")
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
print("fitted sigma:", guide._loc("sigma").exp().item())
```

The PPCA covariance is $W W^\top + \sigma^2 I$, so we check both the recovered $W W^\top$ (up to rotation) and the recovered isotropic scale.

## Categorical Perspective

PPCA is the isotropic restriction of [factor analysis](factor-analysis.md): the per-dimension noise morphism $\mathsf{ObsDim} \to \mathcal{G}(\mathsf{ObsDim})$ is a scalar-shared kernel rather than a $\mathsf{ObsDim}$-indexed plate. The loading-matrix morphism $W$ retains the same [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) prior, expressing a prior measure on $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{ObsDim})$. Marginalising the latent factor $z$ gives the closed-form covariance $W W^\top + \sigma^2 I$, recovered as the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) along the projection $\mathsf{Item} \times \mathsf{LatentDim} \to \mathsf{Item}$.

## See Also

- [Factor Analysis](factor-analysis.md), the free-diagonal generalisation.
- [DSL Guide](../guides/dsl.md) for the morphism-valued prior surface and plate-bind syntax.
