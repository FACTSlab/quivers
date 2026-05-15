# Probabilistic Matrix Factorization

## Overview

[Probabilistic matrix factorization](https://en.wikipedia.org/wiki/Probabilistic_matrix_factorization) ([Salakhutdinov & Mnih 2007](https://papers.nips.cc/paper/2007/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html)) is the standard latent-factor recommender model: each user and each item has a $K$-dimensional latent vector, the expected rating is their inner product, and observed ratings are Normal noise around that inner product:

$$
r_{u, m} \mid U_{:, u}, V_{:, m} \sim \mathcal{N}(\langle U_{:, u}, V_{:, m} \rangle, \sigma_{\text{obs}}^2).
$$

In quivers, the two factor matrices are arrows $U : \mathsf{LatentDim} \to \mathsf{User}$ and $V : \mathsf{LatentDim} \to \mathsf{Movie}$ carrying [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) priors. The bilinear score is the composition $U^\dagger \mathbin{>>} V : \mathsf{User} \to \mathsf{Movie}$, whose `(u, m)` entry is the inner product $\sum_k U_{k, u} V_{k, m}$. Under `algebra real` this composition is the canonical PMF rating-mean matmul.

## QVR Source

```qvr
algebra real

object LatentDim : 2
object User : 1000
object Movie : 500

latent U : LatentDim -> User ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent V : LatentDim -> Movie ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)

let pmf = U.dagger >> V

export pmf
```

## Walkthrough

The two top-level latent declarations introduce the user and item factor matrices as first-class arrows, each with a `K x N_side` tensor whose row covariance is the Kronecker factor on `LatentDim` and column covariance the Kronecker factor on the user / movie plate:

<!-- compile: false -->
```qvr
latent U : LatentDim -> User ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent V : LatentDim -> Movie ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

The `.dagger` modifier on $U$ transposes the morphism to $\mathsf{User} \to \mathsf{LatentDim}$. The composition `U.dagger >> V` contracts along `LatentDim` and recovers the full `(User, Movie)` score matrix; under `algebra real` this is a real matmul and the resulting tensor entry at `(u, m)` is exactly $\sum_k U_{k, u} V_{k, m}$.

Working over discrete `User` and `Movie` plates materialises the full dense score matrix. For very large catalogues the dense materialisation is wasteful and a per-rating gather is preferable; the morphism surface in quivers can lift that gather as a separate fibration $\mathsf{Rating} \to \mathsf{User} \times \mathsf{Movie}$ composed with the bilinear pmf morphism.

## Try it

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)

prog = load("docs/examples/source/pmf.qvr")
model = prog.morphism

# The model tensor materialises the full User x Movie score
# matrix S = U^T @ V. Fit it to a sampled rating matrix.
n_user, n_movie, K = 1000, 500, 2
U_true = torch.randn(K, n_user)
V_true = torch.randn(K, n_movie)
R = U_true.T @ V_true + 0.1 * torch.randn(n_user, n_movie)

opt = torch.optim.Adam(prog.parameters(), lr=5e-2)
for _ in range(300):
    opt.zero_grad()
    loss = (model.tensor - R).pow(2).mean()
    loss.backward()
    opt.step()

print("residual RMSE:", (model.tensor - R).pow(2).mean().sqrt().item())
```

The model is identifiable up to a `K x K` invertible reparameterisation of the latent space; we therefore check recovery in score space `U^T V` rather than in `U`, `V` separately.

## Categorical Perspective

The two factor matrices $U : \mathsf{LatentDim} \to \mathsf{User}$ and $V : \mathsf{LatentDim} \to \mathsf{Movie}$ are arrows in the discrete-object category whose priors live on the hom-objects $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{User})$ and $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{Movie})$. The rating likelihood couples them through the bilinear pairing realised as the composition

$$
\mathsf{User} \xrightarrow{U^\dagger} \mathsf{LatentDim} \xrightarrow{V} \mathsf{Movie}
$$

in the real algebra, whose tensor is the dense score matrix $U^\top V$. The morphism-valued [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) priors carry the [Kronecker covariance](https://en.wikipedia.org/wiki/Kronecker_product) assumed by the original PMF paper as first-class measures on the hom-objects.

## See Also

- [Factor Analysis](factor-analysis.md) for a single-side morphism-valued loading.
- [DSL Guide](../guides/dsl.md) for the morphism-valued prior surface and the [`.dagger`](../api/core/morphisms.md) transpose.
