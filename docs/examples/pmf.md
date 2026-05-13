# Probabilistic Matrix Factorization

## Overview

[Probabilistic matrix factorization](https://en.wikipedia.org/wiki/Probabilistic_matrix_factorization) ([Salakhutdinov & Mnih 2007](https://papers.nips.cc/paper/2007/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html)) is the standard latent-factor recommender model: each user and each item has a $K$-dimensional latent vector, the expected rating is their inner product, and observed ratings are Normal noise around that inner product. The two factor matrices are declared as morphism-valued latents with [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) priors.

## QVR Source

```qvr
object LatentDim : 2
object User : 1000
object Movie : 500
object Rating : 100000

latent U : LatentDim -> User ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent V : LatentDim -> Movie ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)

program pmf : Rating -> Rating
    sigma_user <- HalfCauchy(1.0)
    sigma_item <- HalfCauchy(1.0)
    sigma_obs <- HalfCauchy(1.0)

    u_1 : User <- Normal(0.0, sigma_user)
    u_2 : User <- Normal(0.0, sigma_user)
    v_1 : Movie <- Normal(0.0, sigma_item)
    v_2 : Movie <- Normal(0.0, sigma_item)

    let user_factor_1 = u_1[user_idx]
    let user_factor_2 = u_2[user_idx]
    let item_factor_1 = v_1[movie_idx]
    let item_factor_2 = v_2[movie_idx]

    let mu = user_factor_1 * item_factor_1 + user_factor_2 * item_factor_2
    observe r : Rating <- Normal(mu, sigma_obs)
    return sigma_obs

export pmf
```

## Walkthrough

The two top-level morphism declarations

<!-- compile: false -->
```qvr
latent U : LatentDim -> User ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent V : LatentDim -> Movie ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

place [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) priors on the user and item factor matrices: each is a $K \times N_\mathrm{side}$ tensor whose row covariance is the Kronecker factor on `LatentDim` and column covariance is the Kronecker factor on the user / movie plate. The morphism-valued surface makes the two factor matrices first-class arrows on which subsequent operations (composition, transformation, residuation) can act.

Inside the program body, per-side scale hyperpriors (`sigma_user`, `sigma_item`) and per-latent-dim factor plates (`u_1, u_2, v_1, v_2`) make the rating model end-to-end differentiable. The per-rating row gather `u_1[user_idx]`, `v_1[movie_idx]`, ... pulls back the plate variables along the per-rating fibrations `user_idx : Rating -> User` and `movie_idx : Rating -> Movie`; the elementwise inner product is the rating mean.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)

prog = load("docs/examples/source/pmf.qvr")
model = prog.morphism

n_user, n_movie, n_rating, K = 100, 50, 2000, 2
U_true = torch.randn(n_user, K)
V_true = torch.randn(n_movie, K)
user_idx = torch.randint(0, n_user, (n_rating,))
movie_idx = torch.randint(0, n_movie, (n_rating,))
mu_true = (U_true[user_idx] * V_true[movie_idx]).sum(-1)
r = mu_true + 0.1 * torch.randn(n_rating)

obs = {"user_idx": user_idx, "movie_idx": movie_idx, "r": r}
guide = AutoNormalGuide(model, observed_names={"r"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=2e-2,
)
svi = SVI(model, guide, optim, ELBO())
for _ in range(500):
    loss = svi.step(torch.zeros(n_rating, 1), obs)

mu_pred = (
    torch.stack([guide._loc("u_1"), guide._loc("u_2")], dim=-1)[user_idx]
    * torch.stack([guide._loc("v_1"), guide._loc("v_2")], dim=-1)[movie_idx]
).sum(-1)
print("predicted vs true RMSE:", (mu_pred - mu_true).pow(2).mean().sqrt().item())
```

## Categorical Perspective

The two factor matrices $U : \mathsf{LatentDim} \to \mathsf{User}$ and $V : \mathsf{LatentDim} \to \mathsf{Movie}$ are arrows in the discrete-object category whose priors live on the hom-objects $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{User})$ and $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{Movie})$. The rating likelihood couples them through a bilinear pairing, the composite

$$
\mathsf{Rating} \xrightarrow{(\mathsf{user\_idx}, \mathsf{movie\_idx})} \mathsf{User} \times \mathsf{Movie} \xrightarrow{U^\top V} \mathbb{R} \xrightarrow{\mathcal{N}(\cdot, \sigma_\mathrm{obs}^2)} \mathcal{G}(\mathbb{R}).
$$

The morphism-valued [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) priors are the natural categorical handle for the Kronecker covariance assumed by the original PMF paper.

## See Also

- [Factor Analysis](factor-analysis.md) for a single-side morphism-valued loading.
- [DSL Guide](../guides/dsl.md) for the morphism-valued prior surface and plate-gather idiom.
