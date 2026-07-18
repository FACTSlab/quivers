# Probabilistic Matrix Factorization

## Overview

[Probabilistic matrix factorization](https://en.wikipedia.org/wiki/Probabilistic_matrix_factorization) ([Salakhutdinov & Mnih 2007](https://papers.nips.cc/paper/2007/hash/d7322ed717dedf1eb4e6e52a37ea7bcd-Abstract.html)) is the standard latent-factor recommender model: each user and each item has a $K$-dimensional latent vector, the expected rating is their inner product, and observed ratings are Normal noise around that inner product:

$$
r_{u, m} \mid U_{:, u}, V_{:, m} \sim \mathcal{N}(\langle U_{:, u}, V_{:, m} \rangle, \sigma_{\text{obs}}^2).
$$

In quivers, the two factor matrices are arrows $U : \mathsf{LatentDim} \to \mathsf{User}$ and $V : \mathsf{LatentDim} \to \mathsf{Movie}$ carrying [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) priors. The bilinear score is the composition $U^\dagger \mathbin{>>} V : \mathsf{User} \to \mathsf{Movie}$, whose `(u, m)` entry is the inner product $\sum_k U_{k, u} V_{k, m}$. Under `composition real [level=algebra]` this composition is the canonical PMF rating-mean matmul.

## QVR Source

```qvr
composition real [level=algebra]

object LatentDim : FinSet 2
object User, Movie : FinSet 8

morphism U : LatentDim -> User [role=latent]
morphism V : LatentDim -> Movie [role=latent]

define pmf = U.dagger >> V

export pmf
```

## Walkthrough

The two top-level latent declarations introduce the user and item factor matrices as first-class arrows, each with a `K x N_side` tensor whose row covariance is the Kronecker factor on `LatentDim` and column covariance the Kronecker factor on the user / movie plate:

<!-- compile: false -->
```qvr
morphism U : LatentDim -> User [role=latent, over=[dom, cod]] ~ MatrixNormal(0.0, 1.0, 1.0)
morphism V : LatentDim -> Movie [role=latent, over=[dom, cod]] ~ MatrixNormal(0.0, 1.0, 1.0)
```

The `.dagger` modifier on $U$ transposes the morphism to $\mathsf{User} \to \mathsf{LatentDim}$. The composition `U.dagger >> V` contracts along `LatentDim` and recovers the full `(User, Movie)` score matrix; under `composition real [level=algebra]` this is a real matmul and the resulting tensor entry at `(u, m)` is exactly $\sum_k U_{k, u} V_{k, m}$.

Working over discrete `User` and `Movie` plates materialises the full dense score matrix. For very large catalogues the dense materialisation is wasteful and a per-rating gather is preferable; the morphism surface in quivers can lift that gather as a separate fibration $\mathsf{Rating} \to \mathsf{User} \times \mathsf{Movie}$ composed with the bilinear pmf morphism.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


The QVR source declares the bilinear score $U^\dagger \mathbin{>>} V$ as a deterministic morphism. To get a Bayesian likelihood surface, the snippets below wrap the score with a sparse rating observation: a single Normal site $r_{u, m} \sim \mathcal{N}(S_{u, m}, \sigma^2)$ evaluated at a small batch of `(user, movie)` index pairs. The latent factors $U$, $V$ are the program's learnable parameters; the [`RatingLikelihood`](../api/continuous/morphisms.md) wrapper exposes them to SVI and NUTS without touching the QVR source.

### Generating synthetic data

Pick true $U$, $V$ factor matrices for an 8-user / 8-item catalogue, then draw a handful of `(user, movie, rating)` triples by sampling distinct index pairs (under 10% of the 64 cells) and adding Normal observation noise around the bilinear inner product.

```python
import torch

torch.manual_seed(0)

n_user, n_movie, K = 8, 8, 2
U_true = 0.5 * torch.randn(K, n_user)
V_true = 0.5 * torch.randn(K, n_movie)

n_obs = 5  # under 10% of n_user * n_movie = 64 cells
flat   = torch.randperm(n_user * n_movie)[:n_obs]
u_idx  = (flat // n_movie).long()
m_idx  = (flat %  n_movie).long()
sigma  = 0.1
r_obs  = (U_true[:, u_idx] * V_true[:, m_idx]).sum(0) + sigma * torch.randn(n_obs)
print("density:", n_obs / (n_user * n_movie))
print("ratings:", r_obs.tolist())
```

### SVI fit

Wrap the deterministic pmf score with a Normal rating likelihood inside a [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram), then fit with [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) + [`ELBO`](../api/inference/elbo.md#quivers.inference.objectives.ELBO) + [`SVI`](../api/inference/svi.md#svi). The guide is degenerate here (no `sample` latent sites in the program), so the optimiser walks $U$, $V$ as point variables; the ELBO collapses to the expected negative log-likelihood plus a constant. Print the initial and final ELBO to confirm the optimiser is moving in the right direction.

```python
import torch
import torch.distributions as D
from quivers.dsl import load
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import Unit
from quivers.inference import AutoNormalGuide, ELBO, SVI


class RatingLikelihood(ContinuousMorphism):
    """Normal observation around the bilinear score at one (user, movie) pair."""

    def __init__(self, pmf_prog, sigma=0.1):
        super().__init__(Unit, Euclidean(name="Rating", dim=1))
        self._prog = pmf_prog  # holds U, V as registered parameters
        self._sigma = sigma

    def rsample(self, x, sample_shape=None):
        u = x[..., 0].long()
        m = x[..., 1].long()
        return self._prog.morphism.tensor[u, m] + self._sigma * torch.randn(u.shape[0])

    def log_prob(self, x, y):
        u = x[..., 0].long()
        m = x[..., 1].long()
        mu = self._prog.morphism.tensor[u, m]
        yy = y.squeeze(-1) if y.dim() > 1 and y.shape[-1] == 1 else y
        return D.Normal(mu, self._sigma).log_prob(yy)


torch.manual_seed(0)
n_user, n_movie, K = 8, 8, 2
U_true = 0.5 * torch.randn(K, n_user)
V_true = 0.5 * torch.randn(K, n_movie)
n_obs = 5
flat   = torch.randperm(n_user * n_movie)[:n_obs]
u_idx  = (flat // n_movie).long()
m_idx  = (flat %  n_movie).long()
sigma  = 0.1
r_obs  = (U_true[:, u_idx] * V_true[:, m_idx]).sum(0) + sigma * torch.randn(n_obs)

torch.manual_seed(1)
prog = load("docs/examples/source/pmf.qvr")
lik = RatingLikelihood(prog, sigma=sigma)
inner = MonadicProgram(
    domain=Euclidean(name="Ix", dim=2),
    codomain=Euclidean(name="Rating", dim=1),
    steps=[(("rating",), lik, None, True)],
    return_vars=("rating",),
)
x = torch.stack([u_idx.float(), m_idx.float()], dim=-1)
obs = {"rating": r_obs.unsqueeze(-1)}

guide = AutoNormalGuide(inner, observed_names={"rating"})
optim = torch.optim.Adam(
    list(inner.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(inner, guide, optim, ELBO())
loss0 = svi.step(x, obs)
for _ in range(200):
    loss = svi.step(x, obs)
print(f"ELBO loss: {loss0:.2f} -> {loss:.2f}")
```

The latent factors are identifiable only up to a $K \times K$ invertible reparameterisation, so the fitted $U$, $V$ will differ from the truths even when the score matrix has converged.

### NUTS posterior

The PMF program exposes $U$, $V$ as `[role=latent]` parameters with no explicit prior. To run NUTS we lift them into Normal-prior sample sites via [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) and target the lifted Bayesian model: the lift wraps every parameter as one flat Normal-prior site, after which standard MCMC machinery applies uniformly.

```python
import torch
import torch.distributions as D
from quivers.dsl import load
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import Unit
from quivers.inference import MCMC, NUTSKernel
from quivers.inference import bayesian_lift_parameters


class RatingLikelihood(ContinuousMorphism):
    def __init__(self, pmf_prog, sigma=0.1):
        super().__init__(Unit, Euclidean(name="Rating", dim=1))
        self._prog = pmf_prog
        self._sigma = sigma

    def rsample(self, x, sample_shape=None):
        u = x[..., 0].long()
        m = x[..., 1].long()
        return self._prog.morphism.tensor[u, m] + self._sigma * torch.randn(u.shape[0])

    def log_prob(self, x, y):
        u = x[..., 0].long()
        m = x[..., 1].long()
        mu = self._prog.morphism.tensor[u, m]
        yy = y.squeeze(-1) if y.dim() > 1 and y.shape[-1] == 1 else y
        return D.Normal(mu, self._sigma).log_prob(yy)


torch.manual_seed(0)
n_user, n_movie, K = 8, 8, 2
U_true = 0.5 * torch.randn(K, n_user)
V_true = 0.5 * torch.randn(K, n_movie)
n_obs = 5
flat   = torch.randperm(n_user * n_movie)[:n_obs]
u_idx  = (flat // n_movie).long()
m_idx  = (flat %  n_movie).long()
sigma  = 0.1
r_obs  = (U_true[:, u_idx] * V_true[:, m_idx]).sum(0) + sigma * torch.randn(n_obs)

torch.manual_seed(2)
prog = load("docs/examples/source/pmf.qvr")
lik = RatingLikelihood(prog, sigma=sigma)
inner = MonadicProgram(
    domain=Euclidean(name="Ix", dim=2),
    codomain=Euclidean(name="Rating", dim=1),
    steps=[(("rating",), lik, None, True)],
    return_vars=("rating",),
)
x = torch.stack([u_idx.float(), m_idx.float()], dim=-1)
obs = {"rating": r_obs.unsqueeze(-1)}

lifted, lx, lobs = bayesian_lift_parameters(inner, x, obs, prior_scale=1.0)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
result = mc.run(lifted, lx, lobs)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
```


## Categorical Perspective

The two factor matrices $U : \mathsf{LatentDim} \to \mathsf{User}$ and $V : \mathsf{LatentDim} \to \mathsf{Movie}$ are arrows in the discrete-object category whose priors live on the hom-objects $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{User})$ and $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{Movie})$. The rating likelihood couples them through the bilinear pairing realised as the composition

$$
\mathsf{User} \xrightarrow{U^\dagger} \mathsf{LatentDim} \xrightarrow{V} \mathsf{Movie}
$$

in the real algebra, whose tensor is the dense score matrix $U^\top V$. The morphism-valued [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) priors carry the [Kronecker covariance](https://en.wikipedia.org/wiki/Kronecker_product) assumed by the original PMF paper as first-class measures on the hom-objects.

## See Also

- [Factor Analysis](factor-analysis.md) for a single-side morphism-valued loading.
- [DSL Guide](../guides/dsl-overview.md) for the morphism-valued prior surface and the [`.dagger`](../api/core/morphisms.md) transpose.
