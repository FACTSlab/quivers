# Probabilistic Principal Component Analysis

## Overview

[Probabilistic PCA](https://en.wikipedia.org/wiki/Probabilistic_principal_component_analysis) ([Tipping & Bishop 1999](https://doi.org/10.1111/1467-9868.00196)) factors a data matrix through a low-rank loading matrix $W$ acting on a per-item latent code $z$:

$$
z_i \sim \mathcal{N}(0, I_K), \quad y_i \mid z_i \sim \mathcal{N}(W z_i, \sigma^2 I_D).
$$

The model is identifiable up to a $K \times K$ orthogonal rotation of $W$; the maximum-likelihood $W$ recovers the leading-$K$ [principal components](https://en.wikipedia.org/wiki/Principal_component_analysis) scaled by $\sqrt{\lambda_k - \sigma^2}$, where $\lambda_k$ are the data covariance eigenvalues. PPCA differs from [factor analysis](factor-analysis.md) only in the observation noise: PPCA uses a single isotropic scalar $\sigma$, factor analysis a free diagonal $\psi$.

In quivers, the loading matrix is a [`LatentMorphism`](../api/core/morphisms.md) $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ and the per-item code is $Z : \mathsf{Item} \to \mathsf{LatentDim}$. The model mean is the composition $Z \mathbin{>>} W`. The exported program below supplies explicit Normal priors through `Z_mat` and `W_mat`; the top-level morphisms themselves have no family declaration.

## QVR source

```qvr
# Probabilistic Principal Component Analysis
#
# Probabilistic PCA factors a data matrix through a low-rank
# loading matrix and a per-item latent code, so the model mean
# is a morphism composition under the real algebra.
#
# Structural form:
#
#   Z     : Item      -> LatentDim       per-item latent code
#   W     : LatentDim -> ObsDim          matrix-normal loadings
#   ppca  = Z >> W                       low-rank linear mean
#
# The loading matrix carries a matrix-normal prior whose
# Kronecker covariance V (x) U expresses independent row and
# column correlation. The model is identifiable up to a K x K
# orthogonal rotation of W; the maximum-likelihood W recovers
# the leading-K principal components scaled by sqrt(lambda_k -
# sigma^2).
#
# Reference: [Tipping and Bishop 1999](https://doi.org/10.1111/1467-9868.00196).

composition real [level=algebra]

object LatentDim : FinSet 2
object ObsDim : FinSet 5
object Item : FinSet 32
object Resp : FinSet 160
object Val : Real 1

morphism Z : Item -> LatentDim [role=latent]

morphism W : LatentDim -> ObsDim [role=latent]

define ppca = Z >> W

# Probabilistic surface: every entry of the loading matrix and
# per-item latent code carries an independent Normal(0, 1) prior
# (the matrix-normal special case with V = U = I), the isotropic
# noise scale sigma carries a HalfCauchy(2.5) prior, and the
# observed response is scored under Normal(mu, sigma) where mu is
# the per-row inner product Z[i] . W[d]. The loading matrix is
# sampled transposed (ObsDim -> LatentDim) so the per-Resp inner
# product is two compatible Resp-by-LatentDim gathers.
program ppca_program : Resp -> Val
    sample sigma <- HalfCauchy(2.5)
    sample Z_mat : Item <- Normal(0.0, 1.0) [over=LatentDim, iid_over=Item]
    sample W_mat : ObsDim <- Normal(0.0, 1.0) [over=LatentDim, iid_over=ObsDim]

    let z_row = Z_mat[item_idx]
    let w_row = W_mat[obs_idx]
    let mu = sum(z_row * w_row)

    observe y : Resp <- Normal(mu, sigma)
    return y

export ppca_program
```

## Walkthrough

An [object](../guides/dsl-declarations.md#object) name in QVR has no fixed reading; each position it can occupy gives it a different one, and this file uses four. In `morphism Z : Item -> LatentDim` the codomain is the arrow's own value space, which is what makes `Z` an `Item x LatentDim` real matrix. In `[over=LatentDim]` the object fixes the *event width* instead, the two coordinates a latent code or a loading row spans. In `observe y : Resp <- Normal(mu, sigma)` it fixes the *plate extent*, the 160 response rows, one per `(Item, ObsDim)` cell, which is why an object in that slot must be discrete; what a row holds is not `Resp`'s business but the family's, and `Normal` is what makes each response real. And in the program signature `Resp -> Val` the codomain names the *value space* of what the program returns: `return y` gives back one real response per row, so that space is `Real 1`, declared as `object Val : Real 1`. Reading that codomain as an index instead is the misstep to avoid: a signature `Resp -> Resp` would claim the program returns an element of the response index set, which is a category error the compiler cannot catch, since its only condition on `return` is that the name be bound and it never compares the returned value against the declared codomain.

The two latent declarations introduce the per-item code and the loading matrix as first-class arrows. The composition `Z >> W` is real-algebra matmul: under `composition real [level=algebra]` the `(i, d)` entry of the resulting `Item x ObsDim` tensor is exactly $\sum_k Z_{i,k} W_{k,d}$, the PPCA model mean.

The following matrix-normal declaration is an optional alternative, not part of the source above:

<!-- compile: false -->
```qvr
morphism W : LatentDim -> ObsDim [role=latent] ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

It would place a [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on the loading matrix. The runnable `ppca_program` uses indexed Normal draws instead.

The PPCA / factor analysis distinction lives in the choice of downstream observation kernel applied to the matmul mean: a single shared scalar `sigma` for PPCA, a free diagonal `psi_d` for factor analysis. The morphism surface itself (the `Z >> W` matmul) is shared.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/ppca.qvr")
model = prog.morphism

N, K, Dn = 32, 2, 5
ND = N * Dn
item_idx = torch.arange(N).repeat_interleave(Dn)
obs_idx = torch.arange(Dn).repeat(N)

true_sigma = 0.2
true_Z_mat = torch.randn(N, K)
true_W_mat = torch.randn(Dn, K)
mu_true = (true_Z_mat[item_idx] * true_W_mat[obs_idx]).sum(dim=-1)
y = torch.distributions.Normal(mu_true, true_sigma).sample()

observations = {
    "y": y,
    "item_idx": item_idx,
    "obs_idx": obs_idx,
}
x_in = torch.zeros(ND, 1)
```

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"y", "item_idx", "obs_idx"})
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


## Categorical perspective

PPCA is a pair of arrows in a real-algebra category: the per-item code $Z : \mathsf{Item} \to \mathsf{LatentDim}$ and the loading $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$. Their composition $Z \mathbin{>>} W$ is the [`LatentMorphism`](../api/core/morphisms.md) $\mathsf{Item} \to \mathsf{ObsDim}$ whose tensor is the model mean. Marginalising the latent factor under an isotropic noise kernel recovers the closed-form covariance $W^\top W + \sigma^2 I$ on the observation side.

The morphism-valued [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on $W$ is a measure on the hom-object $\mathbf{Kern}(\mathsf{LatentDim}, \mathsf{ObsDim})$, treating the loading as a first-class arrow rather than a flat vector of entries.

## See also

- [Factor Analysis](factor-analysis.md), the free-diagonal generalisation.
- [DSL Guide](../guides/dsl-overview.md) for the morphism-valued prior surface.


## References

- Michael E. Tipping and Christopher M. Bishop. 1999. Probabilistic principal component analysis. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, 61(3):611–622.
