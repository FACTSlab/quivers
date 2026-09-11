# Factor Analysis

## Overview

Classical [factor analysis](https://en.wikipedia.org/wiki/Factor_analysis) ([Spearman, 1904](https://www.jstor.org/stable/1412107); [Bartholomew, Knott, and Moustaki, 2011](https://doi.org/10.1002/9781119970583)) decomposes a $D$-dimensional observation as a linear-Gaussian transformation of a $K$-dimensional latent factor plus diagonal idiosyncratic noise:

$$
z_i \sim \mathcal{N}(0, I_K), \quad y_i \mid z_i \sim \mathcal{N}(W z_i, \mathrm{diag}(\psi)).
$$

The runnable QVR program below uses one scalar `sigma`, not a free diagonal $\psi$. Its observation-noise structure is thus isotropic and matches [probabilistic PCA](ppca.md), despite the example's historical name. The top-level arrows $Z : \mathsf{Item} \to \mathsf{LatentDim}$ and $W : \mathsf{LatentDim} \to \mathsf{ObsDim}$ demonstrate the corresponding low-rank composition.

## QVR source

```qvr
# Factor Analysis
#
# Classical factor analysis decomposes a D-dimensional
# observation as a linear-Gaussian transformation of a
# K-dimensional latent factor. The model mean factors through a
# learnable loading matrix and a per-item latent code, expressed
# here as a morphism composition under the real algebra.
#
# Structural form:
#
#   Z     : Item      -> LatentDim       per-item latent code
#   W     : LatentDim -> ObsDim          matrix-normal loadings
#   model = Z >> W                       linear-Gaussian mean
#
# The loading matrix carries a matrix-normal prior whose
# Kronecker covariance V (x) U expresses independent row and
# column correlation, the natural prior for a low-rank factor
# decomposition. Factor analysis pairs Z >> W with a free
# diagonal noise (one psi per ObsDim coordinate), whereas PPCA
# collapses that diagonal to a single isotropic scalar; both
# choices are downstream observation kernels rather than
# features of the matrix-valued mean expressed here.

composition real [level=algebra]

object LatentDim : FinSet 2
object ObsDim : FinSet 5
object Item : FinSet 32
object Resp : FinSet 160
object Val : Real 1

morphism Z : Item -> LatentDim [role=latent]

morphism W : LatentDim -> ObsDim [role=latent]

define factor_analysis = Z >> W

# Probabilistic surface: every entry of the loading matrix and
# per-item latent code carries an independent Normal(0, 1) prior
# (the matrix-normal special case with V = U = I), the noise
# scale carries a HalfCauchy(2.5) prior, and the observed
# response is scored under Normal(mu, sigma) where mu is the
# per-row inner product Z[i] . W[d]. The loading matrix is
# sampled transposed (ObsDim -> LatentDim) so the per-Resp
# inner product is two compatible Resp-by-LatentDim gathers.
program factor_analysis_program : Resp -> Val
    sample sigma <- HalfCauchy(2.5)
    sample Z_mat : Item <- Normal(0.0, 1.0) [over=LatentDim, iid_over=Item]
    sample W_mat : ObsDim <- Normal(0.0, 1.0) [over=LatentDim, iid_over=ObsDim]

    let z_row = Z_mat[item_idx]
    let w_row = W_mat[obs_idx]
    let mu = sum(z_row * w_row)

    observe y : Resp <- Normal(mu, sigma)
    return y

export factor_analysis_program
```

## Walkthrough

An [object](../guides/dsl-declarations.md#object) name in QVR has no fixed reading; each position it can occupy gives it a different one, and this file uses four. In `morphism Z : Item -> LatentDim` the codomain is the arrow's own value space, which is what makes `Z` an `Item x LatentDim` real matrix. In `[over=LatentDim]` the object fixes the *event width* instead, the two coordinates a latent code or a loading row spans. In `observe y : Resp <- Normal(mu, sigma)` it fixes the *plate extent*, the 160 response rows, one per `(Item, ObsDim)` cell, which is why an object in that slot must be discrete; what a row holds is not `Resp`'s business but the family's, and `Normal` is what makes each response real. And in the program signature `Resp -> Val` the codomain names the *value space* of what the program returns: `return y` gives back one real response per row, so that space is `Real 1`, declared as `object Val : Real 1`. Reading that codomain as an index instead is the misstep to avoid: a signature `Resp -> Resp` would claim the program returns an element of the response index set, which is a category error the compiler cannot catch, since its only condition on `return` is that the name be bound and it never compares the returned value against the declared codomain.

The two latent declarations introduce the per-item factor and the loading matrix as arrows. Under `composition real [level=algebra]` the composition `Z >> W` is a real-valued matmul: the `(i, d)` entry of the `Item x ObsDim` model tensor is $\sum_k Z_{i,k} W_{k,d}$, the factor analysis model mean.

The following block is an optional extension, not part of the source shown above:

<!-- compile: false -->
```qvr
morphism W : LatentDim -> ObsDim [role=latent] ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

It would place a [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) prior on the loading matrix. The actual `factor_analysis_program` instead samples `W_mat` through an indexed Normal plate.

To implement classical factor analysis, replace scalar `sigma` with a positive `ObsDim`-indexed noise vector and gather the appropriate element for each response row.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/factor_analysis.qvr")
model = prog.morphism

N, K, Dn = 32, 2, 5
ND = N * Dn
item_idx = torch.arange(N).repeat_interleave(Dn)
obs_idx = torch.arange(Dn).repeat(N)

true_sigma = 0.3
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


## Categorical perspective

The top-level mean is the real-algebra composition $Z \mathbin{>>} W$, whose tensor is the low-rank matrix product. In the exported program, `Z_mat` and `W_mat` are explicit Normal plate draws and the scalar `sigma` supplies isotropic observation noise.

## See also

- [Probabilistic PCA](ppca.md), the isotropic-noise special case.
- [DSL Guide: Hierarchical Bayesian Models](../guides/programs-hierarchical.md#hierarchical-models-with-parametric-templates) for the morphism-valued prior surface.


## References

- Charles Spearman. 1904. "General intelligence," objectively determined and measured. *The American Journal of Psychology*, 15(2):201–293.
- David J. Bartholomew, Martin Knott, and Irini Moustaki. 2011. *Latent Variable Models and Factor Analysis: A Unified Approach*, 3rd edition. Wiley.
