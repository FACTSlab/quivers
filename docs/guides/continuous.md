# Continuous Distributions

## ContinuousSpace Hierarchy

Continuous morphisms act between continuous measurable spaces. quivers provides a hierarchy of standard spaces:

```
ContinuousSpace (abstract)
├── Euclidean         , ℝ^n with Lebesgue measure
├── UnitInterval      , (0, 1) with Lebesgue measure
├── Simplex           , standard n-simplex {x ∈ ℝⁿ⁺¹ : x ≥ 0, Σx = 1}
├── PositiveReals     , (0, ∞) with Lebesgue measure
└── ProductSpace      , product of spaces
```

### Euclidean Space

```python
from quivers.continuous.spaces import Euclidean

R3 = Euclidean(3)       # ℝ³
R2_bounded = Euclidean(2, low=0.0, high=1.0)  # [0,1]²
```

### Unit Interval

```python
from quivers.continuous.spaces import UnitInterval

U = UnitInterval()      # (0, 1)
```

### Simplex

```python
from quivers.continuous.spaces import Simplex

S3 = Simplex(3)         # 2-simplex in ℝ³ (3 categories with Σ = 1)
```

### PositiveReals

```python
from quivers.continuous.spaces import PositiveReals

P2 = PositiveReals(2)   # (0, ∞)²
```

### ProductSpace

```python
from quivers.continuous.spaces import ProductSpace, Euclidean, UnitInterval

R3 = Euclidean(3)
U = UnitInterval()
P = ProductSpace(R3, U)  # ℝ³ × (0, 1)
```

## ContinuousMorphism

A continuous morphism $f: X \to Y$ between spaces $X$ and $Y$ defines a conditional distribution $p(y | x)$ via two operations:

- `log_prob(x, y)`: evaluate $\log p(y | x)$
- `rsample(x)`: generate reparameterized samples from $p(\cdot | x)$

```python
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import Euclidean
import torch

class MyMorphism(ContinuousMorphism):
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log density log p(y | x)"""
        raise NotImplementedError()

    def rsample(self, x: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Generate samples: return shape (..., n_samples, |Y|)"""
        raise NotImplementedError()
```

Morphisms are `nn.Module` subclasses, so they are trainable.

## Parameterized Distribution Families

quivers provides 30+ built-in conditional distributions. Each takes an input (domain) and produces learnable parameters via a learned linear map.

### Hand-Written Families

```python
from quivers.continuous.families import (
    ConditionalNormal,
    ConditionalLogitNormal,
    ConditionalBeta,
    ConditionalTruncatedNormal,
    ConditionalDirichlet,
)
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
import torch

# Input: finite set X
X = FinSet("X", 5)
domain = Euclidean(5)  # or FinSet
codomain = Euclidean(3)

# Conditional normal: learns linear maps μ(x), log_σ(x)
normal = ConditionalNormal(domain, codomain)

# Sample from p(y | x)
x = torch.randn(5)
samples = normal.rsample(x, n_samples=100)  # shape (100, 3)

# Log probability
y = torch.randn(3)
log_p = normal.log_prob(x, y)  # scalar or batch
```

### Loc-Scale Families

Standard reparameterizable distributions with learned location and scale:

```python
from quivers.continuous.families import (
    ConditionalCauchy,
    ConditionalLaplace,
    ConditionalGumbel,
    ConditionalLogNormal,
    ConditionalStudentT,
)
```

### Positive-Valued Families

For $\mathbb{R}_{>0}$ output:

```python
from quivers.continuous.families import (
    ConditionalExponential,
    ConditionalGamma,
    ConditionalWeibull,
    ConditionalPareto,
    ConditionalInverseGamma,
)

gamma = ConditionalGamma(domain, codomain)
samples = gamma.rsample(x)  # positive
```

### Unit Interval Families

For $(0, 1)$ output:

```python
from quivers.continuous.families import (
    ConditionalBeta,
    ConditionalKumaraswamy,
    ConditionalContinuousBernoulli,
)
```

### Multivariate Families

```python
from quivers.continuous.families import (
    ConditionalMultivariateNormal,
    ConditionalLowRankMVN,
    ConditionalDirichlet,
    ConditionalWishart,
    ConditionalInverseWishart,
    ConditionalMatrixNormal,
    ConditionalGaussianProcess,
    ConditionalHorseshoe,
)

# Multivariate normal with learned mean and cov
mvn = ConditionalMultivariateNormal(domain, Euclidean(5))
```

`event_rank` per family controls the axis-role surface in the DSL
(see the [DSL guide](dsl.md) on `over <axes>`):

| Family | Event rank | Categorical reading |
|---|---|---|
| `Normal`, `Beta`, `Gamma`, `Exponential`, etc. | 0 | Scalar; every codomain axis is iid by default |
| `MultivariateNormal`, `LowRankMVN`, `Dirichlet`, `OneHotCategorical`, `LogisticNormal`, `GP`, `Horseshoe` | 1 | Vector; one named event axis carries the joint distribution |
| `Wishart`, `InverseWishart`, `MatrixNormal`, `LKJCholesky` | 2 | Matrix; two named event axes carry the joint distribution |

The DSL surface `~ Family over <axes>` requires the axis count to
match the family's event rank exactly; mismatch is a compile-time
error.  In particular, a flat MVN over `dim(A)*dim(B)` (dense
covariance, event_rank 1 with a single named axis whose dim equals
the product) is categorically distinct from a `MatrixNormal` over
`(A, B)` (Kronecker structure `V ⊗ U`, event_rank 2); the surface
keeps the two distinguishable rather than auto-substituting.

#### MatrixNormal: Kronecker-covariance matrix prior

```python
from quivers.continuous.families import ConditionalMatrixNormal

# Matrix-valued kernel: domain -> R^(rows*cols), with samples
# reshaped to (rows, cols) and Kronecker covariance Σ = V ⊗ U.
mn = ConditionalMatrixNormal(domain, Euclidean(rows * cols), rows=4, cols=8)
```

The matrix-Normal `MN(M, U, V)` is the natural prior for a
weight matrix `W : R^d → R^k` whose row and column correlations
factor separately.  When used as a `latent` morphism prior in the
DSL (`latent W : Euclidean(D) -> Euclidean(K) ~ MatrixNormal(loc,
row_scale, col_scale) over (dom, cod)`), the first axis listed in
`over` binds the row covariance and the second the column
covariance.

#### InverseWishart: conjugate covariance prior

```python
from quivers.continuous.families import ConditionalInverseWishart

# Conjugate prior on a d-dim covariance matrix.  Realised as the
# inversion of a Wishart sample with the correct symmetric-matrix
# change-of-variables Jacobian.
iw = ConditionalInverseWishart(domain, Euclidean(d))
```

Conjugate prior for the covariance of a multivariate normal
([Gelman, Carlin, Stern, Dunson, Vehtari & Rubin, 2013](https://doi.org/10.1201/b16018), §3.6).
The change-of-variables Jacobian for inverting a positive-definite
symmetric matrix contributes a `-(d + 1) log det(Σ)` term to the
log-density.

#### Gaussian process: covariance kernel evaluated at inputs

A [Gaussian process](https://en.wikipedia.org/wiki/Gaussian_process)
on `R^D` is a Markov kernel `X^N -> G(R^N)` whose value at any
finite collection of input locations `x_1, ..., x_N in R^D` is a
joint multivariate Normal with covariance `K(x_i, x_j)`.
Categorically, the family departs from the other parametric
families: its "parameters" at each evaluation are the input
locations themselves, not a small neural-net summary of them.
[`ConditionalGaussianProcess`](../api/continuous/families.md#quivers.continuous.families.ConditionalGaussianProcess)
exposes three covariance kernels (RBF, Matern 5/2, linear) with
learnable length scale and amplitude.

```python
from quivers.continuous.families import ConditionalGaussianProcess
from quivers.continuous.spaces import Euclidean
import torch

D, N = 2, 8
gp = ConditionalGaussianProcess(
    Euclidean(name="X", dim=D),
    Euclidean(name="Y", dim=N),
    kernel="rbf",
    length_scale=0.5,
    amplitude=1.0,
)
x = torch.randn(N, D)            # N input locations in R^D
f = gp.rsample(x)                # one GP draw at those inputs, shape (N,)
log_p = gp.log_prob(x, f)        # MVN(0, K(x, x) + jitter * I) density
```

The DSL registers the family as `GP` with event rank 1; one named
axis (the input-location count `N`) carries the joint distribution.
Reference: [Rasmussen & Williams (2006)](http://www.gaussianprocess.org/gpml/).

#### Horseshoe: global-local shrinkage prior

The [horseshoe prior](https://doi.org/10.1093/biomet/asq017) is a
sparse-signal prior whose hierarchical form is

```text
tau ~ HalfCauchy(scale)
lambda_d ~ HalfCauchy(1)            for d = 1, ..., D
beta_d | tau, lambda_d ~ Normal(0, (tau * lambda_d)^2)
```

The marginal density of `beta_d` after integrating the local scale
`lambda_d` has no closed form; the [Carvalho, Polson & Scott
(2010)](https://doi.org/10.1093/biomet/asq017) bound is improper.
[`ConditionalHorseshoe`](../api/continuous/families.md#quivers.continuous.families.ConditionalHorseshoe)
returns the exact marginal via 16-point Gauss-Legendre quadrature
after the change of variables `lambda = tan(pi * t / 2)` which
maps `(0, inf)` to `(0, 1)` with Jacobian
`(pi / 2) * sec^2(pi * t / 2)`.

```python
from quivers.continuous.families import ConditionalHorseshoe
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
import torch

hs = ConditionalHorseshoe(
    FinSet(name="A", cardinality=4),
    Euclidean(name="beta", dim=10),
    scale=0.1,
)
x = torch.tensor([0, 1, 2, 3])
beta = hs.rsample(x)             # shape (4, 10)
log_p = hs.log_prob(x, beta)     # shape (4,), summed over coordinates
```

The DSL registers the family as `Horseshoe` with event rank 1; the
one named axis carries the per-coordinate marginal product.

### Discrete (Categorical)

```python
from quivers.continuous.families import (
    ConditionalBernoulli,
    ConditionalCategorical,
    ConditionalRelaxedBernoulli,
    ConditionalRelaxedOneHotCategorical,
)
```

## Composition: SampledComposition

When composing continuous morphisms, the intermediate is continuous. The result is computed via ancestral sampling:

```python
from quivers.continuous.morphisms import SampledComposition

# f: X → Y (continuous), g: Y → Z (continuous)
f = ConditionalNormal(Euclidean(3), Euclidean(4))
g = ConditionalNormal(Euclidean(4), Euclidean(5))

# Composition: (g ∘ f)(x) samples from g(f(x))
composed = f >> g
assert isinstance(composed, SampledComposition)

# rsample: sample from intermediate
x = torch.randn(3)
z_samples = composed.rsample(x, n_samples=50)  # (50, 5)
```

When an intermediate is discrete (a FinSet), composition uses exact marginalization:

$$p(z | x) = \sum_y p(y | x) p(z | y)$$

## ProductContinuousMorphism

Tensor product of continuous morphisms:

```python
from quivers.continuous.morphisms import ProductContinuousMorphism

f = ConditionalNormal(Euclidean(3), Euclidean(2))
g = ConditionalBeta(Euclidean(3), UnitInterval())

# Product: (f @ g)(x) ~ p(y₁, y₂ | x) = p(y₁ | x) · p(y₂ | x)
fg = f @ g

# Domain and codomain are products
assert fg.domain == f.domain * g.domain  # Euclidean(3) × Euclidean(3)
assert fg.codomain == f.codomain * g.codomain  # ℝ² × (0,1)
```

## DiscreteAsContinuous

Embed a discrete morphism (finite set) as continuous:

```python
from quivers.continuous.boundaries import Embed
from quivers.core.morphisms import morphism
from quivers.core.objects import FinSet

X = FinSet("X", 4)
Y = FinSet("Y", 5)
discrete_f = morphism(X, Y)

# Treat X and Y as uniform distributions
Y_continuous = Euclidean(5)
embedded_f = Embed(discrete_f, target_space=Y_continuous)

# Now can compose with continuous morphisms
y_cont = torch.randn(5)
log_p = embedded_f.log_prob(torch.arange(4), y_cont)
```

## Discretization (Boundary)

Discretize a continuous space into a finite set:

```python
from quivers.continuous.boundaries import Discretize

# Discretize [0, 1] into 20 bins
U = UnitInterval()
discretized = Discretize(U, n_bins=20)

# Maps continuous values to bin indices (as a FinSet of size 20)
sample = torch.tensor(0.35)
bin_idx = discretized(sample)  # integer in [0, 19]
```

## Normalizing Flows

Transform a simple base distribution via a chain of invertible transformations.

### AffineCouplingLayer

An affine coupling layer partitions the input, applies an invertible affine transformation to each partition:

```python
from quivers.continuous.flows import AffineCouplingLayer

dim = 4
layer = AffineCouplingLayer(dim, hidden_dim=32)

# Forward (sampling)
z = torch.randn(10, dim)  # base noise
x, log_det = layer.forward(z)

# Inverse (density evaluation)
x = torch.randn(10, dim)
z, log_det_inv = layer.inverse(x)
```

### ConditionalFlow

A full normalizing flow conditioned on input:

```python
from quivers.continuous.flows import ConditionalFlow

domain = Euclidean(5)
codomain = Euclidean(4)

flow = ConditionalFlow(
    domain=domain,
    codomain=codomain,
    n_layers=6,
    hidden_dim=64,
)

# Sample
x = torch.randn(5)
y_samples = flow.rsample(x, n_samples=50)

# Log probability
y = torch.randn(4)
log_p = flow.log_prob(x, y)
```

## Integration with Discrete

Continuous and discrete morphisms integrate transparently via the `>>` and `@` operators:

```python
# Discrete → Continuous
discrete_f = StochasticMorphism(X, Y)  # X → Y (finite)
cont_g = ConditionalNormal(Euclidean(|Y|), Euclidean(3))

composed = discrete_f >> cont_g  # marginalizes exactly over Y

# Continuous → Discrete
cont_h = ConditionalNormal(Euclidean(3), Euclidean(4))
discrete_k = StochasticMorphism(FinSet("4", 4), FinSet("5", 5))

# Composition via sampling: sample from h, then k
composed2 = cont_h >> discrete_k
```
