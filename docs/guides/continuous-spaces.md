# Continuous Spaces and Morphisms

This page covers the `ContinuousSpace` hierarchy, the
[`ContinuousMorphism`](../api/continuous/morphisms.md#quivers.continuous.morphisms.ContinuousMorphism)
interface, composition surfaces, the discrete /
continuous boundary, and normalizing flows. The full registry of
parameterized distribution families lives in
[continuous families](continuous-families.md).

## ContinuousSpace hierarchy

Continuous morphisms act between continuous measurable spaces.
quivers provides a hierarchy of standard spaces:

```
ContinuousSpace (abstract)
├── Euclidean         R^n with Lebesgue measure
├── UnitInterval      (0, 1) with Lebesgue measure
├── Simplex           standard n-simplex {x in R^(n+1) : x >= 0, sum x = 1}
├── PositiveReals     (0, inf) with Lebesgue measure
└── ProductSpace      product of spaces
```

### Euclidean space

```python
from quivers.continuous.spaces import Euclidean

R3 = Euclidean(name="R3", dim=3)                                  # R^3
R2_bounded = Euclidean(name="R2", dim=2, low=0.0, high=1.0)        # [0,1]^2
```

### Unit interval

```python
from quivers.continuous.spaces import UnitInterval

U = UnitInterval(name="U")     # (0, 1); pass dim=k for [0, 1]^k
```

### Simplex

```python
from quivers.continuous.spaces import Simplex

S3 = Simplex(name="S3", dim=3)   # 2-simplex in R^3 (3 categories with sum = 1)
```

### PositiveReals

```python
from quivers.continuous.spaces import PositiveReals

P2 = PositiveReals(name="P2", dim=2)   # (0, inf)^2
```

### ProductSpace

```python
from quivers.continuous.spaces import ProductSpace, Euclidean, UnitInterval

R3 = Euclidean(name="R3", dim=3)
U = UnitInterval(name="U")
P = ProductSpace(components=(R3, U))   # R^3 x (0, 1); also R3 * U
```

## ContinuousMorphism

A continuous morphism $f : X \to Y$ between spaces $X$ and $Y$
defines a conditional distribution $p(y \mid x)$ via two operations:

- `log_prob(x, y)`: evaluate $\log p(y \mid x)$.
- `rsample(x)`: generate reparameterized samples from $p(\cdot \mid
  x)$.

```python
from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.spaces import Euclidean
import torch

class MyMorphism(ContinuousMorphism):
    def log_prob(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log density log p(y | x)."""
        raise NotImplementedError()

    def rsample(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:
        """Reparameterized samples; shape (*sample_shape, batch, codomain.dim)."""
        raise NotImplementedError()
```

Morphisms are
[`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
subclasses, so they are trainable.

## Composition: SampledComposition

When composing continuous morphisms, the intermediate is
continuous. The result is computed via ancestral sampling:

```python
from quivers.continuous.morphisms import SampledComposition
from quivers.continuous.families import ConditionalNormal

# f: X -> Y (continuous), g: Y -> Z (continuous)
X = Euclidean(name="X", dim=3)
Y = Euclidean(name="Y", dim=4)
Z = Euclidean(name="Z", dim=5)
f = ConditionalNormal(X, Y)
g = ConditionalNormal(Y, Z)

# Composition: (g . f)(x) samples from g(f(x))
composed = f >> g
assert isinstance(composed, SampledComposition)

# rsample: sample from intermediate
x = torch.randn(2, 3)
z_samples = composed.rsample(x, sample_shape=torch.Size((50,)))  # (50, 2, 5)
```

When an intermediate is discrete (a
[`FinSet`](../api/core/objects.md)), composition uses exact
marginalization:

$$p(z \mid x) = \sum_y p(y \mid x) \, p(z \mid y)$$

## ProductContinuousMorphism

Tensor product of continuous morphisms:

```python
from quivers.continuous.morphisms import ProductContinuousMorphism
from quivers.continuous.families import ConditionalNormal, ConditionalBeta

X = Euclidean(name="X", dim=3)
f = ConditionalNormal(X, Euclidean(name="Y1", dim=2))
g = ConditionalBeta(X, UnitInterval(name="Y2"))

# Product: (f @ g)(x) ~ p(y_1, y_2 | x) = p(y_1 | x) . p(y_2 | x)
fg = f @ g

# Domain and codomain are products
assert fg.domain == f.domain * g.domain   # X x X
assert fg.codomain == f.codomain * g.codomain  # R^2 x (0,1)
```

## Discrete / continuous boundary

### DiscreteAsContinuous: embed a discrete morphism

Embed a discrete morphism (finite set) as continuous:

```python
from quivers.continuous.boundaries import Embed
from quivers.core.morphisms import morphism
from quivers.core.objects import FinSet

X = FinSet(name="X", cardinality=4)
Y = FinSet(name="Y", cardinality=5)
discrete_f = morphism(X, Y)

# Treat X and Y as uniform distributions
Y_continuous = Euclidean(name="Y", dim=5)
embedded_f = Embed(domain=X, codomain=Y_continuous)

# Now can compose with continuous morphisms
y_cont = torch.randn(4, 5)
log_p = embedded_f.log_prob(torch.arange(4), y_cont)
```

### Discretize: bin a continuous space

Discretize a continuous space into a finite set:

```python
from quivers.continuous.boundaries import Discretize

# Discretize a bounded interval into 20 bins
# (Discretize requires bounded Euclidean)
U = Euclidean(name="U", dim=1, low=0.0, high=1.0)
discretized = Discretize(U, n_bins=20)

# Maps continuous values to bin indices (as a FinSet of size 20)
sample = torch.tensor(0.35)
bin_idx = discretized(sample)  # integer in [0, 19]
```

### Integration with discrete morphisms

Continuous and discrete morphisms integrate transparently via the
`>>` and `@` operators:

```python
# Discrete to Continuous
discrete_f = morphism(X, Y)  # X -> Y (finite)
cont_g = ConditionalNormal(
    Euclidean(name="YE", dim=Y.size),
    Euclidean(name="Z", dim=3),
)

composed = discrete_f >> cont_g  # exact marginalization over Y

# Continuous to Discrete: not directly composable without an
# explicit Discretize boundary; use Discretize to materialize a
# FinSet target first.
```

## Normalizing flows

A [normalizing
flow](https://en.wikipedia.org/wiki/Flow-based_generative_model)
transforms a simple base distribution via a chain of invertible
transformations.

### AffineCouplingLayer

An [affine coupling
layer](https://doi.org/10.48550/arXiv.1605.08803) partitions the
input and applies an invertible affine transformation to each
partition:

```python
from quivers.continuous.flows import AffineCouplingLayer
from quivers.continuous.spaces import Euclidean

dim = 4
domain = Euclidean(name="X", dim=2)
layer = AffineCouplingLayer(domain, dim, hidden_dim=32)

# Forward (base to target), conditioned on x
x = torch.randn(10, 2)
z = torch.randn(10, dim)
z_out, log_det = layer.forward(x, z)

# Inverse (target to base)
z_back, log_det_inv = layer.inverse(x, z_out)
```

### ConditionalFlow

A full normalizing flow conditioned on input:

```python
from quivers.continuous.flows import ConditionalFlow

domain = Euclidean(name="X", dim=5)
codomain = Euclidean(name="Y", dim=4)

flow = ConditionalFlow(
    domain=domain,
    codomain=codomain,
    n_layers=6,
    hidden_dim=64,
)

# Sample
x = torch.randn(8, 5)
y_samples = flow.rsample(x, sample_shape=torch.Size((50,)))

# Log probability
y = torch.randn(4)
log_p = flow.log_prob(x, y)
```

Additional flow primitives live in
[`quivers.continuous.flows`](../api/continuous/flows.md): masked
autoregressive flows ([MAF](https://doi.org/10.48550/arXiv.1705.07057)),
inverse autoregressive flows
([IAF](https://doi.org/10.48550/arXiv.1606.04934)), neural-spline
flows ([NSF](https://doi.org/10.48550/arXiv.1906.04032)), batch
normalization layers, and LU-parameterized linear layers.

## See also

- [Continuous Families](continuous-families.md): the 30+
  parameterized distribution families used to build
  `ContinuousMorphism`s and the event-rank table that drives the
  axis-role surface.
- [Monadic Programs](programs.md): how `ContinuousMorphism`s
  compose into probabilistic programs.
- [Stochastic Morphisms](stochastic.md): the finite-state
  counterpart and the
  [Giry monad](https://ncatlab.org/nlab/show/Giry+monad) substrate.
