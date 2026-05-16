# Bayesian Neural Network

## Overview

A [Bayesian neural network](https://en.wikipedia.org/wiki/Bayesian_neural_network) ([MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.448)) puts a prior over every weight and recovers a posterior over weights via SVI or MCMC, giving calibrated predictive uncertainty far from the training data. This example builds a three-layer Bayesian *linear* network whose per-layer weight matrices carry [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) priors. The forward pass is a pure categorical composition of [`LatentMorphism`](../api/core/morphisms.md) factors:

$$
\mathsf{Item} \xrightarrow{X} \mathsf{H}_\text{in} \xrightarrow{W_1} \mathsf{H}_1 \xrightarrow{W_2} \mathsf{H}_2 \xrightarrow{W_3} \mathsf{H}_\text{out}.
$$

Each $W_l$ is a learnable morphism whose prior is matrix-normal over its (in, out) [Kronecker covariance](https://en.wikipedia.org/wiki/Kronecker_product), the natural prior for a linear layer. Under `algebra real` the composition `X >> W_1 >> W_2 >> W_3` is a real-valued matmul stack.

## QVR Source

```qvr
algebra real

object Item : 200
object H_in : 4
object H1 : 32
object H2 : 16
object H_out : 2

latent X : Item -> H_in
latent W_1 : H_in -> H1 ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent W_2 : H1 -> H2 ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent W_3 : H2 -> H_out ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)

let bnn = X >> W_1 >> W_2 >> W_3

export bnn
```

## Walkthrough

The three weight declarations

<!-- compile: false -->
```qvr
latent W_1 : H_in -> H1 ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent W_2 : H1 -> H2  ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent W_3 : H2 -> H_out ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

place [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) priors on each per-layer weight tensor. The two axes under `over (dom, cod)` bind positionally to the family's event axes: the input-side cardinality is the row axis and the output-side cardinality is the column axis, so the Kronecker covariance expresses independent row and column correlation in the weight matrix.

The per-item input is itself a learnable morphism `X : Item -> H_in`; in a real workload `X` would be set from data via [`from_data`](../guides/dsl-overview.md). The composition `X >> W_1 >> W_2 >> W_3` is the full forward pass, materialising an `Item x H_out` score tensor under real-algebra matmul.

### Limitation

QVR's pure-composition surface under `algebra real` is strictly linear. There is no pointwise nonlinearity between composed weight matrices and no stochastic observation kernel on top of the discrete codomain `H_out`. The model expressed here is therefore a Bayesian *linear* network with matrix-normal priors on every layer's tensor. A deep nonlinear MLP with Bayesian weights is not currently expressible as a pure latent-morphism composition: a continuous-space surface would have to wrap each `W_l` in a continuous kernel carrying a nonlinearity inside its parameter network. The closest categorical form of the original BNN, with the matrix-normal priors actually entering the inference path, is the per-layer linear composition shown above.

## Try it

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)

prog = load("docs/examples/source/bnn.qvr")
model = prog.morphism

# The model tensor materialises the Item x H_out score matrix.
# Fit it as a low-rank linear network to a target Y by gradient
# descent on the matmul output.
N = 200
H_out = 2
Y = torch.randn(N, H_out)

opt = torch.optim.Adam(prog.parameters(), lr=2e-2)
for _ in range(300):
    opt.zero_grad()
    loss = (model.tensor - Y).pow(2).mean()
    loss.backward()
    opt.step()

print("residual MSE:", (model.tensor - Y).pow(2).mean().item())
```

## Categorical Perspective

Each weight tensor is a morphism in the discrete-object category whose prior measure on the hom-object $\mathbf{Kern}(\mathsf{H}_l, \mathsf{H}_{l + 1})$ is the [`MatrixNormal`](../api/continuous/families.md#quivers.continuous.families.ConditionalMatrixNormal) distribution. The forward pass is the composition

$$
\mathsf{Item} \xrightarrow{X \mathbin{>>} W_1 \mathbin{>>} W_2 \mathbin{>>} W_3} \mathsf{H}_\text{out}
$$

in the real algebra. SVI's mean-field variational guide places an independent Normal posterior on every weight; predictive uncertainty is the marginal over weight samples drawn from this posterior.

## See Also

- [Bayesian Linear Regression](bayesian-regression.md) for the single-layer linear special case.
- [DSL Guide](../guides/dsl-overview.md) for the morphism-valued prior surface and stochastic-kernel composition.
