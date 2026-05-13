# Bayesian Neural Network

## Overview

A [Bayesian neural network](https://en.wikipedia.org/wiki/Bayesian_neural_network) ([MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.448)) puts a prior over every weight and recovers a posterior over weights via SVI or MCMC, giving calibrated predictive uncertainty in regions of input space far from the training data. This example builds a small two-hidden-layer MLP for binary classification whose per-layer weight matrices carry [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) priors.

## QVR Source

```qvr
object H_in : 4
object H1 : 32
object H2 : 16
object H_out : 2
object Item : 200
object Label : 2

type Input = Euclidean 4
type Hidden1 = Euclidean 32
type Hidden2 = Euclidean 16

latent W_1 : H_in -> H1 ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent W_2 : H1 -> H2 ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent W_3 : H2 -> H_out ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)

kernel layer_1 : Input -> Hidden1 ~ Normal [scale=0.5]
kernel layer_2 : Hidden1 -> Hidden2 ~ Normal [scale=0.5]
kernel head : Hidden2 -> Label ~ Bernoulli

let backbone = layer_1 >> layer_2

program bnn : Input -> Label
    h <- backbone
    observe y : Label <- head(h)
    return y

export bnn
```

## Walkthrough

The three top-level declarations

<!-- compile: false -->
```qvr
latent W_1 : H_in -> H1 ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent W_2 : H1 -> H2  ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
latent W_3 : H2 -> H_out ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

place [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) priors on each per-layer weight tensor. The two axes under `over (dom, cod)` bind positionally to the family's event axes: the input-side cardinality is the row axis and the output-side cardinality is the column axis, so the Kronecker covariance expresses independent row and column correlation in the weight matrix.

The continuous-space layers `layer_1` and `layer_2` are stochastic Normal kernels at small init scale; `head : Hidden2 -> Label ~ Bernoulli` is the classification head. The composed [Kleisli](https://en.wikipedia.org/wiki/Kleisli_category) `let backbone = layer_1 >> layer_2` gives the hidden representation; the program body samples it, then scores the observed label under the Bernoulli head.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)

prog = load("docs/examples/source/bnn.qvr")
model = prog.morphism

N = 200
X = torch.randn(N, 4)
y_true = (X.sum(-1) > 0).long()

guide = AutoNormalGuide(model, observed_names={"y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optim, ELBO())
for _ in range(500):
    loss = svi.step(X, {"y": y_true})

with torch.no_grad():
    h = model.steps[0].morphism(X).rsample()
    logits = model.steps[1].morphism(h).rsample()
print("train accuracy:", (logits.argmax(-1) == y_true).float().mean().item())
```

## Categorical Perspective

Each weight tensor is a morphism in the discrete category whose prior measure on the hom-object $\mathbf{Kern}(\mathsf{H}_l, \mathsf{H}_{l + 1})$ is the [matrix-normal](https://en.wikipedia.org/wiki/Matrix_normal_distribution) distribution. The continuous-space forward pass is the Kleisli composite

$$
\mathsf{Input} \xrightarrow{\mathsf{layer}_1} \mathcal{G}(\mathsf{Hidden}_1) \xrightarrow{\mathsf{layer}_2} \mathcal{G}(\mathsf{Hidden}_2) \xrightarrow{\mathsf{head}} \mathcal{G}(\mathsf{Label})
$$

in the Kleisli category over the [Giry monad](https://doi.org/10.1007/BFb0092872). SVI's mean-field variational guide places an independent Normal posterior on every weight; predictive uncertainty is the marginal over weight samples drawn from this posterior.

## See Also

- [Bayesian Linear Regression](bayesian-regression.md) for the linear special case.
- [DSL Guide](../guides/dsl.md) for the morphism-valued prior surface and stochastic-kernel composition.
