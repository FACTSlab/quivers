# Bayesian Neural Network

## Overview

A [Bayesian neural network](https://en.wikipedia.org/wiki/Bayesian_neural_network) ([MacKay 1992](https://doi.org/10.1162/neco.1992.4.3.448)) puts a prior over every weight and recovers a posterior over weights, giving calibrated predictive uncertainty far from the training data. This example fits a [multi-layer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) to a nonlinear regression target: the response is [Normal](https://en.wikipedia.org/wiki/Normal_distribution) about a learned function of the input,

$$
\mu(x),\, \sigma(x) \;=\; \mathrm{MLP}(x), \qquad y_n \;\sim\; \mathcal{N}\!\bigl(\mu(x_n),\, \sigma(x_n)\bigr).
$$

The network emits both the mean and the log-scale, so the model is [heteroscedastic](https://en.wikipedia.org/wiki/Heteroscedasticity): the predictive spread varies with the input rather than being pinned to a single global noise level.

The nonlinearity lives in the morphism's parameter network. A morphism declared `~ Normal` over continuous spaces is a [Kleisli arrow](https://en.wikipedia.org/wiki/Kleisli_category) whose distribution parameters are produced from its input by a [`ParamSource`](../api/continuous/param_source.md#quivers.continuous.param_source.ParamSource), and `[param_source=mlp, hidden_dim=64]` selects an [`MLPSource`](../api/continuous/param_source.md#quivers.continuous.param_source.MLPSource): two hidden layers of width 64 with tanh activations between them. That is where the model departs from a linear map, and it is the reason this example can fit a curve that no linear model can.

## QVR Source

```qvr
object Feature : Real 1
object Target : Real 1
object Resp : FinSet 200

morphism net : Feature -> Target [param_source=mlp, hidden_dim=64] ~ Normal

program bnn : Resp -> Resp
    observe y : Resp <- net(x)
    return y

export bnn
```

## Walkthrough

`object Feature : Real 1` and `object Target : Real 1` declare the input and response as one-dimensional continuous spaces; `object Resp : FinSet 200` is the discrete plate indexing the rows. The plate is what the `observe` step reduces over, so it is the program's domain.

`morphism net : Feature -> Target [param_source=mlp, hidden_dim=64] ~ Normal` declares the network. The `~ Normal` clause makes `net` a conditional-Normal kernel rather than a tensor: applied to an input it returns a distribution over `Target`, parameterised by a mean and a log-scale. Those two numbers are what the parameter network emits, so the concrete module behind `net` is

```
Linear(1, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, 2)
```

Its 4418 weights are the network's parameters. `hidden_dim` sets the width and `param_source` the architecture; `linear`, `identity`, and `attention` are the other choices, and `mlp` is the default for a continuous domain.

`program bnn : Resp -> Resp` then does the only thing left. `observe y : Resp <- net(x)` applies the kernel to the per-row input and scores the observed response under the resulting Normal, accumulating over the `Resp` plate. `x` is a free variable: it never appears in a `sample` or `let`, so it is supplied as host data through the observations dict alongside `y`, exactly as a covariate would be in a regression.

## Try it

> The step counts and NUTS budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need more steps, longer warmup, and multiple chains to converge.

### Generating synthetic data

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/bnn.qvr")
model = prog.morphism

N = 200
x = torch.linspace(-3.0, 3.0, N).unsqueeze(-1)
y = torch.sin(3.0 * x) + 0.1 * torch.randn(N, 1)

x_in = torch.zeros(N, 1)
observations = {"y": y, "x": x}
```

The target is a sine wave in noise, which is the point: it is a function no linear model can represent. `y` carries the `Target` event axis, so it is shaped `(N, 1)` rather than `(N,)`. The dummy `x_in` satisfies the program's `Resp -> Resp` signature; the real covariate enters through `observations["x"]`.

### Fitting the network

```python
optim = torch.optim.Adam(model.parameters(), lr=5e-3)

for step in range(2000):
    optim.zero_grad()
    loss = -model.log_joint(x_in, observations)[0]
    loss.backward()
    optim.step()

print(f"final negative log-likelihood: {float(loss.detach()):.1f}")
```

[`log_joint`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram.log_joint) returns the total log-joint broadcast across the batch axis, so every entry is the same number and `[0]` reads it. Summing instead would count the same scalar once per row.

### Does the nonlinearity earn its keep?

The honest comparison is against the best a linear model can do on the same data, scored the same way. Ordinary least squares gives the optimal line, and its residual spread gives the matching Gaussian noise level.

```python
design = torch.cat([torch.ones(N, 1), x], dim=-1)
beta   = torch.linalg.lstsq(design, y).solution
resid  = y - design @ beta
linear_nll = -torch.distributions.Normal(design @ beta, resid.std()).log_prob(y).sum()

print(f"MLP    negative log-likelihood: {float(loss.detach()):.1f}")
print(f"linear negative log-likelihood: {float(linear_nll):.1f}")
print(f"residual sd of the best line:   {float(resid.std()):.3f}")
```

The line's residual spread lands near 0.7 against a true observation noise of 0.1, which is the quantitative form of the obvious: a straight line through a full period of a sine explains almost nothing, and the noise it infers is really the signal it cannot represent. The MLP's likelihood is several hundred nats better, and the gap is entirely attributable to the tanh layers, since the two models see identical data and differ only in the map from `x` to the Normal's parameters.

### NUTS posterior over the weights

Nothing so far is Bayesian: the fit above is maximum likelihood over the weights. `bnn` declares no `sample` sites, so there is no prior in the source to sample. [`lift_from_log_prob`](../api/inference/lifts.md#quivers.inference.lifts.lift_from_log_prob) supplies one, putting a Normal prior on every weight and scoring the data through a log-density function of your choosing. The result is a [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) whose sites are the weights themselves, which is exactly the Bayesian neural network the overview describes.

```python
from quivers.inference import MCMC, NUTSKernel, lift_from_log_prob

def log_prob_fn(x_unused, y_obs):
    return model.log_joint(x_in, {"y": y_obs, "x": x})[0].reshape(1)

lifted, lift_x, lift_obs = lift_from_log_prob(
    model,
    log_prob_fn=log_prob_fn,
    parameter_prior_scale=1.0,
    target_key="y",
    observations={"y": y},
)

kernel = NUTSKernel(step_size=0.01, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
result = mc.run(lifted, lift_x, lift_obs)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```

`log_prob_fn` runs inside the lifted program's score step, after the sampled weights are substituted into the network's parameter slots, so it reads the current draw rather than the values the fit above left behind. Sampling 4418 weights is a genuine posterior over networks and the budget here is far too small to converge; it demonstrates the surface, not a usable posterior.

## Categorical Perspective

`net : Feature -> Target` is a Kleisli arrow $\mathbb{R} \to \mathcal{G}(\mathbb{R})$ in the [Giry monad](https://doi.org/10.1007/BFb0092872), sending an input to a Gaussian measure over the response. The MLP is not part of that arrow's type: it is the map from the domain into the family's parameter space, and composing it with the Gaussian's parameterisation is what makes the kernel's mean and scale depend nonlinearly on the input.

This is the difference between a nonlinearity in the parameter network and a nonlinearity between composed morphisms. Under a strictly linear algebra a chain `W_1 >> W_2 >> W_3` of tensor morphisms collapses: the composite is a single linear map, and the intermediate objects buy nothing but a rank bound. Placing the nonlinearity inside the kernel's parameter source avoids that collapse without leaving the V-Cat surface, because the kernel was never required to be linear in its input. The [`ParamSource`](../api/continuous/param_source.md#quivers.continuous.param_source.ParamSource) abstraction is the seam: `mlp`, `linear`, `attention`, and `identity` all present the same interface to the family and differ only in what they compute.

Putting a prior on the weights is then a second, independent move. It lifts the deterministic parameter $\theta$ into a sample site, so the model becomes a mixture of networks $\int p(y \mid x, \theta)\, p(\theta)\, d\theta$ rather than a single one.

## See Also

- [Deep Markov Model](deep-markov.md) for the same conditional-Normal kernels threaded through `scan` as a nonlinear state-space model.
- [Bayesian Regression](bayesian-regression.md) for the linear counterpart, where the coefficients carry declared priors in the source.
