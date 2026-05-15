# 1. Your first model

We'll write Bayesian linear regression in QVR end-to-end: define the model, generate some synthetic data, fit it with variational inference, and inspect the posterior. By the end of the chapter you'll have run a complete QVR workflow, and you'll know where to look for the analogues of every step you're used to from another PPL.

## The model

We model `n` real-valued observations $y_i$ as a noisy linear function of a scalar predictor $x_i$:

$$
\beta_0 \sim \mathrm{Normal}(0, 5), \qquad
\beta_1 \sim \mathrm{Normal}(0, 2), \qquad
\sigma \sim \mathrm{HalfNormal}(1), \qquad
y_i \sim \mathrm{Normal}(\beta_0 + \beta_1 x_i, \sigma).
$$

## QVR vs PyMC vs NumPyro

=== "QVR (`.qvr`)"

    ```qvr
    object Item : 100

    program regression : Item -> Item ! Sample, Score
        sigma  <- HalfNormal(1.0)
        beta_0 <- Normal(0.0, 5.0)
        beta_1 <- Normal(0.0, 2.0)
        let mu = beta_0 + beta_1 * x_design
        observe y : Item <- Normal(mu, sigma)
        return y

    export regression
    ```

=== "PyMC"

    ```python
    import pymc as pm

    with pm.Model() as model:
        sigma  = pm.HalfNormal("sigma", 1.0)
        beta_0 = pm.Normal("beta_0", 0.0, 5.0)
        beta_1 = pm.Normal("beta_1", 0.0, 2.0)
        mu     = beta_0 + beta_1 * x_data
        y      = pm.Normal("y", mu, sigma, observed=y_data)
    ```

=== "NumPyro"

    ```python
    import numpyro
    import numpyro.distributions as dist

    def regression(x, y=None):
        sigma  = numpyro.sample("sigma", dist.HalfNormal(1.0))
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 5.0))
        beta_1 = numpyro.sample("beta_1", dist.Normal(0.0, 2.0))
        mu     = beta_0 + beta_1 * x
        numpyro.sample("y", dist.Normal(mu, sigma), obs=y)
    ```

Reading the QVR line by line:

| Line | What it says |
|---|---|
| `object Item : 100` | Declare a finite-set index `Item` of size 100: the row dimension of the data. Domain and codomain are typed objects rather than implicit. |
| `program regression : Item -> Item ! Sample, Score` | A `program` block is the unit of compilation. The `!`-annotation declares the algebraic effects the body uses; `Sample` is monadic draw, `Score` is conditioning. |
| `sigma <- HalfNormal(1.0)` | Bind a random variable. Same as PyMC's `pm.HalfNormal(...)` or NumPyro's `numpyro.sample(...)`. |
| `let mu = beta_0 + beta_1 * x_design` | Deterministic let. The `let`-arithmetic supports `+ - * /`, indexing, broadcasts, and a small standard library (`sum`, `prod`, `cumsum`, `logsumexp`, ...). The free name `x_design` is supplied at fit time via the observations dict (declared by `observed_names` on the guide). |
| `observe y : Item <- Normal(mu, sigma)` | Vectorised conditioned bind, one draw per element of `Item`. The runtime sets `y` to the observed value at inference time and scores the likelihood. |

If you're coming from Pyro/NumPyro/Stan, the only piece without an obvious analogue is `! Sample, Score`: the effect signature. It's a static check that the body uses only effects you declared. You can leave it off and the compiler will infer, but writing it out makes intent explicit.

## Compile and fit

QVR programs compile to `nn.Module`. You can train them with the inference stack quivers ships (built around stochastic variational inference, [Hoffman, Blei, Wang & Paisley, 2013](https://doi.org/10.5555/2567709.2502622)), or with any PyTorch optimizer if you want to drop to raw gradients.

```python
import torch
from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, ELBO, SVI

src = open("regression.qvr").read()
program = loads(src)
model   = program.morphism                   # the compiled MonadicProgram

# Synthetic data.
torch.manual_seed(0)
x_data = torch.randn(100)
y_data = 1.5 + 2.7 * x_data + 0.3 * torch.randn(100)

# Variational inference. ``x_design`` is a free name in the model body;
# we hand it to the trace via ``observed_names`` and the observations dict.
guide     = AutoNormalGuide(model, observed_names={"y", "x_design"})
elbo      = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)

x_tensor = torch.zeros(1, 1)                  # SVI dispatch input (unused)
observations = {"x_design": x_data, "y": y_data}
for step in range(2000):
    loss = svi.step(x_tensor, observations)
    if step % 200 == 0:
        print(f"step {step:4d}  ELBO = {-loss:.3f}")
```

The pattern is identical to Pyro: a [`Guide`](../../api/inference/guide.md) carries the variational family, an [`Objective`](../../api/inference/elbo.md) is the loss, an [`SVI`](../../api/inference/svi.md) driver runs the loop.

The default [`AutoNormalGuide`](../../api/inference/guide.md) is a diagonal-Gaussian variational posterior trained by the pathwise (reparameterised) gradient estimator of [Kingma & Welling (2014)](https://doi.org/10.48550/arXiv.1312.6114); SVI ramps that to mini-batches.

## Inspect the posterior

```python
from quivers.inference import Predictive

# Draw posterior samples by repeatedly calling the guide.
posterior = {k: torch.stack([guide.rsample(x_tensor)[k] for _ in range(1000)])
             for k in ("beta_0", "beta_1", "sigma")}
print("posterior mean beta_0 =", posterior["beta_0"].mean().item())
print("posterior mean beta_1 =", posterior["beta_1"].mean().item())
print("posterior mean sigma  =", posterior["sigma"].mean().item())

predictive = Predictive(model, posterior=guide, num_samples=1000)
y_pred = predictive(x_tensor, {"x_design": x_data})["y"]    # (1000, 100)
print("predictive mean y[0] =", y_pred[:, 0].mean().item())
```

[`Guide.rsample`](../../api/inference/guide.md) returns one `dict[name, Tensor]` posterior draw per call; [`Predictive`](../../api/inference/predictive.md) produces posterior-predictive draws by re-running the model with each posterior sample.

## What's different from Pyro/NumPyro?

Three things:

1. **Types on the outside, names on the inside.** Every program has a typed signature `dom -> cod`; latents in the body are scoped to that signature. In Pyro/NumPyro, names live in a global trace and types are implicit.
2. **Compile, then fit.** `loads` runs the QVR compiler before training: malformed models, type mismatches, undefined references, or shape inconsistencies surface as `CompileError` with line/column information *before* any tensor evaluation runs. Pyro/NumPyro discover most of these only when you call the model.
3. **Effects on the signature.** `! Sample, Score, Marginal, Pure` is a static promise about what the body does. It's optional but lets the compiler reject programs that, say, try to `observe` inside a `Pure` block.

## Try this

- Add `algebra log_prob` at the top of the file. The enrichment changes how compositions accumulate scalars: likelihood-style values stay finite under very small probabilities; sometimes useful for long sequence models.
- Replace [`AutoNormalGuide`](../../api/inference/guide.md) with [`AutoMultivariateNormalGuide`](../../api/inference/guide.md) and watch the recovered correlation between `beta_0` and `beta_1`.
- Drop the `! Sample, Score` annotation, then add `! Pure` and re-run. The second case fails compilation with a typed error pointing to the `observe`.

## Next

[Chapter 2](02-glms.md) lifts this model to a generalized linear model: logistic and Poisson regression, with link functions and posterior-predictive calibration plots.
