# 2. Generalised linear models

Linear regression is the corner case of a generalised linear model ([Nelder & Wedderburn, 1972](https://doi.org/10.2307/2344614)) with an identity link and a normal likelihood. Loosening either gives you a much wider family: logistic regression for binary outcomes, Poisson regression for counts, ordinal regression for graded responses. This chapter does the first two end to end in QVR, with calibration checks on the posterior predictive.

The model template is the same in every case:

$$
\theta_i = g^{-1}(\beta^\top x_i), \qquad y_i \sim \mathcal{F}(\theta_i),
$$

where $g$ is the link function and $\mathcal{F}$ is the response family. We pick $(g^{-1}, \mathcal{F})$ together.

## Logistic regression

Binary response, sigmoid link, Bernoulli likelihood.

=== "QVR"

    ```qvr
    quantale real
    object Item : 200

    program logistic : Item -> Item ! Sample, Score
        beta_0 <- Normal(0.0, 5.0)
        beta_1 <- Normal(0.0, 2.0)
        x      <- Normal(0.0, 1.0)
        let logit = beta_0 + beta_1 * x
        let p     = sigmoid(logit)
        observe y <- Bernoulli(p)
        return y

    export logistic
    ```

=== "PyMC"

    ```python
    with pm.Model() as model:
        beta_0 = pm.Normal("beta_0", 0.0, 5.0)
        beta_1 = pm.Normal("beta_1", 0.0, 2.0)
        p      = pm.invlogit(beta_0 + beta_1 * x_data)
        y      = pm.Bernoulli("y", p, observed=y_data)
    ```

=== "NumPyro"

    ```python
    def logistic(x, y=None):
        beta_0 = numpyro.sample("beta_0", dist.Normal(0.0, 5.0))
        beta_1 = numpyro.sample("beta_1", dist.Normal(0.0, 2.0))
        logit  = beta_0 + beta_1 * x
        numpyro.sample("y", dist.Bernoulli(logits=logit), obs=y)
    ```

QVR's let-arithmetic exposes `sigmoid`, `softmax`, `log_softmax`, `exp`, `log`, `tanh`, `relu`, and a handful of others as builtins so you can name the inverse link explicitly. Fitting is the same as chapter 1:

```python
program = loads(open("logistic.qvr").read())
model   = program.morphism

torch.manual_seed(0)
x_data = torch.randn(200)
true_p = torch.sigmoid(0.3 + 1.5 * x_data)
y_data = torch.bernoulli(true_p)

guide = AutoNormalGuide(model, observed_names={"y"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-3,
)
svi = SVI(model, guide, optimizer, elbo)
for step in range(3000):
    svi.step({"x": x_data}, {"y": y_data})
```

## Posterior-predictive calibration

A logistic regression is well-calibrated if, of all observations where the model predicts $p = 0.7$, roughly 70% actually have $y = 1$.

```python
from quivers.inference import Predictive

pred  = Predictive(model, guide=guide, num_samples=500)
y_hat = pred({"x": x_data})["y"]                    # (500, 200)
p_hat = y_hat.float().mean(dim=0)                    # (200,)

# Bin and check empirical frequency.
bins = torch.linspace(0, 1, 11)
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_hat >= lo) & (p_hat < hi)
    if mask.sum() > 0:
        empirical = y_data[mask].float().mean().item()
        midpoint  = (lo + hi).item() / 2
        print(f"  [{lo:.1f}, {hi:.1f})  n={int(mask.sum()):3d}"
              f"  predicted={midpoint:.2f}  observed={empirical:.2f}")
```

## Poisson regression

Count response, log link, Poisson likelihood.

```qvr
quantale real
object Item : 150

program poisson_reg : Item -> Item ! Sample, Score
    beta_0 <- Normal(0.0, 5.0)
    beta_1 <- Normal(0.0, 2.0)
    x      <- Normal(0.0, 1.0)
    let log_rate = beta_0 + beta_1 * x
    let rate     = exp(log_rate)
    observe y <- Poisson(rate)
    return y

export poisson_reg
```

The fit code is identical to the logistic case; only the model file changes. The `Predictive` machinery handles the count-valued likelihood with no special configuration.

## What the QVR surface gives you here

Three observations from this chapter you may have already noticed:

1. **No `pyro.plate` / `numpyro.plate` wrapping.** Plates are inferred from object cardinalities and the domain/codomain typing. If you need an explicit indexed family (say a per-group intercept), chapter 3 introduces the plate-draw syntax (`v : G <- Normal(0, sigma)`).
2. **`let` is not sampling.** PyMC `pm.Deterministic`, NumPyro `numpyro.deterministic`, Pyro `pyro.deterministic`: every PPL has a different name for "this is a function of random variables, not itself random." QVR uses `let`. The compiler tracks the dependency for autograd.
3. **One module, one program.** Each `.qvr` file declares zero or more objects, optional `quantale`, and zero or more programs. To fit several models, write several files (or several program blocks in one file; `loads` returns the one tagged with `export`).

## Try this

- Add a constant offset (`offset`) to the Poisson model: `let log_rate = beta_0 + beta_1 * x + offset`. The compiler treats `offset` as a free variable and exposes it as a model input alongside `x`.
- Switch the Bernoulli likelihood to a Binomial with a known trial count `n_trials`. The Binomial constructor takes `(n, p)`.
- Run NUTS instead of SVI. Skip ahead to chapter 6, line "Calling NUTS"; it's a five-line change.

## Next

Chapter 3 covers hierarchical models: the place where mean-field VI starts to misbehave and where NUTS earns its keep.
