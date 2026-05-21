# Tutorial 8: Analysis Pipelines

In this tutorial, you will fit a regression to data with one line, inspect the QVR program the formula compiles to, run an SVI fit of a hierarchical model, do NUTS sampling for a Bernoulli regression, then compare two models with PSIS-LOO and run a posterior-predictive check, all on synthetic data.

## Concepts

- **Formula**: A brms / lme4-style string like `"y ~ x + (1 | g)"` that names a response, predictors, and random-effect structure.
- **`fit`**: The one-liner that parses the formula against a dataframe, compiles to a QVR [`MonadicProgram`](../../api/continuous/programs.md), and dispatches to NUTS / HMC / SVI.
- **`BayesianFit`**: The returned record. Carries the parsed `Formula`, the compiled program, the posterior (either an `MCMCResult` or a `Guide`), and the inference-time observations dict.
- **Posterior**: For NUTS / HMC, an [`MCMCResult`](../../api/inference/predictive.md) with `(chain, draw)`-shaped sample tensors per site. For SVI, a fitted [`Guide`](../../api/inference/guide.md) from which you draw via `rsample`.
- **`DataTree`**: The canonical [ArviZ 1.x](https://python.arviz.org/) container for posterior analysis. The [`to_datatree`](../../api/diagnostics/arviz_io.md) adapter wraps a `BayesianFit.posterior` into the right shape so every ArviZ plot, diagnostic, and information criterion just works.

## Setup

The analysis-pipeline surface is gated behind the `formulas`, `data`, and `diagnostics` extras:

```bash
pip install "quivers[formulas,data,diagnostics]"
```

```python
import numpy as np
import pandas as pd
import torch
from quivers.formulas import fit, formula_to_qvr
from quivers.diagnostics import to_datatree, compare, posterior_predictive_check
```

## A first fit: Gaussian regression in one line

Generate 200 rows of synthetic data with a known linear relationship:

```python
torch.manual_seed(0)
np.random.seed(0)

N = 200
x = np.random.randn(N)
y = 1.0 + 2.0 * x + 0.5 * np.random.randn(N)
df = pd.DataFrame({"y": y, "x": x})
```

Fit `y ~ x` under a Gaussian family. SVI is the fastest inference method (it fits a variational [`Guide`](../../api/inference/guide.md) by optimization rather than running an MCMC chain) and is enough for a quick sanity check:

```python
result = fit(
    "y ~ x",
    data=df,
    family="gaussian",
    method="svi",
    num_samples=2000,    # under SVI, num_samples is the number of optimisation steps
    seed=0,
)
```

`result` is a [`BayesianFit`](../../api/formulas/fit.md): a frozen `dx.Model` with these load-bearing fields:

| Field         | What it holds                                                                 |
|---------------|--------------------------------------------------------------------------------|
| `.formula`    | The parsed [`Formula`](../../api/formulas/formula.md): columns, family, response. |
| `.program`    | The compiled [`MonadicProgram`](../../api/continuous/programs.md).                |
| `.posterior`  | An [`MCMCResult`](../../api/inference/predictive.md) (NUTS / HMC) or a [`Guide`](../../api/inference/guide.md) (SVI). |
| `.observations` | The inference-time observations dict (response values + host-data covariates + plate-index codes). |

Inspect the QVR program the formula compiled to:

```python
print(result.qvr_source)
```

```text
object Resp : FinSet 200
program model : Resp -> Resp
    sample intercept <- Normal(0.0, 5.0)
    sample beta_x <- Normal(0.0, 5.0)
    let beta_x_per_row = (beta_x * x)
    sample sigma <- HalfCauchy(2.0)
    let eta = (intercept + beta_x_per_row)
    let mu = eta
    observe y : Resp <- Normal(mu, sigma)
    return y

export model
```

Per-row predictor values do not appear in the program: the `(beta_x * x)` term references a free variable `x` that the inference path resolves from the observations dict at fit time (the *host-data channel*; see [`quivers.inference.trace`](../../api/inference/trace.md)). The intercept latent stays unindexed because the constant-1 design column doesn't need a per-row binding.

You can also emit the source without fitting:

```python
src = formula_to_qvr("y ~ x", data=df)
result.dump_qvr("regression.qvr")          # write the same source to disk
```

### Recover the coefficients

Under SVI the posterior is an `AutoNormalGuide`. Draw from it and check the means against the true values (`intercept=1.0`, `beta_x=2.0`):

```python
draws = [result.posterior.rsample(torch.zeros(1, 1)) for _ in range(200)]
ic = torch.stack([d["intercept"] for d in draws])
bx = torch.stack([d["beta_x"] for d in draws])
print(f"intercept (true 1.0): {ic.mean().item():.3f} ± {ic.std().item():.3f}")
print(f"beta_x    (true 2.0): {bx.mean().item():.3f} ± {bx.std().item():.3f}")
```

```text
intercept (true 1.0): 0.927 ± 0.036
beta_x    (true 2.0): 2.023 ± 0.030
```

SVI's posterior is biased toward the prior under small `num_samples`; bumping to `num_samples=4000` or switching to `method="nuts"` tightens recovery further.

## Random effects: hierarchical model under SVI

Eight groups with their own intercept offsets, twelve rows per group:

```python
torch.manual_seed(0)
np.random.seed(0)

N_g, N_per = 8, 12
g = np.repeat(np.arange(N_g), N_per)
alpha = np.random.randn(N_g) * 0.6   # group-level offsets
x = np.random.randn(N_g * N_per)
y = 1.5 + 0.7 * x + alpha[g] + 0.4 * np.random.randn(N_g * N_per)

df = pd.DataFrame({"y": y, "x": x, "g": g})
```

The formula adds a `(1 | g)` random intercept:

```python
result = fit(
    "y ~ x + (1 | g)",
    data=df,
    family="gaussian",
    method="svi",
    num_samples=4000,
    seed=0,
)
print(result.qvr_source)
```

```text
object Resp : FinSet 96
object g : FinSet 8
program model : Resp -> Resp
    sample intercept <- Normal(0.0, 5.0)
    sample beta_x <- Normal(0.0, 5.0)
    let beta_x_per_row = (beta_x * x)
    sample sigma_g_Intercept <- HalfNormal(1.0)
    sample z_g_Intercept : g <- Normal(0.0, 1.0)
    let alpha_g_per_row = (sigma_g_Intercept * z_g_Intercept[g_idx])
    sample sigma <- HalfCauchy(2.0)
    let eta = ((intercept + beta_x_per_row) + alpha_g_per_row)
    let mu = eta
    observe y : Resp <- Normal(mu, sigma)
    return y

export model
```

The random-intercept structure expands to a `HalfNormal` scale latent `sigma_g_Intercept`, a *standard-Normal* plate draw `z_g_Intercept` of size `|g|`, and a per-row contribution `sigma_g_Intercept * z_g_Intercept[g_idx]`. This is the non-centered parameterization: `alpha_g[i] = sigma * z[i]` is computed deterministically rather than drawn as `alpha_g[i] ~ Normal(0, sigma)`. Non-centered is the default because it eliminates Neal's funnel and gives HMC / NUTS a well-conditioned geometry for sampling variance components; flip to the textbook form with `fit(..., reparameterize="centered")` when you want it.

Recovery:

```python
def mean_param(res, name, n=200):
    draws = torch.stack(
        [res.posterior.rsample(torch.zeros(1, 1))[name] for _ in range(n)]
    )
    return draws.mean().item(), draws.std().item()

for name, true in [
    ("intercept", 1.5),
    ("beta_x", 0.7),
    ("sigma_g_Intercept", 0.6),
    ("sigma", 0.4),
]:
    m, s = mean_param(result, name)
    print(f"{name:20s} (true {true:.2f}): {m:.3f} ± {s:.3f}")
```

```text
intercept            (true 1.50): 1.748 ± 0.035
beta_x               (true 0.70): 0.620 ± 0.038
sigma_g_Intercept    (true 0.60): 0.273 ± 0.014
sigma                (true 0.40): 0.375 ± 0.028
```

Non-centered emission means SVI gives recognisable recovery for the fixed effects on a model this small; the group-level scale is still under-shrunk because mean-field VI generically under-estimates posterior variance, but the funnel that wrecks centered hierarchical models is gone. For tighter posterior estimates on variance components, switch to `method="nuts"`; for a different variational family, pass `guide=AutoMultivariateNormalGuide` (or any other [auto-guide](../../api/inference/guide.md)).

### Knobs: `method`, `guide`, `reparameterize`

Three orthogonal choices control the inference path:

- **`method="nuts" | "hmc" | "svi"`**. The default `"nuts"` is the right call for serious analysis of any model with random effects: it samples the joint posterior with the No-U-Turn extension to HMC, gets variance components right, and produces the `MCMCResult` that PSIS-LOO and posterior-predictive checks consume below. `"svi"` is the fast option for prototyping or for large models where NUTS chains would be too expensive; it fits a [`Guide`](../../api/inference/guide.md) by optimization instead of sampling.
- **`guide=Cls`** (SVI only). Defaults to [`AutoNormalGuide`](../../api/inference/guide.md) (mean-field diagonal Normal). Other choices: [`AutoMultivariateNormalGuide`](../../api/inference/guide.md) for a full-rank Cholesky; [`AutoLowRankMultivariateNormalGuide`](../../api/inference/guide.md) for `O(D·rank)` memory on high-dimensional models; [`AutoLaplaceApproximation`](../../api/inference/guide.md) for a Gaussian fit at the MAP; [`AutoIAFGuide`](../../api/inference/guide.md) for an inverse-autoregressive normalizing flow. Pass the class, not an instance: `fit(..., guide=AutoMultivariateNormalGuide)`.
- **`reparameterize="noncentered" | "centered"`**. Controls how random-effect terms are emitted. The default `"noncentered"` matches Stan / brms expert practice; `"centered"` matches the textbook density. Mathematically identical posteriors, very different sampling geometry.

## Model comparison with PSIS-LOO

Two models on the same Bernoulli outcome: with and without the predictor. The smaller `N` and short MCMC keeps the example fast.

```python
torch.manual_seed(0)
np.random.seed(0)

N = 80
x = np.random.randn(N)
logit = -0.5 + 1.2 * x
y = np.random.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)
df = pd.DataFrame({"y": y, "x": x})

res_with_x = fit(
    "y ~ x", data=df, family="bernoulli", method="nuts",
    num_warmup=100, num_samples=200, num_chains=2, seed=0,
)
res_intercept = fit(
    "y ~ 1", data=df, family="bernoulli", method="nuts",
    num_warmup=100, num_samples=200, num_chains=2, seed=0,
)
```

Compute per-row log-likelihoods under each model's posterior (the ArviZ comparison needs them):

```python
y_obs = torch.as_tensor(y, dtype=torch.float32)
x_t = torch.as_tensor(x, dtype=torch.float32)

def per_row_log_lik(res, y_obs, x_t):
    samples = res.posterior.samples
    ic = samples["intercept"]                         # (chain, draw)
    bx = samples.get("beta_x", torch.zeros_like(ic))  # zero if intercept-only
    logits = ic[..., None] + bx[..., None] * x_t
    p = torch.sigmoid(logits)
    return y_obs * torch.log(p + 1e-12) + (1 - y_obs) * torch.log(1 - p + 1e-12)

ll_with_x = per_row_log_lik(res_with_x, y_obs, x_t)
ll_intercept = per_row_log_lik(res_intercept, y_obs, x_t)
```

Wrap each posterior into an ArviZ `DataTree` with the canonical groups:

```python
idata_with_x = to_datatree(
    res_with_x.posterior,
    observed_data={"y": y_obs},
    log_likelihood={"y": ll_with_x},
)
idata_intercept = to_datatree(
    res_intercept.posterior,
    observed_data={"y": y_obs},
    log_likelihood={"y": ll_intercept},
)
```

[`compare`](../../api/diagnostics/comparison.md) delegates to ArviZ's PSIS-LOO ranking and stacking weights:

```python
print(compare({"with_x": idata_with_x, "intercept_only": idata_intercept}))
```

```text
                rank  elpd    p  elpd_diff  weight   se  dse  warning
with_x             0 -43.0  0.0        0.0    0.98  4.4  0.0    False
intercept_only     1 -50.0  0.0       -6.0    0.02  3.3  3.4    False
```

The predictor model wins decisively: `with_x` gets 0.98 of the stacking weight and the elpd difference (6.0) is multiple standard errors clear of zero.

## Posterior-predictive checks

[`posterior_predictive_check`](../../api/diagnostics/predictive_checks.md) computes the posterior-predictive p-value for a chosen test statistic. Generate replicated responses under each posterior draw of `(intercept, beta_x)` and check whether the observed mean is in the bulk of the replicated distribution:

```python
samples = res_with_x.posterior.samples
ic, bx = samples["intercept"], samples["beta_x"]
p_pp = torch.sigmoid(ic[..., None] + bx[..., None] * x_t)
pp = torch.bernoulli(p_pp).float()         # (chain, draw, N) replicated y's

idata = to_datatree(
    res_with_x.posterior,
    observed_data={"y": y_obs},
    posterior_predictive={"y": pp},
)
ppc = posterior_predictive_check(idata, observed_name="y", statistic="mean")
print(f"observed mean: {float(ppc['observed']):.3f}")
print(f"predictive mean: {ppc['predictive_mean']:.3f}")
print(f"ppp (mean):    {ppc['ppp']:.3f}")
```

```text
observed mean: 0.312
predictive mean: 0.312
ppp (mean):    0.557
```

A posterior-predictive p-value near 0.5 says the observed mean is in the bulk of the replicated distribution, exactly what we want when the model is well-calibrated for that test statistic. Values near 0 or 1 indicate the statistic is poorly captured.

Built-in statistics: `mean`, `median`, `sd`, `var`, `min`, `max`, `q05`, `q95`. The `by=` argument splits the calculation along a named dim (e.g. `by="Verb"` for a per-category check); for arbitrary statistics, pass a callable.

## What about the original formula?

`BayesianFit` keeps both the parsed [`Formula`](../../api/formulas/formula.md) and the compiled program. You can round-trip:

```python
print(result.formula.formula)         # original formula string
print(result.formula.response_name)   # "y"
for col in result.formula.fixed_columns:
    print(f"  {col.qvr_name}: term={col.term!r}, is_intercept={col.is_intercept}")
```

The lens behind the scenes (`FormulaToQVRModule`) is a typed [`dx.Lens`](https://github.com/aaronstevenwhite/didactic) whose forward emits `(Module, FormulaData)`. `FormulaData` is the strict subset of the formula that isn't encoded in the QVR program: the per-row data arrays, the original (pre-`_qvr_name`) identifier names, presentation labels, the formula string itself. Calling `lens.backward(module, complement)` decodes the structural skeleton out of the Module and fuses it with the complement to reproduce the original `Formula` byte-for-byte. See the [formulas API reference](../../api/formulas/compile.md) for the full lens signature.

## Summary

You have:

- Fit a Gaussian regression with one line of code and recovered the true coefficients via SVI.
- Inspected the emitted `.qvr` source and seen how predictors flow in through the host-data channel rather than as latent draws.
- Fit a hierarchical model with `(1 | g)` random intercepts and seen where SVI's mean-field approximation under-shrinks group-level variance.
- Run NUTS on a Bernoulli regression and compared two models via PSIS-LOO with stacking weights.
- Wrapped a `BayesianFit.posterior` into an ArviZ `DataTree` and run a posterior-predictive check.
- Inspected the lens machinery that maps a `Formula` to a QVR `Module`.

## Next

- The [Analysis Pipelines guide](../../guides/analysis-data-and-formulas.md) is the reference for the whole `quivers.formulas` / `quivers.data` / `quivers.diagnostics` surface, including the full prior-override syntax and the [`DatasetSchema`](../../api/data/schema.md) bridge from dataframes to QVR programs without writing a formula.
- The [Variational Inference tutorial](05-variational-inference.md) shows the lower-level path: building a `MonadicProgram` by hand, tracing, and running SVI without the formula surface.
- The [DSL guide](../../guides/dsl-overview.md) is the right place to go if you want to write or hand-edit `.qvr` programs directly.
