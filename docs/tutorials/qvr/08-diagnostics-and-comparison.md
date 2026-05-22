# 8. Diagnostics, posterior-predictive checks, and model comparison

Fitting a model is half the workflow. The other half is asking whether the fit converged, whether the model can reproduce the observed data, and whether one model is preferable to another. Quivers delegates all of these to [ArviZ](https://python.arviz.org/) through a thin adapter at [`quivers.diagnostics`](../../api/diagnostics/index.md). This chapter walks the full diagnostic pipeline on a regression you've already seen in chapter 1.

## The DataTree bridge

ArviZ's lingua franca is an [`xarray.DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html) with named groups: `posterior`, `sample_stats`, `posterior_predictive`, `log_likelihood`, `observed_data`, `constant_data`. [`to_datatree`](../../api/diagnostics/index.md) converts an [`MCMCResult`](../../api/inference/predictive.md) into that tree. Once you have the tree, every ArviZ entry point works out of the box.

```python
import torch
from quivers.dsl import loads
from quivers.inference import NUTSKernel, MCMC, Predictive
from quivers.diagnostics import to_datatree

REGRESSION_SRC = """
object Item : FinSet 100

program regression : Item -> Item
    sample sigma  <- HalfNormal(1.0)
    sample beta_0 <- Normal(0.0, 5.0)
    sample beta_1 <- Normal(0.0, 2.0)
    let mu = beta_0 + beta_1 * x_design
    observe y : Item <- Normal(mu, sigma)
    return y

export regression
"""

program = loads(REGRESSION_SRC)
model = program.morphism

torch.manual_seed(0)
x_data = torch.randn(100)
y_data = 1.5 + 2.7 * x_data + 0.3 * torch.randn(100)

kernel = NUTSKernel(target_accept=0.9)
# Small budget for documentation; bump warmup/samples for real fits.
mcmc = MCMC(kernel, num_warmup=50, num_samples=100, num_chains=2)
result = mcmc.run(model, torch.zeros(1, 1),
                  {"x_design": x_data, "y": y_data})

# Posterior-predictive draws conditional on the fit. Reshape
# ``(num_samples, plate)`` into ``(num_chains, num_samples_per_chain,
# plate)`` for the ArviZ convention.
n_chains, n_per_chain = result.num_chains, result.num_samples
predictive = Predictive(
    model, posterior=result, num_samples=n_chains * n_per_chain,
)
ppc = predictive(torch.zeros(1, 1), {"x_design": x_data})
pp_y = ppc["y"].reshape(n_chains, n_per_chain, 100)

tree = to_datatree(
    result,
    observed_data={"y": y_data},
    posterior_predictive={"y": pp_y},
    constant_data={"x_design": x_data},
    coords={"Item": list(range(100))},
    dims={"y": ["Item"]},
)
```

The `coords` and `dims` arguments are optional but useful: they let ArviZ label axes by name in plots (`Item 0`, `Item 1`, ...) instead of by integer index.

## Convergence diagnostics

```python
import arviz as az

summary = az.summary(tree, var_names=["beta_0", "beta_1", "sigma"])
print(summary)
```

The columns to watch:

- `r_hat`: rank-normalised split-R-hat. Should be < 1.01 for every site.
- `ess_bulk`, `ess_tail`: bulk and tail effective sample sizes. Both should clear 100 per chain (so 400 here, since `num_chains=4`).
- `mcse_mean`, `mcse_sd`: Monte Carlo standard error of the posterior mean and standard deviation. Should be small relative to the posterior `sd`.

Trace and rank plots give you a visual on the same diagnostics:

<!-- python: skip -->
```python
az.plot_trace(tree, var_names=["beta_0", "beta_1", "sigma"])
az.plot_rank(tree,  var_names=["beta_0", "beta_1", "sigma"])
```

A healthy `plot_trace` shows hairy caterpillars overlapping across chains; a healthy `plot_rank` shows uniformly mixed bars across chains. Anything else is a mixing problem and you should run more warmup, raise `target_accept`, or reparameterise.

## Posterior-predictive checks

A posterior-predictive check ([Gelman, Meng & Stern, 1996, *Statistica Sinica* 6:733-807](https://www3.stat.sinica.edu.tw/statistica/J6N4/J6N41/J6N41.htm)) asks whether replicate datasets drawn from the fitted model look like the observed data. The canonical plot:

<!-- python: skip -->
```python
az.plot_ppc(tree, var_names=["y"], num_pp_samples=200)
```

For numeric statistics, quivers ships [`posterior_predictive_check`](../../api/diagnostics/index.md) for a quick scalar comparison:

```python
from quivers.diagnostics import posterior_predictive_check

result_dict = posterior_predictive_check(
    tree, observed_name="y", statistic="mean",
)
print(result_dict)            # observed, posterior mean, p-value
```

A Bayesian p-value near 0.5 indicates the statistic of the observed data sits in the middle of the posterior-predictive distribution. Values near 0 or 1 indicate the model under- or over-predicts that statistic. The standard repertoire is `"mean"`, `"sd"`, `"min"`, `"max"`, `"median"`, with arbitrary user-supplied callables also accepted.

## Comparing models

Suppose you also fit a quadratic-in-`x` version of the same regression. Each fit you want to compare needs a `log_likelihood` group in its DataTree, so populate it during conversion. The Bayesian regression example in `docs/examples/source/bayesian_regression.qvr` shows the canonical way to compute per-observation likelihoods.

<!-- python: skip -->
```python
from quivers.diagnostics import compare

table = compare({"linear": tree_linear, "quadratic": tree_quad})
print(table)
```

[`compare`](../../api/diagnostics/index.md) delegates to [`arviz.compare`](https://python.arviz.org/en/stable/api/generated/arviz.compare.html), which runs PSIS-LOO ([Vehtari, Gelman & Gabry, 2017](https://doi.org/10.1007/s11222-016-9696-4)) on each fit and ranks them by expected log-pointwise predictive density (`elpd_loo`). The first row of the table is the preferred model; the `dse` column gives the standard error of the difference relative to the top model, and pseudo-BMA weights (`weight`) tell you how much each model is worth in a model-average.

A few warning signs to watch for in `compare`'s output:

- `p_loo` (the effective number of parameters) bigger than $N/5$ for $N$ observations: PSIS-LOO is unreliable.
- `k > 0.7` for any observation in the Pareto-k diagnostic ([Vehtari, Simpson, Gelman, Yao & Gabry, 2024](https://doi.org/10.48550/arXiv.1507.02646)): the importance-sampling reweighting is unstable. Re-fit with the observation held out, or use exact LOO.

## Try this

- Run the eight-schools fit from chapter 3 through this pipeline. Plot the trace; the centered parameterisation should show divergences in `sample_stats/diverging`.
- Add a `Cauchy` likelihood model alongside the `Normal` one and compare them on a dataset with outliers.
- Slice the posterior tree by chain (`tree.posterior.sel(chain=0)`) and confirm individual chains are converging to the same place.

## Next

If you want to extend the categorical machinery itself (custom algebras, custom transformations, custom composition rules), continue to the Python-API track starting at [chapter 1](../python/01-first-quiver.md). If you want a deeper formal treatment, the [semantics section](../../semantics/index.md) gives the full denotational semantics of the DSL.


## References

- Aki Vehtari, Andrew Gelman, and Jonah Gabry. 2017. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing*, 27(5):1413–1432.
- Aki Vehtari, Daniel Simpson, Andrew Gelman, Yuling Yao, and Jonah Gabry. 2024. Pareto smoothed importance sampling. *Journal of Machine Learning Research*, 25(72):1–58. arXiv:1507.02646.
