# 3. Hierarchical models

The eight-schools dataset ([Rubin, 1981](https://doi.org/10.3102/10769986006004377)) is the standard stress test for hierarchical-Bayes machinery: eight schools, one observed treatment effect and standard error per school, and the question is how much each school's true effect borrows strength from the global mean. The model is tiny, the posterior geometry is treacherous, and mean-field VI famously collapses on it.

This chapter covers:

- The plate-draw syntax (`v : G <- Normal(0, sigma)`) for per-group random effects.
- Centered vs non-centered parameterizations and how to write each in QVR.
- Running NUTS with R-hat, ESS, and divergence diagnostics.

## Eight schools, centered

$$
\mu \sim \mathrm{Normal}(0, 5), \qquad
\tau \sim \mathrm{HalfNormal}(5), \qquad
\theta_j \sim \mathrm{Normal}(\mu, \tau), \qquad
y_j \sim \mathrm{Normal}(\theta_j, \sigma_j).
$$

The eight $\theta_j$ are a per-group random effect over the group object `School`.

=== "QVR"

    ```text
    object School : FinSet 8
    program eight_schools_centred : School -> School [effects=[Sample, Score]]
        sample mu  <- Normal(0.0, 5.0)
        sample tau <- HalfNormal(5.0)
        sample theta : School <- Normal(mu, tau)
        observe y : School <- Normal(theta, sigma_j)
        return theta

    export eight_schools_centred
    ```

=== "NumPyro"

    ```python
    def eight_schools(sigma_j, y=None):
        mu  = numpyro.sample("mu",  dist.Normal(0., 5.))
        tau = numpyro.sample("tau", dist.HalfNormal(5.))
        with numpyro.plate("schools", 8):
            theta = numpyro.sample("theta", dist.Normal(mu, tau))
            numpyro.sample("y", dist.Normal(theta, sigma_j), obs=y)
    ```

The `theta : School <- Normal(...)` line is a *plate-draw*: it samples one value per element of `School`. The plate index is the object's cardinality (8). The `observe y : School <- ...` line is a *vectorized observe* over the same index.

The compiler synthesizes a `PlateDraw` morphism whose codomain is the product space `School ⊗ Real`; you can index into it like `theta[j]` inside subsequent `let` arithmetic.

#! Centered fails mean-field

The centered parameterization puts `theta_j` *inside* the prior for `mu` and `tau`, which creates a funnel-shaped posterior ([Neal, 2003](https://doi.org/10.1214/aos/1056562461), §8). Mean-field VI doesn't see the funnel and collapses to a tight Gaussian around `tau ≈ 0`. To confirm:

```python
import torch
from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, ELBO, SVI

CENTRED_SRC = """
object School : FinSet 8

program eight_schools_centred : School -> School
    sample mu  <- Normal(0.0, 5.0)
    sample tau <- HalfNormal(5.0)
    sample theta : School <- Normal(mu, tau)
    observe y : School <- Normal(theta, sigma_j)
    return theta

export eight_schools_centred
"""

program = loads(CENTRED_SRC)
model   = program.morphism

sigma_j = torch.tensor([[15., 10., 16., 11., 9., 11., 10., 18.]])   # (1, 8)
y_obs   = torch.tensor([[28., 8., -3., 7., -1., 1., 18., 12.]])     # (1, 8)

guide = AutoNormalGuide(model, observed_names={"y", "sigma_j"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)
x_tensor = torch.zeros(1, 1)
observations = {"sigma_j": sigma_j, "y": y_obs}
for _ in range(50):                              # bump to ~3000 for real fits
    svi.step(x_tensor, observations)
```

You'll see something like `tau ≈ 0.1 ± 0.05`: the diagnostic-textbook signature of a funnel collapse. The true posterior mean of `tau` is closer to 3.

## Non-centered fixes it

The standard fix is to reparameterise ([Papaspiliopoulos, Roberts & Sköld, 2007](https://doi.org/10.1214/088342307000000014)): draw $\eta_j \sim \mathrm{Normal}(0, 1)$ and define $\theta_j = \mu + \tau \cdot \eta_j$ deterministically.

```qvr
object School : FinSet 8
program eight_schools_noncentred : School -> School [effects=[Sample, Score]]
    sample mu  <- Normal(0.0, 5.0)
    sample tau <- HalfNormal(5.0)
    sample eta : School <- Normal(0.0, 1.0)
    let theta = mu + tau * eta
    observe y : School <- Normal(theta, sigma_j)
    return theta

export eight_schools_noncentred
```

Re-running with the non-centered parameterization, `AutoNormalGuide` recovers a posterior with `tau` mean around 3, competitive with NUTS on this small problem.

## NUTS

For the centered parameterization (or when you want to trust the posterior mass exactly), reach for the No-U-Turn Sampler ([Hoffman & Gelman, 2014](https://www.jmlr.org/papers/v15/hoffman14a.html)):

```python
from quivers.inference import NUTSKernel, MCMC

kernel = NUTSKernel(
    target_accept=0.95,              # high target -> smaller step -> fewer divergences
    max_tree_depth=10,
)
# Small budget for documentation; bump warmup/samples for real fits.
mcmc = MCMC(
    kernel,
    num_warmup=50,
    num_samples=100,
    num_chains=2,
    init_strategy="prior",
)

result = mcmc.run(model, x_tensor, {"sigma_j": sigma_j, "y": y_obs})
print("posterior mean tau:", result.samples["tau"].mean().item())
print("R-hat tau:", result.r_hat["tau"].item())
print("ESS tau:", result.ess["tau"].item())
print("divergences:", result.total_divergences)
```

A clean run shows R-hat < 1.01 for every site (rank-normalized split-R-hat, [Vehtari, Gelman, Simpson, Carpenter & Bürkner, 2021](https://doi.org/10.1214/20-BA1221)), ESS in the thousands, and zero divergences. On the centered parameterization you'll see a handful of divergences for `tau` near zero: the diagnostic flag that says "consider non-centered."

Healthy console output looks like:

```text
posterior mean tau: 3.42
R-hat tau: 1.003
ESS tau: 1287.4
divergences: 0
```

If R-hat is above 1.01 anywhere, give NUTS more warmup steps (`num_warmup=2000`). If divergences are nonzero, raise `target_accept` to 0.99 or reparameterize to non-centered. If ESS is small (<100 per chain), the chain is mixing slowly: either run longer or reparameterize.

### Centered or non-centered?

A useful rule of thumb from [Betancourt & Girolami (2013)](https://doi.org/10.48550/arXiv.1312.0906): the centered parameterization is good when the data is informative about $\theta_j$ relative to the prior, that is when the per-group likelihood scale $\sigma_j$ is small relative to the population scale $\tau$. The non-centered parameterization is good in the opposite regime: weak per-group likelihoods, where the posterior pinches toward the prior funnel. Eight schools sits in the weak-likelihood regime ($\sigma_j$ ranges 9-18 against a prior $\tau$ in single digits), so non-centered is the right call.

### Init strategy and mass matrix

`init_strategy="prior"` draws the chain's initial state from the model prior. The other shipped options are `"uniform"` (uniform on the unconstrained space) and `"value"` (you supply a dict). `"prior"` is the safe default: it starts the chain in regions the prior places mass on.

`NUTSKernel(mass_matrix="diagonal")` adapts a diagonal mass matrix during warmup, which works for most hierarchical models. `"dense"` adapts a full matrix and is worth a try when latents are strongly correlated, at quadratic memory cost. `"identity"` skips adaptation entirely and only makes sense for already-well-scaled models.

## Posterior predictive

[`Predictive`](../../api/inference/predictive.md) accepts either a [`Guide`](../../api/inference/guide.md) or an `MCMCResult`:

```python
from quivers.inference import Predictive

pred = Predictive(model, posterior=result, num_samples=100)
y_hat = pred(x_tensor, {"sigma_j": sigma_j})["y"]  # (100, ..., 8)
print("predictive school 1:", y_hat[..., 0].mean().item(),
      "+/-", y_hat[..., 0].std().item())
```

## What you've seen

- **Plate-draws.** `v : G <- F(...)` declares one draw per index of object `G`, producing a vector-valued latent.
- **Non-centered parameterization.** A small-but-essential trick for hierarchical models; QVR doesn't automate it (yet), but writing it explicitly is two lines.
- **Diagnostics on `MCMCResult`.** R-hat, ESS, divergences are first-class fields, not strings in a log.

## Try this

- Run `AutoMultivariateNormal` on the centered parameterization. It can sometimes recover the funnel where mean-field can't.
- Change `target_accept` from 0.95 to 0.8 and watch divergences appear. The trade-off is step size vs trajectory length.
- Add a per-school covariate $x_j$ and lift the model to a varying-intercepts-and-slopes regression.

## Next

[Chapter 4](04-marginalize.md) introduces the `marginalize` block: QVR's typed-scope marginalization surface for discrete latents and mixtures.
