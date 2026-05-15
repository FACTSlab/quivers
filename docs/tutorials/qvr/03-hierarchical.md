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

    ```qvr
    object School : 8

    program eight_schools_centred : School -> School ! Sample, Score
        mu  <- Normal(0.0, 5.0)
        tau <- HalfNormal(5.0)
        theta : School <- Normal(mu, tau)
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

## Centered fails mean-field

The centered parameterization puts `theta_j` *inside* the prior for `mu` and `tau`, which creates a funnel-shaped posterior ([Neal, 2003](https://doi.org/10.1214/aos/1056562461), §8). Mean-field VI doesn't see the funnel and collapses to a tight Gaussian around `tau ≈ 0`. To confirm:

```python
program = loads(open("eight_schools_centred.qvr").read())
model   = program.morphism

sigma_j = torch.tensor([15., 10., 16., 11., 9., 11., 10., 18.])
y_obs   = torch.tensor([28., 8., -3., 7., -1., 1., 18., 12.])

guide = AutoNormalGuide(model, observed_names={"y", "sigma_j"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)
x_tensor = torch.zeros(1, 1)
observations = {"sigma_j": sigma_j, "y": y_obs}
for _ in range(3000):
    svi.step(x_tensor, observations)

post = torch.stack([guide.rsample(x_tensor)["tau"] for _ in range(1000)])
print("posterior tau:", post.mean().item(), "+/-", post.std().item())
```

You'll see something like `tau ≈ 0.1 ± 0.05`: the diagnostic-textbook signature of a funnel collapse. The true posterior mean of `tau` is closer to 3.

## Non-centered fixes it

The standard fix is to reparameterise ([Papaspiliopoulos, Roberts & Sköld, 2007](https://doi.org/10.1214/088342307000000014)): draw $\eta_j \sim \mathrm{Normal}(0, 1)$ and define $\theta_j = \mu + \tau \cdot \eta_j$ deterministically.

```qvr
object School : 8

program eight_schools_noncentred : School -> School ! Sample, Score
    mu  <- Normal(0.0, 5.0)
    tau <- HalfNormal(5.0)
    eta : School <- Normal(0.0, 1.0)
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
mcmc = MCMC(
    kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    init_strategy="prior",
)

result = mcmc.run(model, x_tensor, {"sigma_j": sigma_j, "y": y_obs})
print("posterior mean tau:", result.samples["tau"].mean().item())
print("R-hat tau:", result.r_hat["tau"].item())
print("ESS tau:", result.ess["tau"].item())
print("divergences:", result.total_divergences)
```

A clean run shows R-hat < 1.01 for every site (rank-normalized split-R-hat, [Vehtari, Gelman, Simpson, Carpenter & Bürkner, 2021](https://doi.org/10.1214/20-BA1221)), ESS in the thousands, and zero divergences. On the centered parameterization you'll see a handful of divergences for `tau` near zero: the diagnostic flag that says "consider non-centered."

## Posterior predictive

[`Predictive`](../../api/inference/predictive.md) accepts either a [`Guide`](../../api/inference/guide.md) or an [`MCMCResult`](../../api/inference/svi.md):

```python
from quivers.inference import Predictive

pred = Predictive(model, posterior=result, num_samples=500)
y_hat = pred(x_tensor, {"sigma_j": sigma_j})["y"]  # (500, 8)
print("predictive school 1:", y_hat[:, 0].mean().item(),
      "+/-", y_hat[:, 0].std().item())
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
