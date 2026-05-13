# 4. Mixtures and discrete latents

When a model has a discrete latent variable, you have two options. Sample it (every gradient step pays a Monte Carlo penalty and you lose the score-function variance), or marginalise it out (sum over its support, get a deterministic log-likelihood, gradients flow cleanly). The right choice depends on the support size: if the discrete latent has only a handful of values per observation, marginalising is the obvious win.

QVR makes marginalisation a first-class block. The body of the block runs *once per value* the discrete latent can take; the runtime collects per-value log-likelihoods and combines them under the prior with a `logsumexp`. Mathematically this is an exact integration over the discrete latent; computationally it's the categorical-prior version of the Rao-Blackwellised gradient ([Casella & Robert, 1996](https://doi.org/10.1093/biomet/83.1.81)). The same syntax handles flat mixtures, hierarchical mixtures with grouping, and HMM-shaped models with a per-row latent.

## A two-component Gaussian mixture

Each observation comes from one of two Gaussian clusters; we don't know which.

=== "QVR"

    ```qvr
    quantale real
    object Item : 500
    object K    : 2

    program gmm : Item -> Item ! Sample, Score, Marginal
        probs <- Dirichlet(1.0, 1.0)
        mu_k  : K <- Normal(0.0, 5.0)
        sd_k  : K <- HalfNormal(1.0)

        marginalize z : K <- Categorical(probs) in {
            observe y <- Normal(mu_k[z], sd_k[z])
        }
        return y

    export gmm
    ```

=== "Pyro (enumerated)"

    ```python
    @config_enumerate
    def model(data):
        probs = pyro.sample("probs", dist.Dirichlet(torch.ones(2)))
        mu_k  = pyro.sample("mu_k",  dist.Normal(0., 5.).expand([2]).to_event(1))
        sd_k  = pyro.sample("sd_k",  dist.HalfNormal(1.).expand([2]).to_event(1))
        with pyro.plate("data", len(data)):
            z = pyro.sample("z", dist.Categorical(probs),
                            infer={"enumerate": "parallel"})
            pyro.sample("y", dist.Normal(mu_k[z], sd_k[z]), obs=data)
    ```

=== "Stan"

    ```stan
    data { int N; vector[N] y; }
    parameters {
        simplex[2] probs;
        ordered[2] mu_k;
        vector<lower=0>[2] sd_k;
    }
    model {
        probs ~ dirichlet([1, 1]');
        mu_k  ~ normal(0, 5);
        sd_k  ~ normal(0, 1);
        for (n in 1:N) {
            vector[2] lp;
            for (k in 1:2)
                lp[k] = log(probs[k])
                      + normal_lpdf(y[n] | mu_k[k], sd_k[k]);
            target += log_sum_exp(lp);
        }
    }
    ```

The `marginalize z : K <- Categorical(probs) in { ... }` block is exactly the Stan `log_sum_exp` pattern, expressed once and instantiated for every row of the response. The `! Marginal` effect annotation makes the marginalisation visible at the program signature.

## Posterior over the discrete latent

After fitting, you sometimes want to know *which cluster* an observation belongs to. Quivers exposes the per-row responsibilities at the marginalized block's location:

```python
program = loads(open("gmm.qvr").read())
model   = program.morphism

torch.manual_seed(0)
true_mu = torch.tensor([-2.0, 2.0])
true_sd = torch.tensor([0.5, 0.7])
z_true  = torch.bernoulli(torch.full((500,), 0.6)).long()
y_data  = torch.randn(500) * true_sd[z_true] + true_mu[z_true]

guide = AutoNormalGuide(model, observed_names={"y"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)
for _ in range(3000):
    svi.step({}, {"y": y_data})

from quivers.inference import responsibilities
resp = responsibilities(model, guide, {"y": y_data}, latent="z")  # (500, 2)
print("posterior P(z=1 | y[:5]):", resp[:5, 1].tolist())
```

The `responsibilities` helper takes the marginalised block's name (`z`) and returns the per-row posterior `P(z = k | y_n, θ)` averaged over posterior samples of `θ`. There is no analogous helper to install in Pyro; you build it by hand from `enumerate`-trace post-processing.

## Hierarchical mixtures with grouping

Suppose each observation belongs to one of `G` groups, and the categorical mixture proportions vary by group. The marginalisation has to respect group membership: the log-likelihood over the discrete latent gets aggregated *per group*, not per row.

<!-- compile: false -->
```qvr
quantale real
object Item : 1000
object G    : 20
object K    : 3

program grouped_mixture : Item -> Item ! Sample, Score, Marginal
    group : Item <- Categorical(uniform_prior)
    probs : G    <- Dirichlet(1.0, 1.0, 1.0)
    mu_k  : K    <- Normal(0.0, 5.0)
    sd_k  : K    <- HalfNormal(1.0)

    marginalize z : K <- Categorical(probs)
        over G via group
        in {
            observe y <- Normal(mu_k[z], sd_k[z])
        }
    return y

export grouped_mixture
```

The `over G via group` clause says "every Item is fibred over G by `group`, and the marginalisation is per group, not per row." The block contributes

$$
\sum_{g \in G}\ \log\!\sum_{k=1}^{K}\exp\!\left[\log \pi_{g,k} + \sum_{n:\ \mathrm{group}(n)=g}\ \log f(y_n \mid \mu_k, \sigma_k)\right]
$$

to the log-density, which is the right Kan extension along the fibration `Item -> G` and matches Stan's `target += log_mix(probs[g], ll_item[i])` accumulation.

## When to marginalise vs sample

| Discrete support per row | Recommendation |
|---|---|
| Small (`K` < 20), categorical | `marginalize`; the cost is `K`× the body, gradients are exact. |
| Moderate (`K` < 100), categorical | `marginalize` if budget allows. |
| Large or unbounded | Sample with a discrete proposal; consider Gumbel-softmax relaxations. |
| Continuous-discrete mixture | `marginalize` the discrete part, sample the continuous part. |

## Try this

- Initialise the GMM with `K = 4` and watch what happens to the recovered `mu_k`. (Hint: mixture models have a label-switching identifiability problem, [Stephens, 2000](https://doi.org/10.1111/1467-9868.00265); the standard fix is `ordered[K] mu_k` in Stan; in QVR you'd add a `let mu_k_sorted = sort(mu_k)` constraint or use an ordered prior.)
- Convert the grouped mixture to a `marginalize` without the `over` / `via` clauses and observe the difference: per-row marginalisation versus per-group.
- Combine with chapter 3's plate-draws: a hierarchical mixture where each group has its own `mu_k` drawn from a hyperprior.

## Next

Chapter 5 looks at sequence-shaped models: HMMs, state-space models, and the chart-shaped deduction surface.
