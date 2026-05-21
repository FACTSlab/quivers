# 4. Mixtures and discrete latents

When a model has a discrete latent variable, you have two options. Sample it (every gradient step pays a Monte Carlo penalty and you lose the score-function variance), or marginalize it out (sum over its support, get a deterministic log-likelihood, gradients flow cleanly). The right choice depends on the support size: if the discrete latent has only a handful of values per observation, marginalizing is the obvious win.

QVR makes marginalization a first-class block. The body of the block runs *once per value* the discrete latent can take; the runtime collects per-value log-likelihoods and combines them under the prior with a `logsumexp`. Mathematically this is an exact integration over the discrete latent; computationally it's the categorical-prior version of the Rao-Blackwellised gradient ([Casella & Robert, 1996](https://doi.org/10.1093/biomet/83.1.81)). The same syntax handles flat mixtures, hierarchical mixtures with grouping, and HMM-shaped models with a per-row latent.

## A two-component Gaussian mixture

Each observation comes from one of two Gaussian clusters; we don't know which.

=== "QVR"

    ```text
    object Item : FinSet 500
    object K : FinSet 2
    program gmm : Item -> Item [effects=[Sample, Score, Marginal]]
        sample probs : K <- HalfNormal(1.0)
        sample mu_k  : K <- Normal(0.0, 5.0)
        sample sd_k  : K <- HalfNormal(1.0)

        marginalize z : K <- Dirichlet(probs)
            observe y : Item <- Normal(mu_k[z], sd_k[z])
        return y

    export gmm
    ```

=== "Pyro (enumerated)"

    <!-- python: skip -->
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

The `marginalize z : K <- Categorical(probs)` block, with its body of observe steps indented underneath, is exactly the Stan `log_sum_exp` pattern, expressed once and instantiated for every row of the response. The `effects=[Marginal]` entry in the option block makes the marginalization visible at the program signature.

#! Fitting the mixture

(See [`docs/examples/source/mixture_model.qvr`](../../examples/source/mixture_model.qvr) for the full end-to-end version with grouped marginalisation and the `factor` patterns needed to drive the body. The snippet below shows the shape of the fit; for a running version copy from the example.)

<!-- python: skip -->
```python
import torch
from quivers.dsl import loads
from quivers.inference import AutoNormalGuide, ELBO, SVI

GMM_SRC = """
object Item : FinSet 500
object K    : 2

program gmm : Item -> Item
    sample probs : K <- HalfNormal(1.0)
    sample mu_k  : K <- Normal(0.0, 5.0)
    sample sd_k  : K <- HalfNormal(1.0)

    marginalize z : K <- Dirichlet(probs)
        observe y : Item <- Normal(mu_k[z], sd_k[z])
    return y

export gmm
"""

program = loads(GMM_SRC)
model   = program.morphism

torch.manual_seed(0)
true_mu = torch.tensor([-2.0, 2.0])
true_sd = torch.tensor([0.5, 0.7])
z_true  = torch.bernoulli(torch.full((500,), 0.6)).long()
y_data  = (torch.randn(500) * true_sd[z_true] + true_mu[z_true]).unsqueeze(0)

guide = AutoNormalGuide(model, observed_names={"y"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)
x_tensor = torch.zeros(1, 1)
observations = {"y": y_data}
for _ in range(20):                            # bump to ~3000 for real fits
    svi.step(x_tensor, observations)
```

The `marginalize` block is integrated out exactly at every SVI step, so the gradients on `mu_k`, `sd_k`, and `probs` flow through a smooth `logsumexp`. No discrete-sampling variance contaminates the ELBO.

## Hierarchical mixtures with grouping

Suppose each observation belongs to one of `G` groups, and the categorical mixture proportions vary by group. The marginalization has to respect group membership: the log-likelihood over the discrete latent gets aggregated *per group*, not per row. The `marginalize` header declares the grouping plate (`over G`); each observe inside the body carries its own `via <idx>` clause naming the fibration from its response plate into the grouping plate.

```text
object Item : FinSet 1000
object G : FinSet 20
object K : FinSet 3
program grouped_mixture : Item -> Item [effects=[Sample, Score, Marginal]]
    sample group : Item <- HalfNormal(1.0)
    sample probs : G    <- HalfNormal(1.0)
    sample mu_k  : K    <- Normal(0.0, 5.0)
    sample sd_k  : K    <- HalfNormal(1.0)

    marginalize z : K <- Categorical(probs) [over=G]
        observe y : Item <- Normal(mu_k[z], sd_k[z]) [via=group]
    return y

export grouped_mixture
```

The `[over=G]` entry on the marginalize step's option block declares the grouping plate; the `[via=group]` entry on each observe says "every row of `y` is fibred over `G` by `group`, and the marginalization is per group, not per row." The `group : Item <- HalfNormal(1.0)` line names a per-row fibration into `G`: today the DSL doesn't have a dedicated fibration declaration, so the canonical idiom is to declare it as if it were a per-row latent and then supply the integer indices through the observations dict at fit time (cf. `docs/examples/source/mixture_model.qvr`). The block contributes

$$
\sum_{g \in G}\ \log\!\sum_{k=1}^{K}\exp\!\left[\log \pi_{g,k} + \sum_{n:\ \mathrm{group}(n)=g}\ \log f(y_n \mid \mu_k, \sigma_k)\right]
$$

to the log-density, which is the right Kan extension along the fibration `Item -> G` and matches Stan's `target += log_mix(probs[g], ll_item[i])` accumulation. A grouped block can contain multiple observes, each with its own `[via=<idx>]` entry, when several heterogeneous response axes share the same per-group class indicator; the per-axis log-likelihoods scatter-sum into the same `(|G|, K)` accumulator before the log-sum-exp.

## When to marginalize vs sample

The decision is a straight cost-benefit:

- `marginalize` costs roughly `K × (body cost)` per evaluation. The reward is exact gradients with respect to the discrete-prior parameters and zero Monte Carlo variance on the discrete latent.
- Sampling the discrete latent with `ScoreFunction` (REINFORCE) costs `(body cost)` per evaluation but pays Monte Carlo variance that scales with $K$ and with the variance of the per-component log-likelihood. In practice, REINFORCE gradients are noisy enough that you usually need 10x to 100x more SVI steps to match a marginalized run, swamping the per-step savings.

The crossover is around `K ≈ 32`: below that, `marginalize` wins on wall-clock to convergence; above that, the K× factor on the body becomes painful and a relaxed continuous proxy (Gumbel-softmax, [Maddison, Mnih & Teh, 2017](https://doi.org/10.48550/arXiv.1611.00712)) tends to be the better tradeoff.

| Discrete support per row | Recommendation |
|---|---|
| `K` ≤ 8 | `marginalize`, always. |
| 8 < `K` ≤ 32 | `marginalize` unless the body itself is expensive. |
| 32 < `K` ≤ 100 | Profile both; relaxed continuous if the body is heavy. |
| `K` > 100 or unbounded | Gumbel-softmax or score-function with a learned baseline. |
| Continuous-discrete mixture | `marginalize` the discrete part, reparameterize the continuous part. |

```mermaid
flowchart LR
    A["program block"] -- "marginalize z : K" --> B["body runs K times,<br/>once per z value"]
    B --> C["logsumexp over K"]
    C --> D["score added to ELBO"]
    A --> D
```

## Try this

- Initialise the GMM with `K = 4` and watch what happens to the recovered `mu_k`. (Hint: mixture models have a label-switching identifiability problem, [Stephens, 2000](https://doi.org/10.1111/1467-9868.00265); the standard fix is `ordered[K] mu_k` in Stan; in QVR you'd add a `let mu_k_sorted = sort(mu_k)` constraint or use an ordered prior.)
- Convert the grouped mixture to a `marginalize` without the `over` / `via` clauses and observe the difference: per-row marginalization versus per-group.
- Combine with chapter 3's plate-draws: a hierarchical mixture where each group has its own `mu_k` drawn from a hyperprior.

## Next

[Chapter 5](05-time-series.md) looks at sequence-shaped models: HMMs, state-space models, and the chart-shaped deduction surface.
