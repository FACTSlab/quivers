# Latent Dirichlet Allocation

## Overview

[Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) ([Blei, Ng & Jordan 2003](https://www.jmlr.org/papers/v3/blei03a.html)) is the canonical [topic model](https://en.wikipedia.org/wiki/Topic_model). Each document draws a topic mixture $\theta_d$ from a [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior; each topic draws a vocabulary distribution $\phi_k$ from a Dirichlet prior; each word in a document draws a topic $z_{d,n}$ from $\theta_d$ and then a token $w_{d,n}$ from $\phi_{z_{d,n}}$. The per-word topic assignment is integrated out by a scoped `marginalize` block, yielding the closed-form per-word [Categorical](https://en.wikipedia.org/wiki/Categorical_distribution) marginal

$$
p(w_{d,n} \mid \theta_d, \phi) \;=\; \sum_{k=1}^{K} \theta_d[k]\,\phi_k[w_{d,n}],
$$

computed via [log-sum-exp](https://en.wikipedia.org/wiki/LogSumExp) over the $K$ topic atoms.

## QVR Source

```qvr
composition log_prob [level=algebra]

object Doc : FinSet 20
object Topic : FinSet 3
object Word : FinSet 200

program lda(alpha : Real, beta : Real) : Word -> Word [effects=[Score]]
    sample theta : Doc <- Dirichlet(alpha) [over=Topic, iid_over=Doc]
    sample phi : Topic <- Dirichlet(beta) [over=Word, iid_over=Topic]

    marginalize z : Topic <- Categorical(theta) [over=Doc, reduction=logsumexp]
        observe w : Word <- Categorical(phi[z]) [via=word_idx]

    return theta

export lda
```

## Walkthrough

`object Doc : FinSet 20`, `object Topic : FinSet 3`, `object Word : FinSet 200` declare the three discrete plates: $D = 20$ documents, $K = 3$ topics, $V = 200$ vocabulary items. `composition log_prob [level=algebra]` selects the log-probability semiring so the `Score` effect on the program accumulates log-densities additively.

The two `sample` steps draw the document-topic and topic-vocabulary simplex matrices under symmetric Dirichlet priors:

- `sample theta : Doc <- Dirichlet(alpha) [over=Topic, iid_over=Doc]` draws a $D \times K$ matrix in which each row is an independent symmetric-$\alpha$ Dirichlet over the topic simplex. The `over=Topic` clause names the family's event axis (Dirichlet event-rank 1), and `iid_over=Doc` asserts each row is independent.
- `sample phi : Topic <- Dirichlet(beta) [over=Word, iid_over=Topic]` is the symmetric construction: a $K \times V$ matrix whose every row is an independent Dirichlet over the vocabulary simplex.

The scoped marginalize block

<!-- compile: false -->
```qvr
marginalize z : Topic <- Categorical(theta) [over=Doc, reduction=logsumexp]
    observe w : Word <- Categorical(phi[z]) [via=word_idx]
```

introduces the per-word topic latent $z$ under a Categorical prior parameterised by the document-shaped `theta`. The body's `Categorical(phi[z])` looks up the chosen topic's vocabulary row and scores the observed token `w`. The `[over=Doc]` grouping plate accumulates each observation into its document; `[via=word_idx]` names the runtime fibration from each word position into its document. The agenda evaluates the `[reduction=logsumexp]` reduction over the $K$ topics, integrating $z$ out by pushforward along the projection $\Phi \times \mathsf{Topic} \to \Phi$ and recovering the closed-form per-word mixture marginal $\sum_k \theta_d[k]\,\phi_k[w]$.

Finally `return theta` projects the program's joint kernel onto the document-topic mixture; the per-document topic mixture is the natural quantity to inspect after fitting.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
import torch.nn.functional as F
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/lda.qvr")
fit  = prog.lda(alpha=1.0, beta=0.5)
model = fit.morphism

D, T, V, Npd = 20, 3, 200, 10

# Sharp topic-word and topic-mixture matrices give a recoverable corpus.
phi_true   = F.softmax(torch.randn(T, V) * 2.0, dim=-1)
theta_true = F.softmax(torch.randn(D, T) * 1.5, dim=-1)

word_idx, tokens = [], []
for d in range(D):
    for _ in range(Npd):
        z  = torch.distributions.Categorical(theta_true[d]).sample()
        wt = torch.distributions.Categorical(phi_true[z]).sample()
        word_idx.append(d)
        tokens.append(int(wt))
word_idx = torch.tensor(word_idx, dtype=torch.long)
w        = torch.tensor(tokens,   dtype=torch.long)

N = w.shape[0]
observations = {"w": w, "word_idx": word_idx}
x_in = torch.zeros(N, 1)
```

The per-document topic mixture `theta` and per-topic vocabulary distribution `phi` remain unobserved continuous simplex sites; the per-word topic `z` is integrated out by the marginalize block.

### SVI fit

```python
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"w", "word_idx"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(100)]
print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical Perspective

The discrete per-word topic $z : \mathsf{Topic}$ is integrated out by the [pushforward](https://en.wikipedia.org/wiki/Pushforward_measure) along the projection $\Phi \times \mathsf{Topic} \to \Phi$. The grouped marginalize block scatter-sums per-topic per-row log-likelihoods into the `Doc`-indexed accumulator before the log-sum-exp reduction, so the per-document topic mixture is the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) along the per-word document fibration $\mathsf{Word} \to \mathsf{Doc}$ in $\mathbf{Kern}$.

## See Also

- [Bayesian Gaussian Mixture Model](mixture-model.md) for a simpler grouped `marginalize` over a discrete latent.


## References

- David M. Blei, Andrew Y. Ng, and Michael I. Jordan. 2003. Latent Dirichlet allocation. *Journal of Machine Learning Research*, 3:993–1022.
