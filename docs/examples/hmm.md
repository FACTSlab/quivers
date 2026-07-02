# Hidden Markov Model (Discrete)

## Overview

A discrete-state, discrete-emission [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) expressed as a V-enriched categorical network over finite sets. The initial-state, transition, and emission morphisms are row-stochastic matrices in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://doi.org/10.1007/BFb0092872); the row-wise [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior is set via the morphism-prior surface, and the runtime-variable `repeat` combinator threads `n_steps` transition applications before the final emission. Composition with the chosen [algebra](../api/core/algebras.md) determines whether the same morphism computes the forward marginal (product algebra) or the Viterbi path (tropical algebra).

## QVR Source

```qvr
composition product_fuzzy [level=algebra]

object State : FinSet 8
object Obs : FinSet 16

morphism initial, transition : State -> State [role=latent]
morphism emission : State -> Obs [role=latent]

define n_step = repeat(transition) >> emission
define hmm = initial >> n_step

export hmm
```

## Walkthrough

`composition product_fuzzy as algebra` selects the standard multiplicative composition of probabilities along paths; switching to the [tropical (max-plus) algebra](https://en.wikipedia.org/wiki/Tropical_semiring) reinterprets the same composed morphism as the Viterbi recurrence.

`object State : FinSet 8` and `object Obs : FinSet 16` are finite discrete spaces. The three `morphism` declarations `initial : State -> State`, `transition : State -> State`, and `emission : State -> Obs` introduce row-stochastic matrices as `[role=latent]` arrows. The natural row-wise prior is a [symmetric Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution#Symmetric_case) on each row, attached via the axis-role surface `~ Dirichlet(1.0) [over=cod, iid_over=dom]`: the event axis sits on the codomain (each row is one simplex), and the domain axis is asserted as iid (independent rows). The axis count must match the family's event rank; [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) has event rank one, so a single `over` axis is required.

`let n_step = repeat(transition) >> emission` is the runtime-variable Kleisli composition `T^n >> E`; `repeat` builds an n-step matrix by repeated squaring for $O(\log n)$ matrix multiplications, with $n$ supplied via `prog(n_steps=N)`. `let hmm = initial >> n_step` prepends the initial-state distribution so the exported pipeline is `1 -> Obs`, mapping no input to an n-step marginal over the observation alphabet.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

Initialise the HMM's `initial`, `transition` and `emission` row-stochastic-algebra logits under a fixed seed (these stand in for the ground-truth generative weights), then forward-sample an observation histogram from the n-step composed pipeline `prog(n_steps=K)`. The returned `(State, Obs)` tensor is the fuzzy-algebra n-step joint; normalising it gives a categorical we draw counts from.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/hmm.qvr")

# Ground-truth row-stochastic logits.
for _, p in prog.named_parameters():
    p.data.copy_(torch.randn_like(p))

n_steps = 5
true_marginal = prog(n_steps=n_steps).detach()
probs = (true_marginal / true_marginal.sum()).flatten()
N = 500
obs_counts = torch.distributions.Multinomial(
    total_count=N, probs=probs,
).sample().reshape(true_marginal.shape)
print("counts shape:", obs_counts.shape, "sum:", int(obs_counts.sum()))
```

### SVI fit

Re-initialise the row-stochastic logits and recover the n-step joint that produced the synthetic histogram by minimising the cross-entropy between the normalised `prog(n_steps=n_steps)` and the observed counts. The optimiser walks the algebra's row-stochastic submodule directly.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/hmm.qvr")
for _, p in prog.named_parameters():
    p.data.copy_(torch.randn_like(p))
n_steps = 5
true_marginal = prog(n_steps=n_steps).detach()
probs = (true_marginal / true_marginal.sum()).flatten()
N = 500
obs_counts = torch.distributions.Multinomial(
    total_count=N, probs=probs,
).sample().reshape(true_marginal.shape)

# Fresh logits for fitting.
torch.manual_seed(1)
for _, p in prog.named_parameters():
    p.data.copy_(torch.randn_like(p))

optim = torch.optim.Adam(list(prog.parameters()), lr=5e-2)
losses = []
for _ in range(150):
    optim.zero_grad()
    pred = prog(n_steps=n_steps)
    pred_norm = pred / (pred.sum() + 1e-12)
    loss = -(obs_counts * (pred_norm + 1e-12).log()).sum()
    loss.backward()
    optim.step()
    losses.append(float(loss.detach()))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

The HMM has no explicit `sample` priors; its `initial`, `transition` and `emission` latents are row-stochastic-algebra parameters. [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) lifts each `nn.Parameter` into a Normal-prior sample site so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) has a continuous unconstrained state space. The likelihood scores the observed histogram against the normalised n-step joint emitted by `prog(n_steps=K)`.

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel, lift_from_log_prob

torch.manual_seed(0)
prog = load("docs/examples/source/hmm.qvr")
for _, p in prog.named_parameters():
    p.data.copy_(torch.randn_like(p))
n_steps = 5
true_marginal = prog(n_steps=n_steps).detach()
probs = (true_marginal / true_marginal.sum()).flatten()
N = 500
obs_counts = torch.distributions.Multinomial(
    total_count=N, probs=probs,
).sample().reshape(true_marginal.shape)

def hmm_log_prob(x, counts):
    pred = prog(n_steps=n_steps)
    pred_norm = pred / (pred.sum() + 1e-12)
    return (counts * (pred_norm + 1e-12).log()).sum().expand(x.shape[0])

lifted, lx, lobs = lift_from_log_prob(
    prog,
    log_prob_fn=hmm_log_prob,
    parameter_prior_scale=1.0,
    target_key="obs_counts",
    x=torch.zeros(1, 1),
    observations={"obs_counts": obs_counts},
)
kernel = NUTSKernel(step_size=0.01, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(lifted, lx, lobs)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical Perspective

The forward algorithm and the Viterbi algorithm are the same composed morphism evaluated in different [algebras](https://ncatlab.org/nlab/show/algebra): under product, composition multiplies probabilities and summation marginalizes; under tropical, composition adds log-probabilities and summation maximizes. Quivers makes this explicit: switching `algebra` changes the V-enriched composition rule without touching the program text.

The row-wise Dirichlet prior is the standard conjugate prior for a categorical kernel; declaring it via `over cod iid over dom` resolves the axis-role ambiguity that distinguishes a flat Dirichlet on $|State|\cdot|State|$ entries (wrong: not row-stochastic) from $|State|$ independent simplex draws (right). The categorical reading: each row of $T$ is a fiber of the dependent kernel $\prod_{c \,:\, \mathrm{State}} \mathcal{G}(\mathrm{State})$, so the prior factors as a product of independent simplex priors.


## References

- Michèle Giry. 1982. A categorical approach to probability theory. In Bernhard Banaschewski, editor, *Categorical Aspects of Topology and Analysis*, volume 915 of *Lecture Notes in Mathematics*, pages 68–85. Springer, Berlin, Heidelberg.
