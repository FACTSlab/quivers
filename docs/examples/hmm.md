# Finite-state path composition

## Overview

The exported `hmm` is a finite-state path-composition example inspired by a discrete [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model). Its top-level morphisms live under `product_fuzzy`, have sigmoid-constrained entries, and have no `Dirichlet` declarations. They are thus not row-stochastic HMM kernels. The same source also contains `hmm_program`, which draws Dirichlet rows but currently leaves `transition_rows` unused in its marginalized observation body.

## QVR source

```qvr
# Discrete Hidden Markov Model
#
# A classic K-state HMM as a V-enriched categorical network over
# finite sets. The composed pipeline runs an initial-state
# distribution into a runtime-variable number of transition
# steps and a final emission row, so the same model computes
# n-step marginals for any horizon.
#
# Structural form:
#
#   initial    : State -> State          initial-state row
#   transition : State -> State          row-stochastic kernel
#   emission   : State -> Obs            row-stochastic emission
#   hmm        = initial >> repeat(transition) >> emission
#
# Each row of every row-stochastic matrix carries a Dirichlet
# prior via the axis-role surface (over cod iid over dom), the
# standard conjugate prior for a discrete Markov chain.

composition product_fuzzy [level=algebra]

object State : FinSet 8
object Obs : FinSet 16
object Step : FinSet 12
object StateDist : Real 8

morphism initial, transition : State -> State [role=latent]
morphism emission : State -> Obs [role=latent]

define n_step = repeat(transition) >> emission
define hmm = initial >> n_step

# Probabilistic surface for transpile: every row of every
# row-stochastic matrix carries a Dirichlet(1) prior (the
# uniform conjugate prior the header describes; equivalently the
# Bayesian-Laplace add-one prior of Manning and Schuetze 1999,
# section 6.2). The initial-state vector is one Dirichlet draw
# on the State simplex, so it carries no plate annotation; the
# transition kernel allocates one Dirichlet row per source state
# (iid over the State axis); the emission kernel allocates one
# Dirichlet row per latent state over the Obs simplex.
#
# Step is the plate the emitted sequence is observed over and
# Obs is the alphabet each emission takes its value in, so the
# two are separate objects. StateDist is the value space of the
# returned initial-state vector, a point of the State simplex
# embedded in R^8.
program hmm_program : Step -> StateDist
    sample initial_row <- Dirichlet(1.0) [over=State]
    sample transition_rows : State <- Dirichlet(1.0) [over=State, iid_over=State]
    sample emission_rows : State <- Dirichlet(1.0) [over=Obs, iid_over=State]

    marginalize state <- Categorical(initial_row) [reduction=logsumexp]
        observe obs : Step <- Categorical(emission_rows[state])

    return initial_row

export hmm
export hmm_program
```

## Walkthrough

`composition product_fuzzy [level=algebra]` selects product and noisy-OR aggregation for fuzzy relations. It is not the sum-product semiring used by the HMM forward algorithm. A Viterbi implementation would also require log weights and max aggregation; changing this declaration alone is not a validated conversion of the current parameters.

`object State : FinSet 8` and `object Obs : FinSet 16` are finite discrete spaces: the latent state set and the emission alphabet. `object Step : FinSet 12` is the separate plate the emitted sequence is observed over, and `object StateDist : Real 8` is the value space of the returned initial-state vector, a point of the State simplex embedded in $\mathbb{R}^8$. The three `[role=latent]` declarations introduce learned relation tensors. To build a normalized categorical kernel, add a row-wise [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution#Symmetric_case) family or change base through a row-normalizing transformation.

`repeat` builds the algebraic power of `transition` by repeated squaring, with $n$ supplied via `prog(n_steps=N)`. Under the current algebra, the result is a product-fuzzy relation. Also note that `initial : State -> State`, so the exported domain is `State`, not a singleton initial-distribution object.

`hmm_program` is the separate probabilistic surface exported from the same source. Its three `sample` steps draw the initial-state vector, the transition kernel, and the emission kernel from symmetric `Dirichlet(1.0)` priors: `[over=State]` names the simplex axis of the initial row, which carries no plate annotation because it is a single draw, and `iid_over=State` allocates one independent Dirichlet row per source state for `transition_rows` and per latent state for `emission_rows`. The `marginalize` block integrates the latent `state` out of the observation by log-sum-exp over the eight `State` atoms, so the observation body scores `obs` under `Categorical(emission_rows[state])` alone and `transition_rows` never enters the likelihood.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

Draw the three row-stochastic parameter blocks from their own Dirichlet priors, draw one latent state from the initial row, and emit a `Step`-long sequence from that state's emission row. `transition_rows` never enters the likelihood under this body, so it is drawn from the prior and clamped alongside the other two; the emitted sequence is a single-state categorical mixture rather than a full transition sequence.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/hmm.qvr")
model = prog.hmm_program

n_state, n_emit, n_step = 8, 16, 12

true_initial_row = torch.distributions.Dirichlet(torch.ones(n_state)).sample()
true_transition_rows = torch.distributions.Dirichlet(
    torch.ones(n_state),
).sample((n_state,))
true_emission_rows = torch.distributions.Dirichlet(
    torch.ones(n_emit),
).sample((n_state,))

state = torch.distributions.Categorical(true_initial_row).sample()
obs = torch.distributions.Categorical(true_emission_rows[state]).sample(
    (n_step,),
)

observations = {"obs": obs}
x_in = torch.zeros(n_step, 1)
```

### SVI fit

Re-initialise the raw relation logits and recover the n-step joint that produced the synthetic histogram by minimising the cross-entropy between the normalised `prog(n_steps=n_steps)` and the observed counts. The optimiser walks the sigmoid-constrained entries of the product-fuzzy algebra directly.

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

The exported `hmm` composition has no `sample` priors of its own; its `initial`, `transition`, and `emission` latents are plain algebra parameters with sigmoid-constrained entries. [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) lifts each `nn.Parameter` into a Normal-prior sample site so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) has a continuous unconstrained state space. The likelihood scores the observed histogram against the normalised n-step joint emitted by `prog(n_steps=K)`.

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


## Categorical perspective

HMM forward and Viterbi recurrences can be expressed as matrix products over sum-product and max-plus semirings. The current example instead demonstrates runtime-variable powering under Quivers' product-fuzzy rule. A normalized HMM needs different parameter constraints and aggregation.


## References

- Michèle Giry. 1982. A categorical approach to probability theory. In Bernhard Banaschewski, editor, *Categorical Aspects of Topology and Analysis*, volume 915 of *Lecture Notes in Mathematics*, pages 68–85. Springer, Berlin, Heidelberg.
