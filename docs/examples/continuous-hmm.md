# Continuous State-Space Model

## Overview

A continuous state-space model extends the HMM to continuous latent states and observations, with a state transition function and an observation function, both stochastic. This example demonstrates the `scan` combinator for threading state through a sequence, the `observe` statement for Bayesian filtering, and the separation of generative and inference programs over the same morphisms.

## QVR Source

```qvr
object State : Real 16
object Obs : Real 8

morphism transition : State -> State [scale=0.1] ~ Normal
morphism emission : State -> Obs [scale=0.1] ~ Normal

program generative_step : State -> State
    sample s_new <- transition

    observe o <- emission(s_new)
    return s_new

morphism inference_cell : Obs * State -> State [scale=0.1] ~ Normal

define filter = scan(inference_cell)

morphism decoder : State -> Obs [scale=0.1] ~ Normal

define filter_and_reconstruct = scan(inference_cell) >> decoder

export generative_step
export filter_and_reconstruct
```

## Walkthrough

`object State : Real 16` and `object Obs : Real 8` introduce the two Euclidean spaces: a 16-dimensional latent and an 8-dimensional observation.

`morphism transition : State -> State [scale=0.1] ~ Normal` evolves the latent state by one time step under a Normal kernel whose mean is a learned linear function of the previous state and whose prior scale is 0.1. `morphism emission : State -> Obs [scale=0.1] ~ Normal` projects a state to an observation under the same kernel family.

`program generative_step : State -> State` is a one-step monadic program: `sample s_new <- transition` draws the new latent state from the transition kernel, `observe o <- emission(s_new)` scores an observation against the emission kernel, and `return s_new` projects the program's joint kernel onto the new state. To unroll over time, this single-step program is composed with itself via `repeat` or threaded through `scan`.

`morphism inference_cell : Obs * State -> State [scale=0.1] ~ Normal` is a recurrent cell that incorporates a new observation into the running state estimate. `define filter = scan(inference_cell)` constructs a temporal-recurrence morphism that threads state across a sequence of observations.

`morphism decoder : State -> Obs [scale=0.1] ~ Normal` decodes a state back to observation space; `define filter_and_reconstruct = scan(inference_cell) >> decoder` composes the scan with the decoder so the exported pipeline filters and reconstructs in one composite.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

Pick ground-truth linear dynamics for the latent state and a random emission matrix, then forward-sample a trajectory of latent states and observations of length `T`. The single-step program `generative_step : State -> State` reads the previous state as input; the per-step pair `(s_new, o)` is supplied as the observation dict.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/continuous_hmm.qvr")
model = prog.morphism

T = 32
state_dim = 16
obs_dim = 8
A = 0.9 * torch.eye(state_dim)
C = 0.3 * torch.randn(obs_dim, state_dim)
s = torch.zeros(T + 1, state_dim)
o = torch.zeros(T, obs_dim)
for t in range(T):
    s[t + 1] = s[t] @ A.T + 0.1 * torch.randn(state_dim)
    o[t] = s[t + 1] @ C.T + 0.1 * torch.randn(obs_dim)
state_prev = s[:T]
sites = {"s_new": s[1:T + 1], "o": o}
```

### SVI fit

The exported program is a single-step [`MonadicProgram`](../api/continuous/programs.md#quivers.continuous.programs.MonadicProgram) with no explicit priors on the kernel parameter networks; [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) lifts each leaf parameter into a unit-Normal sample site so [`AutoNormalGuide`](../api/inference/guide.md#quivers.inference.guides.AutoNormalGuide) can build a mean-field surrogate. With both `s_new` and `o` observed, the ELBO is the parameter-marginal log-likelihood of the full trajectory.

```python
from quivers.inference import bayesian_lift_parameters
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(1)
prog = load("docs/examples/source/continuous_hmm.qvr")
inner = prog.morphism
model, x_lift, obs_lift = bayesian_lift_parameters(
    inner, state_prev, sites, prior_scale=1.0,
)

guide = AutoNormalGuide(model, observed_names={"s_new", "o"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optim, ELBO())

loss0 = svi.step(x_lift, obs_lift)
losses = [svi.step(x_lift, obs_lift) for _ in range(300)]
loss_final = sum(losses[-20:]) / 20.0
oracle_ll = inner.log_joint(state_prev, sites).sum().item()
print(f"initial ELBO loss: {loss0:.1f}")
print(f"final ELBO loss:   {loss_final:.1f}")
print(f"oracle -log p:     {-oracle_ll:.1f}")
```

### NUTS posterior

The lifted model exposes one Normal sample site per leaf parameter; [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) samples them directly under small warmup and sample budgets.

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
prog = load("docs/examples/source/continuous_hmm.qvr")
model, x_lift, obs_lift = bayesian_lift_parameters(
    prog.morphism, state_prev, sites, prior_scale=1.0,
)

kernel = NUTSKernel(step_size=0.05, max_tree_depth=3, target_accept=0.8)
mc = MCMC(kernel, num_warmup=15, num_samples=15, num_chains=1)
result = mc.run(model, x_lift, obs_lift)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
```

## Categorical Perspective

The `scan` combinator implements [Kleisli composition](https://en.wikipedia.org/wiki/Kleisli_category) threaded through time. Given a step morphism $f : S \to S$ in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) (where $S$ carries both state and noise), `scan` produces the $n$-fold composition $f^n$ while collecting all intermediate results. Because Kleisli composition is associative, the computation decomposes into local single-step updates, which is why online/streaming inference works: each filtering step depends only on the previous belief and the current observation, not on the full history.

The generative program `generative_step` and the filtering pipeline `filter_and_reconstruct` are built from the same underlying kernels but run in opposite directions: the generative path threads the `transition` and `emission` morphisms forward to produce states and observations, while the filtering path threads `inference_cell` over the observed sequence and uses `observe` to invert the observation morphism. The inversion is Bayes' rule expressed as conditioning in the Kleisli category, and the `scan` combinator threads it through the full sequence.
