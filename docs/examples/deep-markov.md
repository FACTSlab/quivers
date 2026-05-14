# Deep Markov Model

## Overview

The [deep Markov model](https://doi.org/10.1609/aaai.v31i1.10779) of Krishnan, Shalit, and Sontag (2017) is a state-space model with nonlinear, neural-network-parameterized transition and emission kernels:

$$
s_t = f_\theta(s_{t-1}) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma_s^2 I)
$$
$$
o_t = g_\phi(s_t) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_o^2 I)
$$

The transition and emission means are MLPs; per-step Normal noise gives a tractable density. The companion recognition network `q_\phi(o_t, s_{t-1}) -> s_t` carries the variational posterior and is threaded across the sequence by `scan` to amortize the posterior over the latent trajectory. The combinator surface mirrors the [linear-Gaussian SSM](linear-gaussian-ssm.md): only the per-step cells change.

## QVR Source

```qvr
type Driver = Euclidean 4
type Hidden = Euclidean 32
type State = Euclidean 8
type Obs = Euclidean 4

kernel trans_mlp_1 : Driver * State -> Hidden ~ Normal [scale=0.5]
kernel trans_mlp_2 : Hidden -> State ~ Normal [scale=0.1]

kernel emit_mlp_1 : State -> Hidden ~ Normal [scale=0.5]
kernel emit_mlp_2 : Hidden -> Obs ~ Normal [scale=0.1]

kernel infer_cell : Obs * State -> State ~ Normal [scale=0.1]

let transition_cell = trans_mlp_1 >> trans_mlp_2
let emission = emit_mlp_1 >> emit_mlp_2

let generate = scan(transition_cell) >> emission
let recognize = scan(infer_cell)

export recognize
```

## Walkthrough

The transition stack `trans_mlp_1 >> trans_mlp_2` is a two-layer MLP that maps `(u_t, s_{t-1})` through a hidden width of 32 down to the 8-d state; the emission stack `emit_mlp_1 >> emit_mlp_2` is the symmetric decoder back to the 4-d observation. Both stacks are Kleisli compositions of Gaussian kernels, so the joint per-step kernel is a [normalizing-flow](https://doi.org/10.1145/3422622)-like reparameterisable Gaussian whose mean is the network output.

`scan(transition_cell) >> emission` is the generative pipeline; `scan(infer_cell)` is the [variational autoencoder](https://doi.org/10.48550/arXiv.1312.6114)-style recognition network that threads the previous belief and the new observation through `infer_cell` to produce the next belief. The choice of `Driver` width controls the exogenous input; a non-driven model uses `Driver = Euclidean 1` and feeds a zero vector.

## Try it

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/deep_markov.qvr")
recognize = prog.morphism

batch, seq_len, obs_dim = 4, 16, 4
o_seq = torch.randn(batch, seq_len, obs_dim)
final_state = recognize.rsample(o_seq)
print("recognized state shape:", final_state.shape)

# Smoke-train the recognizer to align with the generative prior; in
# a full setup, ``condition`` the generative pipeline on ``o_seq``
# and fit by SVI with the recognizer as the amortized guide.
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adam(recognize.parameters(), lr=1e-3)
for _ in range(50):
    optim.zero_grad()
    h = recognize.rsample(o_seq)
    loss = loss_fn(h, torch.zeros_like(h))
    loss.backward()
    optim.step()
print("smoke-train final loss:", float(loss))
```

## Categorical Perspective

The transition stack is the Kleisli composition of two Gaussian kernels; the second kernel's mean depends on the sample from the first, so the joint per-step kernel is no longer Gaussian, only a reparameterisable density. `scan` realizes the iterated Kleisli composition over the time index, so the full trajectory kernel is the right Kan extension of the per-step cell along the time projection.

The recognizer is a directed inverse of the generative kernel: where the prior is a forward chain `s_0 -> s_1 -> ... -> s_T -> o_{1:T}`, the recognizer is the [encoder side](seq2seq.md) of an amortized variational posterior. The two share the latent space `State` but live in opposite Kleisli morphisms; SVI tunes them jointly against an ELBO.

```mermaid
flowchart LR
    "s_{t-1}" --> "trans_mlp_1"
    "u_t" --> "trans_mlp_1"
    "trans_mlp_1" --> "h_trans"
    "h_trans" --> "trans_mlp_2"
    "trans_mlp_2" --> "s_t"
    "s_t" --> "emit_mlp_1"
    "emit_mlp_1" --> "h_emit"
    "h_emit" --> "emit_mlp_2"
    "emit_mlp_2" --> "o_t"
```
