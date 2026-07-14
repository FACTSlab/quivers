# LSTM Language Model

## Overview

A Bayesian [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735) language model. The recurrent cell is a parametric [`program`](../guides/dsl-programs-and-lets.md#program-declarations) that draws the four standard gates (`i`, `f`, `o`, `g`) from [`LogitNormal`](../api/continuous/families.md) and [`Normal`](../api/continuous/families.md) priors, computes the canonical cell update `c_t = f_t * h_{t-1} + i_t * g_t`, and emits the per-step hidden output `h_t = o_t * tanh(c_t)`. The per-position hidden output is projected onto the vocabulary by a Categorical `lm_head` so the program's `observe` step scores the next-token target end to end.

## QVR Source

```qvr
object Token : FinSet 256
object Embedded : Real 64
object Hidden : Real 128

morphism tok_embed : Token -> Embedded [role=embed]
morphism gate_i, gate_f, gate_o : Embedded * Hidden -> Hidden ~ LogitNormal
morphism cell_cand : Embedded * Hidden -> Hidden [scale=0.5] ~ Normal
morphism lm_head : Hidden -> Token ~ Categorical

program lstm_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
    sample i_gate <- gate_i(x_t, h_prev)
    sample f_gate <- gate_f(x_t, h_prev)
    sample o_gate <- gate_o(x_t, h_prev)
    sample g_cand <- cell_cand(x_t, h_prev)

    let c_new = f_gate * h_prev + i_gate * g_cand
    let two_c = 2.0 * c_new
    let sig_2c = sigmoid(two_c)
    let tanh_c = 2.0 * sig_2c - 1.0
    let h_new = o_gate * tanh_c
    return h_new

define backbone = tok_embed >> scan(lstm_cell)

program lstm_lm : Token -> Token
    sample h <- backbone

    observe next_token : Token <- lm_head(h)
    return next_token

export lstm_lm
```

## Walkthrough

### Cell equations

The parametric program `lstm_cell` realizes the canonical LSTM update. Each gate is a Bayesian Kleisli morphism `Embedded * Hidden -> Hidden`; `LogitNormal` constrains the gate activations to $(0, 1)$ in expectation. The cell candidate is a `Normal` Kleisli morphism with `scale = 0.5`. Inside the program body:

| Step | DSL | Meaning |
|---|---|---|
| input gate | `i_gate <- gate_i(x_t, h_prev)` | $i_t = \sigma(W_i [x_t, h_{t-1}])$ |
| forget gate | `f_gate <- gate_f(x_t, h_prev)` | $f_t = \sigma(W_f [x_t, h_{t-1}])$ |
| output gate | `o_gate <- gate_o(x_t, h_prev)` | $o_t = \sigma(W_o [x_t, h_{t-1}])$ |
| candidate | `g_cand <- cell_cand(x_t, h_prev)` | $g_t = \phi(W_g [x_t, h_{t-1}])$ |
| cell update | `let c_new = f_gate * h_prev + i_gate * g_cand` | $c_t = f_t \odot h_{t-1} + i_t \odot g_t$ |
| hidden | `let h_new = o_gate * tanh_c` | $h_t = o_t \odot \tanh(c_t)$ |

`tanh` is realized from [`sigmoid`](../guides/dsl-programs-and-lets.md#indexed-gather-in-let) via the identity $\tanh(x) = 2\,\sigma(2x) - 1$.

### State threading

`scan(lstm_cell)` is an iterated Kleisli composition along the sequence: the threaded state is the per-step hidden output $h_t$, which the Categorical `lm_head` reads at the terminal position to score the next token. The cell-state vector $c_t$ is computed inside the cell at every step from the threaded $h_{t-1}$ and used immediately to form $h_t$ via the output-gate / $\tanh$ post-composition. Because `scan` threads a single codomain, this presentation folds the canonical LSTM's separate $c$-channel into the local cell body: the long-term cell-state memory channel that a two-state LSTM exposes is not propagated across time steps here, and the recurrence reduces to $h_t = o_t \odot \tanh(f_t \odot h_{t-1} + i_t \odot g_t)$.

```mermaid
flowchart LR
    x_t["x_t"] --> gate_i["gate_i"]
    x_t["x_t"] --> gate_f["gate_f"]
    x_t["x_t"] --> gate_o["gate_o"]
    x_t["x_t"] --> cell_cand["cell_cand"]
    h_prev["h_prev"] --> gate_i["gate_i"]
    h_prev["h_prev"] --> gate_f["gate_f"]
    h_prev["h_prev"] --> gate_o["gate_o"]
    h_prev["h_prev"] --> cell_cand["cell_cand"]
    gate_f["gate_f"] --> c_new["c_new"]
    cell_cand["cell_cand"] --> c_new["c_new"]
    gate_i["gate_i"] --> c_new["c_new"]
    h_prev["h_prev"] --> c_new["c_new"]
    gate_o["gate_o"] --> h_new["h_new"]
    c_new["c_new"] --> h_new["h_new"]
    h_new["h_new"] --> scan["scan"]
```

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

Initialise the model's stochastic-weight parameters under a fixed seed (these stand in for the ground-truth generative weights), then forward-sample a corpus of token sequences by drawing random prompts and reading off the next-token target through [`rsample`](../api/continuous/programs.md). The corpus is a `(batch, seq_len)` int64 prompt tensor paired with a `(batch,)` next-token target.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/lstm_lm.qvr")
model = prog.morphism

for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)

batch, seq_len, vocab = 4, 8, 256
prompts = torch.randint(0, vocab, (batch, seq_len))
targets = model.rsample(prompts)
print("prompts:", prompts.shape, prompts.dtype)
print("targets:", targets.shape, targets.dtype)
```

### SVI fit

Re-initialise the parameters and recover next-token weights from the synthetic corpus with [`AutoNormalGuide`](../api/inference/guide.md) + [`ELBO`](../api/inference/elbo.md) + [`SVI`](../api/inference/svi.md). The loss is the negative ELBO under a Categorical likelihood on the `next_token` site.

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/lstm_lm.qvr")
model = prog.morphism

for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)
batch, seq_len, vocab = 4, 8, 256
prompts = torch.randint(0, vocab, (batch, seq_len))
targets = model.rsample(prompts)
observations = {"next_token": targets}

torch.manual_seed(1)
for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)

guide = AutoNormalGuide(model, observed_names={"next_token"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(prompts, observations)]
for _ in range(30):
    losses.append(svi.step(prompts, observations))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

The LSTM's four gates and cell candidate are kernel Bayesian morphisms whose weights live as `nn.Parameter`s inside the program. [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) lifts those parameters into Normal-prior sample sites so [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) has a continuous unconstrained state space. The likelihood scores the next-token target via the Categorical [`lm_head`](../api/continuous/families.md) applied to a forward sample of the hidden state.

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel, bayesian_lift_parameters

torch.manual_seed(0)
prog = load("docs/examples/source/lstm_lm.qvr")
model = prog.morphism
for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)
batch, seq_len, vocab = 4, 8, 256
prompts = torch.randint(0, vocab, (batch, seq_len))
targets = model.rsample(prompts)
observations = {"next_token": targets}

h_shape = tuple(model._step_h.rsample(prompts).shape)
lifted, lx, lobs = bayesian_lift_parameters(
    model, prompts, observations,
    prior_scale=1.0,
    additional_latents={"h": h_shape},
)
kernel = NUTSKernel(step_size=0.005, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1)
result = mc.run(lifted, lx, lobs)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical Perspective

The cell program denotes a Kleisli morphism $\mathrm{Embedded} \times \mathrm{Hidden} \to \mathcal{G}(\mathrm{Hidden})$ in the Kleisli category of the [Giry monad](https://doi.org/10.1007/BFb0092872); `scan(lstm_cell)` is its iterated composition over the sequence. The Categorical head closes the composite with a finite-set codomain, and `observe next_token` accumulates per-batch categorical log-likelihood through a [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension).


## References

- Michèle Giry. 1982. A categorical approach to probability theory. In Bernhard Banaschewski, editor, *Categorical Aspects of Topology and Analysis*, volume 915 of *Lecture Notes in Mathematics*, pages 68–85. Springer, Berlin, Heidelberg.
- Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. *Neural Computation*, 9(8):1735–1780.
