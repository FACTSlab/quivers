# GRU Language Model

## Overview

A Bayesian [GRU](https://doi.org/10.3115/v1/D14-1179) language model. The recurrent cell follows the canonical GRU equations with [`LogitNormal`](../api/continuous/families.md) update and reset gates and a [`Normal`](../api/continuous/families.md) candidate, and a `Categorical` `lm_head` projects the per-position hidden state onto the vocabulary so the program's `observe` step scores the next-token target.

## QVR Source

```qvr
object Token : 256

type Embedded = Euclidean 64
type Hidden = Euclidean 128

embed tok_embed : Token -> Embedded

kernel gate_z : Embedded * Hidden -> Hidden ~ LogitNormal
kernel gate_r : Embedded * Hidden -> Hidden ~ LogitNormal
kernel lm_head : Hidden -> Token ~ Categorical

program gru_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
    z <- gate_z(x_t, h_prev)
    r <- gate_r(x_t, h_prev)

    let reset_hidden = r * h_prev

    h_cand <- Normal(reset_hidden, 0.5)

    let z_complement = 1.0 - z
    let h_new = z_complement * h_prev + z * h_cand

    return h_new

let backbone = tok_embed >> scan(gru_cell)

program gru_lm : Token -> Token
    h <- backbone
    observe next_token : Token <- lm_head(h)
    return next_token

export gru_lm
```

## Walkthrough

### Cell equations

| Step | DSL | Meaning |
|---|---|---|
| update gate | `z <- gate_z(x_t, h_prev)` | $z_t = \sigma(W_z [x_t, h_{t-1}])$ |
| reset gate | `r <- gate_r(x_t, h_prev)` | $r_t = \sigma(W_r [x_t, h_{t-1}])$ |
| reset-gated state | `let reset_hidden = r * h_prev` | $r_t \odot h_{t-1}$ |
| candidate | `h_cand <- Normal(reset_hidden, 0.5)` | $\tilde h_t = \phi(W \,[x_t, r_t \odot h_{t-1}])$ |
| update | `let h_new = z_complement * h_prev + z * h_cand` | $h_t = (1 - z_t)\,h_{t-1} + z_t \,\tilde h_t$ |

The candidate is drawn from a Normal centered on the reset-gated previous state; the update-gate convex combination $(1 - z_t)\,h_{t-1} + z_t \,\tilde h_t$ interpolates between persistence and the new candidate.

### State threading

`scan(gru_cell)` threads the hidden state $h_t$ across the sequence; the `Categorical` `lm_head` scores the next-token target from the terminal state $h_T$.

```mermaid
flowchart LR
    "x_t" --> "gate_z"
    "x_t" --> "gate_r"
    "h_prev" --> "gate_z"
    "h_prev" --> "gate_r"
    "gate_r" --> "reset_hidden"
    "h_prev" --> "reset_hidden"
    "reset_hidden" --> "h_cand"
    "x_t" --> "h_cand"
    "h_prev" --> "h_new"
    "gate_z" --> "h_new"
    "h_cand" --> "h_new"
    "h_new" --> "scan"
```

## Try it

```python
import torch
from quivers.dsl import load

prog = load("docs/examples/source/gru_lm.qvr")
model = prog.morphism

torch.manual_seed(0)
inputs = torch.randint(0, 256, (32, 16))

next_tokens = model.rsample(inputs)
print(next_tokens.shape)                  # torch.Size([32])

samples = torch.stack([model.rsample(inputs) for _ in range(64)])
modes = samples.mode(dim=0).values
print(modes.shape)                        # torch.Size([32])
```

## Categorical Perspective

The GRU cell is a Kleisli morphism $\mathrm{Embedded} \times \mathrm{Hidden} \to \mathcal{G}(\mathrm{Hidden})$ in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category; `scan(gru_cell)` is its iterated composition along the sequence. The Categorical head and `observe` step close the composite into the LM likelihood by accumulating per-batch categorical log-probabilities.
