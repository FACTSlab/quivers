# Vanilla RNN Language Model

## Overview

The simplest recurrent language model in the gallery: a single Bayesian [Kleisli morphism](https://ncatlab.org/nlab/show/Kleisli+category) [`cell`](../api/continuous/morphisms.md) `: Embedded * Hidden -> Hidden` updates the hidden state from the current input and the previous state, and a `Categorical` [`lm_head`](../api/continuous/families.md) projects the per-position hidden state onto the vocabulary so the program can `observe` the next-token target. The model exercises the [`scan`](../guides/dsl.md#scan-temporal-recurrence) combinator for threading state across a sequence and the minimal end-to-end LM wiring in the DSL.

## QVR Source

```qvr
object Token : 256

type Embedded = Euclidean 64
type Hidden = Euclidean 128

embed tok_embed : Token -> Embedded

kernel cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
kernel lm_head : Hidden -> Token ~ Categorical

let backbone = tok_embed >> scan(cell)

program vanilla_rnn_lm : Token -> Token
    h <- backbone
    observe next_token : Token <- lm_head(h)
    return next_token

export vanilla_rnn_lm
```

## Walkthrough

Tokens are embedded into the 64-dimensional `Embedded` space, then `scan(cell)` threads a 128-dimensional hidden state across the sequence: at each step the cell consumes the concatenated `(x_t, h_{t-1})` and emits `h_t`. The terminal hidden state $h_T$ summarizes the whole prefix; the `Categorical` [`lm_head`](../api/continuous/families.md) maps it to a Categorical distribution over the 256-symbol vocabulary, and the program's `observe next_token` step conditions on the next-token target tensor.

```mermaid
flowchart LR
    "tok" --> "embed"
    "embed" --> "scan(cell)"
    "scan(cell)" --> "h_T"
    "h_T" --> "lm_head"
    "lm_head" --> "next_token"
```

## Try it

The recurrent backbone forward-samples through [`MonadicProgram.rsample`](../api/continuous/programs.md). Each call draws fresh weights from the priors and returns the predicted next token at the end of the input window.

```python
import torch
from quivers.dsl import load

prog = load("docs/examples/source/vanilla_rnn_lm.qvr")
model = prog.morphism

torch.manual_seed(0)
inputs = torch.randint(0, 256, (32, 16))   # batch=32, seq_len=16

next_tokens = model.rsample(inputs)
print(next_tokens.shape)                   # torch.Size([32])

# Posterior-predictive ensemble: average over weight samples
samples = torch.stack([model.rsample(inputs) for _ in range(64)])
modes = samples.mode(dim=0).values
print(modes.shape)                         # torch.Size([32])
```

## Categorical Perspective

The model is a Kleisli morphism $\mathrm{Token} \to \mathcal{G}(\mathrm{Token})$ in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category. [`scan(cell)`](../guides/dsl.md#scan-temporal-recurrence) is the recursive [fold](https://ncatlab.org/nlab/show/fold) along the sequence in the Kleisli category: each step composes the previous step's output kernel with the new cell. The closing Categorical head observes the next-token label as a sub-probability kernel in $\mathcal{G}_{\le 1}$.
