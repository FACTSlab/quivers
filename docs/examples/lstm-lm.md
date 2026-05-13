# LSTM Language Model

## Overview

A Bayesian [LSTM](https://doi.org/10.1162/neco.1997.9.8.1735) language model. The recurrent cell is a parametric [`program`](../guides/dsl.md#program) that draws the four standard gates (`i`, `f`, `o`, `g`) from [`LogitNormal`](../api/continuous/families.md) and [`Normal`](../api/continuous/families.md) priors, applies the canonical cell update `c_t = f_t * c_{t-1} + i_t * g_t`, and emits `h_t = o_t * tanh(c_t)`. The per-position cell state is projected onto the vocabulary by a Categorical `lm_head` so the program's `observe` step scores the next-token target end to end.

## QVR Source

```qvr
object Token : 256

type Embedded = Euclidean 64
type Hidden = Euclidean 128

embed tok_embed : Token -> Embedded

kernel gate_i : Embedded * Hidden -> Hidden ~ LogitNormal
kernel gate_f : Embedded * Hidden -> Hidden ~ LogitNormal
kernel gate_o : Embedded * Hidden -> Hidden ~ LogitNormal
kernel cell_cand : Embedded * Hidden -> Hidden ~ Normal [scale=0.5]
kernel lm_head : Hidden -> Token ~ Categorical

program lstm_cell(x_t, c_prev) : Embedded * Hidden -> Hidden
    i_gate <- gate_i(x_t, c_prev)
    f_gate <- gate_f(x_t, c_prev)
    o_gate <- gate_o(x_t, c_prev)
    g_cand <- cell_cand(x_t, c_prev)

    let c_new = f_gate * c_prev + i_gate * g_cand
    let two_c = 2.0 * c_new
    let sig_2c = sigmoid(two_c)
    let tanh_c = 2.0 * sig_2c - 1.0
    let h_new = o_gate * tanh_c

    return c_new

let backbone = tok_embed >> scan(lstm_cell)

program lstm_lm : Token -> Token
    h <- backbone
    observe next_token : Token <- lm_head(h)
    return next_token

export lstm_lm
```

## Walkthrough

### Cell equations

The parametric program `lstm_cell` realises the canonical LSTM update. Each gate is a Bayesian Kleisli morphism `Embedded * Hidden -> Hidden`; `LogitNormal` constrains the gate activations to $(0, 1)$ in expectation. The cell candidate is a `Normal` Kleisli morphism with `scale = 0.5`. Inside the program body:

| Step | DSL | Meaning |
|---|---|---|
| input gate | `i_gate <- gate_i(x_t, c_prev)` | $i_t = \sigma(W_i [x_t, c_{t-1}])$ |
| forget gate | `f_gate <- gate_f(x_t, c_prev)` | $f_t = \sigma(W_f [x_t, c_{t-1}])$ |
| output gate | `o_gate <- gate_o(x_t, c_prev)` | $o_t = \sigma(W_o [x_t, c_{t-1}])$ |
| candidate | `g_cand <- cell_cand(x_t, c_prev)` | $g_t = \phi(W_g [x_t, c_{t-1}])$ |
| cell update | `let c_new = f_gate * c_prev + i_gate * g_cand` | $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ |
| hidden | `let h_new = o_gate * tanh_c` | $h_t = o_t \odot \tanh(c_t)$ |

`tanh` is realised from [`sigmoid`](../guides/dsl.md#indexed-gather-in-let) via the identity $\tanh(x) = 2\,\sigma(2x) - 1$.

### State threading

`scan(lstm_cell)` is an iterated Kleisli composition along the sequence: the threaded state is the cell state $c_t$ (the LSTM's long-range memory channel). The hidden vector $h_t$ is computed inside the cell on every step but does not need to be threaded separately, since the Categorical `lm_head` is parameterised by the terminal $c_T$ and absorbs the output-gate / $\tanh$ post-composition into its own learnt linear map.

```mermaid
flowchart LR
    "x_t" --> "gate_i"
    "x_t" --> "gate_f"
    "x_t" --> "gate_o"
    "x_t" --> "cell_cand"
    "c_prev" --> "gate_i"
    "c_prev" --> "gate_f"
    "c_prev" --> "gate_o"
    "c_prev" --> "cell_cand"
    "gate_f" --> "c_new"
    "cell_cand" --> "c_new"
    "gate_i" --> "c_new"
    "c_prev" --> "c_new"
    "c_new" --> "scan"
    "gate_o" --> "h_new"
    "c_new" --> "h_new"
```

## Try it

```python
import torch
from quivers.dsl import load

prog = load("docs/examples/source/lstm_lm.qvr")
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

The cell program denotes a Kleisli morphism $\mathrm{Embedded} \times \mathrm{Hidden} \to \mathcal{G}(\mathrm{Hidden})$ in the Kleisli category of the [Giry monad](https://doi.org/10.1007/BFb0092872); `scan(lstm_cell)` is its iterated composition over the sequence. The Categorical head closes the composite with a finite-set codomain, and `observe next_token` accumulates per-batch categorical log-likelihood through a [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension).
