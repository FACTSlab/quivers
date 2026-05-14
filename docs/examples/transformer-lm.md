# Transformer Language Model

## Overview

A multi-layer Bayesian transformer used as a causal language model. The architecture follows the canonical encoder block of the [Transformer](https://doi.org/10.48550/arXiv.1706.03762): four stacked layers, each with four-head self-attention via [`fan`](../guides/dsl.md#fan-out-diagonal-morphism), an attention output projection, a two-stage feed-forward sub-block, and two small residual Bayesian morphisms. The final `lm_head` is a `Categorical` morphism over the `Token` vocabulary, so the program's `observe` step scores the next-token target under a [Categorical likelihood](https://en.wikipedia.org/wiki/Categorical_distribution).

## QVR Source

```qvr
object Token : 256

type Latent = Euclidean 64
type HeadOut = Euclidean 16
type FFHidden = Euclidean 128

embed tok_embed : Token -> Latent

kernel head[4] : Latent -> HeadOut ~ Normal [scale=0.1]
kernel attn_proj : Latent -> Latent ~ Normal [scale=0.1]
kernel ff_up : Latent -> FFHidden ~ Normal
kernel ff_down : FFHidden -> Latent ~ Normal [scale=0.1]
kernel residual_attn : Latent -> Latent ~ Normal [scale=0.01]
kernel residual_ff : Latent -> Latent ~ Normal [scale=0.01]
kernel lm_head : Latent -> Token ~ Categorical

let layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
let backbone = tok_embed >> stack(layer, 4)

program transformer_lm : Token -> Token
    h <- backbone
    observe next_token : Token <- lm_head(h)
    return next_token

export transformer_lm
```

## Walkthrough

### Multi-head attention

`kernel head[4] : Latent -> HeadOut ~ Normal [scale=0.1]` declares four independent attention heads via the [replicate](../guides/dsl.md#replicated-declarations) form `head[N]`. Each head is a Bayesian Kleisli morphism `Latent -> HeadOut` (HeadOut is 16-dimensional, so the four heads together cover the 64-dimensional `Latent`). [`fan(head)`](../guides/dsl.md#fan-out-diagonal-morphism) runs the four heads in parallel on the same input and concatenates the outputs, the standard multi-head wiring.

### Layer block

<!-- compile: false -->
```qvr
let layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
```

After the multi-head attention, `attn_proj` mixes the head outputs back into `Latent`, `residual_attn` is a small-scale Bayesian shortcut that plays the role of the standard residual `+` (the prior centered near identity), and the `ff_up >> ff_down` pair is the standard 2-layer position-wise feed-forward block.

### Deep stack

[`stack(layer, 4)`](../guides/dsl.md#stack-independent-multi-layer) creates four independent deep copies of `layer`, each with its own parameters (unlike [`repeat`](../guides/dsl.md#repeat-iterated-composition), which weight-ties the iterations). The full backbone is `tok_embed >> stack(layer, 4)`, mapping the input token sequence to a per-position `Latent` representation.

### Language-model head

The closing `kernel lm_head : Latent -> Token ~ Categorical` is a Kleisli morphism `Latent -> Token`; per position it produces a Categorical distribution over the 256-symbol vocabulary, and the program's `observe next_token` step accumulates the per-position categorical log-likelihood against the supplied target tensor.

## Try it

```python
import torch
from quivers.dsl import load

prog = load("docs/examples/source/transformer_lm.qvr")
model = prog.morphism

torch.manual_seed(0)
inputs = torch.randint(0, 256, (16, 8))

next_tokens = model.rsample(inputs)
print(next_tokens.shape)                  # torch.Size([16, 8])

samples = torch.stack([model.rsample(inputs) for _ in range(32)])
modes = samples.mode(dim=0).values
print(modes.shape)                        # torch.Size([16, 8])
```

## Categorical Perspective

The model denotes a Kleisli morphism $\mathrm{Token} \to \mathcal{G}(\mathrm{Token})$ in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category, assembled by composition of replicated heads, an output projection, residual mixers, and a two-stage feed-forward block. [`stack`](../guides/dsl.md#stack-independent-multi-layer) is independent multi-layer deep composition; [`fan`](../guides/dsl.md#fan-out-diagonal-morphism) is the diagonal followed by parallel composition, the categorical realization of multi-head attention. The Categorical head accumulates per-position log-likelihood as a sub-probability kernel.
