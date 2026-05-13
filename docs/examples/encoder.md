# Sequence Encoder

## Overview

A transformer-style stacked encoder: token embedding followed by `N` independent self-attention plus feed-forward blocks composed with `stack`. Each block carries multi-head attention through `fan(head)` plus a two-stage feed-forward sub-block, with residual-style continuous morphisms wired in between. The encoder maps a token sequence to a sequence-aligned latent representation suitable for downstream tasks (classification, retrieval, decoder cross-attention).

## QVR Source

```qvr
object Token : 256

type Latent = Euclidean 64
type HeadOut = Euclidean 16
type FFHidden = Euclidean 128

embed tok_embed : Token -> Latent

continuous head[4] : Latent -> HeadOut ~ Normal [scale=0.1]
continuous attn_proj : Latent -> Latent ~ Normal [scale=0.1]
continuous residual_attn : Latent -> Latent ~ Normal [scale=0.01]
continuous ff_up : Latent -> FFHidden ~ Normal
continuous ff_down : FFHidden -> Latent ~ Normal [scale=0.1]
continuous residual_ff : Latent -> Latent ~ Normal [scale=0.01]

let block = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
let encoder = tok_embed >> stack(block, 4)

export encoder
```

## Walkthrough

The pipeline reads left to right. Discrete `Token` indices are lifted into the continuous `Latent` space by an embedding lookup. Each of the four stacked `block`s applies, in order: a `fan(head)` multi-head self-attention that splits the 64-dimensional representation across four 16-dimensional heads and concatenates the outputs, an `attn_proj` recombination through `Latent -> Latent`, a small `residual_attn` adjustment, a `ff_up` expansion to 128 dimensions, an `ff_down` projection back to 64, and a final `residual_ff` adjustment. `stack(block, 4)` instantiates four copies with independent parameters.

## Try it

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/encoder.qvr")
batch = torch.randint(0, 256, (8, 32))  # (batch, seq_len)
representation = prog(batch)
```

Pair the encoder with a classification head (a [`continuous`](../api/dsl/ast_nodes.md) morphism to a label space) and condition on observed labels, then fit with [`SVI`](../api/inference/svi.md) and an [`AutoNormalGuide`](../api/inference/guide.md). For unconditional pretraining over a corpus, pair it with the [decoder](decoder.md) example below.

## Categorical Perspective

The encoder is a Kleisli morphism $\mathrm{Token} \to \mathcal{G}(\mathrm{Latent})$ in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category. `stack(block, 4)` is the fourfold sequential composition of independent stochastic blocks; `fan(head)` is the categorical diagonal followed by tensor-product fan-out across the four head morphisms. Residual continuous morphisms are small-scale perturbations around the identity, which give a [residual network](https://doi.org/10.1109/CVPR.2016.90)-style inductive bias without leaving the Kleisli category.
