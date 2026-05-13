# Sequence Decoder (Language-Model Head)

## Overview

A transformer-style stacked decoder that consumes a token sequence and projects the final latent representation onto a 256-dimensional logits space. Each block carries multi-head self-attention through `fan(head)`, a two-stage feed-forward, and residual continuous morphisms; the closing `lm_head` `continuous` morphism produces per-position logits over the `Token` vocabulary suitable for next-token prediction.

## QVR Source

```qvr
object Token : 256

type Latent = Euclidean 64
type HeadOut = Euclidean 16
type FFHidden = Euclidean 128
type Logits = Euclidean 256

embed tok_embed : Token -> Latent

continuous head[4] : Latent -> HeadOut ~ Normal [scale=0.1]
continuous attn_proj : Latent -> Latent ~ Normal [scale=0.1]
continuous residual_attn : Latent -> Latent ~ Normal [scale=0.01]
continuous ff_up : Latent -> FFHidden ~ Normal
continuous ff_down : FFHidden -> Latent ~ Normal [scale=0.1]
continuous residual_ff : Latent -> Latent ~ Normal [scale=0.01]

continuous lm_head : Latent -> Logits ~ Normal [scale=0.1]

let block = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
let decoder = tok_embed >> stack(block, 4) >> lm_head

export decoder
```

## Walkthrough

The decoder is structurally identical to the [encoder](encoder.md) up to the projection head: token embedding into a 64-dimensional `Latent`, four `stack`ed self-attention + feed-forward blocks, and a final `lm_head` projecting onto a 256-dimensional `Logits` space (one logit per vocabulary token). Conditioning the logits on observed next-token targets (via a downstream `Categorical` observation) reduces inference to standard next-token-prediction language modeling.

## Language model

Wrap the decoder in a `program` that observes the next-token target via a `Categorical` over the logits:

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

prog = load("docs/examples/source/decoder.qvr")

batch = torch.randint(0, 256, (32, 64))   # (batch, seq_len)
targets = torch.randint(0, 256, (32, 64)) # next-token labels

logits = prog(batch)                       # (32, 64, 256)
log_probs = logits.log_softmax(dim=-1)
nll = -log_probs.gather(-1, targets.unsqueeze(-1)).mean()
```

For full Bayesian training, pair `decoder` with an `AutoNormalGuide` and an `ELBO` objective; conditioning on `targets` yields the per-position negative-log-likelihood loss familiar from standard language modeling.

## Categorical Perspective

The decoder denotes a Kleisli morphism $\mathrm{Token} \to \mathcal{G}(\mathrm{Logits})$ followed by a softmax change-of-base into the [Markov quantale](../semantics/quantales.md), so the composite arrow lands in a categorical distribution over the vocabulary. Composing the encoder and decoder (via tensor product on a shared latent space) gives an encoder/decoder architecture; conditioning on observed targets uses the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) along the natural projection that the [`observe`](../guides/dsl.md) construct realises.
