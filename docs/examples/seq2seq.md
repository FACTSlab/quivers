# Sequence-to-Sequence (Encoder + Decoder)

## Overview

A single transformer-style encoder-decoder ([Sutskever, Vinyals, and Le 2014](https://doi.org/10.48550/arXiv.1409.3215); [Vaswani et al. 2017](https://doi.org/10.48550/arXiv.1706.03762)) combining both halves in one example. The encoder is a stacked self-attention + feed-forward backbone on the source vocabulary; the decoder is a parallel stacked backbone on the target vocabulary; a `cross` Bayesian morphism merges the two latent streams into a single `Combined` representation, and a Categorical `lm_head` scores the next target token. Composing the encoder and decoder via [`@`](../guides/dsl.md) and following with `cross >> lm_head` gives a Kleisli morphism $\mathrm{Source} \times \mathrm{Target} \to \mathcal{G}(\mathrm{Target})$.

## QVR Source

```qvr
object Source : 256
object Target : 256

type Latent = Euclidean 64
type HeadOut = Euclidean 16
type FFHidden = Euclidean 128
type Combined = Euclidean 128

embed src_embed : Source -> Latent
embed tgt_embed : Target -> Latent

kernel enc_head[4] : Latent -> HeadOut ~ Normal [scale=0.1]
kernel enc_attn_proj : Latent -> Latent ~ Normal [scale=0.1]
kernel enc_residual_attn : Latent -> Latent ~ Normal [scale=0.01]
kernel enc_ff_up : Latent -> FFHidden ~ Normal
kernel enc_ff_down : FFHidden -> Latent ~ Normal [scale=0.1]
kernel enc_residual_ff : Latent -> Latent ~ Normal [scale=0.01]

kernel dec_head[4] : Latent -> HeadOut ~ Normal [scale=0.1]
kernel dec_attn_proj : Latent -> Latent ~ Normal [scale=0.1]
kernel dec_residual_attn : Latent -> Latent ~ Normal [scale=0.01]
kernel dec_ff_up : Latent -> FFHidden ~ Normal
kernel dec_ff_down : FFHidden -> Latent ~ Normal [scale=0.1]
kernel dec_residual_ff : Latent -> Latent ~ Normal [scale=0.01]

kernel cross : Combined -> Combined ~ Normal [scale=0.1]
kernel lm_head : Combined -> Target ~ Categorical

let enc_block = fan(enc_head) >> enc_attn_proj >> enc_residual_attn >> enc_ff_up >> enc_ff_down >> enc_residual_ff
let dec_block = fan(dec_head) >> dec_attn_proj >> dec_residual_attn >> dec_ff_up >> dec_ff_down >> dec_residual_ff

let encoder = src_embed >> stack(enc_block, 4)
let decoder = tgt_embed >> stack(dec_block, 4)
let backbone = (encoder @ decoder) >> cross

program seq2seq : Source * Target -> Target
    h <- backbone
    observe next_token : Target <- lm_head(h)
    return next_token

export seq2seq
```

## Walkthrough

### Encoder

`src_embed >> stack(enc_block, 4)` is the non-autoregressive encoder: source tokens are embedded into the 64-dimensional `Latent` space and run through four independent stacked self-attention + feed-forward blocks. Each block uses four-head fan via `fan(enc_head)`, an `enc_attn_proj` recombination, two small residual Bayesian morphisms, and a two-stage feed-forward sub-block. [`stack`](../guides/dsl.md#stack-independent-multi-layer) gives each layer its own parameters.

### Decoder

`tgt_embed >> stack(dec_block, 4)` mirrors the encoder structure on the target side with its own independent parameters. In a strict causal decoder the runtime supplies a causal mask to the per-step self-attention; in this categorical surface the mask is a runtime concern, not a structural one.

### Cross-composition

`(encoder @ decoder) >> cross` runs the encoder and decoder in parallel via the [tensor product](https://ncatlab.org/nlab/show/tensor+product) `@` of Kleisli morphisms and then merges the two 64-dimensional latent streams into a 128-dimensional `Combined` representation through the `cross` Bayesian morphism. `cross` plays the role of [cross-attention](https://doi.org/10.48550/arXiv.1706.03762) between source and target.

### Language-model head

The closing `kernel lm_head : Combined -> Target ~ Categorical` maps the combined representation onto a Categorical distribution over the target vocabulary; the program's `observe next_token` step accumulates the per-position categorical log-likelihood against the supplied target tensor.

```mermaid
flowchart LR
    "src" --> "src_embed"
    "tgt" --> "tgt_embed"
    "src_embed" --> "encoder"
    "tgt_embed" --> "decoder"
    "encoder" --> "cross"
    "decoder" --> "cross"
    "cross" --> "lm_head"
    "lm_head" --> "next_token"
```

## Try it

The program's domain is the product object `Source * Target`, so the runtime input is a `(batch, 2)` tensor whose two columns are the source and target token indices. The encoder reads the source column, the decoder reads the target column, and the model returns one predicted next-target-token per batch element. Treating each `(source, target)` row as a single Bayesian-LM example, a sequence is processed by flattening the position axis into the batch dimension.

```python
import torch
from quivers.dsl import load

prog = load("docs/examples/source/seq2seq.qvr")
model = prog.morphism

torch.manual_seed(0)
batch, seq_len = 16, 8
source = torch.randint(0, 256, (batch, seq_len))
target = torch.randint(0, 256, (batch, seq_len))
joint  = torch.stack([source.flatten(), target.flatten()], dim=-1)  # (128, 2)

next_token = model.rsample(joint)
print(next_token.shape)                        # torch.Size([128])

samples = torch.stack([model.rsample(joint) for _ in range(32)])
print(samples.shape)                           # torch.Size([32, 128])
```

## Categorical Perspective

The seq2seq model denotes a Kleisli morphism $\mathrm{Source} \times \mathrm{Target} \to \mathcal{G}(\mathrm{Target})$ in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category. The encoder and decoder are independent Kleisli morphisms over distinct objects; the [tensor product](https://ncatlab.org/nlab/show/tensor+product) `@` is their strong-monoidal product, and `cross` is the merge that closes the bilinear pairing into a single combined latent. The Categorical head puts a finite-set codomain on the composite, and `observe` is the [right Kan extension](https://ncatlab.org/nlab/show/Kan+extension) closing the LM likelihood.
