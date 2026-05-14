# Bidirectional RNN Masked Language Model

## Overview

A bidirectional RNN used as a masked language model in the spirit of [BERT](https://doi.org/10.18653/v1/N19-1423). Two independently-parameterized recurrent cells scan the token sequence forward and backward; the [tensor product](https://ncatlab.org/nlab/show/tensor+product) `@` runs the two directional Kleisli morphisms in parallel and a `combine` morphism merges their outputs into a single 128-dimensional `Combined` representation. The Categorical `lm_head` scores the masked-token target from the bidirectional context.

## QVR Source

```qvr
object Token : 256

type Embedded = Euclidean 64
type FwdHidden = Euclidean 64
type BwdHidden = Euclidean 64
type Combined = Euclidean 128

embed tok_embed : Token -> Embedded

kernel fwd_cell : Embedded * FwdHidden -> FwdHidden ~ Normal [scale=0.1]
kernel bwd_cell : Embedded * BwdHidden -> BwdHidden ~ Normal [scale=0.1]
kernel combine : Combined -> Combined ~ Normal [scale=0.1]
kernel lm_head : Combined -> Token ~ Categorical

let forward_path = tok_embed >> scan(fwd_cell)
let backward_path = tok_embed >> scan(bwd_cell)
let backbone = (forward_path @ backward_path) >> combine

program bidirectional_rnn_lm : Token -> Token
    h <- backbone
    observe masked_token : Token <- lm_head(h)
    return masked_token

export bidirectional_rnn_lm
```

## Walkthrough

### Two independent scans

`forward_path = tok_embed >> scan(fwd_cell)` and `backward_path = tok_embed >> scan(bwd_cell)` are two independent Kleisli morphisms `Token -> Hidden`. They use distinct cells with independent parameters; the runtime supplies the reversed sequence to the backward path so the same `scan` machinery realizes the right-to-left pass.

### Parallel composition

`(forward_path @ backward_path) >> combine` runs the two directional paths in parallel via the [tensor product](https://ncatlab.org/nlab/show/tensor+product) `@` of Kleisli morphisms in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category. The result lives in `FwdHidden * BwdHidden`, which by the type aliases above has total dimension 128, matching `Combined`. The `combine` Bayesian morphism is the merge that mixes the two streams into a single combined representation.

### Masked LM head

The Categorical `lm_head : Combined -> Token` scores a masked-token target conditional on bidirectional context: at any position the prediction is conditioned on both the left and the right context, so this is an encoder rather than a causal LM.

```mermaid
flowchart LR
    "tok" --> "embed"
    "embed" --> "fwd"
    "embed" --> "bwd"
    "fwd" --> "combine"
    "bwd" --> "combine"
    "combine" --> "lm_head"
    "lm_head" --> "masked_token"
```

## Try it

```python
import torch
from quivers.dsl import load

prog = load("docs/examples/source/bidirectional_rnn_lm.qvr")
model = prog.morphism

torch.manual_seed(0)
inputs = torch.randint(0, 256, (32, 16))

masked_predictions = model.rsample(inputs)
print(masked_predictions.shape)              # torch.Size([32])

samples = torch.stack([model.rsample(inputs) for _ in range(64)])
modes = samples.mode(dim=0).values
print(modes.shape)                           # torch.Size([32])
```

## Categorical Perspective

The model denotes a Kleisli morphism $\mathrm{Token} \to \mathcal{G}(\mathrm{Token})$ assembled by `@`-product of two independent scan-folds and a merge. The tensor product `@` is the strong-monoidal product of the Kleisli category; `combine` is the merge $\mathrm{Hidden}^2 \to \mathcal{G}(\mathrm{Combined})$ that pulls the bilinear pairing back onto a single object. The Categorical head closes with the masked-token likelihood as a sub-probability kernel.
