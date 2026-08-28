# Transformer-shaped language model

## Overview

This model borrows the parallel-head and stacked-block shape of a [Transformer](https://doi.org/10.48550/arXiv.1706.03762), then scores a token with a Categorical head. The `head` morphisms are independent MLP-parameterized Normal kernels. The source contains no query-key dot products, softmax over positions, causal mask, layer normalization, or additive residual connection. It is thus a wiring demonstration rather than a Transformer implementation or causal language model.

## QVR source

```qvr
# Multi-Layer Bayesian Transformer Language Model
#
# A multi-layer Bayesian transformer used as a causal language
# model. Token indices are embedded into a Latent
# representation, passed through two independent attention plus
# feed-forward layers via stack(layer, 2), and projected back
# onto the Token vocabulary via a Categorical lm_head.
#
# Generative structure:
#
#   h_attn   ~ fan(head)(h) >> attn_proj           four-head attention
#   h_res    ~ residual_attn(h_attn)               attention residual
#   h_ff     ~ ff_up(h_res) >> ff_down             feed-forward block
#   h        ~ residual_ff(h_ff)                   feed-forward residual
#   next_t   ~ Categorical(lm_head(h))             next-token target
#
# stack(layer, 2) composes two independent copies of the
# attention plus feed-forward block, each carrying its own
# Normal-prior weights drawn from the morphism declarations.
#
# Resp is the plate: it indexes the 32 scored rows, one
# next-token target per context. Token is the vocabulary, so it
# is the value space of what lm_head draws and of what the
# program returns.
#
# Reference: [Vaswani et al. 2017](https://doi.org/10.48550/arXiv.1706.03762).

object Token : FinSet 32
object Resp : FinSet 32
object Latent : Real 16
object HeadOut : Real 4
object FFHidden : Real 32

morphism tok_embed : Token -> Latent [role=embed]
morphism head : Latent -> HeadOut [replicate=4, param_source=mlp] ~ Normal
morphism attn_proj : Latent -> Latent [param_source=mlp] ~ Normal
morphism ff_up : Latent -> FFHidden [param_source=mlp] ~ Normal
morphism ff_down : FFHidden -> Latent [param_source=mlp] ~ Normal
morphism residual_attn, residual_ff : Latent -> Latent [param_source=mlp] ~ Normal
morphism lm_head : Latent -> Token ~ Categorical

define layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
define backbone = tok_embed >> stack(layer, 2)

program transformer_lm : Token -> Token
    sample h <- backbone

    observe next_token : Resp <- lm_head(h)
    return next_token

export transformer_lm
```

## Walkthrough

### Parallel heads

`morphism head : Latent -> HeadOut [replicate=4, param_source=mlp] ~ Normal` declares four independent kernels. [`fan(head)`](../guides/dsl-declarations.md#fan-out-diagonal-morphism) runs them in parallel on the same input and concatenates their four-dimensional outputs. This matches the dimensional wiring of four heads but does not compute attention.

### Layer block

<!-- compile: false -->
```qvr
define layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
```

`attn_proj` mixes the parallel outputs back into `Latent`. `residual_attn` and `residual_ff` are sequential stochastic morphisms, not additions of a saved input, despite their names. `ff_up >> ff_down` is a two-stage MLP-shaped kernel composition.

### Deep stack

[`stack(layer, 2)`](../guides/dsl-declarations.md#stack-independent-multi-layer) creates two independent deep copies of `layer`, each with its own parameters (unlike [`repeat`](../guides/dsl-declarations.md#repeat-iterated-composition), which weight-ties the iterations). The full backbone is `tok_embed >> stack(layer, 2)`, mapping the input token sequence to a per-position `Latent` representation.

### Language-model head

The closing `morphism lm_head : Latent -> Token ~ Categorical` is a Kleisli morphism `Latent -> Token`; per position it produces a Categorical distribution over the thirty-two-symbol vocabulary, and the program's `observe next_token` step accumulates the per-position categorical log-likelihood against the supplied target tensor.

The two `FinSet` objects sit in different positions and mean different things. `Resp : FinSet 32` fills the observe step's index slot, so it is the plate: 32 scored rows. `Token : FinSet 32` fills `lm_head`'s codomain and the program's own codomain, so it is the value space: the 32 outcomes a draw ranges over, and the space the returned `next_token` lives in. That the two happen to have the same cardinality here is a coincidence of this example's sizing, not a shared role.

## Try it

> The short fits below demonstrate the API. Assess convergence with multiple chains and diagnostics before interpreting a posterior.


### Generating synthetic data

Fix the model's parameters to a chosen draw, then run one forward [`trace`](../api/inference/trace.md) so the latent representation `h` and the next-token targets generated from it are jointly consistent. `true_h` names the ground truth for the latent `h` site, and shipping it in the observations dict is what clamps it: an unclamped `h` is redrawn on every call, which leaves any reference joint non-deterministic. The corpus is 32 single-token contexts, one per element of the `Resp` plate, paired with 32 next-token targets. A one-position context does not test cross-position attention or causal masking: `fan(head)` inside `stack` folds a multi-position axis into the feature axis, so the composite as written scores one position per row.

```python
import torch
from quivers.dsl import load
from quivers.inference.trace import trace

torch.manual_seed(0)
prog = load("docs/examples/source/transformer_lm.qvr")
model = prog.morphism

for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)

rows, vocab = 32, 32
x_in = torch.randint(0, vocab, (rows,))
with torch.no_grad():
    forward = trace(model, x_in)
true_h = forward.sites["h"].value.detach()
next_token = forward.sites["next_token"].value.detach()

observations = {"next_token": next_token, "h": true_h}
print("x_in:", tuple(x_in.shape))
print("true_h:", tuple(true_h.shape))
print("next_token:", tuple(next_token.shape))
```

### SVI fit

Re-initialise the parameters and recover the next-token weights from the synthetic corpus with [`AutoNormalGuide`](../api/inference/guide.md) + [`ELBO`](../api/inference/elbo.md) + [`SVI`](../api/inference/svi.md). The transformer's per-particle Monte-Carlo log-density makes each step relatively expensive; a short run is enough to verify that the negative ELBO falls.

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

torch.manual_seed(0)
prog = load("docs/examples/source/transformer_lm.qvr")
model = prog.morphism

for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)
rows, vocab = 32, 32
contexts = torch.randint(0, vocab, (rows,))
targets = model.rsample(contexts)
observations = {"next_token": targets}

torch.manual_seed(1)
for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)

guide = AutoNormalGuide(model, observed_names={"next_token"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=5e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(contexts, observations)]
for _ in range(8):
    losses.append(svi.step(contexts, observations))

print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### HMC posterior

The proper Bayesian model has both the parameters $\theta$ and the per-position latent $h$ as random variables: $p(\theta, h \mid x, y) \propto p(\theta) \, p(h \mid x, \theta) \, p(y \mid h, \theta)$. [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) lifts both: Normal priors on every `nn.Parameter`, plus the intermediate `h` site exposed through `additional_latents`. The lifted log-density is deterministic given the full $(\theta, h)$ state. The transformer's full `log_joint` walks every step in the stack, so NUTS's tree expansion is prohibitively expensive at this lifted dimension; we use [`HMCKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.HMCKernel) with a single leapfrog step to keep the run tractable while preserving the same target distribution.

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, HMCKernel, bayesian_lift_parameters

torch.manual_seed(0)
prog = load("docs/examples/source/transformer_lm.qvr")
model = prog.morphism
for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)
rows, vocab = 32, 32
contexts = torch.randint(0, vocab, (rows,))
targets = model.rsample(contexts)
observations = {"next_token": targets}

h_shape = tuple(model._step_h.rsample(contexts).shape)
lifted, lx, lobs = bayesian_lift_parameters(
    model, contexts, observations,
    prior_scale=1.0,
    additional_latents={"h": h_shape},
)
# The full transformer log_joint is expensive; use fixed-step HMC
# with one leapfrog step per sample to keep the run tractable.
# NUTS with the same target produces the same chain mathematically
# at much higher cost.
kernel = HMCKernel(step_size=0.001, num_steps=1, target_accept=0.6)
mc     = MCMC(kernel, num_warmup=3, num_samples=3, num_chains=1)
result = mc.run(lifted, lx, lobs)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical perspective

The model composes replicated stochastic branches, a projection, and two-stage feed-forward kernels. [`stack`](../guides/dsl-declarations.md#stack-independent-multi-layer) makes independently parameterized layers, while [`fan`](../guides/dsl-declarations.md#fan-out-diagonal-morphism) copies the input across parallel branches. Neither combinator by itself implements self-attention.


## References

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. arXiv preprint arXiv:1706.03762.
- Michèle Giry. 1982. A categorical approach to probability theory. In Bernhard Banaschewski, editor, *Categorical Aspects of Topology and Analysis*, volume 915 of *Lecture Notes in Mathematics*, pages 68–85. Springer, Berlin, Heidelberg.
