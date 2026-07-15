# Transformer Language Model

## Overview

A multi-layer Bayesian transformer used as a causal language model. The architecture follows the canonical encoder block of the [Transformer](https://doi.org/10.48550/arXiv.1706.03762): a stack of independent layers, each with multi-head self-attention via [`fan`](../guides/dsl-declarations.md#fan-out-diagonal-morphism), an attention output projection, a two-stage feed-forward sub-block, and two small residual Bayesian morphisms. The final `lm_head` is a `Categorical` morphism over the `Token` vocabulary, so the program's `observe` step scores the next-token target under a [Categorical likelihood](https://en.wikipedia.org/wiki/Categorical_distribution).

## QVR Source

```qvr
object Token : FinSet 32
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

    observe next_token : Token <- lm_head(h)
    return next_token

export transformer_lm
```

## Walkthrough

### Multi-head attention

`morphism head : Latent -> HeadOut [replicate=4, param_source=mlp] ~ Normal` declares four independent attention heads via the [replicate](../guides/dsl-declarations.md#replicated-declarations) attribute on a single morphism. Each head is a Bayesian Kleisli morphism `Latent -> HeadOut`; `HeadOut` is four-dimensional, so the four heads together cover the sixteen-dimensional `Latent`. [`fan(head)`](../guides/dsl-declarations.md#fan-out-diagonal-morphism) runs the four heads in parallel on the same input and concatenates the outputs, the standard multi-head wiring.

### Layer block

<!-- compile: false -->
```qvr
define layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
```

After the multi-head attention, `attn_proj` mixes the head outputs back into `Latent`, `residual_attn` is a small-scale Bayesian shortcut that plays the role of the standard residual `+` (the prior centered near identity), and the `ff_up >> ff_down` pair is the standard two-layer position-wise feed-forward block.

### Deep stack

[`stack(layer, 2)`](../guides/dsl-declarations.md#stack-independent-multi-layer) creates two independent deep copies of `layer`, each with its own parameters (unlike [`repeat`](../guides/dsl-declarations.md#repeat-iterated-composition), which weight-ties the iterations). The full backbone is `tok_embed >> stack(layer, 2)`, mapping the input token sequence to a per-position `Latent` representation.

### Language-model head

The closing `morphism lm_head : Latent -> Token ~ Categorical` is a Kleisli morphism `Latent -> Token`; per position it produces a Categorical distribution over the thirty-two-symbol vocabulary, and the program's `observe next_token` step accumulates the per-position categorical log-likelihood against the supplied target tensor.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

Initialise the transformer's stochastic-weight parameters under a fixed seed (these stand in for the ground-truth generative weights), then forward-sample a corpus of single-token contexts and read off the next-token target through [`rsample`](../api/continuous/programs.md). The transformer's stacked attention block expects sequence-axis-1 inputs, so the corpus is a `(batch, 1)` int64 context tensor paired with a `(batch, 1)` next-token target.

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)
prog = load("docs/examples/source/transformer_lm.qvr")
model = prog.morphism

for _, p in model.named_parameters():
    p.data.copy_(torch.randn_like(p) * 0.3)

batch, seq_len, vocab = 2, 1, 32
contexts = torch.randint(0, vocab, (batch, seq_len))
targets = model.rsample(contexts)
print("contexts:", contexts.shape, contexts.dtype)
print("targets: ", targets.shape, targets.dtype)
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
batch, seq_len, vocab = 2, 1, 32
contexts = torch.randint(0, vocab, (batch, seq_len))
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
batch, seq_len, vocab = 2, 1, 32
contexts = torch.randint(0, vocab, (batch, seq_len))
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


## Categorical Perspective

The model denotes a Kleisli morphism $\mathrm{Token} \to \mathcal{G}(\mathrm{Token})$ in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category, assembled by composition of replicated heads, an output projection, residual mixers, and a two-stage feed-forward block. [`stack`](../guides/dsl-declarations.md#stack-independent-multi-layer) is independent multi-layer deep composition; [`fan`](../guides/dsl-declarations.md#fan-out-diagonal-morphism) is the diagonal followed by parallel composition, the categorical realization of multi-head attention. The Categorical head accumulates per-position log-likelihood as a sub-probability kernel.


## References

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. arXiv preprint arXiv:1706.03762.
- Michèle Giry. 1982. A categorical approach to probability theory. In Bernhard Banaschewski, editor, *Categorical Aspects of Topology and Analysis*, volume 915 of *Lecture Notes in Mathematics*, pages 68–85. Springer, Berlin, Heidelberg.
