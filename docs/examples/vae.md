# Variational Autoencoder

## Overview

A [variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) ([Kingma & Welling 2014](https://arxiv.org/abs/1312.6114)) learns latent representations by training an encoder, which maps observations to a distribution over latent codes, and a decoder, which maps latent codes back to observations, jointly under the [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) objective. The quivers idiom expresses both networks as [Kleisli](https://en.wikipedia.org/wiki/Kleisli_category) morphisms for the [Giry monad](https://doi.org/10.1007/BFb0092872) and wires them with explicit `>>` composition into two execution paths: a generative path (prior to decoder) and a reconstruction path (encoder to decoder).

## QVR Source

```qvr
object Pixel : FinSet 8
object Latent : Real 4
object EncoderHidden, DecoderHidden : Real 16
object ObsSpace : Real 8
object UnitSpace : Real 1

morphism pixel_embed : Pixel -> EncoderHidden [role=embed]
morphism enc_deep : EncoderHidden -> EncoderHidden [param_source=mlp] ~ Normal
morphism enc_to_latent : EncoderHidden -> Latent ~ Normal

define encoder = pixel_embed >> stack(enc_deep, 1) >> enc_to_latent

morphism prior : UnitSpace -> Latent ~ Normal

morphism dec_1 : Latent -> DecoderHidden ~ Normal
morphism dec_deep : DecoderHidden -> DecoderHidden [param_source=mlp] ~ Normal
morphism dec_to_obs : DecoderHidden -> ObsSpace ~ Normal

define decoder = dec_1 >> stack(dec_deep, 1) >> dec_to_obs

define generative = prior >> decoder
define reconstruct = encoder >> decoder

program vae_program : UnitSpace -> ObsSpace
    sample z <- prior
    observe Y <- decoder(z)
    return Y

export vae_program
```

## Walkthrough

The encoder begins with `morphism pixel_embed : Pixel -> EncoderHidden [role=embed]`, a deterministic embedding lookup mapping the discrete `Pixel` object into the continuous `EncoderHidden` space. The `stack(enc_deep, 1)` combinator inserts one independently-parameterized stochastic Normal hidden layer, distinct from `repeat(enc_deep, 1)` which would weight-tie. The final `enc_to_latent` projects to the latent space at small init scale.

The decoder mirrors the encoder: an initial `dec_1` lifts the latent code into the decoder hidden width, one stacked deep layer `stack(dec_deep, 1)` adds depth, and `dec_to_obs` projects to the observation space at tight init scale (the reconstruction should be more precise than the encoding).

The two top-level compositions

<!-- compile: false -->
```qvr
define generative = prior >> decoder
define reconstruct = encoder >> decoder
```

express the VAE's two execution paths as explicit [Kleisli composition](https://en.wikipedia.org/wiki/Kleisli_category). The `generative` path samples a latent from the standard-normal prior and decodes it, used for sampling new data. The `reconstruct` path encodes observed data and decodes the resulting latent code, the path traversed by the [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) reconstruction term during training. Both paths share the decoder; the relationship between generation and inference is a matter of which morphism precedes the decoder in the composition chain.

## Try it

> The SVI step counts and NUTS warmup, sample, and chain budgets in the snippets below are illustrative: each block is sized to run in tens of seconds and demonstrate the API surface. Production fits typically need 10x to 100x more SVI steps, longer NUTS warmup, and multiple chains to actually converge to the data-generating parameters.


### Generating synthetic data

```python
import torch
from quivers.dsl import load
from quivers.inference.trace import trace

torch.manual_seed(0)
prog = load("docs/examples/source/vae.qvr")
model = prog.morphism

N = 32
unit = torch.zeros(N, 1)
x_in = unit

# Run one forward trace under the program's own parameters so the
# captured latent and observation are jointly consistent under the
# generative kernel.
with torch.no_grad():
    forward = trace(model, unit)
true_z = forward.sites["z"].value.detach()
Y = forward.sites["Y"].value.detach()

observations = {"Y": Y, "z": true_z}
print("Y shape:", Y.shape)
```

The exported `generative` composition is a [Kleisli](https://en.wikipedia.org/wiki/Kleisli_category) morphism `UnitSpace -> ObsSpace`; `rsample` runs the full prior-then-decoder ancestral path so the synthetic batch comes from the model itself at its current (random) parameter values, then lift the entire parameter vector into a Bayesian model for SVI and NUTS.

### SVI fit

```python
from quivers.inference import (
    AutoNormalGuide, ELBO, SVI, lift_from_log_prob,
)

model, x_in, observations = lift_from_log_prob(
    prog,
    log_prob_fn=prog.morphism.log_prob,
    parameter_prior_scale=1.0,
    target_key="Y",
    x=unit,
    observations={"Y": Y},
)

torch.manual_seed(1)
guide = AutoNormalGuide(model, observed_names={"Y"})
optim = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optim, ELBO(num_particles=1))

losses = [svi.step(x_in, observations) for _ in range(50)]
print(f"initial loss: {losses[0]:.2f}")
print(f"final loss:   {losses[-1]:.2f}")
```

### NUTS posterior

```python
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(2)
# The lifted parameter vector is high-dimensional, so a small
# step size and shallow tree keep one full chain inside a
# documentation-friendly budget.
kernel = NUTSKernel(step_size=0.005, max_tree_depth=3, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=5, num_samples=5, num_chains=1)
result = mc.run(model, x_in, observations)

print(f"acceptance:  {float(result.acceptance_rates.mean()):.2f}")
print(f"divergences: {int(result.divergence_counts.sum())}")
```


## Categorical Perspective

The encoder and decoder are both [Kleisli](https://en.wikipedia.org/wiki/Kleisli_category) morphisms for the [Giry monad](https://doi.org/10.1007/BFb0092872); their two compositions `prior >> decoder` and `encoder >> decoder` correspond to the generative and reconstruction paths. They share the decoder but differ in which morphism produces the latent code. The `embed` operation acts as a functor from the category of discrete objects to the category of [Euclidean spaces](https://en.wikipedia.org/wiki/Euclidean_space), letting the encoder accept a discrete input and feed it into continuous stochastic layers. The `stack(f, N)` combinator is iterated independent composition: $f_1 \circ f_2 \circ \cdots \circ f_N$ with $N$ fresh copies of $f$ (no weight sharing), distinct from `repeat(f, N) = f^N`.

The [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) decomposes categorically into a reconstruction term, the faithfulness of `encoder >> decoder`, and a KL term, the distance from the prior in the enriched hom-space $\mathbf{Kern}(\mathsf{Pixel}, \mathsf{Latent})$.

## See Also

- [Probabilistic PCA](ppca.md) for a linear-Gaussian latent-variable model.
- [DSL Guide](../guides/dsl-overview.md) for the morphism composition surface (`>>`, `stack`, `embed`).


## References

- Diederik P. Kingma and Max Welling. 2013. Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.
