# Variational Autoencoder

## Overview

A [variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) ([Kingma & Welling 2014](https://arxiv.org/abs/1312.6114)) learns latent representations by training an encoder, which maps observations to a distribution over latent codes, and a decoder, which maps latent codes back to observations, jointly under the [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) objective. The quivers idiom expresses both networks as [Kleisli](https://en.wikipedia.org/wiki/Kleisli_category) morphisms for the [Giry monad](https://doi.org/10.1007/BFb0092872) and wires them with explicit `>>` composition into two execution paths: a generative path (prior to decoder) and a reconstruction path (encoder to decoder).

## QVR Source

```qvr
object Pixel : 784

type Latent = Euclidean 16
type EncoderHidden = Euclidean 256
type DecoderHidden = Euclidean 256
type ObsSpace = Euclidean 784
type UnitSpace = Euclidean 1

embed pixel_embed : Pixel -> EncoderHidden
kernel enc_deep : EncoderHidden -> EncoderHidden ~ Normal
kernel enc_to_latent : EncoderHidden -> Latent ~ Normal [scale=0.5]

let encoder = pixel_embed >> stack(enc_deep, 3) >> enc_to_latent

kernel prior : UnitSpace -> Latent ~ Normal

kernel dec_1 : Latent -> DecoderHidden ~ Normal
kernel dec_deep : DecoderHidden -> DecoderHidden ~ Normal
kernel dec_to_obs : DecoderHidden -> ObsSpace ~ Normal [scale=0.1]

let decoder = dec_1 >> stack(dec_deep, 2) >> dec_to_obs

let generative = prior >> decoder
let reconstruct = encoder >> decoder

export generative
```

## Walkthrough

The encoder begins with `embed pixel_embed : Pixel -> EncoderHidden`, a deterministic lookup mapping the discrete `Pixel` object into the continuous `EncoderHidden` space. The `stack(enc_deep, 3)` combinator creates three independently-parameterized stochastic Normal layers, distinct from `repeat(enc_deep, 3)` which would weight-tie. The final `enc_to_latent` projects to the 16-dimensional latent space at small init scale.

The decoder mirrors the encoder: an initial `dec_1` lifts the latent code into the decoder hidden width, two stacked deep layers `stack(dec_deep, 2)` add depth, and `dec_to_obs` projects to the 784-dimensional observation space at tight init scale (the reconstruction should be more precise than the encoding).

The two top-level compositions

<!-- compile: false -->
```qvr
let generative = prior >> decoder
let reconstruct = encoder >> decoder
```

express the VAE's two execution paths as explicit [Kleisli composition](https://en.wikipedia.org/wiki/Kleisli_category). The `generative` path samples a latent from the standard-normal prior and decodes it, used for sampling new data. The `reconstruct` path encodes observed data and decodes the resulting latent code, the path traversed by the [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) reconstruction term during training. Both paths share the decoder; the relationship between generation and inference is a matter of which morphism precedes the decoder in the composition chain.

## Try it

```python
import torch
from quivers.dsl import load

torch.manual_seed(0)

prog = load("docs/examples/source/vae.qvr")
generative = prog.morphism

B = 4
unit = torch.zeros(B, 1)
sample = generative.rsample(unit)
print("sample shape:", sample.shape)
```

The exported `generative` composition is a [Kleisli](https://en.wikipedia.org/wiki/Kleisli_category) morphism `UnitSpace -> ObsSpace`; `rsample` runs the full prior-then-decoder ancestral path. Replacing `generative` with the named composition `reconstruct` from the source threads observed pixels through the encoder and back through the decoder for ELBO-style reconstruction training.

## Categorical Perspective

The encoder and decoder are both [Kleisli](https://en.wikipedia.org/wiki/Kleisli_category) morphisms for the [Giry monad](https://doi.org/10.1007/BFb0092872); their two compositions `prior >> decoder` and `encoder >> decoder` correspond to the generative and reconstruction paths. They share the decoder but differ in which morphism produces the latent code. The `embed` operation acts as a functor from the category of discrete objects to the category of [Euclidean spaces](https://en.wikipedia.org/wiki/Euclidean_space), letting the encoder accept a discrete input and feed it into continuous stochastic layers. The `stack(f, N)` combinator is iterated independent composition: $f_1 \circ f_2 \circ \cdots \circ f_N$ with $N$ fresh copies of $f$ (no weight sharing), distinct from `repeat(f, N) = f^N`.

The [ELBO](https://en.wikipedia.org/wiki/Evidence_lower_bound) decomposes categorically into a reconstruction term, the faithfulness of `encoder >> decoder`, and a KL term, the distance from the prior in the enriched hom-space $\mathbf{Kern}(\mathsf{Pixel}, \mathsf{Latent})$.

## See Also

- [Probabilistic PCA](ppca.md) for a linear-Gaussian latent-variable model.
- [DSL Guide](../guides/dsl-overview.md) for the morphism composition surface (`>>`, `stack`, `embed`).
