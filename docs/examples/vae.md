# Deep Variational Autoencoder

## Overview

A Variational Autoencoder learns latent representations by training an encoder (data to latent code) and decoder (latent code to reconstruction) jointly. This example demonstrates embedding (discrete-to-continuous), the `stack` combinator for deep layers, distribution parameters (`scale`), and Kleisli composition (`>>`) to wire encoder and decoder into generative and reconstruction paths.

## QVR Source

```qvr
object Pixel : 784
type Latent = Euclidean 16
type EncoderHidden = Euclidean 256
type DecoderHidden = Euclidean 256
type ObsSpace = Euclidean 784
type UnitSpace = Euclidean 1

embed pixel_embed : Pixel -> EncoderHidden

continuous enc_deep : EncoderHidden -> EncoderHidden ~ Normal
continuous enc_to_latent : EncoderHidden -> Latent ~ Normal [scale=0.5]

let encoder = pixel_embed >> stack(enc_deep, 3) >> enc_to_latent

continuous prior : UnitSpace -> Latent ~ Normal
continuous dec_1 : Latent -> DecoderHidden ~ Normal
continuous dec_deep : DecoderHidden -> DecoderHidden ~ Normal
continuous dec_to_obs : DecoderHidden -> ObsSpace ~ Normal [scale=0.1]

let decoder = dec_1 >> stack(dec_deep, 2) >> dec_to_obs
let generative = prior >> decoder
let reconstruct = encoder >> decoder

output generative
```

## Walkthrough

The program declares a discrete `object Pixel : 784` (a flattened 28x28 image) and several continuous Euclidean types for the latent space (16-d), hidden layers (256-d), and observation space (784-d). `UnitSpace` (1-d) serves as a trivial input to the prior, a standard pattern for unconditional distributions.

`embed pixel_embed : Pixel -> EncoderHidden` maps discrete pixel indices into continuous vectors. The `embed` keyword is distinct from `continuous` because embeddings are deterministic lookups, not stochastic sampling operations. It bridges the discrete object space and the continuous Euclidean space so that downstream stochastic layers can operate on the result.

The encoder stacks three identical stochastic layers via `stack(enc_deep, 3)`, which composes `enc_deep` with itself three times. This avoids declaring separate variables for each layer. The `[scale=0.5]` parameter on `enc_to_latent` sets the standard deviation of the output distribution, controlling how much stochasticity the bottleneck layer introduces.

The decoder mirrors the encoder: `dec_1` expands from 16-d latent space to 256-d hidden, `stack(dec_deep, 2)` applies two hidden layers, and `dec_to_obs` projects back to 784-d observations. The decoder's `[scale=0.1]` gives a tighter output distribution than the encoder's `[scale=0.5]`, reflecting that reconstruction should be more precise than encoding.

Two compositions define the model's two execution paths. `let generative = prior >> decoder` samples a latent code from the prior and decodes it (used for generating new data). `let reconstruct = encoder >> decoder` encodes observed data and decodes it (used for training via the ELBO). The `>>` operator is Kleisli composition: the output distribution of the left morphism feeds into the right morphism.

`output generative` marks the generative path as the primary exposed model.

## DSL Features

- **`embed`**: Deterministic map from a discrete `object` space to a continuous `Euclidean` type. Required because `continuous` morphisms cannot directly consume discrete objects.
- **`continuous` keyword**: Marks stochastic morphisms as differentiable, so the runtime can apply reparameterization for gradient-based training.
- **Distribution parameters** (`~ Normal [scale=0.5]`): Inline control of distribution shape per morphism.
- **`stack(f, n)`**: Composes morphism `f` with itself `n` times. The repetition count is fixed at compile time (contrast with `repeat`, which is runtime-variable).
- **Kleisli composition (`>>`)**: Chains stochastic morphisms; the codomain of the left must match the domain of the right, checked statically.
- **`let` bindings**: Name intermediate compositions (`encoder`, `decoder`) for reuse. Both are used in defining `generative` and `reconstruct`.

## Python Usage

<!-- TODO: add working Python usage example -->

## Categorical Perspective

The encoder and decoder are both Kleisli morphisms in the Kleisli category over the Giry monad. Their two compositions correspond to the model's two execution paths: `prior >> decoder` is the generative path (sample latent, then decode), while `encoder >> decoder` is the reconstruction path (encode observed data, then decode). These share the decoder but differ in how the latent code is produced, making the relationship between generation and inference a matter of which morphism precedes the decoder in the composition chain.

The `embed` operation acts as a functor from the category of discrete objects to the category of Euclidean spaces, providing the bridge that lets the encoder accept discrete input and feed it into continuous stochastic layers. The `stack` combinator expresses iterated self-composition of an endomorphism ($f^n$), and composition with `>>` remains associative throughout, so the pipeline `pixel_embed >> stack(enc_deep, 3) >> enc_to_latent` is itself a well-typed Kleisli morphism that can be named and composed further.
