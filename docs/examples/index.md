# Examples Gallery

Complete `.qvr` programs spanning neural architectures, probabilistic models, and linguistic applications. Each example compiles to a trainable `nn.Module`. Click through for a walkthrough of the code and its categorical interpretation.

All source files are in `src/quivers/dsl/examples/`.

## Neural Architectures

### [Multi-Layer Bayesian Transformer](transformer.md)

A 4-layer Bayesian transformer with multi-head attention (4 independent heads via `replicate` + `fan`) and feed-forward blocks, using the `stack` combinator for deep composition. Each layer has independently-parameterized morphisms.

**Features:** `embed`, `continuous`, `stack`, `head[4]` replicate, `fan`, `>>`, `let`, `type`

```qvr
object Token : 256
type Latent = Euclidean 64
type HeadOut = Euclidean 16
type FFHidden = Euclidean 128

embed tok_embed : Token -> Latent

continuous head[4] : Latent -> HeadOut ~ Normal [scale=0.1]
continuous attn_proj : Latent -> Latent ~ Normal [scale=0.1]
continuous ff_up : Latent -> FFHidden ~ Normal
continuous ff_down : FFHidden -> Latent ~ Normal [scale=0.1]
continuous residual_attn : Latent -> Latent ~ Normal [scale=0.01]
continuous residual_ff : Latent -> Latent ~ Normal [scale=0.01]

let layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
let transformer = tok_embed >> stack(layer, 4)

output transformer
```

---

### [Bayesian Vanilla RNN](vanilla-rnn.md)

A recurrent network using `scan` to thread hidden state across the input sequence. The cell has a product domain to accept both the current input and previous hidden state.

**Features:** `embed`, `continuous`, `scan`, `>>` composition, `type`, product domain

```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 128
type Output = Euclidean 64

embed tok_embed : Token -> Embedded

continuous cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
continuous output_proj : Hidden -> Output ~ Normal [scale=0.1]

let rnn = tok_embed >> scan(cell) >> output_proj

output rnn
```

---

### [Bayesian LSTM](lstm.md)

An LSTM cell expressed as a monadic program and wrapped with `scan` for temporal recurrence. Demonstrates gate activations from LogitNormal priors and tanh approximation via `2 * sigmoid(2x) - 1`.

**Features:** `program`, `scan`, `continuous`, `draw`, `let` arithmetic, `sigmoid`, `LogitNormal`, `type`

```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 64
type State = Euclidean 128
type Output = Euclidean 32

embed tok_embed : Token -> Embedded

continuous gate_i : Embedded * State -> Hidden ~ LogitNormal
continuous gate_f : Embedded * State -> Hidden ~ LogitNormal
continuous gate_o : Embedded * State -> Hidden ~ LogitNormal
continuous cell_cand : Embedded * State -> Hidden ~ Normal [scale=0.5]

program lstm_cell(x_t, state_prev) : Embedded * State -> State
    draw i_gate ~ gate_i(x_t, state_prev)
    draw f_gate ~ gate_f(x_t, state_prev)
    draw o_gate ~ gate_o(x_t, state_prev)
    draw g_cand ~ cell_cand(x_t, state_prev)

    let c_new = f_gate * g_cand + i_gate * g_cand
    let two_c = 2.0 * c_new
    let sig_2c = sigmoid(two_c)
    let tanh_c = 2.0 * sig_2c - 1.0
    let h_new = o_gate * tanh_c

    return (c_new, h_new)

continuous output_proj : State -> Output ~ Normal [scale=0.1]

let lstm = tok_embed >> scan(lstm_cell) >> output_proj

output lstm
```

---

### [Bayesian GRU](gru.md)

A Gated Recurrent Unit cell expressed as a monadic program with update and reset gates controlling information flow. Demonstrates inline distribution syntax with `<-`.

**Features:** `program`, `scan`, `continuous`, `draw`, `<-` bind syntax, `let` arithmetic, `LogitNormal`, `type`

```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 128
type Output = Euclidean 64

embed tok_embed : Token -> Embedded

continuous gate_z : Embedded * Hidden -> Hidden ~ LogitNormal
continuous gate_r : Embedded * Hidden -> Hidden ~ LogitNormal

program gru_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
    draw z ~ gate_z(x_t, h_prev)
    draw r ~ gate_r(x_t, h_prev)

    let reset_hidden = r * h_prev

    h_cand <- Normal(reset_hidden, 0.5)

    let z_complement = 1.0 - z
    let h_new = z_complement * h_prev + z * h_cand

    return h_new

continuous output_proj : Hidden -> Output ~ Normal [scale=0.1]

let gru = tok_embed >> scan(gru_cell) >> output_proj

output gru
```

---

### [Bayesian Elman Network](elman-rnn.md)

A Bayesian Elman network decomposing the recurrent cell into a transition stage followed by a near-identity context copy. Demonstrates composition within the scan combinator.

**Features:** `embed`, `continuous`, `scan`, `>>` composition, product domain, `type`

```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 128
type Output = Euclidean 64

embed tok_embed : Token -> Embedded

continuous transition : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
continuous context_copy : Hidden -> Hidden ~ Normal [scale=0.01]
continuous output_proj : Hidden -> Output ~ Normal [scale=0.1]

let cell = transition >> context_copy
let elman = tok_embed >> scan(cell) >> output_proj

output elman
```

---

### [Bayesian Bidirectional RNN](bidirectional-rnn.md)

A bidirectional RNN that processes sequences in both directions using `scan`, then combines the final hidden states. The tensor product `@` runs forward and backward paths in parallel.

**Features:** `embed`, `continuous`, `scan`, `@` tensor product, `>>` composition, product domain, `type`

```qvr
object Token : 256
type Embedded = Euclidean 64
type FwdHidden = Euclidean 64
type BwdHidden = Euclidean 64
type Combined = Euclidean 128
type Output = Euclidean 32

embed tok_embed : Token -> Embedded

continuous fwd_cell : Embedded * FwdHidden -> FwdHidden ~ Normal [scale=0.1]

let forward_path = tok_embed >> scan(fwd_cell)

continuous bwd_cell : Embedded * BwdHidden -> BwdHidden ~ Normal [scale=0.1]

let backward_path = tok_embed >> scan(bwd_cell)

continuous combine : Combined -> Output ~ Normal [scale=0.1]

let birnn = (forward_path @ backward_path) >> combine

output birnn
```

---

## Generative Models

### [Deep Variational Autoencoder](vae.md)

A VAE with multi-layer encoder and decoder networks using `stack` for deep layers. The encoder maps observations through 3 hidden layers to a latent distribution; the decoder maps latent codes through 3 hidden layers back to observation space.

**Features:** `embed`, `continuous`, `stack`, `>>` composition, `type`

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

---

### [Bayesian Gaussian Mixture Model](mixture-model.md)

A Bayesian GMM with 4 components. Demonstrates hierarchical priors (`Gamma` on precision), the `softplus` built-in, division for normalization, and soft mixture observations.

**Features:** `program`, `<-` bind syntax, `Gamma`, `Exponential`, `softplus`, `/` division, `observe`

```qvr
object Unit : 1
object Obs : 1

program gmm : Unit -> Obs
    mu_1 <- Normal(0.0, 3.0)
    mu_2 <- Normal(0.0, 3.0)
    mu_3 <- Normal(0.0, 3.0)
    mu_4 <- Normal(0.0, 3.0)
    tau_1 <- Gamma(2.0, 1.0)
    tau_2 <- Gamma(2.0, 1.0)
    tau_3 <- Gamma(2.0, 1.0)
    tau_4 <- Gamma(2.0, 1.0)

    let sigma_1 = 1.0 / softplus(tau_1)
    let sigma_2 = 1.0 / softplus(tau_2)
    let sigma_3 = 1.0 / softplus(tau_3)
    let sigma_4 = 1.0 / softplus(tau_4)

    weight_1 <- Exponential(1.0)
    weight_2 <- Exponential(1.0)
    weight_3 <- Exponential(1.0)
    weight_4 <- Exponential(1.0)

    let total = weight_1 + weight_2 + weight_3 + weight_4
    let p1 = weight_1 / total
    let p2 = weight_2 / total
    let p3 = weight_3 / total
    let mix_mu = p1 * mu_1 + p2 * mu_2 + p3 * mu_3 + (1.0 - p1 - p2 - p3) * mu_4
    let mix_sigma = p1 * sigma_1 + p2 * sigma_2 + p3 * sigma_3 + (1.0 - p1 - p2 - p3) * sigma_4

    observe x ~ Normal(mix_mu, mix_sigma)
    return x

output gmm
```

---

### [Hidden Markov Model (Discrete)](hmm.md)

A classic discrete HMM using stochastic morphisms (Markov kernels). `repeat(transition)` without a count creates a `RepeatMorphism` whose step count is set at runtime via `prog(n_steps=N)`.

**Features:** `stochastic`, `repeat` (runtime-variable), `>>` composition, `quantale`

```qvr
quantale product_fuzzy

object State : 8
object Obs : 16

stochastic initial : State -> State
stochastic transition : State -> State
stochastic emission : State -> Obs

let n_step = repeat(transition) >> emission
let hmm = initial >> n_step

output hmm
```

---

### [Continuous State-Space Model](continuous-hmm.md)

A continuous-state hidden Markov model using `scan` for temporal recurrence. Includes both a generative direction (monadic program sampling state-observation pairs) and an inference direction (scan-based Bayesian filtering over observation sequences).

**Features:** `continuous`, `program`, `scan`, `>>`, `draw`, `observe`, `type`

```qvr
type State = Euclidean 16
type Obs = Euclidean 8

continuous transition : State -> State ~ Normal [scale=0.1]
continuous emission : State -> Obs ~ Normal [scale=0.1]

program generative_step : State -> State
    draw s_new ~ transition

    observe o ~ emission(s_new)

    return s_new

continuous inference_cell : Obs * State -> State ~ Normal [scale=0.1]

let filter = scan(inference_cell)

continuous decoder : State -> Obs ~ Normal [scale=0.1]

let filter_and_reconstruct = scan(inference_cell) >> decoder

output filter_and_reconstruct
```

---

## Formal Grammars

### [Probabilistic Context-Free Grammar](pcfg.md)

A learnable PCFG expressed as a pair of morphisms in the Kleisli category of the Giry monad: a branching morphism $N \to N \otimes N$ and a lexicalization morphism $N \to T$. The compiler inspects morphism types to assemble a CKY deductive system.

**Features:** `stochastic`, `parser` (morphism rules), `quantale`, product codomain (`N * N`)

```qvr
quantale product_fuzzy

object N : 10
object T : 64

stochastic binary_rules : N -> N * N
stochastic lexical_rules : N -> T

let pcfg = parser(
    rules=[binary_rules, lexical_rules],
    start=0
)

output pcfg
```

---

### [Weighted Combinatory Categorial Grammar](ccg.md)

A weighted CCG parser composed from rule schema primitives: `evaluation` (forward/backward application), `harmonic_composition` (forward/backward composition), and `crossed_composition` (forward/backward crossed composition).

**Features:** `category`, `parser`, `object`, rule schemas, `terminal=`

```qvr
category S, NP, N, VP, PP
object Token : 256

let grammar = parser(
    rules=[evaluation, harmonic_composition, crossed_composition],
    terminal=Token,
    start=S
)

output grammar
```

---

### [Type-Logical Grammar (Lambek Calculus)](type-logical.md)

A weighted parser based on the non-commutative Lambek calculus, assembled from schema primitives: `evaluation`, `adjunction_units` (Lambek lifting), `tensor_introduction`, and `tensor_projection`.

**Features:** `category`, `parser`, `object`, rule schemas, `terminal=`

```qvr
category S, NP, N, VP, PP
object Token : 256

let grammar = parser(
    rules=[evaluation, adjunction_units, tensor_introduction, tensor_projection],
    terminal=Token,
    start=S
)

output grammar
```

---

### [Multimodal Type-Logical Grammar](multimodal-tlg.md)

A multimodal type-logical grammar (Moortgat 1997) with modal type constructors. Uses `constructors=[slash, diamond]` to generate diamond-modal categories alongside standard slash categories.

**Features:** `category`, `parser`, `object`, rule schemas, `constructors`, `diamond` modality

```qvr
category S, NP, N, VP, PP
object Token : 256

let tlg = parser(
    rules=[evaluation, adjunction_units, modal_introduction, modal_elimination],
    terminal=Token,
    constructors=[slash, diamond],
    depth=1,
    start=S
)

output tlg
```

---

### [Custom Rules of Inference](custom-rules.md)

A parser using explicitly declared rules of inference in sequent-style notation with universally quantified pattern variables, instead of built-in schema primitives.

**Features:** `category`, `rule`, `parser`, `terminal=`, `\` backslash, `=>`

```qvr
category S, NP, N, VP, PP
object Token : 256

rule forward_app(X, Y) : X/Y, Y => X
rule backward_app(X, Y) : Y, X\Y => X
rule forward_comp(X, Y, Z) : X/Y, Y/Z => X/Z
rule backward_comp(X, Y, Z) : Y\Z, X\Y => X\Z

let grammar = parser(
    rules=[forward_app, backward_app, forward_comp, backward_comp],
    terminal=Token,
    start=S
)

output grammar
```

---

## Probabilistic Programs

### [Bayesian Linear Regression](bayesian-regression.md)

The simplest meaningful probabilistic program: a two-parameter linear model with a `HalfCauchy` prior on noise scale. Demonstrates the core `draw`/`let`/`observe` pattern.

**Features:** `program`, `<-` bind syntax, `HalfCauchy`, `let` arithmetic, `observe`

```qvr
object Predictor : 1
object Response : 1

program bayesian_regression : Predictor -> Response
    sigma <- HalfCauchy(2.0)
    beta_0 <- Normal(0.0, 5.0)
    beta_1 <- Normal(0.0, 2.0)
    x <- Normal(0.0, 1.0)

    let mu = beta_0 + beta_1 * x

    observe y ~ Normal(mu, sigma)
    return y

output bayesian_regression
```

---

## Feature Index

The table below shows which DSL features each example demonstrates.

| Example | `program` | `continuous` | `stochastic` | `embed` | `>>` | `@` | `fan` | `stack` | `scan` | `repeat` | `<-` | `let` arith | `observe` | Grammar | Built-ins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [Transformer](transformer.md) | | ✓ | | ✓ | ✓ | | ✓ | ✓ | | | | | | | |
| [Vanilla RNN](vanilla-rnn.md) | | ✓ | | ✓ | ✓ | | | | ✓ | | | | | | |
| [LSTM](lstm.md) | ✓ | ✓ | | ✓ | ✓ | | | | ✓ | | ✓ | ✓ | | | `sigmoid` |
| [GRU](gru.md) | ✓ | ✓ | | ✓ | ✓ | | | | ✓ | | ✓ | ✓ | | | |
| [Elman](elman-rnn.md) | | ✓ | | ✓ | ✓ | | | | ✓ | | | | | | |
| [Bidirectional](bidirectional-rnn.md) | | ✓ | | ✓ | ✓ | ✓ | | | ✓ | | | | | | |
| [VAE](vae.md) | | ✓ | | ✓ | ✓ | | | ✓ | | | | | | | |
| [GMM](mixture-model.md) | ✓ | | | | | | | | | | ✓ | ✓ | ✓ | | `softplus` |
| [HMM (discrete)](hmm.md) | | | ✓ | | ✓ | | | | | ✓ | | | | | |
| [HMM (continuous)](continuous-hmm.md) | ✓ | ✓ | | | ✓ | | | | ✓ | | | | ✓ | | |
| [PCFG](pcfg.md) | | | ✓ | | | | | | | | | | | `parser` | |
| [CCG](ccg.md) | | | | | | | | | | | | | | `parser` | |
| [Lambek](type-logical.md) | | | | | | | | | | | | | | `parser` | |
| [Multimodal TLG](multimodal-tlg.md) | | | | | | | | | | | | | | `parser` | |
| [Custom Rules](custom-rules.md) | | | | | | | | | | | | | | `parser` | |
| [Regression](bayesian-regression.md) | ✓ | | | | | | | | | | ✓ | ✓ | ✓ | | |
