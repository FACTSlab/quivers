# Examples Gallery

Complete `.qvr` programs spanning neural architectures, probabilistic models, and linguistic applications. Each example compiles to a trainable `nn.Module`. Click through for a walkthrough of the code and its categorical interpretation.

All source files are in `docs/examples/source/`.


## Sequence Architectures

Bayesian sequence models for language modeling: each example declares a tokeniser-style embed, a sequence processor, and a Categorical [`lm_head`](../api/continuous/morphisms.md) so the program's `observe` step scores the next- or masked-token target under a Categorical likelihood. Forward sampling via [`MonadicProgram.rsample`](../api/continuous/programs.md) runs the model on synthetic inputs.

The Elman SRN (Elman 1990, [doi:10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)) is the same Kleisli morphism `Embedded * Hidden -> Hidden` threaded by [`scan`](../guides/dsl-declarations.md#scan-temporal-recurrence) as the vanilla RNN example below: the "context units" of the classical SRN are exactly the `h_{t-1}` argument that `scan` passes into the cell at each step, so there is no separate Elman example.

### [Vanilla RNN Language Model](vanilla-rnn-lm.md)

A single-cell recurrent language model. The cell is a Bayesian [Kleisli morphism](https://ncatlab.org/nlab/show/Kleisli+category) `Embedded * Hidden -> Hidden` with Normal weight priors; `scan` threads the hidden state across the token sequence; a Categorical `lm_head` scores the next-token target.

**Distinguishing feature:** [`scan`](../guides/dsl-declarations.md#scan-temporal-recurrence) for temporal recurrence over a sequence-valued input.

```qvr
object Token : 256

type Embedded = Euclidean 64
type Hidden = Euclidean 128

embed tok_embed : Token -> Embedded

kernel cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
kernel lm_head : Hidden -> Token ~ Categorical

let backbone = tok_embed >> scan(cell)

program vanilla_rnn_lm : Token -> Token
    h <- backbone
    observe next_token : Token <- lm_head(h)
    return next_token

export vanilla_rnn_lm
```

---

### [LSTM Language Model](lstm-lm.md)

A Bayesian LSTM ([Hochreiter and Schmidhuber 1997](https://doi.org/10.1162/neco.1997.9.8.1735)). The recurrent cell is a parametric [`program`](../guides/dsl-programs-and-lets.md#program-declarations) that draws the four standard gates (`i`, `f`, `o`, `g`) from `LogitNormal` and `Normal` priors, updates the cell state by `c_t = f_t * c_{t-1} + i_t * g_t`, and emits `h_t = o_t * tanh(c_t)`.

**Distinguishing feature:** parametric `program` cell composed with `scan`, exercising `let` arithmetic and the `sigmoid` builtin to realize tanh as `2 * sigmoid(2x) - 1`.

```qvr
object Token : 256

type Embedded = Euclidean 64
type Hidden = Euclidean 128

embed tok_embed : Token -> Embedded

kernel gate_i : Embedded * Hidden -> Hidden ~ LogitNormal
kernel gate_f : Embedded * Hidden -> Hidden ~ LogitNormal
kernel gate_o : Embedded * Hidden -> Hidden ~ LogitNormal
kernel cell_cand : Embedded * Hidden -> Hidden ~ Normal [scale=0.5]
kernel lm_head : Hidden -> Token ~ Categorical

program lstm_cell(x_t, c_prev) : Embedded * Hidden -> Hidden
    i_gate <- gate_i(x_t, c_prev)
    f_gate <- gate_f(x_t, c_prev)
    o_gate <- gate_o(x_t, c_prev)
    g_cand <- cell_cand(x_t, c_prev)

    let c_new = f_gate * c_prev + i_gate * g_cand
    let two_c = 2.0 * c_new
    let sig_2c = sigmoid(two_c)
    let tanh_c = 2.0 * sig_2c - 1.0
    let h_new = o_gate * tanh_c

    return c_new

let backbone = tok_embed >> scan(lstm_cell)

program lstm_lm : Token -> Token
    h <- backbone
    observe next_token : Token <- lm_head(h)
    return next_token

export lstm_lm
```

---

### [GRU Language Model](gru-lm.md)

A Bayesian GRU ([Cho et al. 2014](https://doi.org/10.3115/v1/D14-1179)). The cell draws update and reset gates from `LogitNormal`, builds a candidate from the reset-gated previous state via a Normal kernel, and interpolates with the update gate.

**Distinguishing feature:** parametric `program` cell with interior `let` arithmetic and indexed Normal arguments, all composed under `scan`.

```qvr
object Token : 256

type Embedded = Euclidean 64
type Hidden = Euclidean 128

embed tok_embed : Token -> Embedded

kernel gate_z : Embedded * Hidden -> Hidden ~ LogitNormal
kernel gate_r : Embedded * Hidden -> Hidden ~ LogitNormal
kernel lm_head : Hidden -> Token ~ Categorical

program gru_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
    z <- gate_z(x_t, h_prev)
    r <- gate_r(x_t, h_prev)

    let reset_hidden = r * h_prev

    h_cand <- Normal(reset_hidden, 0.5)

    let z_complement = 1.0 - z
    let h_new = z_complement * h_prev + z * h_cand

    return h_new

let backbone = tok_embed >> scan(gru_cell)

program gru_lm : Token -> Token
    h <- backbone
    observe next_token : Token <- lm_head(h)
    return next_token

export gru_lm
```

---

### [Bidirectional RNN Masked Language Model](bidirectional-rnn-lm.md)

A bidirectional RNN used as a masked language model in the spirit of [BERT](https://doi.org/10.18653/v1/N19-1423). Two independent recurrent paths scan the sequence forward and backward; the tensor product `@` runs them in parallel in the [Giry monad](https://doi.org/10.1007/BFb0092872)'s Kleisli category, and a Categorical `lm_head` scores the masked-token target from the combined representation.

**Distinguishing feature:** the [tensor product](../guides/morphisms.md) `@` to run the forward and backward Kleisli morphisms in parallel.

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

---

### [Transformer Language Model](transformer-lm.md)

A four-layer Bayesian transformer ([Vaswani et al. 2017](https://doi.org/10.48550/arXiv.1706.03762)) with four-head attention. The `kernel head[4]` declaration creates four independently-parameterized heads; `fan(head)` runs them in parallel and concatenates outputs, and `stack(layer, 4)` produces four independent transformer blocks.

**Distinguishing feature:** [`stack`](../guides/dsl-declarations.md#stack-independent-multi-layer) for independent deep copies and [`fan`](../guides/dsl-declarations.md#fan-out-diagonal-morphism) for parallel multi-head attention.

```qvr
object Token : 256

type Latent = Euclidean 64
type HeadOut = Euclidean 16
type FFHidden = Euclidean 128

embed tok_embed : Token -> Latent

kernel head[4] : Latent -> HeadOut ~ Normal [scale=0.1]
kernel attn_proj : Latent -> Latent ~ Normal [scale=0.1]
kernel ff_up : Latent -> FFHidden ~ Normal
kernel ff_down : FFHidden -> Latent ~ Normal [scale=0.1]
kernel residual_attn : Latent -> Latent ~ Normal [scale=0.01]
kernel residual_ff : Latent -> Latent ~ Normal [scale=0.01]
kernel lm_head : Latent -> Token ~ Categorical

let layer = fan(head) >> attn_proj >> residual_attn >> ff_up >> ff_down >> residual_ff
let backbone = tok_embed >> stack(layer, 4)

program transformer_lm : Token -> Token
    h <- backbone
    observe next_token : Token <- lm_head(h)
    return next_token

export transformer_lm
```

---

### [Sequence-to-Sequence Model](seq2seq.md)

A single transformer-style encoder-decoder ([Sutskever, Vinyals, and Le 2014](https://doi.org/10.48550/arXiv.1409.3215); [Vaswani et al. 2017](https://doi.org/10.48550/arXiv.1706.03762)) combining both halves in one example. The encoder maps a source sequence to a Latent representation, the decoder maps a target prefix to its own Latent, and a `cross` morphism merges the two streams before the Categorical `lm_head` scores the next target token.

**Distinguishing feature:** simultaneous use of `stack` (deep per-side blocks) and `@` (parallel encoder/decoder composition) to assemble the two halves of an encoder/decoder.

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

---

## Generative Models

### [Deep Variational Autoencoder](vae.md)

A VAE with multi-layer encoder and decoder networks using `stack` for deep layers. The encoder maps observations through 3 hidden layers to a latent distribution; the decoder maps latent codes through 3 hidden layers back to observation space.

**Features:** `embed`, `kernel`, `stack`, `>>` composition, `type`

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

    observe x <- Normal(mix_mu, mix_sigma)
    return x

export gmm
```

---

### [Hidden Markov Model (Discrete)](hmm.md)

A classic discrete HMM using stochastic morphisms (Markov kernels). `repeat(transition)` without a count creates a `RepeatMorphism` whose step count is set at runtime via `prog(n_steps=N)`.

**Features:** `kernel`, `repeat` (runtime-variable), `>>` composition, `algebra`

```qvr
algebra product_fuzzy

object State : 8
object Obs : 16

kernel initial : State -> State
kernel transition : State -> State
kernel emission : State -> Obs

let n_step = repeat(transition) >> emission
let hmm = initial >> n_step

export hmm
```

---

### [Continuous State-Space Model](continuous-hmm.md)

A continuous-state hidden Markov model using `scan` for temporal recurrence. Includes both a generative direction (monadic program sampling state-observation pairs) and an inference direction (scan-based Bayesian filtering over observation sequences).

**Features:** `kernel`, `program`, `scan`, `>>`, `<-`, `observe`, `type`

```qvr
type State = Euclidean 16
type Obs = Euclidean 8

kernel transition : State -> State ~ Normal [scale=0.1]
kernel emission : State -> Obs ~ Normal [scale=0.1]

program generative_step : State -> State
    s_new <- transition

    observe o <- emission(s_new)

    return s_new

kernel inference_cell : Obs * State -> State ~ Normal [scale=0.1]

let filter = scan(inference_cell)

kernel decoder : State -> Obs ~ Normal [scale=0.1]

let filter_and_reconstruct = scan(inference_cell) >> decoder

export filter_and_reconstruct
```

---

## Formal Grammars

### [Probabilistic Context-Free Grammar](pcfg.md)

A learnable PCFG declared as an agenda-based weighted deduction over chart-spans `span(I, J, N)`. Branching and lexical-anchor rules are sequents; the lexicon block ships learnable per-entry log-weights; the `LogProb` semiring carries differentiable inside scores.

**Features:** `deduction`, `atoms`, sequent rules, `lexicon`, `semiring LogProb`

```qvr
object Term : 16

deduction PCFG : Term -> Term {
    atoms { S, NP, VP, Det, N, V, the, a, cat, dog, sleeps, runs, span, leaf }

    rule branch  : span(I, K, B), span(K, J, C) |- span(I, J, A)
    rule anchor  : leaf(I, T)                    |- span(I, J, A)

    lexicon {
        "the"    : Det = the    @ learnable
        "cat"    : N   = cat    @ learnable
        "sleeps" : V   = sleeps @ learnable
    }

    semiring  LogProb
    start     S
    depth     6
}
```

---

### [Weighted Combinatory Categorial Grammar](ccg.md)

A weighted CCG parser whose six structural combinators, forward / backward application, harmonic composition, crossed composition, are each one sequent rule over `span(I, J, X)`. The slash constructors `Fwd(X, Y) ≡ X/Y` and `Bwd(X, Y) ≡ X\Y` are user-declared atoms.

**Features:** `deduction`, `atoms`, sequent rules, slash constructors, `semiring LogProb`

```qvr
object Term : 16

deduction CCG : Term -> Term {
    atoms { NP, S, N, VP, PP, Fwd, Bwd, span }

    rule fwd_app    : span(I, K, Fwd(X, Y)), span(K, J, Y)       |- span(I, J, X)
    rule bwd_app    : span(I, K, Y),         span(K, J, Bwd(X, Y)) |- span(I, J, X)
    rule fwd_comp   : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, Z)) |- span(I, J, Fwd(X, Z))
    rule bwd_comp   : span(I, K, Bwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Bwd(X, Z))
    rule fwd_xcomp  : span(I, K, Fwd(X, Y)), span(K, J, Bwd(Y, Z)) |- span(I, J, Bwd(X, Z))
    rule bwd_xcomp  : span(I, K, Fwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Fwd(X, Z))

    semiring  LogProb
    start     S
    depth     6
}
```

---

### [Type-Logical Grammar (Lambek Calculus)](type-logical.md)

A weighted parser based on the non-commutative Lambek calculus: right and left application, plus product introduction and elimination over the tensor constructor `Tns(A, B) ≡ A⊗B`.

**Features:** `deduction`, slash + tensor constructors, sequent rules, `semiring LogProb`

```qvr
object Term : 16

deduction Lambek : Term -> Term {
    atoms { S, NP, N, VP, PP, Fwd, Bwd, Tns, span }

    rule right_app    : span(I, K, Fwd(A, B)), span(K, J, B)         |- span(I, J, A)
    rule left_app     : span(I, K, B),         span(K, J, Bwd(A, B)) |- span(I, J, A)
    rule tensor_intro : span(I, K, A),         span(K, J, B)         |- span(I, J, Tns(A, B))
    rule tensor_left  : span(I, J, Tns(A, B))                        |- span(I, J, A)
    rule tensor_right : span(I, J, Tns(A, B))                        |- span(I, J, B)

    semiring  LogProb
    start     S
    depth     6
}
```

---

### [Multimodal Type-Logical Grammar](multimodal-tlg.md)

A multimodal type-logical grammar (Moortgat 1997) extending the Lambek calculus with unary modal constructors `Dia(A) ≡ ◇A` and `Box(A) ≡ □A`. The deduction licenses base right / left application together with modal introduction and elimination.

**Features:** `deduction`, modal constructors, unary + binary sequent rules

```qvr
object Term : 16

deduction MMTLG : Term -> Term {
    atoms { S, NP, N, VP, PP, Fwd, Bwd, Dia, Box, span }

    rule right_app  : span(I, K, Fwd(A, B)), span(K, J, B)         |- span(I, J, A)
    rule left_app   : span(I, K, B),         span(K, J, Bwd(A, B)) |- span(I, J, A)
    rule dia_intro  : span(I, J, A)                                |- span(I, J, Dia(A))
    rule dia_elim   : span(I, J, Dia(A))                           |- span(I, J, A)

    semiring  LogProb
    start     S
    depth     6
}
```

---

### [Custom Sequent Rules](custom-rules.md)

An AB grammar declared from the rule level up: each combinator is one sequent in the `deduction { … }` block. Single-uppercase identifiers (`X`, `Y`, `Z`, `I`, `J`, `K`) bind as pattern variables; every other identifier in a rule pattern must appear in the surrounding `atoms { … }` block.

**Features:** `deduction`, `atoms`, sequent rules, pattern variables

```qvr
object Term : 16

deduction AB : Term -> Term {
    atoms { S, NP, N, VP, PP, Fwd, Bwd, span }

    rule fwd_app  : span(I, K, Fwd(X, Y)), span(K, J, Y)         |- span(I, J, X)
    rule bwd_app  : span(I, K, Y),         span(K, J, Bwd(X, Y)) |- span(I, J, X)
    rule fwd_comp : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, Z)) |- span(I, J, Fwd(X, Z))
    rule bwd_comp : span(I, K, Bwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Bwd(X, Z))

    semiring  LogProb
    start     S
    depth     6
}
```

---

## Probabilistic Programs

### [Bayesian Linear Regression](bayesian-regression.md)

The simplest meaningful probabilistic program: a two-parameter linear model with a `HalfCauchy` prior on noise scale. Demonstrates the core bind/let/observe pattern.

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

    observe y <- Normal(mu, sigma)
    return y

export bayesian_regression
```

---

## Feature Index

The table below shows which DSL features each example demonstrates.

| Example | `program` | `kernel` | `embed` | `>>` | `@` | `fan` | `stack` | `scan` | `repeat` | `<-` | `let` arith | `observe` | Grammar | Built-ins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [Vanilla RNN LM](vanilla-rnn-lm.md) | ✓ | ✓ | ✓ | ✓ | | | | ✓ | | ✓ | | ✓ | | |
| [LSTM LM](lstm-lm.md) | ✓ | ✓ | ✓ | ✓ | | | | ✓ | | ✓ | ✓ | ✓ | | `sigmoid` |
| [GRU LM](gru-lm.md) | ✓ | ✓ | ✓ | ✓ | | | | ✓ | | ✓ | ✓ | ✓ | | |
| [Bidirectional RNN LM](bidirectional-rnn-lm.md) | ✓ | ✓ | ✓ | ✓ | ✓ | | | ✓ | | ✓ | | ✓ | | |
| [Transformer LM](transformer-lm.md) | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | | | ✓ | | ✓ | | |
| [Seq2Seq](seq2seq.md) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | | | ✓ | | ✓ | | |
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
