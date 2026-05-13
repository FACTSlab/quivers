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

export transformer
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

export rnn
```

---

### [Bayesian LSTM](lstm.md)

An LSTM cell expressed as a monadic program and wrapped with `scan` for temporal recurrence. Demonstrates gate activations from LogitNormal priors and tanh approximation via `2 * sigmoid(2x) - 1`.

**Features:** `program`, `scan`, `continuous`, `<-`, `let` arithmetic, `sigmoid`, `LogitNormal`, `type`

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
    i_gate <- gate_i(x_t, state_prev)
    f_gate <- gate_f(x_t, state_prev)
    o_gate <- gate_o(x_t, state_prev)
    g_cand <- cell_cand(x_t, state_prev)

    let c_new = f_gate * g_cand + i_gate * g_cand
    let two_c = 2.0 * c_new
    let sig_2c = sigmoid(two_c)
    let tanh_c = 2.0 * sig_2c - 1.0
    let h_new = o_gate * tanh_c

    return (c_new, h_new)

continuous output_proj : State -> Output ~ Normal [scale=0.1]

let lstm = tok_embed >> scan(lstm_cell) >> output_proj

export lstm
```

---

### [Bayesian GRU](gru.md)

A Gated Recurrent Unit cell expressed as a monadic program with update and reset gates controlling information flow. Demonstrates inline distribution syntax with `<-`.

**Features:** `program`, `scan`, `continuous`, `<-` bind syntax, `let` arithmetic, `LogitNormal`, `type`

```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 128
type Output = Euclidean 64

embed tok_embed : Token -> Embedded

continuous gate_z : Embedded * Hidden -> Hidden ~ LogitNormal
continuous gate_r : Embedded * Hidden -> Hidden ~ LogitNormal

program gru_cell(x_t, h_prev) : Embedded * Hidden -> Hidden
    z <- gate_z(x_t, h_prev)
    r <- gate_r(x_t, h_prev)

    let reset_hidden = r * h_prev

    h_cand <- Normal(reset_hidden, 0.5)

    let z_complement = 1.0 - z
    let h_new = z_complement * h_prev + z * h_cand

    return h_new

continuous output_proj : Hidden -> Output ~ Normal [scale=0.1]

let gru = tok_embed >> scan(gru_cell) >> output_proj

export gru
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

export elman
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

export birnn
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

export hmm
```

---

### [Continuous State-Space Model](continuous-hmm.md)

A continuous-state hidden Markov model using `scan` for temporal recurrence. Includes both a generative direction (monadic program sampling state-observation pairs) and an inference direction (scan-based Bayesian filtering over observation sequences).

**Features:** `continuous`, `program`, `scan`, `>>`, `<-`, `observe`, `type`

```qvr
type State = Euclidean 16
type Obs = Euclidean 8

continuous transition : State -> State ~ Normal [scale=0.1]
continuous emission : State -> Obs ~ Normal [scale=0.1]

program generative_step : State -> State
    s_new <- transition

    observe o <- emission(s_new)

    return s_new

continuous inference_cell : Obs * State -> State ~ Normal [scale=0.1]

let filter = scan(inference_cell)

continuous decoder : State -> Obs ~ Normal [scale=0.1]

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

### [Event-Structure Latent-Class Model](event-structure.md)

A four-class telicity × durativity latent-class model over cloze and proportion responses. Exercises indexed binds, a parametric `random_intercepts` template instantiated 8 times for crossed random intercepts on subject, verb, sense, and item, an ordinal monotone spline via `cumsum` of `HalfNormal` increments, indexed observes against a runtime `observations` dict, and scoped `marginalize` for coordinate marginalisation.

**Features:** `program`, parametric templates, indexed bind `v : A <- F(args)`, `observe r : N <- F(args)`, scoped `marginalize`, `cumsum`, `HalfNormal`

<!-- compile: false -->
```qvr
program random_intercepts (G : FinSet, scale : Real) : G -> 1
    sigma <- HalfNormal(scale)
    v : G <- Normal(0.0, sigma)
    return v

program event_structure : Item -> Item
    intercept_cloze <- Normal(0.0, 1.0)
    by_subj_cloze <- random_intercepts(SubjCloze, 1.0)
    by_verb_cloze <- random_intercepts(Verb,     1.0)
    duration_incr_cloze : Item <- HalfNormal(1.0)
    let duration_eff_cloze = cumsum(duration_incr_cloze)

    marginalize cloze_resp : RespCloze <- Bernoulli(intercept_cloze) in {
        observe cloze_resp : RespCloze <- Bernoulli(intercept_cloze)
    }
    return intercept_cloze
```

---

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
