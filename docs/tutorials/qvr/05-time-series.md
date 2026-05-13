# 5. Sequence models

Sequences turn up everywhere: time series, text, RNN-shaped models, hidden Markov models, state-space models. The QVR surface has three constructs aimed at sequence-shaped problems:

- **Plate-draws** for IID-along-an-index data (chapter 3 used this for `theta : School <- Normal(...)`).
- **`scan`** for sequential evaluation: a per-step cell function fold-applied along the sequence dimension.
- **The deduction layer** for chart-shaped problems whose computation is not a simple left-to-right scan (CKY, Earley, forward-backward, Viterbi). That layer is its own subject; chapter 7 points at it and the deduction guide covers the full surface.

This chapter covers the first two and ends with a concrete HMM written with `scan` plus a `marginalize` block.

## Plates revisited

A *plate-draw* binds one value per index of a finite-set object:

<!-- compile: false -->
```qvr
object School : 8
theta : School <- Normal(mu, tau)
```

is the QVR analogue of NumPyro's `with plate("schools", 8): theta = sample("theta", dist.Normal(mu, tau))`. The result has shape `(8,)`; subsequent `let` arithmetic broadcasts over it. A *vectorised observe* over a plate has the same shape:

<!-- compile: false -->
```qvr
observe y : School <- Normal(theta, sigma_j)
```

Plates are good for IID structure: nothing about index `j+1` depends on what happened at `j`. For genuinely sequential data, you want `scan`.

## `scan` for left-to-right evaluation

`scan` takes a *cell* whose signature is `Input * Hidden -> Hidden` and lifts it to operate along the sequence dimension of an input tensor. The cell may be a `continuous` morphism (one-step state update under a Gaussian transition) or a `program` block (one-step update with its own random draws).

```qvr
quantale real
object Token : 256
type Embedded = Euclidean 64
type Hidden   = Euclidean 128
type Output   = Euclidean 64

embed tok_embed : Token -> Embedded

continuous cell        : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
continuous output_proj : Hidden -> Output             ~ Normal [scale=0.1]

let rnn = tok_embed >> scan(cell) >> output_proj
export rnn
```

The pipeline reads as: embed each token to a 64-dim vector, fold the cell over the sequence dimension (with zero or learned initial state), project the final hidden state to a 64-dim output. The compiler infers the sequence axis from the input tensor's second dimension at evaluation time.

If you've used Haskell's `mapAccumL` or NumPy's `np.cumsum`, this is the same idea generalised to a learnable cell.

## A hidden Markov model

HMMs ([Rabiner, 1989](https://doi.org/10.1109/5.18626)) sit at the intersection of `scan` (the per-step transition is sequential) and `marginalize` (the discrete state is integrated out). Here's a two-state HMM with continuous Gaussian emissions:

<!-- compile: false -->
```qvr
quantale real
object Step : 100
object State : 2

program hmm : Step -> Step ! Sample, Score, Marginal
    init_logits  : State        <- Normal(0.0, 1.0)
    trans_logits : State * State <- Normal(0.0, 1.0)
    mu_emit      : State        <- Normal(0.0, 5.0)
    sd_emit      : State        <- HalfNormal(1.0)

    let init_probs  = softmax(init_logits)
    let trans_probs = softmax_rows(trans_logits)

    marginalize z : Step * State <- ForwardLattice(init_probs, trans_probs) in {
        observe y : Step <- Normal(mu_emit[z], sd_emit[z])
    }
    return y

export hmm
```

A few things to point out:

- `init_logits : State` and `mu_emit : State` are plate-draws over the state object: one initial-logit per state, one emission mean per state.
- `trans_logits : State * State` is a plate over a product object: a 2×2 matrix of transition logits.
- `softmax_rows` is a `let`-arithmetic builtin that row-normalises a 2D tensor.
- `marginalize z : Step * State <- ForwardLattice(...)` is the HMM-shaped marginalisation. `ForwardLattice` is the structured family that exposes the forward-algorithm log-likelihood (i.e. the marginal `log p(y | params)` after summing over all state sequences). The runtime contracts this against the categorical-prior and the per-step emission log-likelihoods using a forward-backward pass; gradients flow through the full forward pass.

The Pyro analogue would be a custom `infer={"enumerate": "parallel"}` model with manual axis-shape juggling; NumPyro's `numpyro.contrib.control_flow.scan` does the per-step recursion but the discrete-state forward algorithm is still your responsibility. QVR's `ForwardLattice` family wraps the recursion for you.

## State-space (Kalman) models

For continuous-state sequences, replace the discrete-state marginalisation with a Gaussian transition and a Gaussian emission. The marginal likelihood is closed-form via Kalman smoothing ([Kalman, 1960](https://doi.org/10.1115/1.3662552)):

<!-- compile: false -->
```qvr
quantale real
object Step : 100
type State = Euclidean 4
type Obs   = Euclidean 2

program kalman : Step -> Step ! Sample, Score
    A     <- Normal(0.0, 1.0)              # transition matrix (4 * 4)
    Q     <- HalfNormal(0.5)                # state noise scale
    H     <- Normal(0.0, 1.0)              # emission matrix (2 * 4)
    R     <- HalfNormal(0.5)                # observation noise scale

    let prior_mean = zeros(4)
    let prior_cov  = identity(4)

    observe y : Step <- KalmanSmoother(prior_mean, prior_cov, A, Q, H, R)
    return y

export kalman
```

The `KalmanSmoother` family computes the marginal likelihood of the observation sequence under the linear-Gaussian state-space model in closed form; gradients flow through the smoother's filter-then-smooth pass.

## Try this

- Convert the HMM to use a `scan` cell with explicit per-step draws (no `ForwardLattice`), then run NUTS to sample the latent state sequence. Compare runtime to the marginalised version.
- Make the HMM hierarchical: each sequence has its own transition matrix drawn from a hyperprior. (Chapter 3's plate-draw applies.)
- Replace the Gaussian emissions with Poisson emissions for count data; the marginalised structure is identical.

## Next

Chapter 6 surveys the inference algorithms: nine variational guides, four objectives, HMC and NUTS, two hybrid samplers. We'll work out which combination fits which kind of model.
