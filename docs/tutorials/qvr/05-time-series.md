# 5. Sequence models

Sequences turn up everywhere: time series, text, RNN-shaped models, hidden Markov models, state-space models. The QVR surface has three constructs aimed at sequence-shaped problems:

- **Plate-draws** for IID-along-an-index data (chapter 3 used this for `theta : School <- Normal(...)`).
- **`scan`** for sequential evaluation: a per-step cell function fold-applied along the sequence dimension.
- **The deduction layer** for chart-shaped problems whose computation is not a simple left-to-right scan (CKY, Earley, forward-backward, Viterbi). That layer is its own subject; chapter 7 points at it and the deduction guide covers the full surface.

This chapter covers the first two, then walks through a discrete HMM and a linear-Gaussian state-space model.

## Plates revisited

A *plate-draw* binds one value per index of a finite-set object:

<!-- compile: false -->
```qvr
object School : 8
theta : School <- Normal(mu, tau)
```

is the QVR analogue of NumPyro's `with plate("schools", 8): theta = sample("theta", dist.Normal(mu, tau))`. The result has shape `(8,)`; subsequent `let` arithmetic broadcasts over it. A *vectorized observe* over a plate has the same shape:

<!-- compile: false -->
```qvr
observe y : School <- Normal(theta, sigma_j)
```

Plates are good for IID structure: nothing about index `j+1` depends on what happened at `j`. For genuinely sequential data, you want `scan`.

## `scan` for left-to-right evaluation

`scan` takes a *cell* whose signature is `Input * Hidden -> Hidden` and lifts it to operate along the sequence dimension of an input tensor. The cell may be a `kernel ... ~ Family` morphism (one-step state update under a Gaussian transition) or a `program` block (one-step update with its own random draws).

```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden   = Euclidean 128
type Output   = Euclidean 64

embed tok_embed : Token -> Embedded

kernel cell        : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
kernel output_proj : Hidden -> Output            ~ Normal [scale=0.1]

let rnn = tok_embed >> scan(cell) >> output_proj
export rnn
```

The pipeline reads as: embed each token to a 64-dim vector, fold the cell over the sequence dimension (with zero or learned initial state), project the final hidden state to a 64-dim output. The compiler infers the sequence axis from the input tensor's second dimension at evaluation time.

If you've used Haskell's `mapAccumL` or NumPy's `np.cumsum`, this is the same idea generalized to a learnable cell.

## A discrete hidden Markov model

HMMs ([Rabiner, 1989](https://doi.org/10.1109/5.18626)) factor as an initial distribution, a row-stochastic transition kernel, and a row-stochastic emission kernel. In QVR's enriched setting they compose directly with `>>`. Here's the canonical K-state HMM with categorical emissions, lifted from `docs/examples/source/hmm.qvr`:

```qvr
algebra product_fuzzy

object State : 8
object Obs : 16

latent initial    : State -> State ~ Dirichlet(1.0) over cod iid over dom
latent transition : State -> State ~ Dirichlet(1.0) over cod iid over dom
latent emission   : State -> Obs   ~ Dirichlet(1.0) over cod iid over dom

let n_step = repeat(transition) >> emission
let hmm    = initial >> n_step

export hmm
```

Two points to call out:

- Every kernel is a row-stochastic matrix with a row-wise [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior. The axis-role surface `~ Dirichlet(1.0) over cod iid over dom` says each row of the matrix is an independent simplex draw, indexed by the domain object: the conjugate prior for a discrete Markov chain.
- `repeat(transition)` is the runtime-variable repetition combinator: at evaluation time it folds the transition kernel against itself for the requested number of steps. The same model produces n-step marginals for any horizon.

The Pyro analogue uses `infer={"enumerate": "parallel"}` and walks the chain with axis-shape juggling. NumPyro's `numpyro.contrib.control_flow.scan` does the per-step recursion explicitly. QVR's compositional surface treats the chain as a single morphism: the runtime contracts initial, transition, and emission in the algebra's tensor-and-join structure.

## State-space models

For continuous-state sequences, the per-step transition is a Gaussian kernel and so is the emission. The canonical linear-Gaussian SSM whose forward filter is the Kalman filter ([Kalman, 1960](https://doi.org/10.1115/1.3662552)) appears in `docs/examples/source/linear_gaussian_ssm.qvr`:

```qvr
type Driver = Euclidean 2
type State  = Euclidean 4
type Obs    = Euclidean 2

kernel transition_cell : Driver * State -> State ~ Normal [scale=0.1]
kernel emission        : State -> Obs           ~ Normal [scale=0.1]
kernel filter_cell     : Obs * State -> State   ~ Normal [scale=0.1]

let generate = scan(transition_cell) >> emission
let filter   = scan(filter_cell)

export filter
```

`scan(transition_cell)` folds the per-step Gaussian transition along the sequence dimension; composing with `emission` produces the generative model. The dual `filter` pipeline scans the same shape with the conditioning kernel.

For a fully nonlinear (deep) variant where transition and emission are neural Gaussians, see `docs/examples/source/continuous_hmm.qvr` and `docs/examples/source/deep_markov.qvr`.

## Try this

- Run NUTS on the discrete HMM and check Rhat for `initial`, `transition`, `emission`.
- Make the linear-Gaussian SSM hierarchical: each sequence has its own transition cell drawn from a hyperprior. (Chapter 3's plate-draw applies.)
- Swap the linear `transition_cell` for the deep-Markov nonlinear kernel and compare ELBO convergence.

## Next

[Chapter 6](06-inference-zoo.md) surveys the inference algorithms: nine variational guides, four objectives, HMC and NUTS, two hybrid samplers. We'll work out which combination fits which kind of model.
