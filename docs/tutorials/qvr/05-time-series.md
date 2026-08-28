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
object School : FinSet 8
theta : School <- Normal(mu, tau)
```

is the QVR analogue of NumPyro's `with plate("schools", 8): theta = sample("theta", dist.Normal(mu, tau))`. The result has shape `(8,)`; subsequent `let` arithmetic broadcasts over it. A *vectorized observe* over a plate has the same shape:

<!-- compile: false -->
```qvr
observe y : School <- Normal(theta, sigma_j)
```

Plates are good for IID structure: nothing about index `j+1` depends on what happened at `j`. For genuinely sequential data, you want `scan`.

## `scan` for left-to-right evaluation

`scan` takes a *cell* whose signature is `Input * Hidden -> Hidden` and lifts it to operate along the sequence dimension of an input tensor. The `*` in `Input * Hidden -> Hidden` is "the cell takes two inputs in parallel (an input vector and the previous hidden state) and returns one". That's all you need to read it; the categorical reading (binary product in a monoidal category) is in chapter 7. The cell may be a `kernel ... ~ Family` morphism (one-step state update under a Gaussian transition) or a `program` block (one-step update with its own random draws).

```text
object Token : FinSet 256
object Embedded : Real 64
object Hidden : Real 128
object Output : Real 64

morphism tok_embed : Token -> Embedded [role=embed]

morphism cell : Embedded * Hidden -> Hidden ~ Normal
morphism output_proj : Hidden -> Output ~ Normal
define rnn = tok_embed >> scan(cell) >> output_proj
export rnn
```

The pipeline reads as: embed each token to a 64-dim vector, fold the cell over the sequence dimension (with zero or learned initial state), project the final hidden state to a 64-dim output. The compiler infers the sequence axis from the input tensor's second dimension at evaluation time.

If you've used Haskell's `mapAccumL` or NumPy's `np.cumsum`, this is the same idea generalized to a learnable cell.

## A discrete hidden Markov model

An HMM ([Rabiner, 1989](https://doi.org/10.1109/5.18626)) normally factors as an initial distribution, a row-stochastic transition kernel, and a row-stochastic emission kernel. The following block is the finite-relation demonstration exported by `docs/examples/source/hmm.qvr`:

```qvr
composition product_fuzzy [level=algebra]
object State : FinSet 8
object Obs : FinSet 16
morphism initial : State -> State [role=latent]
morphism transition : State -> State [role=latent]
morphism emission : State -> Obs [role=latent]
define n_step = repeat(transition) >> emission
define hmm    = initial >> n_step

export hmm
```

Two limitations matter:

- The three declarations have no `Dirichlet` families, and the module uses `product_fuzzy`, so their learned tensors are sigmoid-constrained fuzzy relations rather than row-stochastic kernels. This exported `hmm` should thus be read as a path-composition example, not as a normalized HMM likelihood.
- `repeat(transition)` repeats composition of the relation before the emission. Its output is determined by the selected algebra; under `product_fuzzy` it is neither a sum-product forward probability nor a Viterbi score.

### Axis-role syntax in 60 seconds

The clause `~ Family(args) over <axis> [iid over <axis>]` attaches a prior to a kernel `A -> B`. The axes name *which* dimensions of the kernel tensor the family describes and *which* dimensions are replicated independently.

- `over cod` puts the family on the codomain axis. For `transition : State -> State` declared as `~ Dirichlet(1.0) over cod`, each draw is a simplex over the 8 codomain states.
- `iid over dom` says the prior is replicated independently across the domain axis. So `over cod iid over dom` gives 8 independent simplex draws, one per domain row, which is exactly a row-stochastic matrix with a row-wise Dirichlet prior.
- Order matters because the family is built outward: first you say what one row looks like (`over cod`), then you say how rows replicate (`iid over dom`).

Other useful combinations:

| Pattern | Meaning |
|---|---|
| `~ Normal(0, 1) over cod` | one Gaussian vector over the codomain (no replication) |
| `~ Normal(0, 1) over cod iid over dom` | per-row independent Gaussian vectors |
| `~ Normal(0, 1) iid over cod iid over dom` | full IID matrix of scalars |
| `~ Dirichlet(0.1) over cod iid over dom` | sparse row-stochastic prior (concentration < 1) |

If neither `over` nor `iid over` is given, the family is broadcast scalar-wise. The compiler synthesises the appropriate `PlateDraw` internally; you don't need to think about plates.

The separate `hmm_program` in the example source demonstrates row-wise `Dirichlet` draws and categorical observations. Its current marginalized body uses `initial_row` and `emission_rows` but does not use the sampled `transition_rows`, so it is not yet a multi-step HMM likelihood.

## State-space models

For continuous-state sequences, the per-step transition and emission can be Gaussian kernels. The following state-space demonstration appears in `docs/examples/source/linear_gaussian_ssm.qvr`:

```text
object Driver : Real 2
object State : Real 4
object Obs : Real 2

morphism transition_cell : Driver * State -> State ~ Normal
morphism emission : State -> Obs ~ Normal
morphism filter_cell : Obs * State -> State ~ Normal
define generate = scan(transition_cell) >> emission
define filter = scan(filter_cell)

export filter
```

`scan(transition_cell)` folds the per-step Gaussian transition along the sequence dimension; composing with `emission` produces the generative path. `filter` scans a separately learned Gaussian kernel over observations and states. The source does not implement the closed-form Kalman update ([Kalman, 1960](https://doi.org/10.1115/1.3662552)).

For a fully nonlinear (deep) variant where transition and emission are neural Gaussians, see `docs/examples/source/continuous_hmm.qvr` and `docs/examples/source/deep_markov.qvr`.

## Try this

- Add `Dirichlet` row priors to a normalized HMM program, then run NUTS and inspect R-hat for its latent parameters.
- Make the linear-Gaussian SSM hierarchical: each sequence has its own transition cell drawn from a hyperprior. (Chapter 3's plate-draw applies.)
- Swap the linear `transition_cell` for the deep-Markov nonlinear kernel and compare ELBO convergence.

## Next

[Chapter 6](06-inference-zoo.md) surveys the inference algorithms: eleven concrete guide classes, seven objectives, HMC and NUTS, and two hybrid approaches.


## References

- Lawrence R. Rabiner. 1989. A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2):257–286.
- Rudolf E. Kalman. 1960. A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1):35–45.
