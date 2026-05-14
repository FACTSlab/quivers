# Hidden Markov Model (Discrete)

## Overview

A discrete-state, discrete-emission [hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model) expressed as a V-enriched categorical network over finite sets. The initial-state, transition, and emission morphisms are row-stochastic matrices in the [Kleisli category](https://ncatlab.org/nlab/show/Kleisli+category) of the [Giry monad](https://doi.org/10.1007/BFb0092872); the row-wise [Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution) prior is set via the morphism-prior surface, and the runtime-variable `repeat` combinator threads `n_steps` transition applications before the final emission. Composition with the chosen [quantale](../api/core/quantales.md) determines whether the same morphism computes the forward marginal (product quantale) or the Viterbi path (tropical quantale).

## QVR Source

```qvr
quantale product_fuzzy

object State : 8
object Obs : 16

latent initial : State -> State ~ Dirichlet(1.0) over cod iid over dom
latent transition : State -> State ~ Dirichlet(1.0) over cod iid over dom
latent emission : State -> Obs ~ Dirichlet(1.0) over cod iid over dom

let n_step = repeat(transition) >> emission
let hmm = initial >> n_step

export hmm
```

## Walkthrough

`quantale product_fuzzy` selects the standard multiplicative composition of probabilities along paths; switching to the [tropical (max-plus) quantale](https://en.wikipedia.org/wiki/Tropical_semiring) reinterprets the same composed morphism as the Viterbi recurrence.

`object State : 4` and `object Obs : 8` are finite discrete spaces. `latent f : A -> B ~ Dirichlet(1.0) over cod iid over dom` declares `f` as a row-stochastic matrix whose every row is an independent [symmetric Dirichlet](https://en.wikipedia.org/wiki/Dirichlet_distribution#Symmetric_case) simplex draw: the event axis sits on the codomain (each row is one simplex), and the domain axis is asserted as iid (independent rows). The axis count must match the family's event rank; `Dirichlet` has event rank 1, so a single `over` axis is required.

`let n_step = repeat(transition) >> emission` is the runtime-variable Kleisli composition `T^n >> E`; `repeat` builds an n-step matrix by repeated squaring for $O(\log n)$ matrix multiplications, with $n$ supplied via `prog(n_steps=N)`. `let hmm = initial >> n_step` prepends the initial-state distribution so the exported pipeline is `1 -> Obs`, mapping no input to an n-step marginal over the observation alphabet.

## Try it

```python
import torch
from quivers.dsl import load

prog = load("docs/examples/source/hmm.qvr")
marginal_3 = prog(n_steps=3)
marginal_10 = prog(n_steps=10)
print("3-step Obs marginal:", marginal_3.tensor.shape)
print("10-step Obs marginal:", marginal_10.tensor.shape)
```

The latent matrices are torch parameters; gradient-based estimation against an observed emission histogram is a standard SVI loop using [`AutoNormalGuide`](../api/inference/guide.md). For sequence-conditional posteriors, swap `product_fuzzy` for `tropical` and read off the most-likely path.

## Categorical Perspective

The forward algorithm and the Viterbi algorithm are the same composed morphism evaluated in different [quantales](https://ncatlab.org/nlab/show/quantale): under product, composition multiplies probabilities and summation marginalises; under tropical, composition adds log-probabilities and summation maximises. Quivers makes this explicit: switching `quantale` changes the V-enriched composition rule without touching the program text.

The row-wise Dirichlet prior is the standard conjugate prior for a categorical kernel; declaring it via `over cod iid over dom` resolves the axis-role ambiguity that distinguishes a flat Dirichlet on $|State|\cdot|State|$ entries (wrong: not row-stochastic) from $|State|$ independent simplex draws (right). The categorical reading: each row of $T$ is a fibre of the dependent kernel $\prod_{c \,:\, \mathrm{State}} \mathcal{G}(\mathrm{State})$, so the prior factors as a product of independent simplex priors.
