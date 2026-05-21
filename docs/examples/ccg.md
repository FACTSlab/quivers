# Weighted Combinatory Categorial Grammar

## QVR Source

```qvr
# Weighted Combinatory Categorial Grammar
#
# A learnable weighted CCG parser declared as an agenda-based
# weighted deduction over chart items. Categories carry slash
# constructors Fwd(X, Y) = X/Y and Bwd(X, Y) = X\Y; chart items
# are span(I, J, X) triples; the semiring is LogProb for
# differentiable inside scores.
#
# Deduction:
#
#   fwd_app    : X/Y, Y       |- X           forward application
#   bwd_app    : Y,   X\Y     |- X           backward application
#   fwd_comp   : X/Y, Y/Z     |- X/Z         forward composition
#   bwd_comp   : Y\Z, X\Y     |- X\Z         backward composition
#   fwd_xcomp  : X/Y, Y\Z     |- X\Z         forward crossed composition
#   bwd_xcomp  : Y/Z, X\Y     |- X/Z         backward crossed composition
#
# CCG is the internal language of a closed monoidal category:
# the slashes are internal homs, application is the counit of
# the hom-tensor adjunction, and composition is composition of
# internal hom morphisms.

object Term : FinSet 16

deduction CCG : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms NP, S, N, VP, PP, Fwd, Bwd, span, the, cat, sleeps, barks
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    rule fwd_comp : span(I, K, Fwd(X, Y)), span(K, J, Fwd(Y, Z)) |- span(I, J, Fwd(X, Z)) #[learnable]
    rule bwd_comp : span(I, K, Bwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Bwd(X, Z)) #[learnable]
    rule fwd_xcomp : span(I, K, Fwd(X, Y)), span(K, J, Bwd(Y, Z)) |- span(I, J, Bwd(X, Z)) #[learnable]
    rule bwd_xcomp : span(I, K, Fwd(Y, Z)), span(K, J, Bwd(X, Y)) |- span(I, J, Fwd(X, Z)) #[learnable]
    lexicon
        "the" : Fwd(NP, N) = the #[learnable]
        "cat" : N = cat #[learnable]
        "sleeps" : Bwd(S, NP) = sleeps #[learnable]
        "barks" : Bwd(S, NP) = barks #[learnable]
```

## Overview

Combinatory Categorial Grammar (CCG) is expressed as an agenda-based weighted deduction whose items are chart spans `span(I, J, X)` (token range `[I, J)` carrying category `X`). The structural combinators of CCG, forward and backward application, harmonic composition, and crossed composition, each become one sequent rule. The semiring is `LogProb`, so inside scores flow as differentiable tensors back to whatever axiom / rule weights the user marks `learnable`.

## Walkthrough

`object Term : FinSet 16` declares a finite carrier for chart items; the concrete cardinality is irrelevant because the deduction reasons symbolically over constructor-tagged tuples, not over enumerated elements of `Term`.

`atoms NAME, NAME, ...` lists every identifier the rules may match literally, category atoms (`NP`, `S`, `N`, `VP`, `PP`), slash constructors (`Fwd`, `Bwd`), and the chart-item constructor (`span`). Identifiers not listed here that appear in a rule pattern are bound as wildcards; the convention is single uppercase letters (`X`, `Y`, `Z`, `I`, `J`, `K`).

Each `rule` is a sequent: premises on the left of `|-`, conclusion on the right. `Fwd(X, Y)` constructs the forward-slash category `X/Y`; `Bwd(X, Y)` constructs the backward-slash category `X\Y`. Adjacent spans whose end / start indices agree fire whichever rule's pattern matches their categories.

`semiring LogProb` selects log-space inside scores. `start S` declares the goal category for a successful parse. `depth 6` bounds derivation depth to keep the agenda finite.

## DSL Features

- **`deduction { … }` block**: declares the agenda-based weighted deduction in a single record. The block's seven irreducible parameters, item algebra (via `atoms`), rule set, semiring, axiom source, goal predicate, start symbol, depth bound, are field-by-field.
- **`atoms NAME, NAME, ...`**: closes the constructor universe. Every identifier appearing in a rule pattern must be either an atom or a single-uppercase wildcard variable.
- **Sequent rules**: arbitrary-arity premises on the left of `|-`, single conclusion on the right; rules with one premise are unary chart rules, with two are binary, and so on.
- **Slash constructors**: `Fwd(X, Y)` and `Bwd(X, Y)` are user-declared atoms, not built-in syntax. The combinators are theorems in this presentation.

## Try it

Every `#[learnable]` lexicon entry and every `#[learnable]` rule
exposes a real `nn.Parameter` on the compiled
[`DeductionSystem`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem). The
system is callable: `ded(sentence)` returns a
[`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) whose
`goal_weight()` is the differentiable log-marginal
$\log Z(s; \mathbf{w}) = \log \sum_d \exp \langle \mathbf{w}, \phi(d) \rangle$
summed over every derivation $d$ that the start symbol licenses for
the input. Fitting the lexicon and rule weights together is then a
regression-style problem: minimise $-\sum_n \log Z(s_n)$ over a
corpus of sentences. The
[`quivers.stochastic.deduction`](../api/stochastic/deduction.md) module ships the
two standard surfaces.

### MAP fit (Adam on rule & lexicon weights)

```python
import torch
from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction, sample_corpus

torch.manual_seed(0)
prog = load("docs/examples/source/ccg.qvr")
ded  = prog.deductions["CCG"]

corpus = [["the", "cat", "sleeps"], ["the", "cat", "barks"]]

history = adam_fit_deduction(
    ded, corpus, steps=300, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} → {history[-1]:.2f}")  # strictly decreasing

# Forward-sample under the fitted parameters and check the
# dominant length-3 yield recovers the training corpus.
draws = sample_corpus(ded, length=3, n_samples=32, seed=0)
print("dominant yield:", max(set(map(tuple, draws)), key=draws.count))
# → ("the cat sleeps",)
```

`adam_fit_deduction` maximises the corpus log-marginal under an
optional Normal prior on the parameters; `prior_scale=1.0` gives MAP
under a unit Normal. `sample_corpus` enumerates yields of the chosen
length and draws from the categorical defined by their chart weights; exact forward sampling because the chart marginalises the
derivation forest.

### NUTS (full Bayesian posterior)

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel
from quivers.stochastic.deduction import nuts_program_from_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/ccg.qvr")
ded  = prog.deductions["CCG"]

corpus = [["the", "cat", "sleeps"], ["the", "cat", "barks"]]

model, x, observations = nuts_program_from_deduction(
    ded, corpus, prior_scale=1.0,
)

kernel = NUTSKernel(step_size=0.1, max_tree_depth=4, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=50, num_samples=50, num_chains=2)
result = mc.run(model, x, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
posterior_means = {
    name: float(samples.mean()) for name, samples in result.samples.items()
}
print("posterior mean log-weights:", posterior_means)
```

`nuts_program_from_deduction` lifts every learnable parameter of the
deduction into a [`Normal(0, σ)`](../api/continuous/families.md) sample
site and adds the corpus log-marginal $\log Z$ to the joint via a
[`score`](../api/program.md) step. The standard
[`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) drives the
posterior $p(\mathbf{w} \mid s_1, \ldots, s_N) \propto p(\mathbf{w})
\cdot \prod_n Z(s_n; \mathbf{w})$. The same Bayesian object
[`bayesian_regression`](bayesian-regression.md) fits, with the chart
total in place of the Gaussian likelihood.

## Categorical Perspective

CCG is the internal language of a closed monoidal category. The forward slash `X/Y` and backward slash `X\Y` are internal hom-objects (exponentials); the application rule is the counit of the hom-tensor adjunction, `[Y, X] ⊗ Y → X`. Composition corresponds to chaining adjunctions: given `X/Y` and `Y/Z`, transitivity yields `X/Z`. Crossed composition relies on a braiding isomorphism to swap argument order. The type of an expression completely determines what it can combine with, because the closed structure forces all combination to go through the adjunction.

## Semiring Selection

The choice of semiring affects the parser's behavior: `LogProb` accumulates inside log-probabilities (numerically stable, differentiable); `Viterbi` returns the highest-weight derivation; `Counting` counts distinct derivations; `Boolean` checks membership without weights. The same deduction block serves all four objectives via the `semiring` field.
