# Type-Logical Grammar (Lambek Calculus)

## QVR Source

```qvr
# Weighted Type-Logical Grammar
#
# The non-commutative Lambek calculus L expressed as an
# agenda-based weighted deduction. Categories carry slash (Fwd,
# Bwd) and tensor (Tns) constructors; chart items are span(I,
# J, X) triples.
#
# Deduction:
#
#   right_app    : X/Y, Y    |- X              forward application
#   left_app     : Y,   X\Y  |- X              backward application
#   tensor_intro : A,   B    |- Tns(A, B)      product introduction
#   tensor_left  : Tns(A, B) |- A              left projection
#   tensor_right : Tns(A, B) |- B              right projection
#
# The Lambek calculus is resource-sensitive (every hypothesis is
# used exactly once) and order-preserving (no permutation). It
# is the internal logic of a biclosed monoidal category; the
# residuation law A (x) B |- C iff A |- C/B iff B |- A\C makes
# L the free residuated monoid over its atoms.

object Term : FinSet 16

deduction Lambek : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, VP, PP, Fwd, Bwd, Tns, span, every, dog, barks
    rule right_app : span(I, K, Fwd(A, B)), span(K, J, B) |- span(I, J, A) #[learnable]
    rule left_app : span(I, K, B), span(K, J, Bwd(A, B)) |- span(I, J, A) #[learnable]
    rule tensor_intro : span(I, K, A), span(K, J, B) |- span(I, J, Tns(A, B)) #[learnable]
    rule tensor_left : span(I, J, Tns(A, B)) |- span(I, J, A) #[learnable]
    rule tensor_right : span(I, J, Tns(A, B)) |- span(I, J, B) #[learnable]
    lexicon
        "every" : Fwd(NP, N) = every #[learnable]
        "dog" : N = dog #[learnable]
        "barks" : Bwd(S, NP) = barks #[learnable]
```

## Overview

Type-logical grammar, grounded in the non-commutative Lambek calculus, is a resource-conscious approach to syntax: every hypothesis is used exactly once and argument order is preserved. The deduction above lists slash (`Fwd`, `Bwd`) and tensor (`Tns`) constructors over chart-spans `span(I, J, X)`, and licenses right / left application, product introduction, and product elimination as sequent rules.

## Walkthrough

`atoms NAME, NAME, ...` declares the constructor vocabulary. Category atoms are `S`, `NP`, `N`, `VP`, `PP`; structural constructors are `Fwd(A, B) ≡ A/B`, `Bwd(A, B) ≡ A\B`, `Tns(A, B) ≡ A⊗B`. The chart-item constructor `span(I, J, X)` packages a derivation covering tokens `[I, J)` carrying category `X`. Single-uppercase identifiers (`A`, `B`, `I`, `J`, `K`) appearing in rule patterns bind as wildcards.

The rules realize the four logical core operations of the Lambek calculus:

- **`right_app`**: modus ponens for forward slash: `A/B, B ⊢ A`.
- **`left_app`**: modus ponens for backward slash: `B, A\B ⊢ A`.
- **`tensor_intro`**: product introduction: adjacent derivations of `A` and `B` combine into a derivation of `A⊗B`.
- **`tensor_left` / `tensor_right`**: product elimination: a derivation of `A⊗B` projects to derivations of either component over the same span.

Together these rules yield the equational theory of the residuated monoid. The agenda runs to depth 6 by default; the `LogProb` semiring accumulates inside log-probabilities that flow back as gradients to learnable axiom weights.

## DSL Features

- **Sequent rules with arbitrary arity**: rule bodies declare premises on the left of `|-` and a single conclusion on the right; the compiler routes unary patterns to unary chart cells and binary patterns to binary chart cells.
- **Resource sensitivity is structural**: there is no contraction or weakening rule, so every premise in a sequent must match a distinct chart cell.
- **Order preservation**: pattern variables appear in textual order; the parser enforces left-to-right span composition.
- **Tensor and slash as user atoms**: there is no special syntax, `Tns`, `Fwd`, `Bwd` are atoms declared in the `atoms NAME, NAME, ...` block and may be replaced or extended by the user.

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
prog = load("docs/examples/source/type_logical.qvr")
ded  = prog.deductions["Lambek"]

corpus = [["every", "dog", "barks"]] * 4

history = adam_fit_deduction(
    ded, corpus, steps=300, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} → {history[-1]:.2f}")  # strictly decreasing

# Forward-sample under the fitted parameters and check the
# dominant length-3 yield recovers the training corpus.
draws = sample_corpus(ded, length=3, n_samples=32, seed=0)
print("dominant yield:", max(set(map(tuple, draws)), key=draws.count))
# → ("every dog barks",)
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
prog = load("docs/examples/source/type_logical.qvr")
ded  = prog.deductions["Lambek"]

corpus = [["every", "dog", "barks"]] * 4

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

The Lambek calculus is the internal language of a residuated monoidal category (biclosed monoidal category). The tensor `⊗` is the monoidal product; the two slashes are its left and right adjoints. The residuation laws

```
A ⊗ B  ⊢  C   iff   A  ⊢  C/B   iff   B  ⊢  A\C
```

are the statement of that adjunction. Because there is no contraction (copying) or weakening (discarding), every derivation consumes its input span exactly once; the agenda's span indexing enforces this by attaching each item to a single token range.

## Connections to Other Formalisms

The Lambek calculus is strictly more expressive than context-free grammar (handling extraction, gapping) but remains decidable and efficiently parseable. Compared to CCG it is more restricted: CCG implicitly permits structural rules (weakening, contraction) that the Lambek calculus does not. The multimodal extensions (see [multimodal-tlg](multimodal-tlg.md)) introduce controlled structural operators that license specific deviations from strict linearity.
