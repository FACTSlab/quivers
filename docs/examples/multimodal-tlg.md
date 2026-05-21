# Multimodal Type-Logical Grammar

## QVR Source

```qvr
# Multimodal Type-Logical Grammar
#
# An extension of the Lambek calculus with unary type
# constructors Dia and Box that form a residuated pair (Dia
# adjoint to Box). The structural rules associated with each
# modality are controlled by the grammar's structural component;
# here we license modal introduction and elimination, leaving
# additional structural postulates as further user-added rules.
#
# Deduction:
#
#   right_app : X/Y, Y     |- X            forward application
#   left_app  : Y,   X\Y   |- X            backward application
#   dia_intro : A          |- Dia(A)       modal introduction
#   dia_elim  : Dia(A)     |- A            modal elimination
#
# Categories carry slash (Fwd, Bwd) and modal (Dia, Box)
# constructors; chart items are span(I, J, X) triples.

object Term : FinSet 16

deduction MMTLG : Term -> Term [semiring=LogProb, start=S, depth=3, tolerance=1e-5]
    atoms S, NP, N, VP, PP, Fwd, Bwd, Dia, Box, span, the, dog, barks
    rule right_app : span(I, K, Fwd(A, B)), span(K, J, B) |- span(I, J, A) #[learnable]
    rule left_app : span(I, K, B), span(K, J, Bwd(A, B)) |- span(I, J, A) #[learnable]
    rule dia_intro : span(I, J, A) |- span(I, J, Dia(A)) #[learnable, bounded]
    rule dia_elim : span(I, J, Dia(A)) |- span(I, J, A) #[learnable, bounded]
    lexicon
        "the" : Fwd(NP, N) = the #[learnable]
        "dog" : N = dog #[learnable]
        "barks" : Bwd(S, NP) = barks #[learnable]
```

## Overview

Multimodal type-logical grammar (Moortgat 1997) extends the Lambek calculus with unary modal type constructors `Dia` (`◇`) and `Box` (`□`) that form a residuated pair (`◇ ⊣ □`). The structural rules associated with each modality are controlled by the grammar's structural component; the deduction above licenses base right / left application together with modal introduction and elimination, leaving further structural postulates as user-added rules.

## Walkthrough

`atoms NAME, NAME, …` declares category atoms (`S`, `NP`, `N`, `VP`, `PP`), slash constructors (`Fwd`, `Bwd`), and modal constructors (`Dia`, `Box`). Chart items are `span(I, J, X)` triples.

- **`right_app` / `left_app`**: modus ponens for forward / backward slash, exactly as in the base Lambek calculus.
- **`dia_intro`**: modal introduction: a derivation of `A` lifts to a derivation of `Dia(A)` over the same span. This is the unit of the modality's monadic structure on the category of formulas.
- **`dia_elim`**: modal elimination: a derivation of `Dia(A)` projects back to a derivation of `A` over the same span. Combined with structural rules licensed under the modality, `dia_elim` is what permits controlled exchange / weakening / contraction within modal-marked subderivations.

A richer fragment would add explicit modal structural rules (modal exchange, modal contraction) and the `Box` introduction / elimination duals; both are sequent rules in the same style.

## DSL Features

- **Sequent rules with arbitrary arity**: rule premises can be unary or binary; the agenda dispatches each pattern to the appropriate chart-cell shape.
- **Modal constructors as atoms**: `Dia` and `Box` are user-declared atoms in the same vocabulary as the slash constructors. There is no built-in modal syntax.
- **Depth bounding**: the `depth=3` agenda bound plus the `#[bounded]` marker on the modal rules keeps modal nesting finite; without it the category space is unbounded (every `A` admits `Dia(A)`, `Dia(Dia(A))`, and so on).

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
prog = load("docs/examples/source/multimodal_tlg.qvr")
ded  = prog.deductions["MMTLG"]

corpus = [["the", "dog", "barks"]] * 4

history = adam_fit_deduction(
    ded, corpus, steps=300, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} → {history[-1]:.2f}")  # strictly decreasing

# Forward-sample under the fitted parameters and check the
# dominant length-3 yield recovers the training corpus.
draws = sample_corpus(ded, length=3, n_samples=32, seed=0)
print("dominant yield:", max(set(map(tuple, draws)), key=draws.count))
# → ("the dog barks",)
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
prog = load("docs/examples/source/multimodal_tlg.qvr")
ded  = prog.deductions["MMTLG"]

corpus = [["the", "dog", "barks"]] * 4

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

The diamond `◇` acts as a monad on the category of formulas: its unit `η : X → ◇X` (modal introduction) lifts any formula into the modal regime, and its multiplication `μ : ◇(◇X) → ◇X` (idempotence) collapses nested modals when the corresponding structural rule is licensed. The standard Lambek calculus is the internal language of a residuated monoidal category with no extra structure. Adding the diamond monad and licensing modal structural rules permits controlled relaxation of resource sensitivity: inside the monad, structural rules like exchange, weakening, and contraction become available without contaminating the surrounding linear context.

## Linguistic Applications

Multimodal type-logical grammar handles cases where strict linearity must be relaxed:

- **Extraction and long-range dependencies**: extracted elements marked as modal can permute with intermediate functors, threading through multiple clause boundaries (wh-questions, relative clauses).
- **Non-constituent coordination**: modal operators can license extracting a common context, letting two fragments coordinate even if they are not standard constituents.
- **Gapping**: a gapped functor with a modal type can be reused across a coordination boundary.
