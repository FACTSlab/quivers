# Montague NLI

## Overview

A two-stage natural-language-inference architecture composed entirely out of QVR's weighted-deduction surface: a Montague-style grammar that derives logical forms from token spans, then an entailment prover that closes the LFs under modus ponens and functorial substitution. Everything is declared in `atoms` and `binders` plus `rule` sequents; no grammar formalism or proof system is baked into the language.

## QVR Source

```qvr
object Term : FinSet 8

deduction Montague : Term -> Term [semiring=LogProb, start=S, depth=10]
    atoms NP, S, N, Fwd, Bwd, span, App, Var, dog_p, cat_p, animal_p, bark_p, walk_p, forall_t, exists_t, implies_t, and_t
    binders Lam
    rule fwd_app : span(I, K, Fwd(X, Y), F), span(K, J, X, A) |- span(I, J, Y, App(F, A)) #[learnable]
    rule bwd_app : span(I, K, X, A), span(K, J, Bwd(X, Y), F) |- span(I, J, Y, App(F, A)) #[learnable]
    lexicon
        "dog"    : N          = Lam(x, App(dog_p, Var(x)))      #[learnable]
        "cat"    : N          = Lam(x, App(cat_p, Var(x)))      #[learnable]
        "animal" : N          = Lam(x, App(animal_p, Var(x)))   #[learnable]
        "barks"  : Bwd(NP, S) = Lam(x, App(bark_p, Var(x)))     #[learnable]
        "walks"  : Bwd(NP, S) = Lam(x, App(walk_p, Var(x)))     #[learnable]
        "every"  : Fwd(N, Fwd(Bwd(NP, S), S)) = Lam(P, Lam(Q, App(forall_t, Lam(x, App(App(implies_t, App(Var(P), Var(x))), App(Var(Q), Var(x))))))) #[learnable]
        "some"   : Fwd(N, Fwd(Bwd(NP, S), S)) = Lam(P, Lam(Q, App(exists_t, Lam(x, App(App(and_t, App(Var(P), Var(x))), App(Var(Q), Var(x))))))) #[learnable]

deduction Prover : Term -> Term [semiring=LogProb, depth=8]
    atoms Claim, Implies, App, Var
    binders Lam
    rule modus_ponens : Claim(P), Claim(Implies(P, Q)) |- Claim(Q) #[learnable]
    rule app_subst : Claim(App(F, X)), Claim(Implies(X, Y)) |- Claim(App(F, Y)) #[learnable]

program fit_grammar : Term -> Term
    let chart = parse(Montague, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

program fit_pipeline : Term -> Term
    let pipeline = compose(Montague, Prover)
    let chart = parse(pipeline, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export fit_grammar
```

## Walkthrough

The grammar half of the module declares atomic category constructors (`NP`, `S`, `N`), slash constructors (`Fwd`, `Bwd`), the chart-item constructor `span(I, J, X, F)` that packages a derivation covering tokens `[I, J)` of category `X` with logical form `F`, a function-application LF combinator `App`, a variable-occurrence constructor `Var`, the predicate constants `dog_p`, `cat_p`, `animal_p`, `bark_p`, `walk_p`, and the logical connectives `forall_t`, `exists_t`, `implies_t`, `and_t` used in determiner denotations. The `binders Lam` block tells the compiler that `Lam`'s first argument is a binding site, so every bound variable is alpha-renamed to a fresh canonical symbol per term construction and structural equality on the chart is [alpha-equivalence](https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence) on the surface.

Forward and backward application (`fwd_app` and `bwd_app`) are the two sequents that combine a slash-typed span with its complement, building the conclusion's logical form by applying the head's LF to its argument's LF via `App`. The lexicon ships learnable log-weights per entry: common nouns and intransitive verbs are unary predicates `Lam(x, App(p, Var(x)))`, and determiners are honest [generalised quantifiers](https://en.wikipedia.org/wiki/Generalized_quantifier) in continuation form, with bound variables `P`, `Q`, `x` introduced by `Lam` and substituted by the application rules. See [Montague (1973)](https://doi.org/10.1007/978-94-010-2506-5_10) for the type-driven semantic compositionality this fragment instantiates.

The prover half closes the resulting `Claim` items under modus ponens and a functorial-substitution rule: from `Claim(App(F, X))` and `Claim(Implies(X, Y))`, conclude `Claim(App(F, Y))`. Both deductions share the `binders Lam` declaration so capture-avoiding substitution lives at the compile-time canonical-variable layer.

The two top-level programs expose the grammar and the composed pipeline as differentiable parsers. `fit_grammar` calls `parse(Montague, sentence)` against the host-supplied token list and `score`s the chart's goal weight (the inside log-marginal) into the program. `fit_pipeline` uses `compose(Montague, Prover)` to feed the grammar's logical forms into the prover's chart, so the prover's goal weight inherits the grammar's gradient flow into the lexicon's log-weights.

## Try it

The grammar denotes a sentence as a logical form: a structured term over the user-declared free term algebra. Lambda terms with bound variables (`Lam(x, App(bark_p, Var(x)))`) are honest object-level lambda calculus: the `binders Lam` declaration tells the compiler that `Lam`'s first argument is a binding site, so every bound variable is alpha-renamed to a fresh canonical symbol at compile time and the chart's structural identity collapses alpha-equivalent terms.

### SVI

```python
import torch
from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/montague_nli.qvr")
grammar = prog.deductions["Montague"]
prover  = prog.deductions["Prover"]

# Training corpus: pairs of (premise, hypothesis) sentences whose
# joint NLI label we want the grammar to fit. The Montague chart
# derives a logical form for each sentence; the prover then closes
# the resulting Claims under modus ponens to check entailment.
corpus = [
    (["every", "dog", "barks"], ["some", "dog", "barks"]),
    (["every", "cat", "walks"], ["some", "cat", "walks"]),
]

# Touch every premise + hypothesis once so the deduction's lazy
# rule-weight ParameterDicts allocate every binding tuple they
# will see during fitting.
for premise, hypothesis in corpus:
    grammar(premise).goal_weight()
    grammar(hypothesis).goal_weight()

# Adam over the grammar's lexicon + rule log-weights. The loss is
# minus the corpus log-marginal: a higher chart total at the start
# symbol of each sentence under the current parameters.
all_sentences = [s for pair in corpus for s in pair]
history = adam_fit_deduction(
    grammar, all_sentences, steps=200, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} -> {history[-1]:.2f}")

for sentence in all_sentences:
    lf = next(
        item[4] for item, _ in grammar(sentence).chart.items()
        if isinstance(item, tuple) and item[:1] == ("span",)
           and len(item) >= 5 and item[3] == ("atom", "S")
    )
    print(f"LF({' '.join(sentence)}) = {lf}")
```

### Entailment via the Prover (full NLI loss)

The prover deduction closes the Claims under modus ponens and a functorial substitution rule. An NLI label is the prover's goal weight at `Claim(hypothesis_lf)` after seeding the chart with `Claim(premise_lf)`.

```python
import torch
from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/montague_nli.qvr")
grammar = prog.deductions["Montague"]
prover  = prog.deductions["Prover"]

def derive_lf(sentence):
    """Run the grammar; return the logical form at span(0, n, S)."""
    chart = grammar(sentence)
    n = len(sentence)
    target = ("span", 0, n, ("atom", "S"))
    for item, _ in chart.chart.items():
        if isinstance(item, tuple) and item[:4] == target:
            return item[4] if len(item) > 4 else None
    return None

def entailment_log_score(premise, hypothesis):
    """Compute log P(prover derives Claim(hypothesis_lf) given
    premise_lf as an axiom). The prover's goal weight at the
    hypothesis Claim is the NLI score; gradients flow back into the
    grammar's lexicon and rule weights."""
    p_lf = derive_lf(premise)
    h_lf = derive_lf(hypothesis)
    if p_lf is None or h_lf is None:
        return torch.tensor(float("-inf"))
    # Seed the prover with the premise as an axiom; the prover
    # closes the chart and reports the weight at Claim(h_lf).
    prover_chart = prover([(("Claim", p_lf), torch.tensor(0.0))])
    return prover_chart.try_weight(("Claim", h_lf))

# A toy entailment corpus: (premise, hypothesis, label) triples
# where label = 1 when the hypothesis is entailed.
nli = [
    (["every", "dog", "barks"], ["some", "dog", "barks"], 1),
    (["every", "cat", "walks"], ["some", "cat", "walks"], 1),
    (["every", "dog", "barks"], ["every", "cat", "walks"], 0),
]

params = list(grammar.parameters()) + list(prover.parameters())
optim  = torch.optim.Adam(params, lr=5e-2)
for step in range(150):
    optim.zero_grad()
    loss = torch.zeros(())
    for premise, hypothesis, label in nli:
        score = entailment_log_score(premise, hypothesis)
        if torch.isfinite(score):
            # Bernoulli NLI loss: -log p(label | score).
            log_p_entailed = score
            log_p_not = torch.log1p(-torch.exp(score)).clamp(min=-50.0) \
                if score.item() < 0 else torch.tensor(-50.0)
            loss = loss - (label * log_p_entailed + (1 - label) * log_p_not)
    loss.backward()
    optim.step()

print("final loss:", float(loss.detach()))
```

### NUTS posterior

Full Bayesian inference uses [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel) over the same model. For models declaring explicit `sample` priors NUTS samples them directly; models whose latents are `[role=latent]` parameters are lifted into a Normal-prior Bayesian model with [`bayesian_lift_parameters`](../api/inference/lifts.md#quivers.inference.lifts.bayesian_lift_parameters) so the standard MCMC machinery applies uniformly.

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel

torch.manual_seed(0)
prog = load("docs/examples/source/montague_nli.qvr")
model = prog.morphism

# Construct ``x`` and ``observations`` exactly as in the SVI block
# above. For models with no explicit ``sample`` priors, lift the
# parameters into a Bayesian model under unit Normal priors:
#   from quivers.inference import bayesian_lift_parameters
#   model, x, observations = bayesian_lift_parameters(
#       model, x, observations, prior_scale=1.0,
#   )

kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=30, num_samples=30, num_chains=2)
result = mc.run(model, x, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
```


## Categorical Perspective

Each `deduction` block denotes a weighted relation in the [agenda-based deduction semiring](../semantics/composition-rules.md): an arrow $\mathrm{Term} \to \mathrm{Term}$ in the [LogProb algebra](../semantics/algebras.md) whose underlying tensor is the chart of derivable items keyed by their derivation log-weights. Composing grammar with prover is composition in the same enriched category, so the gradient of the prover's goal weight flows back through the grammar's lexicon entries during training.


## References

- Richard Montague. 1973. The proper treatment of quantification in ordinary English. In K. J. J. Hintikka, J. M. E. Moravcsik, and P. Suppes, editors, *Approaches to Natural Language*, pages 221–242. Springer, Dordrecht.
