# Montague NLI

## Overview

A two-stage natural-language-inference architecture composed entirely out of QVR's weighted-deduction surface: a Montague-style grammar that derives a logical form for every token span, then an entailment prover that closes those logical forms under three syllogistic rules. Everything is declared in `atoms` and `binders` plus `rule` sequents; no grammar formalism and no proof system is baked into the language.

The two halves share one term language, so the prover's items are literally the terms the grammar builds. That is what lets a single Adam step move the lexicon's log-weights in response to an entailment error.

## QVR Source

```qvr
object Term : FinSet 8

deduction Montague : Term -> Term [semiring=LogProb, start=S, depth=12]
    atoms S, N, VP, Nom, Art, Cop, DetEvery, DetSome, QEvery, QSome, span, App, Var, Every, Some, dog_p, cat_p, animal_p, bark_p, walk_p, every_q, some_q, an_q, is_q
    binders Lam
    rule every_np : span(I, K, DetEvery, D), span(K, J, N, P) |- span(I, J, QEvery, P) #[learnable]
    rule some_np : span(I, K, DetSome, D), span(K, J, N, P) |- span(I, J, QSome, P) #[learnable]
    rule every_s : span(I, K, QEvery, P), span(K, J, VP, Q) |- span(I, J, S, Every(P, Q)) #[learnable]
    rule some_s : span(I, K, QSome, P), span(K, J, VP, Q) |- span(I, J, S, Some(P, Q)) #[learnable]
    rule art_n : span(I, K, Art, D), span(K, J, N, P) |- span(I, J, Nom, P) #[learnable]
    rule cop_nom : span(I, K, Cop, D), span(K, J, Nom, P) |- span(I, J, VP, P) #[learnable]
    lexicon
        "dog"    : N  = Lam(x, App(dog_p, Var(x)))      #[learnable]
        "cat"    : N  = Lam(x, App(cat_p, Var(x)))      #[learnable]
        "animal" : N  = Lam(x, App(animal_p, Var(x)))   #[learnable]
        "barks"  : VP = Lam(x, App(bark_p, Var(x)))     #[learnable]
        "walks"  : VP = Lam(x, App(walk_p, Var(x)))     #[learnable]
        "every"  : DetEvery = every_q #[learnable]
        "some"   : DetSome  = some_q  #[learnable]
        "an"     : Art      = an_q    #[learnable]
        "is"     : Cop      = is_q    #[learnable]

deduction Prover : Term -> Term [semiring=LogProb, depth=12]
    atoms Claim, Every, Some, Nonempty, App, Var
    binders Lam
    rule barbara : Claim(Every(P, Q)), Claim(Every(Q, R)) |- Claim(Every(P, R)) #[learnable]
    rule darii : Claim(Some(P, Q)), Claim(Every(Q, R)) |- Claim(Some(P, R)) #[learnable]
    rule ex_import : Claim(Every(P, Q)), Claim(Nonempty(P)) |- Claim(Some(P, Q)) #[learnable]

program fit_grammar : Term -> Term
    let chart = parse(Montague, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export fit_grammar
```

## Walkthrough

The grammar half declares the categories (`S`, `N`, `VP`, `Nom`, and the closed-class categories `Art`, `Cop`, `DetEvery`, `DetSome`), the chart-item constructor `span(I, J, X, F)` that packages a derivation covering tokens `[I, J)` of category `X` with logical form `F`, the logical-form constructors `App`, `Var`, `Every`, and `Some`, and the predicate constants `dog_p`, `cat_p`, `animal_p`, `bark_p`, `walk_p`. The `binders Lam` block tells the compiler that `Lam`'s first argument is a binding site, so every bound variable is alpha-renamed to a fresh canonical symbol per lexicon entry and structural equality on the chart is [alpha-equivalence](https://en.wikipedia.org/wiki/Lambda_calculus#Alpha_equivalence) on the surface. This matters for the prover: because "dog" is compiled once, the predicate it denotes is one canonical term, and the prover's pattern variables bind it by structural equality wherever it occurs.

### Why the sentence LF is built in normal form

One constraint on the surface drives the whole design, so it is worth stating precisely. A deduction rule's conclusion is a constructor tree over the rule's pattern variables, and the runtime instantiates it with [`instantiate`](../api/stochastic/agenda.md#quivers.stochastic.agenda.instantiate), which performs structural substitution and nothing else. There is no beta-rule anywhere: a rule conclusion cannot call a function, no chart item is normalised on the way in, and the `binders` machinery only alpha-renames lexicon logical forms at compile time.

Thus a determiner cannot denote a continuation-form lambda term such as `Lam(P, Lam(Q, App(forall_t, ...)))` and rely on later beta-reduction against its arguments. Nothing would reduce the redex; the chart would carry `App(App(Lam(...), dog_LF), bark_LF)` forever, and no prover rule could pattern-match through the unreduced application to find the quantifier.

The grammar therefore builds the sentence logical form in normal form directly, in the [generalized quantifier](https://en.wikipedia.org/wiki/Generalized_quantifier) style of [Barwise and Cooper (1981)](https://doi.org/10.1007/BF00350139): a determiner denotes a relation between two sets, written here as the binary constructors `Every(P, Q)` and `Some(P, Q)`. The determiner's quantificational force rides on its category, and the sentence rule that consumes that category (`every_s` or `some_s`) builds the matching constructor. The restrictor and the scope remain genuine lambda terms, so the binder machinery does real work at the predicate level, where it is what makes `Lam(x, App(dog_p, Var(x)))` a single canonical object rather than a name-dependent one. See [Montague (1973)](https://doi.org/10.1007/978-94-010-2506-5_10) for the type-driven compositionality this fragment instantiates.

Two consequences follow, and both are visible in the source. First, the closed-class words carry no logical form of their own: `every`, `some`, `an`, and `is` name themselves with a constant (`every_q` and friends) so that their log-weights stay learnable, while their category is what drives the rules. This is the same convention the [CCG example](ccg.md) uses. Second, the article and the copula are semantically vacuous, so `art_n` and `cop_nom` pass the noun's predicate through unchanged, which is what makes `every dog is an animal` and `every animal barks` come out as `Every(DOG, ANIMAL)` and `Every(ANIMAL, BARK)` over the *same* `ANIMAL` term.

### The prover

The prover's items are `Claim(phi)` for `phi` a sentence logical form the grammar built, plus `Nonempty(P)` for an explicit existence premise. Read the constructors as

$$\mathtt{Every}(P, Q) \;=\; \forall x.\, P(x) \to Q(x), \qquad \mathtt{Some}(P, Q) \;=\; \exists x.\, P(x) \wedge Q(x), \qquad \mathtt{Nonempty}(P) \;=\; \exists x.\, P(x).$$

All three rules are classically valid on that reading. `barbara` (every $P$ is $Q$, every $Q$ is $R$, thus every $P$ is $R$) needs no existence assumption. `darii` (some $P$ is $Q$, every $Q$ is $R$, thus some $P$ is $R$) is valid because the existential premise carries its own witness.

`ex_import` is the one that needs care. `Every(P, Q)` alone does **not** entail `Some(P, Q)`: on the standard first-order reading a universal over an empty restrictor is vacuously true while the corresponding existential is false, so `every dog barks` does not by itself license `some dog barks`. The inference is valid only under [existential import](https://en.wikipedia.org/wiki/Square_of_opposition#Existential_import), that is, only when the restrictor is known to be non-empty. The rule therefore takes `Nonempty(P)` as a second premise rather than smuggling the assumption into the rule's shape, and a caller that wants the inference has to supply that premise. The Python below supplies it explicitly, for exactly one noun, and says so.

## Try it

Every `#[learnable]` lexicon entry and every `#[learnable]` rule exposes a real `nn.Parameter` on the compiled [`DeductionSystem`](../api/stochastic/agenda.md#quivers.stochastic.agenda.DeductionSystem). The system is callable: `ded(sentence)` returns a [`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) whose [`goal_weight`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView.goal_weight) is the differentiable inside log-marginal at the start symbol.

### MAP fit on the grammar

```python
import torch
from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/montague_nli.qvr")
grammar = prog.deductions["Montague"]
prover = prog.deductions["Prover"]

corpus = [
    ["every", "dog", "barks"],
    ["some", "dog", "barks"],
    ["every", "cat", "walks"],
    ["every", "dog", "is", "an", "animal"],
    ["some", "dog", "is", "an", "animal"],
    ["every", "animal", "barks"],
]

# Touch every sentence once so the deduction's lazily allocated
# lexicon- and rule-weight ParameterDicts materialise every binding
# tuple the fit will see.
for sentence in corpus:
    grammar(sentence).goal_weight()

history = adam_fit_deduction(
    grammar, corpus, steps=200, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} -> {history[-1]:.2f}")


def sentence_lf(sentence):
    """The logical form the grammar assigns to the whole sentence,
    paired with the chart's goal weight for that derivation."""
    chart = grammar(sentence)
    target = ("span", 0, len(sentence), ("atom", "S"))
    for item, _ in chart.chart.items():
        if isinstance(item, tuple) and item[:4] == target:
            return item[4], chart.goal_weight()
    raise ValueError(f"no S-derivation for {' '.join(sentence)!r}")


for sentence in corpus:
    lf, _ = sentence_lf(sentence)
    print(f"LF({' '.join(sentence)}) = {lf}")
```

Every logical form printed there is in normal form: its head is `Every` or `Some`, never an `App` whose function is a `Lam`. That is the property the prover depends on.

### Entailment via the prover

The prover takes an axiom list of `(item, log_weight)` pairs. Seeding `Claim(premise_lf)` at the *grammar's* goal weight for that premise, rather than at a constant, is what puts the lexicon's parameters into the prover's chart: the weight of every derived claim is a sum that includes it, so a gradient at `Claim(hypothesis_lf)` reaches back through the grammar.

The NLI corpus below is three entailments and one non-entailment. The third pair is the interesting one. It is licensed only by the explicit `Nonempty(DOG)` premise, which asserts that the dog predicate has a witness; without it, `every dog barks` would not entail `some dog barks`, and the prover would (correctly) derive nothing.

```python
def noun_pred(noun):
    """The one-place predicate a common noun denotes."""
    chart = grammar([noun])
    target = ("span", 0, 1, ("atom", "N"))
    for item, _ in chart.chart.items():
        if isinstance(item, tuple) and item[:4] == target:
            return item[4]
    raise ValueError(f"{noun!r} is not a common noun in the lexicon")


# (premises, nouns-asserted-non-empty, hypothesis, label)
nli = [
    # Barbara. Valid with no existence assumption.
    ([["every", "dog", "is", "an", "animal"], ["every", "animal", "barks"]],
     [], ["every", "dog", "barks"], 1),
    # Darii. The existential premise carries its own witness.
    ([["some", "dog", "is", "an", "animal"], ["every", "animal", "barks"]],
     [], ["some", "dog", "barks"], 1),
    # Subalternation. Valid ONLY given the existence premise, which
    # is why "dog" is listed as non-empty here and nowhere else.
    ([["every", "dog", "barks"]], ["dog"], ["some", "dog", "barks"], 1),
    # Not entailed: nothing in the premises is about cats.
    ([["every", "dog", "is", "an", "animal"], ["every", "animal", "barks"]],
     [], ["every", "cat", "walks"], 0),
]


def entailment_score(premises, nonempty_nouns, hypothesis):
    """Score Claim(hypothesis_lf) against the prover's chart.

    Returns the log-weight at the hypothesis claim together with the
    log-normaliser over every claim the prover holds, so that their
    difference is log p(hypothesis | premises): a proper conditional
    distribution over the prover's conclusions. The log-weight is the
    LogProb semiring's zero (-inf) when the prover derives no such
    claim."""
    axioms = []
    for premise in premises:
        lf, log_z = sentence_lf(premise)
        axioms.append((("Claim", lf), log_z))
    for noun in nonempty_nouns:
        axioms.append((("Claim", ("Nonempty", noun_pred(noun))), torch.zeros(())))
    h_lf, _ = sentence_lf(hypothesis)
    chart = prover(axioms)
    claims = torch.stack(
        [w for item, w in chart.chart.items() if item[0] == "Claim"]
    )
    return chart.try_weight(("Claim", h_lf)), torch.logsumexp(claims, dim=0)


# Touch every prover chart once so its rule-weight ParameterDicts
# allocate every binding tuple before the optimiser reads them.
for premises, nonempty, hypothesis, label in nli:
    entailment_score(premises, nonempty, hypothesis)

params = list(grammar.parameters()) + list(prover.parameters())
optim = torch.optim.Adam(params, lr=5e-2)

history, grad_at_start = [], []
for step in range(150):
    optim.zero_grad()
    loss = torch.zeros(())
    for premises, nonempty, hypothesis, label in nli:
        score, log_norm = entailment_score(premises, nonempty, hypothesis)
        if label == 1:
            assert torch.isfinite(score), (
                f"{' '.join(hypothesis)!r} is labelled entailed but the "
                f"prover cannot derive it"
            )
            loss = loss - (score - log_norm)
        else:
            # The prover is sound on this fragment, so it derives no
            # claim for a non-entailment at all: p(entailed) is
            # exactly zero and the Bernoulli term -log(1 - 0) is
            # exactly zero. Assert the soundness rather than add a
            # term that is identically zero.
            assert not torch.isfinite(score), (
                f"{' '.join(hypothesis)!r} is labelled not entailed but the "
                f"prover derived it"
            )
    loss.backward()
    if step == 0:
        grad_at_start = [
            p for p in params if p.grad is not None and float(p.grad.abs().max()) > 0
        ]
    history.append(float(loss.detach()))
    optim.step()

print(f"NLI loss: {history[0]:.3f} -> {history[-1]:.3f}")
assert torch.isfinite(loss), "loss must be finite"
assert history[-1] < history[0], "the fit must reduce the NLI loss"

# Gradients are read at the first step, before the fit drives them
# toward zero by converging.
print(f"parameters with a non-zero gradient: {len(grad_at_start)} / {len(params)}")
assert grad_at_start, "no parameter received a gradient"

for premises, nonempty, hypothesis, label in nli:
    score, log_norm = entailment_score(premises, nonempty, hypothesis)
    p_entail = float(torch.exp(score - log_norm).detach())
    print(f"p(entailed | {' '.join(hypothesis)}) = {p_entail:.3f}  (label {label})")
```

The final loop reports a calibrated probability per pair. The non-entailment sits at exactly `0.000`, because the prover derives no claim for it at all rather than because it learned to score it low; the prover is sound on this fragment by construction, so the negative supplies no gradient. The three entailments are what the fit actually moves.

### NUTS posterior

Full Bayesian inference over the grammar's log-weights uses [`NUTSKernel`](../api/inference/mcmc.md#quivers.inference.mcmc.NUTSKernel). [`nuts_program_from_deduction`](../api/stochastic/deduction.md#quivers.stochastic.deduction.nuts_program_from_deduction) lifts every learnable parameter into a [`Normal`](../api/continuous/families.md) sample site and adds the corpus log-marginal to the joint via a `score` step, so the standard MCMC machinery applies unchanged.

```python
from quivers.inference import MCMC, NUTSKernel
from quivers.stochastic.deduction import nuts_program_from_deduction

torch.manual_seed(0)
model, x, observations = nuts_program_from_deduction(
    grammar, corpus, prior_scale=1.0,
)

kernel = NUTSKernel(step_size=0.05, max_tree_depth=4, target_accept=0.8)
mc = MCMC(kernel, num_warmup=30, num_samples=30, num_chains=2)
result = mc.run(model, x, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
```

## Categorical Perspective

Each `deduction` block denotes a weighted relation in the [agenda-based deduction semiring](../semantics/composition-rules.md): an arrow $\mathrm{Term} \to \mathrm{Term}$ in the [LogProb algebra](../semantics/algebras.md) whose underlying tensor is the chart of derivable items keyed by their derivation log-weights.

The two deductions are chained by hand rather than by a `compose(...)` step, and the reason is worth stating. `compose(D1, D2)` lifts `D1`'s *goal items* into `D2`'s axiom list, which is the right operation only when the two systems agree on an item algebra. Here they do not: the grammar's goal items are `span(0, n, S, phi)` five-tuples, while the prover's rules match `Claim(phi)`. Feeding the former to the latter yields a chart in which no rule can fire. Passing `Claim(phi)` explicitly, weighted by the grammar's goal weight, is the composite that actually typechecks, and it preserves the property that matters: the prover's score is a differentiable function of the grammar's lexicon weights, so gradient descent on an entailment error is gradient descent on the lexicon.

## Limitations

The fragment is deliberately small, and two limits are structural rather than incidental.

First, there is no beta-reduction on the deduction surface, so the grammar cannot use continuation-form determiner denotations and recover a readable logical form; it builds the normal form directly instead. A grammar that genuinely needed higher-order denotations (quantifier raising, for instance, or the scope ambiguities the [quantifier-scope example](quantifier-scope.md) encodes in its categories) would have to encode the reduction in its rules or its categories, not rely on the runtime to normalise.

Second, a rule pattern cannot mention a nullary constant inside a logical form. Rule patterns compile atoms to a tagged `("atom", name)` pair while the lexicon's logical-form evaluator emits a bare `(name,)` tuple, so a pattern like `Claim(Quant(every_t, P, Q))` never matches a chart item. That is why quantificational force is carried by the determiner's category here rather than by a constant in the logical form. The workaround costs one category per determiner, which is why the fragment has `DetEvery` and `DetSome` instead of a single `Det`.

## References

- Jon Barwise and Robin Cooper. 1981. [Generalized quantifiers and natural language](https://doi.org/10.1007/BF00350139). *Linguistics and Philosophy*, 4(2):159–219.
- Richard Montague. 1973. [The proper treatment of quantification in ordinary English](https://doi.org/10.1007/978-94-010-2506-5_10). In K. J. J. Hintikka, J. M. E. Moravcsik, and P. Suppes, editors, *Approaches to Natural Language*, pages 221–242. Springer, Dordrecht.
