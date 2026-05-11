# Event-Structure Latent-Class Model

## Overview

A four-class latent-class model (telicity × durativity) over cloze and proportion responses, with crossed random intercepts on subject, verb, sense, and item, and an ordinal monotone spline for duration. The program exercises every Bayesian-modelling construct in the DSL: plate-draws, a parametric `random_intercepts` template instantiated eight times, an ordinal monotone spline via `cumsum` of `HalfNormal` increments, vectorised observes against a runtime `observations` dict, and a program-level `marginalize` step for the discrete latent class.

## QVR Source

```qvr
object Verb : 100
object Sense : 50
object Item : 1000
object SubjCloze : 200
object SubjProp : 200
object RespCloze : 5000
object RespProp : 5000

program random_intercepts (G : FinSet, scale : Real) : G -> 1
    draw sigma ~ HalfNormal(scale)
    draw v : G -> 1 ~ Normal(0.0, sigma)
    return v

program event_structure : Item -> Item
    draw prob_durative ~ Uniform(0.0, 1.0)
    draw prob_telic_given_dur ~ Beta(10.0, 1.0)
    draw prob_telic_given_nodur ~ Beta(1.0, 1.0)

    draw intercept_cloze ~ Normal(0.0, 1.0)
    draw intercept_prop ~ Normal(0.0, 1.0)
    draw telicity_coef_cloze ~ HalfNormal(1.0)
    draw telicity_coef_prop ~ HalfNormal(1.0)
    draw durativity_coef_prop ~ HalfNormal(1.0)

    draw by_subj_cloze  ~ random_intercepts(SubjCloze, 1.0)
    draw by_verb_cloze  ~ random_intercepts(Verb,      1.0)
    draw by_sense_cloze ~ random_intercepts(Sense,     1.0)
    draw by_item_cloze  ~ random_intercepts(Item,      1.0)

    draw by_subj_prop   ~ random_intercepts(SubjProp,  1.0)
    draw by_verb_prop   ~ random_intercepts(Verb,      1.0)
    draw by_sense_prop  ~ random_intercepts(Sense,     1.0)
    draw by_item_prop   ~ random_intercepts(Item,      1.0)

    draw duration_incr_cloze : Item -> 11 ~ HalfNormal(1.0)
    draw duration_incr_prop  : Item -> 11 ~ HalfNormal(1.0)

    let duration_eff_cloze = cumsum(duration_incr_cloze)
    let duration_eff_prop  = cumsum(duration_incr_prop)

    observe cloze_resp[n] ~ Bernoulli(intercept_cloze) for n in RespCloze
    observe prop_resp[n]  ~ Bernoulli(intercept_prop)  for n in RespProp

    marginalize cloze_resp
    marginalize prop_resp

    return intercept_cloze

output event_structure
```

## Walkthrough

### Cardinalities

The cardinalities of `Verb`, `Sense`, `Item`, `SubjCloze`, `SubjProp`, `RespCloze`, and `RespProp` are fit-time-determined placeholders for the static parser; they bound the indexing sets for the plates and the vectorised observes but do not constrain the actual response tensors at runtime.

### Latent classes

The class prior factors through a cell parameterisation: $\mathrm{prob\_durative}$ marginalises durativity, and the two conditionals $\mathrm{prob\_telic\_given\_dur}$ and $\mathrm{prob\_telic\_given\_nodur}$ give telicity given durativity. The four cells of $\{\pm\mathrm{telic}\} \times \{\pm\mathrm{durative}\}$ are recovered as products of these factors.

### Parametric random-intercepts template

```qvr
program random_intercepts (G : FinSet, scale : Real) : G -> 1
    draw sigma ~ HalfNormal(scale)
    draw v : G -> 1 ~ Normal(0.0, sigma)
    return v
```

The template denotes a dependent kernel

$$
\llbracket \mathsf{random\_intercepts} \rrbracket \;:\; \prod_{G : \mathbf{FinSet}} \prod_{\mathrm{scale} : \mathbb{R}_{>0}} \mathbf{Kern}(G, \mathbf{1}),
$$

a half-normal scale hyperprior followed by a $G$-indexed Normal-$(0, \sigma)$ plate. Each of the eight call sites — `by_subj_cloze`, `by_verb_cloze`, ..., `by_item_prop` — inlines a fresh α-renamed copy of the body, so every grouping factor contributes an independent $\sigma$ and an independent per-level plate. The eight call sites realise crossed random intercepts on the cloze and proportion sides of the experiment.

A named `continuous` morphism cannot bundle a fresh scale draw per call because morphism reference is invocation, not instantiation; parametric programs *are* instantiated freshly at each call, which is the right categorical handle for prior reuse.

### Plate-draws

```qvr
draw v : G -> 1 ~ Normal(0.0, sigma)
draw duration_incr_cloze : Item -> 11 ~ HalfNormal(1.0)
```

A plate-draw `draw v : A -> K ~ F(args)` denotes the Kleisli morphism $A \to \mathcal{G}(K)$ given by independent $F$-draws indexed by $A$, equivalently a single arrow $\mathbf{1} \to \mathcal{G}(K^A)$ via the natural isomorphism $\mathbf{Kern}(\mathbf{1}, K^A) \cong \mathbf{Kern}(A, K)$. Numeric codomains are interpreted as `Euclidean(K)`.

### Ordinal monotone spline

The eleven duration levels carry a monotone-increasing effect via the `cumsum` parameterisation: per-level positive increments are drawn from `HalfNormal(1.0)` and accumulated into partial sums.

```qvr
draw duration_incr_cloze : Item -> 11 ~ HalfNormal(1.0)
let duration_eff_cloze = cumsum(duration_incr_cloze)
```

Because `HalfNormal` has support $[0, \infty)$, the partial sums are monotone-increasing by construction. The `let` step lifts the deterministic measurable map $\mathrm{cumsum}$ into the Kleisli category as a Dirac kernel.

### Vectorised observes

```qvr
observe cloze_resp[n] ~ Bernoulli(intercept_cloze) for n in RespCloze
```

This denotes the sub-probabilistic kernel $\Phi \to \mathcal{G}_{\le 1}(\Phi)$ with score $\prod_{n \in \mathrm{RespCloze}} p_{\mathrm{Bern}}(r_{\mathrm{obs}}(n); \theta(n, \phi))$. The response buffer $r_{\mathrm{obs}}$ is supplied at runtime via the `observations` dict, keyed by the response identifier `cloze_resp`.

### Discrete-latent marginalisation

```qvr
marginalize cloze_resp
marginalize prop_resp
```

The `marginalize` step pushes the accumulated joint measure forward through the projection $\pi : \Phi \times C \to \Phi$, integrating out the named coordinate by log-sum-exp on the accumulated log-likelihood. The two `marginalize` steps sum out the discrete latent class across the cloze and proportion sides of the experiment.

## Python Usage

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

program = load("event_structure.qvr")

observations = {
    "cloze_resp": cloze_response_tensor,   # shape (n_cloze_resp,)
    "prop_resp":  prop_response_tensor,    # shape (n_prop_resp,)
}

guide = AutoNormalGuide(program, observed_names={"cloze_resp", "prop_resp"})
elbo  = ELBO(model=program, guide=guide)
svi   = SVI(model=program, guide=guide)

optimizer = torch.optim.Adam(
    list(program.parameters()) + list(guide.parameters()), lr=1e-2,
)

for step in range(5000):
    loss = svi.step(item_input, observations=observations, optimizer=optimizer)
```

## Categorical Perspective

The program denotes a $\mathbf{Kern}$-morphism

$$
\llbracket \mathsf{event\_structure} \rrbracket \;:\; \mathrm{Data} \to \mathcal{G}\bigl(\mathrm{LatentClass} \times \mathrm{Item}\bigr),
$$

assembled by Kleisli composition of its step denotations. The vectorised observes accumulate Bernoulli log-likelihoods per response into a sub-probability kernel in $\mathcal{G}_{\le 1}$; the `marginalize` steps push forward through projection to integrate out the discrete latent class. The eight calls to `random_intercepts` are eight distinct fibres of the dependent kernel $\prod_{G : \mathbf{FinSet}} \prod_{\mathrm{scale} : \mathbb{R}} \mathbf{Kern}(G, \mathbf{1})$; substitution-and-α-rename at each call site is sound by the standard substitution lemma for the body's denotation function.
