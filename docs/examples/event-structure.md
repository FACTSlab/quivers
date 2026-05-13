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
    sigma <- HalfNormal(scale)
    v : G <- Normal(0.0, sigma)
    return v

program event_structure : Item -> Item
    prob_durative <- Uniform(0.0, 1.0)
    prob_telic_given_dur <- Beta(10.0, 1.0)
    prob_telic_given_nodur <- Beta(1.0, 1.0)

    intercept_cloze <- Normal(0.0, 1.0)
    intercept_prop <- Normal(0.0, 1.0)
    telicity_coef_cloze <- HalfNormal(1.0)
    telicity_coef_prop <- HalfNormal(1.0)
    durativity_coef_prop <- HalfNormal(1.0)

    by_subj_cloze  <- random_intercepts(SubjCloze, 1.0)
    by_verb_cloze  <- random_intercepts(Verb,      1.0)
    by_sense_cloze <- random_intercepts(Sense,     1.0)
    by_item_cloze  <- random_intercepts(Item,      1.0)

    by_subj_prop   <- random_intercepts(SubjProp,  1.0)
    by_verb_prop   <- random_intercepts(Verb,      1.0)
    by_sense_prop  <- random_intercepts(Sense,     1.0)
    by_item_prop   <- random_intercepts(Item,      1.0)

    duration_incr_cloze : Item <- HalfNormal(1.0)
    duration_incr_prop  : Item <- HalfNormal(1.0)

    let duration_eff_cloze = cumsum(duration_incr_cloze)
    let duration_eff_prop  = cumsum(duration_incr_prop)

    marginalize cloze_resp : RespCloze <- Bernoulli(intercept_cloze) in {
        observe cloze_resp : RespCloze <- Bernoulli(intercept_cloze)
    }
    marginalize prop_resp : RespProp <- Bernoulli(intercept_prop) in {
        observe prop_resp : RespProp <- Bernoulli(intercept_prop)
    }

    return intercept_cloze

export event_structure
```

## Walkthrough

### Cardinalities

The cardinalities of `Verb`, `Sense`, `Item`, `SubjCloze`, `SubjProp`, `RespCloze`, and `RespProp` are fit-time-determined placeholders for the static parser; they bound the indexing sets for the plates and the vectorised observes but do not constrain the actual response tensors at runtime.

### Latent classes

The class prior factors through a cell parameterisation: $\mathrm{prob\_durative}$ marginalises durativity, and the two conditionals $\mathrm{prob\_telic\_given\_dur}$ and $\mathrm{prob\_telic\_given\_nodur}$ give telicity given durativity. The four cells of $\{\pm\mathrm{telic}\} \times \{\pm\mathrm{durative}\}$ are recovered as products of these factors.

### Parametric random-intercepts template

```qvr
program random_intercepts (G : FinSet, scale : Real) : G -> 1
    sigma <- HalfNormal(scale)
    v : G <- Normal(0.0, sigma)
    return v
```

The template denotes a dependent kernel

$$
\llbracket \mathsf{random\_intercepts} \rrbracket \;:\; \prod_{G : \mathbf{FinSet}} \prod_{\mathrm{scale} : \mathbb{R}_{>0}} \mathbf{Kern}(G, \mathbf{1}),
$$

a half-normal scale hyperprior followed by a $G$-indexed Normal-$(0, \sigma)$ plate. Each of the eight call sites, `by_subj_cloze`, `by_verb_cloze`, ..., `by_item_prop`, inlines a fresh α-renamed copy of the body, so every grouping factor contributes an independent $\sigma$ and an independent per-level plate. The eight call sites realise crossed random intercepts on the cloze and proportion sides of the experiment.

A named `continuous` morphism cannot bundle a fresh scale draw per call because morphism reference is invocation, not instantiation; parametric programs *are* instantiated freshly at each call, which is the right categorical handle for prior reuse.

### Indexed binds

<!-- compile: false -->
```qvr
v : G <- Normal(0.0, sigma)
duration_incr_cloze : Item <- HalfNormal(1.0)
```

An indexed bind `v : A <- F(args)` denotes the Kleisli morphism $A \to \mathcal{G}(K)$ given by independent $F$-draws indexed by $A$, equivalently a single arrow $\mathbf{1} \to \mathcal{G}(K^A)$ via the natural isomorphism $\mathbf{Kern}(\mathbf{1}, K^A) \cong \mathbf{Kern}(A, K)$. The per-fiber codomain $K = \mathsf{cod}(F)$ is taken from the family.

### Ordinal monotone spline

The eleven duration levels carry a monotone-increasing effect via the `cumsum` parameterisation: per-level positive increments are drawn from `HalfNormal(1.0)` and accumulated into partial sums.

<!-- compile: false -->
```qvr
duration_incr_cloze : Item <- HalfNormal(1.0)
let duration_eff_cloze = cumsum(duration_incr_cloze)
```

Because `HalfNormal` has support $[0, \infty)$, the partial sums are monotone-increasing by construction. The `let` step lifts the deterministic measurable map $\mathrm{cumsum}$ into the Kleisli category as a Dirac kernel.

### Indexed observes

<!-- compile: false -->
```qvr
observe cloze_resp : RespCloze <- Bernoulli(intercept_cloze)
```

This denotes the sub-probabilistic kernel $\Phi \to \mathcal{G}_{\le 1}(\Phi)$ with score $\prod_{n \in \mathrm{RespCloze}} p_{\mathrm{Bern}}(r_{\mathrm{obs}}(n); \theta(n, \phi))$. The response buffer $r_{\mathrm{obs}}$ is supplied at runtime via the `observations` dict, keyed by the response identifier `cloze_resp`.

### Coordinate marginalisation

<!-- compile: false -->
```qvr
marginalize cloze_resp : RespCloze <- Bernoulli(intercept_cloze) in {
    observe cloze_resp : RespCloze <- Bernoulli(intercept_cloze)
}
```

The scoped `marginalize c : A <- F in { … }` step introduces the coordinate `c` bound to a kernel `F`, optionally `A`-indexed, with the `{ … }` body as its integration scope. At the end of the scope, the accumulated joint measure on $\Phi \times C$ is pushed forward through the projection $\pi : \Phi \times C \to \Phi$, integrating out `c` by log-sum-exp on the accumulated log-likelihood; `c` then falls out of scope.

## Python Usage

```python
import torch
from quivers.dsl import load
from quivers.inference import AutoNormalGuide, ELBO, SVI

program = load("event_structure.qvr")
model = program.morphism  # underlying MonadicProgram

observations = {
    "cloze_resp": cloze_response_tensor,   # shape (n_cloze_resp,)
    "prop_resp":  prop_response_tensor,    # shape (n_prop_resp,)
}

guide = AutoNormalGuide(model, observed_names={"cloze_resp", "prop_resp"})
elbo  = ELBO(num_particles=1)
optimizer = torch.optim.Adam(
    list(model.parameters()) + list(guide.parameters()), lr=1e-2,
)
svi = SVI(model, guide, optimizer, elbo)

for step in range(5000):
    loss = svi.step(item_input, observations)
```

## Categorical Perspective

The program denotes a $\mathbf{Kern}$-morphism

$$
\llbracket \mathsf{event\_structure} \rrbracket \;:\; \mathrm{Data} \to \mathcal{G}\bigl(\mathrm{LatentClass} \times \mathrm{Item}\bigr),
$$

assembled by Kleisli composition of its step denotations. The vectorised observes accumulate Bernoulli log-likelihoods per response into a sub-probability kernel in $\mathcal{G}_{\le 1}$; the `marginalize` steps push forward through projection on the corresponding trace coordinates. The eight calls to `random_intercepts` are eight distinct fibres of the dependent kernel $\prod_{G : \mathbf{FinSet}} \prod_{\mathrm{scale} : \mathbb{R}} \mathbf{Kern}(G, \mathbf{1})$; substitution-and-α-rename at each call site is sound by the standard substitution lemma for the body's denotation function.
