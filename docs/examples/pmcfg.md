# Probabilistic Multiple Context-Free Grammar

## QVR Source

```qvr
object Term : FinSet 16

deduction PMCFG : Term -> Term [semiring=LogProb, start=NP, depth=4]
    atoms NP, N, Det, V, who, S, RC, sg, span, the, man, dog, mary, john, saw, met
    rule transitive_obj_gap : span(I, K, NP), span(K, J, V) |- sg(I, J, J, J) #[learnable]
    rule relative_clause : span(P, Pp, who), sg(Pp, J1, J1, J2) |- span(P, J2, RC) #[learnable]
    rule modify_n : span(I, J, N), span(J, K, RC) |- span(I, K, N) #[learnable]
    rule np_det_n : span(I, J, Det), span(J, K, N) |- span(I, K, NP) #[learnable]
    lexicon
        "the"  : Det = the   #[learnable]
        "man"  : N   = man   #[learnable]
        "dog"  : N   = dog   #[learnable]
        "who"  : who = who   #[learnable]
        "Mary" : NP  = mary  #[learnable]
        "John" : NP  = john  #[learnable]
        "saw"  : V   = saw   #[learnable]
        "met"  : V   = met   #[learnable]
```

## Overview

A Probabilistic Multiple Context-Free Grammar (PMCFG) is a Multiple Context-Free Grammar (MCFG; Seki, Matsumura, Fujii, Kasami 1991) with a probability (or, more generally, a semiring weight) attached to every production. PMCFG is to MCFG what PCFG is to CFG: the same rule set, decorated with learnable weights, fitted to data via the chart's inside marginal $\log Z(s) = \log \sum_d \exp \langle \mathbf{w}, \phi(d) \rangle$.

The defining feature of MCFG (and therefore PMCFG) is that each non-terminal $A$ has a fixed rank $k(A) \ge 1$ and generates *tuples of strings*, not just single strings. A production rewrites a non-terminal as a linear combination of its premises' tuple components: each component of the conclusion is a concatenation of terminals and components drawn from the RHS non-terminals. The rank-1 case is exactly CFG; ranks $\ge 2$ make discontinuous constituents expressible.

This example uses MCFG to model English **WH-movement in relative clauses**, a textbook motivation for the formalism. In a noun phrase like

> "the man who Mary saw"

the relative pronoun *who* appears at the left edge of the embedded clause, but its grammatical role is the *object* of *saw*: there is a gap immediately after the verb that *who* discontinuously fills. A CFG cannot derive this with a flat single-yield non-terminal because the filler and the gap straddle a constituent boundary. MCFG handles it by giving the gapped clause a **rank-2 non-terminal** whose two components straddle the gap site.

## Walkthrough

`sg(I1, J1, I2, J2)` is the rank-2 item for an *S with an NP gap*. Its first yield component spans `[I1, J1)` (the prefix before the gap) and its second component spans `[I2, J2)` (the suffix after the gap). For the input `[the, man, who, Mary, saw]` parsed as `the man who Mary saw`, the gapped clause is `Mary saw _` with the gap at the very end, giving

```
sg(3, 5, 5, 5)         # component 1 = "Mary saw"; component 2 = ""
```

The relative-clause production

```
rule relative_clause : span(P, Pp, who), sg(Pp, J1, J1, J2) |- span(P, J2, RC)
```

implements the linear yield function `RC(w x y) :- who(w) sg(x, y)`: the WH-word, the prefix component of the sg, and the suffix component are concatenated *in input order* into a single rank-1 RC item. The variable `Pp` appearing in both the `who` span and the start of the sg's first component enforces input-adjacency between the WH-filler and the subject of the gapped clause. Likewise `J1` appearing both at the end of the first sg component and at the start of the second pins the gap site.

`modify_n` and `np_det_n` are the standard CFG productions for N-modification and the determiner-noun NP. The compiler allocates one `nn.Parameter` per distinct binding tuple it observes at run time, so each production becomes a weighted edge in the chart and the goal weight at `span(0, 5, NP)` is the inside log-probability of the full NP under the grammar.

## DSL Features

- **Tuple-valued chart items** for non-terminals of rank $\ge 2$. The `sg(I1, J1, I2, J2)` item is a four-position structural pattern; the chart engine pattern-matches it the same way it does the rank-1 `span(I, J, X)`.
- **Linear yield functions** as ordinary sequent rules. Concatenation and component permutation across premises are expressed by where each variable appears in the conclusion.
- **`#[learnable]` weights on every production**, lexicon entries and rules alike. The bindings-keyed parameter dictionary stores one log-weight per distinct binding tuple, giving the same partial-application weight surface as a per-production-instantiation PCFG.

## Try it

The deduction system is callable: `ded(sentence)` returns a [`ChartView`](../api/stochastic/agenda.md#quivers.stochastic.agenda.ChartView) whose `goal_weight()` is the differentiable log-marginal $\log Z(s; \mathbf{w})$ summed over every derivation the start symbol licenses for the input. Fitting the lexicon and rule weights together is then a regression-style problem; the [`quivers.stochastic.deduction`](../api/stochastic/deduction.md) module ships the two standard surfaces.

### MAP fit (Adam on rule & lexicon weights)

```python
import torch
from quivers.dsl import load
from quivers.stochastic.deduction import adam_fit_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/pmcfg.qvr")
ded  = prog.deductions["PMCFG"]

corpus = [
    ["the", "man", "who", "Mary", "saw"],
    ["the", "dog", "who", "John", "met"],
]

history = adam_fit_deduction(
    ded, corpus, steps=300, lr=5e-2, prior_scale=1.0,
)
print(f"loss: {history[0]:.2f} → {history[-1]:.2f}")  # strictly decreasing

for sentence in corpus:
    log_z = float(ded(sentence).goal_weight().detach())
    print(f"  log Z({' '.join(sentence)}) = {log_z:.2f}")
```

### NUTS (full Bayesian posterior)

```python
import torch
from quivers.dsl import load
from quivers.inference import MCMC, NUTSKernel
from quivers.stochastic.deduction import nuts_program_from_deduction

torch.manual_seed(0)
prog = load("docs/examples/source/pmcfg.qvr")
ded  = prog.deductions["PMCFG"]

corpus = [
    ["the", "man", "who", "Mary", "saw"],
    ["the", "dog", "who", "John", "met"],
]

model, x, observations = nuts_program_from_deduction(
    ded, corpus, prior_scale=1.0,
)

kernel = NUTSKernel(step_size=0.1, max_tree_depth=4, target_accept=0.8)
mc     = MCMC(kernel, num_warmup=50, num_samples=50, num_chains=2)
result = mc.run(model, x, observations)

print("acceptance:", float(result.acceptance_rates.mean()))
print("divergences:", int(result.divergence_counts.sum()))
```

## Categorical Perspective

An MCFG production with $k$-component non-terminals is a hyperedge in a multi-coloured directed hypergraph whose nodes are tuple-positioned chart items. The chart's least pre-fixed point on the LogProb-enriched lattice is the sum over every derivation (Goodman 1999); each derivation contributes the product of the log-weights of its rule firings, lifted through the bindings-keyed parameter dictionary. The PMCFG inside algorithm is the agenda-driven evaluation of that fixed point; WH-movement is recovered as the linear yield function that interleaves a filler with the components of a higher-rank non-terminal.

The framework imposes no built-in commitment to rank-1 (CFG) items. Higher-rank PMCFG, MCFG, LCFRS, and PLCFRS all share the same chart implementation: only the rule patterns and their conclusion arities change.
