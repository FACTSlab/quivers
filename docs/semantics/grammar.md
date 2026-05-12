# Weighted Deduction Fragment

The QVR weighted-deduction fragment is declared by the
`deduction NAME : Domain -> Codomain { … }` block. A single block
realises grammar-style parsers, type-theoretic proof systems,
Datalog-shaped fixed-point evaluations, and graph algorithms as
parameter settings on the same agenda engine. See
[Weighted Deduction Systems](../guides/deduction.md) for the user
guide; this page gives the formal denotation.

## 1. Item algebra

A `deduction` block fixes a single item algebra `I` via the
`atoms { … }` field. The atoms are the closed set of nullary and
parametric constructor symbols; an item `i ∈ I` is a constructor
application `c(a_1, …, a_n)` with `c` an atom and `a_1, …, a_n`
arbitrary nested items. Concretely, `I` is the free algebra over
the atoms quotiented by the underlying carrier object `Term`'s
identification of structurally equal applications.

Pattern variables (single-uppercase identifiers in a rule body)
range over `I`; the unifier supplies first-order substitutions.

## 2. Rules

A rule declaration

```
rule r : π_1, …, π_m |- π
```

introduces a hyperedge in the rule-system multicategory: a
universally quantified sequent

$$
\frac{\pi_1[\bar X / \bar\pi] \quad \cdots \quad \pi_m[\bar X / \bar\pi]}{\pi[\bar X / \bar\pi]} \quad (r)
$$

over substitutions `[\bar X / \bar\pi]` for the rule's free pattern
variables. The hyperedge carries a log-weight `β_r ∈ K` for the
declared semiring `K` (zero by default, learnable per axiom-source
entry).

The collection of all rules forms a rule system
`Σ : I^{op} × I → K-Vect` assigning to each pair `(i, i')` the
`K`-module spanned by `Σ`-derivations `i ⊢ i'`. Rules are
hyperedges in the multicategory of items; the agenda fires each
rule by substitution along a pattern morphism into the chart.

## 3. Semiring

The semiring field selects the scoring quantale `K`:

| Field value | $\oplus$ (plus) | $\otimes$ (times) | Use |
|-------------|-----------------|--------------------|-----|
| `LogProb` | logsumexp | + | marginal log-probability |
| `Viterbi` | max | + | best-derivation decoding |
| `Boolean` | or | and | recognition |
| `Counting` | + | × | derivation counts |
| `ProductFuzzy` | max | × | fuzzy membership |

The chart is enriched over `K`; the agenda enumerates derivations
under `K`'s monoidal operations.

## 4. Axiom injector

A deduction needs an *axiom injector*
`ax : Input → List(I × K)` producing the initial chart from an
input. The block admits three surface forms:

- `lexicon { "word" : Cat = lf @ learnable, … }` — label-indexed
  lookup table inline.
- `lexicon from "path.tsv" with learnable` — same shape loaded
  from a TSV.
- `axioms = some_kernel_morphism` — a declared Kleisli morphism
  `Input → List(I × K)`.

## 5. Goal and depth

`start s` declares the start atom; the goal predicate is
"there exists a derivation whose conclusion is `c(s, …)` for
some configured wrapper constructor (e.g. a `span(0, n, s)` over
the full input range)."

`depth d` bounds the maximum derivation depth so the agenda
terminates on any finite input.

## 6. Chart denotation

Fix the deduction `D = (I, Σ, K, ax, goal, s, d)` and an input
`w`. The *chart* is the function

$$
\alpha : I \to K, \qquad
\alpha[i] = \bigoplus_{\text{deriv } d : i_1, …, i_k \vdash i}
            \bigotimes_{\ell=1}^{k} \alpha[i_{\ell}]
$$

— the `K`-join over all `Σ`-derivations of `i`, of the product
of the children's weights. The base case is the axiom injector:
`α[i] = w` for every `(i, w) ∈ ax(input)`, all other unproven
items at `⊥`.

The denotation of the deduction is the goal weight:

$$
\llbracket D \rrbracket(w) = \bigoplus_{i ∈ \mathrm{goal}} α[i].
$$

When `K = LogProb` this is the inside log-probability of the
input under the rule system; when `K = Viterbi` it is the
best-derivation score; when `K = Boolean` it is the membership
predicate `w ∈ L(D)`.

## 7. Strategy independence

The chart is the least pre-fixed point of the rule-system functor
in the `K`-enriched lattice (Tarski-Knaster). The agenda is the
*operational realisation* of the fixed-point computation; a
strategy is a tuple `(Agenda, π, stop)` with `Agenda` a queue
discipline (FIFO / LIFO / priority), `π` a priority function, and
`stop` a termination predicate.

Goodman 1999 §3 (semiring parsing) establishes that for any
`K` and any pair of well-formed strategies, the resulting chart
value at every item is identical. The runtime picks a default
strategy from rule arities and the semiring's algebraic properties
(CKY-sweep for context-free + idempotent, A\* for weighted +
idempotent + admissible heuristic, semi-naïve for Datalog-shaped,
Viterbi for `(max, ⊗)`, depth-first for proof-search-shaped).

## 8. Charts as first-class differentiable values

The chart's underlying weight storage is a `torch.Tensor` (dense)
or `dict[Item, torch.Tensor]` (sparse) with `requires_grad`
flowing from rule-weight parameters through the agenda's
`semiring.times` / `semiring.plus` operations.

- **Finite-iteration** deductions (no fixpoint cycles): autodiff
  through the bounded number of semiring operations is automatic.
- **Fixed-point** deductions (cyclic): the gradient `∂C^* / ∂θ`
  is solved via the implicit-function theorem at convergence
  (`torch.linalg.solve` over `∂F/∂C^*`).
- **Inside-outside-friendly** semirings (commutative + idempotent
  + with inverses, e.g. `LogProb`, `Viterbi`): the runtime emits
  an analytic outside computation (Eisner & Goldlust 2005) as a
  second agenda pass that propagates outside weights from the
  goal back to all items.

## 9. Program-fragment integration

The Bayesian-modelling step kinds, effect signatures, and the
`over`-modifier introduce additional productions in the QVR
grammar. The shapes below mirror the tree-sitter source at
`grammars/qvr/grammar.js`; semantics is given in
[Programs §2.1–§2.8 and §3a](programs.md).

```ebnf
typed_program_param := IDENT ':' param_kind
param_kind          := object_kind | scalar_kind | morphism_kind
object_kind         := 'FinSet' | 'Space' | 'Object'
scalar_kind         := 'Real'   | 'Nat'
morphism_kind       := 'Mor' '[' type_expr ',' type_expr ']'

effect_set          := effect (',' effect)*
effect              := 'Sample' | 'Score' | 'Marginal' | 'Pure'

bind_step           := var_pattern [ ':' type_expr ] '<-' IDENT
                       [ '(' draw_arg_list ')' ]

observe_step        := 'observe' IDENT [ ':' type_expr ] '<-' IDENT
                       [ '(' draw_arg_list ')' ]

marginalize_step    := 'marginalize' IDENT [ ':' type_expr ] '<-' IDENT
                       [ '(' draw_arg_list ')' ]
                       'in' '{' program_step* '}'

let_index           := IDENT '[' let_arith (',' let_arith)* ']'

program_decl        := 'program' IDENT [ '(' param_list ')' ]
                       ':' type_expr '->' type_expr
                       [ '!' effect_set ]
                       [ 'over' IDENT ]
                       program_step* 'return' return_pattern
```

A `program_decl` is *parametric* iff its parameter list contains any `typed_program_param`; the walker dispatches parametric programs to the call-site inliner rather than to the runtime program compiler. A program declared with `! effect_set` has its body checked against the declared capability set: the actual effects of the body must form a subset of `effect_set`, and `! Pure` rejects any `bind_step` / `observe_step` / `marginalize_step`. A program declared with `over M` is a posterior block consuming the latents of model `M`; the consumed latents appear as data parameters in the program's parameter list.

## References

- Shieber, Schabes & Pereira (1995). [*Principles and implementation of deductive parsing*](https://doi.org/10.1016/0743-1066(95)00035-I). Journal of Logic Programming 24(1–2):3–36.
- Goodman (1999). [*Semiring parsing*](https://aclanthology.org/J99-4004/). Computational Linguistics 25(4):573–605.
- Pereira & Warren (1983). [*Parsing as deduction*](https://doi.org/10.3115/981311.981338). In Proceedings of the 21st Annual Meeting of the Association for Computational Linguistics, pp. 137–144.
- Klein & Manning (2001). [*Parsing and hypergraphs*](https://doi.org/10.1007/1-4020-2295-6_18). In Proceedings of the Seventh International Workshop on Parsing Technologies (IWPT), pp. 123–134.
- Knuth (1977). [*A generalization of Dijkstra's algorithm*](https://doi.org/10.1016/0020-0190(77)90002-3). Information Processing Letters 6(1):1–5.
- Nederhof (2003). [*Weighted deductive parsing and Knuth's algorithm*](https://doi.org/10.1162/089120103321337467). Computational Linguistics 29(1):135–143.
- Eisner, Goldlust & Smith (2005). [*Compiling Comp Ling: Practical weighted dynamic programming and the Dyna language*](https://aclanthology.org/H05-1036/). In Proceedings of HLT-EMNLP, pp. 281–290.
- McAllester (2002). [*On the complexity analysis of static analyses*](https://doi.org/10.1145/581771.581774). Journal of the ACM 49(4):512–537.
