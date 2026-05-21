# Weighted Deduction Fragment

The QVR weighted-deduction fragment is declared by the
`deduction NAME : Domain -> Codomain { … }` block. A single block
realizes grammar-style parsers, type-theoretic proof systems,
Datalog-shaped fixed-point evaluations, and graph algorithms as
parameter settings on the same agenda engine. See
[Weighted Deduction Systems](../guides/deduction.md) for the user
guide; this page gives the formal denotation.

## 1. Item algebra

A `deduction` block fixes a single item algebra `I` via the
`atoms NAME, NAME, ...` clause. The atoms are the closed set of nullary and
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
declared semiring `K`.

The collection of all rules forms a rule system
`Σ : I^{op} × I → K-Vect` assigning to each pair `(i, i')` the
`K`-module spanned by `Σ`-derivations `i ⊢ i'`. Rules are
hyperedges in the multicategory of items; the agenda fires each
rule by substitution along a pattern morphism into the chart.

### 2.1 Learnable rule weights (`#[learnable]`)

A rule may carry a trailing pragma `#[learnable]`:

```
rule r : π_1, …, π_m |- π #[learnable]
```

Rather than the default constant log-weight $\beta_r$, the rule's
contribution is a *bindings-keyed* family of learnable
parameters: at each firing the matcher produces a binding map
$\sigma : \mathrm{Vars}(r) \to I$ assigning each free pattern
variable to a ground item, and the conclusion's weight becomes

$$
\beta_r(\sigma) \;=\; \otimes_{\ell=1}^{m} \alpha[\sigma(\pi_\ell)] \;\otimes\; \theta_{r, \sigma},
$$

where $\theta_{r, \sigma} \in K$ is a learnable scalar
parameter allocated lazily on the first observation of the
binding tuple $\sigma$ (one parameter per distinct $\sigma$
observed at run time, keyed by a canonical string form of
$\sigma$). The parameters live on a per-deduction `nn.Module`
side-table that participates in the model's `.parameters()`.

Two specialisations of the surface deserve explicit names:

* When the rule's conclusion contains no free wildcards (every
  binding has a unique ground value), $\theta_{r, \sigma}$
  reduces to a single global rule weight $\theta_r$, recovering
  the PCFG-style per-production weight.
* When the rule's conclusion is wildcarded (e.g. `branch : ... |-
  span(I, J, A)` with $A$ free), each distinct ground value of
  the wildcard pulls a distinct parameter from the dictionary,
  recovering the *partial-application weight* of [§2.2](#22-parent-rule-specialisation-parent).

### 2.2 Parent rule specialisation (`parent=`)

A rule may declare a *parent* via the `parent=` option in its
pragma:

```
rule r' : π'_1, …, π'_m |- π' #[learnable, parent=r]
```

The conclusion weight then composes additively in the log semiring
with the parent's bindings-keyed weight on the *same* bindings:

$$
\beta_{r'}(\sigma) \;=\; \otimes_\ell \alpha[\sigma(\pi'_\ell)]
\;\otimes\; \theta_{r', \sigma}
\;\otimes\; \theta_{r, \sigma}.
$$

The parent chain is acyclic and terminates at a non-parented
rule. This recovers a *hierarchical* / *back-off* parametrisation
in which the specialisation contributes a correction term over
the parent's bindings-keyed weight without retraining the parent
from scratch.

### 2.3 Bounded rule weights (`bounded`)

A learnable rule whose pragma additionally lists `bounded` is
reparameterised to keep its surface log-weight non-positive:

$$
\theta_{r, \sigma} \;=\; -\mathrm{softplus}(\mathit{raw}_{r,\sigma}),
$$

so $\theta_{r, \sigma} \le 0$ for every binding. This is the
correct parameterisation when the rule appears in a closed cycle
of the deduction graph and the semiring is non-idempotent
(`LogProb`, `Counting`, `Inside`): the cycle's accumulated
log-weight is then bounded above by zero, the Kleene-star series
$\sum_{n \ge 0} (\bigotimes^n \theta_{r, \sigma})$ converges, and
the chart's fixed point is finite (see [§7](#7-strategy-independence)
on convergence vs. divergence detection).

## 3. Semiring

The semiring field selects the scoring algebra `K`:

| Field value | $\oplus$ (plus) | $\otimes$ (times) | Use |
|-------------|-----------------|--------------------|-----|
| `LogProb` | logsumexp | + | marginal log-probability |
| `Viterbi` | max | + | best-derivation decoding |
| `Boolean` | or | and | recognition |
| `Counting` | + | × | derivation counts |

The chart is enriched over `K`; the agenda enumerates derivations
under `K`'s monoidal operations.

## 3a. Binders and alpha-renaming

A deduction may declare a `binders` block listing constructor
names whose first argument is a *bound variable*:

```
deduction D : … {
    atoms NP, S, Bwd, App, Var, bark
    binders Lam
    …
}
```

For every $c \in \mathsf{binders}(D)$, an item of the form
$c(x, t)$ denotes the same equivalence class as $c(y,
t[x := y])$ for any fresh $y$. The lexicon-LF compiler enforces
this *structurally* by alpha-renaming every binder application
at construction time:

$$
c(x, t) \;\longmapsto\; c(\mathsf{fresh}, t[x := \mathsf{fresh}]),
$$

with $\mathsf{fresh}$ drawn from a per-deduction fresh-symbol
pool. Two lexicon entries that both write `Lam(x, body)` in
source therefore produce chart items with *distinct* canonical
bound names, so the chart's structural identity on item tuples
coincides with alpha-equivalence on the source terms.

The compiler additionally collects every bound variable name
appearing in any lexicon LF and extends the constructor set
$\mathsf{atoms}(D)$ with those names for the duration of LF
compilation, so the user need not duplicate them in `atoms`.
References to a bound name inside a binder body resolve through
the alpha-renamed canonical symbol.

The compiler exposes capture-avoiding substitution as a
[`let`-expression builtin](#103-let-expression-builtins) `subst(t,
v, w)`: because every binder has been alpha-renamed to a unique
canonical symbol at compile time, naive structural substitution
is already capture-avoiding.

## 3b. Item shape and pattern arity

Span axioms emitted by an indented `lexicon` body carry the
lexical category in their fourth slot and (when any rule
references an LF binding, or the deduction declares a
`binders` block) the logical form in their fifth slot:

$$
\mathsf{item} \;=\; (\mathsf{span}, i, j, \mathit{cat}) \quad\text{or}\quad
(\mathsf{span}, i, j, \mathit{cat}, \mathit{lf}).
$$

The compiler detects whether the deduction *uses LFs* by
scanning rule patterns for any 4-arity `span(I, J, C, F)`
expression, and turns the LF slot on or off uniformly per
deduction. Patterns of the shorter form `span(I, J, C)` are
auto-extended at compile time with a fresh LF wildcard so both
surface forms match the same chart shape; this preserves
backwards compatibility for LF-free deductions while enabling
the LF-binding surface where needed.

## 4. Axiom injector

A deduction needs an *axiom injector*
`ax : Input → List(I × K)` producing the initial chart from an
input. The block admits three surface forms:

- `lexicon` block with `"word" : Cat = lf #[learnable]` entries: label-indexed
  lookup table inline.
- `lexicon from "path.tsv" with learnable`: same shape loaded
  from a TSV.
- `axioms = some_kernel_morphism`: a declared Kleisli morphism
  `Input → List(I × K)`.

## 5. Goal, depth, and convergence

`start s` declares the start atom; the goal predicate is
"there exists a derivation whose conclusion is `c(s, …)` for
some configured wrapper constructor (e.g. a `span(0, n, s)` over
the full input range)."

`depth d` bounds the maximum *constructor-tree depth* of the
conclusion category of any rule firing. The compiler installs a
uniform side-condition on every rule that rejects firings whose
conclusion category exceeds depth $d$ in the constructor tree
(counting every wrapping $(c, a_1, \dots, a_n)$ tuple whose head
is a category constructor as one unit of depth). This bounds the
chart's category-constructor tower for rule systems with growing
unary intro / elim cycles (modal, continuation, residuation),
without privileging any particular category constructor.

`tolerance ε` controls convergence detection in the chart's
weight aggregation. For non-idempotent semirings with cyclic
rule graphs, each re-derivation of an item along a new path
increases its weight via `semiring.plus`, even when the item is
already in the chart; with bounded cycle weight (e.g. every
cycle's bounded total log-weight $\le 0$ via [§2.3 `bounded`](#23-bounded-rule-weights-bounded)),
the sequence of updates converges to a Kleene-star limit, and
the chart terminates re-firings on an item once successive
updates fall below $\varepsilon$. With a divergent cycle the
agenda hits `max_iterations` instead (the correct rejection of
an ill-posed model). Setting $\varepsilon = 0$ recovers strict
structural equality and the default acyclic-friendly semantics.

`max_iterations N` caps the number of agenda pops; the agenda
raises on exceeding it. Default is $10^5$.

## 6. Chart denotation

Fix the deduction `D = (I, Σ, K, ax, goal, s, d)` and an input
`w`. The *chart* is the function

$$
\alpha : I \to K, \qquad
\alpha[i] = \bigoplus_{\text{deriv } d : i_1, …, i_k \vdash i}
            \bigotimes_{\ell=1}^{k} \alpha[i_{\ell}]
$$

: the `K`-join over all `Σ`-derivations of `i`, of the product
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
*operational realization* of the fixed-point computation; a
strategy is a tuple `(Agenda, π, stop)` with `Agenda` a queue
discipline (FIFO / LIFO / priority), `π` a priority function, and
`stop` a termination predicate.

Goodman 1999 §3 (semiring parsing) establishes that for any
`K` and any pair of well-formed strategies, the resulting chart
value at every item is identical. The runtime picks a default
strategy from rule arities and the semiring's algebraic properties
(CKY-sweep for context-free + idempotent, A\* for weighted +
idempotent + admissible heuristic, semi-naive for Datalog-shaped,
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

## 10. Chart access from program bodies

A `program` body may invoke a compiled deduction $D$ by name
through the `parse` builtin:

```
let chart = parse(D, sentence)
```

The right-hand side denotes a *chart-view* object whose API
exposes the chart's K-presheaf as differentiable measurable maps
on the surrounding trace context:

| Method | Denotation |
|---|---|
| `chart.weight(item)` | $\alpha[i]$ at the ground item $i$ (raises if absent). |
| `chart.try_weight(item, default)` | $\alpha[i]$ or `default` or the semiring's zero. |
| `chart.enumerate(pattern)` | the indexed family $\{(i, \alpha[i]) \mid \pi \sim i\}$ for items matching pattern $\pi$. |
| `chart.goal_weight()` | $\bigoplus_{i \in \mathrm{goal}} \alpha[i]$, the deduction's inside-marginal. |
| `chart.embedding(item)` | the vector embedding of an item under the attached encoder, if any. |
| `chart.attached_loss` | the cumulative loss declared on attached `loss` blocks. |
| `chart.semiring` | the chart's underlying semiring $K$. |

Each of these returns a `torch.Tensor` carrying gradients
through the deduction's learnable parameters (lexicon log-weights
and the rule-weight side-table from [§2.1](#21-learnable-rule-weights-learnable)).

The natural lift into the program's log-joint is via a
[`score` step](programs.md#27a-score-factor):

```
let chart = parse(D, sentence)
score log_Z = chart.goal_weight()
```

The body's denotation extends the program's log-joint by the
inside log-marginal of the input under the deduction's current
parameters: SVI and NUTS over the program's parameter set then
descend to the deduction's parameters automatically.

### 10.1 Composing deductions

The let-expression builtin `compose(D_1, D_2)` builds a deduction
system whose axiom injector chains $D_1$'s goal items into
$D_2$'s axiom set:

$$
\mathsf{ax}_{D_1 \mathbin{\mathrm{c}} D_2}(\mathit{inp})
\;=\;
\mathsf{goal}\_\mathsf{items}\bigl(D_1(\mathit{inp})\bigr)
\;\cup\;
\mathsf{ax}_{D_2}(\mathit{inp}).
$$

The result's semiring, agenda, and goal are inherited from
$D_2$; the parameters of *both* factors appear in the composed
system's `.parameters()` via cross-attachment of the
`_axiom_module` / `_rule_module` side-tables. A composed
deduction is therefore directly usable inside `parse(..., …)`
calls without any further wrapper.

The composition is associative up to chart equality of the
intermediate goal-item set, and identity-on-the-right under the
trivial deduction with no rules.

### 10.2 Forward sampling

The [`quivers.deduction.sample_corpus`](../api/stochastic/deduction/sample.md#quivers.stochastic.deduction.sample.sample_corpus)
helper realises exact forward sampling from a deduction's
distribution over yields: it enumerates the deduction's surface
vocabulary to a chosen length, evaluates $\alpha[\mathsf{goal}]$
for every candidate yield, and draws categorically from the
softmax of the resulting log-weights.

### 10.3 Let-expression builtins

The let-expression compiler ships three deduction-related
builtins, all callable from inside a `program` or lexicon LF
body:

| Builtin | Signature | Denotation |
|---|---|---|
| `parse(D, x)` | `(DeductionSystem, Input) -> ChartView` | invoke $D$ on input $x$. |
| `compose(D_1, D_2)` | `(DeductionSystem, DeductionSystem) -> DeductionSystem` | as in [§10.1](#101-composing-deductions). |
| `subst(t, v, w)` | `(Term, Term, Term) -> Term` | structural substitution; capture-avoiding under the [`binders` invariant](#3a-binders-and-alpha-renaming). |

`subst(t, v, w)` walks $t$ replacing every subterm structurally
equal to $v$ with $w$. Because every binder application has
been alpha-renamed to a fresh canonical symbol at lexicon-LF
compile time, naive structural substitution is already
capture-avoiding: no two distinct bound variables share a
canonical name in any reachable term.

## 9. Program-fragment integration

The Bayesian-modeling step kinds, effect signatures, the
`over`-modifier, the operadic-contraction call, and the
transformation-composition operator introduce additional
productions in the QVR grammar. The shapes below mirror the
tree-sitter source at `grammars/qvr/grammar.js`; semantics is
given in [Programs](programs.md), [Expressions](expressions.md),
and [Composition Rules](composition-rules.md).

```ebnf
# The unified option block is the same production every declaration
# kind (morphism, kernel, contraction, deduction, lexicon-from-file,
# pragma) reads. Values are bare identifiers (flags), identifiers
# (named options), numeric / string literals, comma-separated lists,
# or single function-call expressions.
option_block        := '[' option_entry (',' option_entry)* ']'
option_entry        := IDENT [ '=' option_value ]
option_value        := IDENT
                     | INT | FLOAT | STRING
                     | '[' option_value (',' option_value)* ']'
                     | IDENT '(' option_value (',' option_value)* ')'

# Outer / inner pragmas. The outer form decorates the next
# declaration; the inner form decorates the enclosing module.
pragma_outer        := '#['  pragma_entry (',' pragma_entry)* ']'
pragma_inner        := '#!['  pragma_entry (',' pragma_entry)* ']'
pragma_entry        := IDENT [ '=' option_value ]

deduction_decl      := 'deduction' IDENT ':' type_expr '->' type_expr
                       [ option_block ]
                       deduction_body_entry*

deduction_body_entry
                    := deduction_atoms
                     | deduction_binders
                     | deduction_rule
                     | deduction_lexicon
                     | deduction_lexicon_from_file

deduction_atoms     := 'atoms' IDENT (',' IDENT)*
deduction_binders   := 'binders' IDENT (',' IDENT)*

deduction_rule      := 'rule' IDENT ':' premise_list ('|-' | '⊢')
                       conclusion [ rule_pragma ]
rule_pragma         := '#[' rule_pragma_entry (',' rule_pragma_entry)* ']'
rule_pragma_entry   := 'learnable'
                     | 'bounded'
                     | 'parent' '=' IDENT

# Deduction-level options recognised by the agenda compiler;
# extend the standard ``[k=v, ...]`` option list.
deduction_option    := 'semiring'   '=' IDENT
                     | 'start'      '=' IDENT
                     | 'depth'      '=' INT
                     | 'tolerance'  '=' NUMBER
                     | 'max_iterations' '=' INT
                     | 'axioms'     '=' IDENT
                     | 'signature'  '=' IDENT
                     | 'encoder'    '=' IDENT

typed_program_param := IDENT ':' param_kind
param_kind          := object_kind | scalar_kind | morphism_kind
object_kind         := 'FinSet' | 'Space' | 'Object'
scalar_kind         := 'Real'   | 'Nat'
morphism_kind       := 'Mor' '[' type_expr ',' type_expr ']'

effect_set          := effect (',' effect)*
effect              := 'Sample' | 'Score' | 'Marginal' | 'Pure'

bind_step           := var_pattern [ ':' type_expr ] '<-' IDENT
                       [ '(' draw_arg_list ')' ]

observe_step        := 'observe' IDENT [ ':' type_expr ]
                       '<-' IDENT [ '(' draw_arg_list ')' ]
                       [ option_block ]
# Option-block entries recognised on observe steps include
# ``via = idx`` (a fibration into a marginalize grouping plate)
# and any family-specific keyword overrides.

score_step          := 'score' IDENT '=' let_arith

marginalize_step    := 'marginalize' IDENT [ ':' type_expr ] '<-' IDENT
                       [ '(' draw_arg_list ')' ]
                       [ option_block ]
                       NEWLINE INDENT program_step+ DEDENT
# Option-block entries on marginalize steps include
# ``over = G`` (or ``over = product(G_1, …)`` / ``over = [G_1, …]``
# for a product grouping plate) and ``reduction = R`` selecting
# the aggregation (R ∈ {logsumexp, sum, mean}, default logsumexp).
# Each observe step inside the body carries its own
# ``via = idx`` option naming the fibration into the shared plate;
# the runtime scatter-sums each per-(N_m, K) log-likelihood into
# the same (|G|, K) accumulator before the reduction.

let_index           := IDENT '[' let_arith (',' let_arith)* ']'

program_decl        := 'program' IDENT [ '(' param_list ')' ]
                       ':' type_expr '->' type_expr
                       [ option_block ]
                       NEWLINE INDENT program_step* return_step DEDENT
# Option-block entries on program declarations include
# ``effects = [E_1, ..., E_n]`` (the declared effect signature; each
# E_i is one of {Sample, Score, Marginal, Pure}) and
# ``over = M`` (posterior consumer over model M's latents).

# Top-level composition-rule selection. The single `composition`
# keyword takes an optional `as LEVEL` clause naming the declared
# algebraic level; the optional indented body declares a fresh
# rule inline.
composition_decl    := 'composition' IDENT
                       [ 'as' composition_level ]
                       [ NEWLINE INDENT composition_rule_entry+ DEDENT ]

composition_level   := 'algebra' | 'semigroupoid'
                     | 'bilinear_form' | 'rule'

composition_rule_entry
                    := IDENT '(' IDENT (',' IDENT)* ')' '=' let_expr
                     | IDENT '=' let_expr

# Operadic n-ary contraction declaration.
contraction_decl    := 'contraction' IDENT
                       '(' contraction_input (',' contraction_input)* ')'
                       ':' type_expr '->' type_expr
                       'rule' IDENT
                       'wiring' STRING

contraction_input   := IDENT ':' type_expr '->' type_expr

# Transformation-valued expressions used inside change_base.
# Distinct from morphism composition `>>` (see Expressions § 2.7).
trans_compose       := expr '>>>' expr

# Method-call postfix on a morphism expression. The change_base
# argument is any expression that resolves to a transformation
# (singleton, constructor call, let-bound trans, or `>>>` chain).
change_base_postfix := expr '.' 'change_base' '(' expr ')'

# Operadic call site: `IDENT(arg_1, ..., arg_n)` resolves to a
# registered contraction or a parametric program template.
morphism_call       := IDENT '(' IDENT (',' IDENT)* ')'
```

A `program_decl` is *parametric* iff its parameter list contains any `typed_program_param`; the walker dispatches parametric programs to the call-site inliner rather than to the runtime program compiler. A program declared with `! effect_set` has its body checked against the declared capability set: the actual effects of the body must form a subset of `effect_set`, and `! Pure` rejects any `bind_step` / `observe_step` / `marginalize_step`. A program declared with `over M` is a posterior block consuming the latents of model `M`; the consumed latents appear as data parameters in the program's parameter list.

A `composition_decl` selects the module's underlying composition rule. With no body and no `as` clause, the keyword resolves the named rule from the built-in catalogue and registers it. With an `as LEVEL` clause but no body, the resolved built-in rule is verified to match the declared algebraic level (`algebra`, `semigroupoid`, `bilinear_form`, or `rule`, the last covering any `CompositionRule`). With a body, the entries declare the rule's operations inline; the `as LEVEL` clause fixes the algebraic level, and the compiler verifies that the required entries (`tensor_op`, `join`, plus `unit`, `zero` for `algebra`) are present. See [Composition Rules](composition-rules.md) for the formal denotation.

A `contraction_decl` declares an n-ary operadic morphism whose action contracts its input morphisms under the named composition rule using the wiring spec. Call sites `IDENT(arg_1, …, arg_n)` route through `morphism_call`; the compiler resolves `IDENT` against the contraction registry, the parametric-program template table, and the morphism scope in that order. See [Expressions § 2.13](expressions.md#213-operadic-contraction-call) for the call-site denotation.

## References

- Shieber, Schabes & Pereira (1995). [*Principles and implementation of deductive parsing*](https://doi.org/10.1016/0743-1066(95)00035-I). Journal of Logic Programming 24(1–2):3–36.
- Goodman (1999). [*Semiring parsing*](https://aclanthology.org/J99-4004/). Computational Linguistics 25(4):573–605.
- Pereira & Warren (1983). [*Parsing as deduction*](https://doi.org/10.3115/981311.981338). In Proceedings of the 21st Annual Meeting of the Association for Computational Linguistics, pp. 137–144.
- Klein & Manning (2001). [*Parsing and hypergraphs*](https://doi.org/10.1007/1-4020-2295-6_18). In Proceedings of the Seventh International Workshop on Parsing Technologies (IWPT), pp. 123–134.
- Knuth (1977). [*A generalization of Dijkstra's algorithm*](https://doi.org/10.1016/0020-0190(77)90002-3). Information Processing Letters 6(1):1–5.
- Nederhof (2003). [*Weighted deductive parsing and Knuth's algorithm*](https://doi.org/10.1162/089120103321337467). Computational Linguistics 29(1):135–143.
- Eisner, Goldlust & Smith (2005). [*Compiling Comp Ling: Practical weighted dynamic programming and the Dyna language*](https://aclanthology.org/H05-1036/). In Proceedings of HLT-EMNLP, pp. 281–290.
- McAllester (2002). [*On the complexity analysis of static analyses*](https://doi.org/10.1145/581771.581774). Journal of the ACM 49(4):512–537.
