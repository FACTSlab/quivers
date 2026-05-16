# DSL Declarations

This page covers every declaration form a `.qvr` module can carry
apart from `program` blocks and operadic contractions, which have
their own pages ([Programs and Let-Expressions](dsl-programs-and-lets.md),
[Contractions](dsl-contractions.md)). The grammar summary lives in
the [Overview](dsl-overview.md#grammar).

## Algebra

Choose the enriching algebra (optional, defaults to `product_fuzzy`):

<!-- compile: false -->
```qvr
algebra product_fuzzy
algebra boolean
algebra lukasiewicz
algebra godel
algebra tropical
algebra max_plus
algebra log_prob
algebra markov
algebra real
algebra probability
algebra counting
```

The keyword `algebra` resolves a name against the built-in
composition-rule registry and verifies the registered rule is at
the [`Algebra`](../api/core/algebras.md) level. Three weaker levels
are also surface-declarable:

<!-- compile: false -->
```qvr
semigroupoid material_impl
bilinear_form some_bf
composition_rule any_rule
```

A [semigroupoid](https://ncatlab.org/nlab/show/semigroupoid) rule
promises associativity but not an identity; `bilinear_form` promises
neither; `composition_rule` is permissive and accepts any rule in
the registry. Operations that need the identity element
([`identity(A)`](../api/core/morphisms.md#quivers.core.morphisms.identity),
[`f.dagger`](../api/core/morphisms.md#quivers.core.morphisms.Morphism.dagger),
[`f.trace(A)`](../api/core/morphisms.md#quivers.core.morphisms.Morphism.trace),
[`cup(A)`](../api/core/morphisms.md#quivers.core.morphisms.cup),
[`cap(A)`](../api/core/morphisms.md#quivers.core.morphisms.cap))
compile only under `algebra`. See [Composition
Rules](../semantics/composition-rules.md) for the algebraic
hierarchy and the operadic contraction surface.

A composition rule can also be defined inline via a body block.
Each entry is a `let`-expression:

<!-- compile: false -->
```qvr
algebra my_godel {
    tensor_op(a, b) = a * b
    join(t) = sum(t)
    unit = 1.0
    zero = 0.0
}

semigroupoid my_semi {
    tensor_op(a, b) = a * b
    join(t) = sum(t)
}

bilinear_form my_bf {
    tensor_op(a, b) = (a + b) * 0.5
    join(t) = sum(t)
}
```

`algebra` bodies require `tensor_op`, `join`, `unit`, `zero` (with
optional `negation` and `meet`); `semigroupoid` and `bilinear_form`
bodies require only `tensor_op` and `join`. The compiler verifies
the entry set against the keyword's algebraic level.

## Object

Three surface forms:

<!-- compile: false -->
```qvr
# 1. anonymous-element FinSet of given cardinality, or a TypeExpr
object X : 3          # FinSet(name="X", cardinality=3)
object Y : 4
object XY : X * Y     # ProductSet(components=(X, Y))
object Sum : X + Y    # CoproductSet(components=(X, Y))

# 2. FreeMonoid, bounded Kleene closure over a FinSet of generators.
object Strings = FreeMonoid(X, max_length=4)

# 3. Residuated universe.
object Cat = FreeResiduated(Atoms, depth=4, ops=[slash])
```

Object-shaped aliases (resolvable to a
[`SetObject`](../api/core/objects.md)) are interchangeable with the
underlying object, so `latent f : Pair -> X` is well-formed when
`Pair = X * Y`.

## Morphism

Declare a learnable or fixed morphism:

<!-- compile: false -->
```qvr
# Latent (learnable)
latent f : X -> Y

# With init scale
latent g : Y -> Z [scale=0.3]

# Observed (fixed)
observed h : X -> X = identity(X)
```

### Latent morphism prior

A `latent` morphism declaration accepts a
`~ Family(args) [options] [axis_role_clause]` clause that puts a
prior on its representing tensor. The literal `(args)` carry the
prior's hyperparameters at declaration time (in contrast to
`kernel`'s `~ Family` clause, whose parameters come from a neural
network at sample time). The compiler desugars the prior into a
fresh sample site whose value is the tensor representing the
morphism, making the morphism a deterministic wrap of that random
tensor.

<!-- compile: false -->
```qvr
# Free-parameter morphism (point estimate; MLE / MAP target).
latent W : Euclidean(D) -> Euclidean(K)

# The same morphism with a Matrix-Normal prior on its tensor.
latent W : Euclidean(D) -> Euclidean(K)
    ~ MatrixNormal(0.0, 1.0, 1.0) over (dom, cod)
```

The [axis-role clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid-over)
is fully detailed alongside the program-step bind and observe forms.

### Init recipes (`[init=auto]`)

The `[init=auto]` option overrides the default `randn * scale` init
with the active algebra's saturation-free recipe, derived from
[`Algebra.init_spec(depth, intermediate_size)`](../api/core/algebras.md).
For ProductFuzzy, Boolean, Gödel, Łukasiewicz, and Probability the
recipe is applied through
[`LatentMorphism`](../api/core/morphisms.md#quivers.core.morphisms.LatentMorphism)'s
sigmoid bijector (the raw parameter is set via `logit(value)`); for
the other algebras the recipe is applied to the raw parameter
directly.

<!-- compile: false -->
```qvr
latent W : Euclidean(D) -> Euclidean(K) [init=auto]
```

The per-algebra recipes ensure that a fresh model with `[init=auto]`
on every latent starts with intermediate activations close to the
algebra's neutral element, avoiding saturated sigmoids on
truth-valued algebras and exploding products on multiplicative
algebras. See
[`recommend_init`](../api/diagnostics/index.md) and
[`saturation_warnings`](../api/diagnostics/index.md) in the
[fitting and diagnostics guide](analysis-fitting-and-diagnostics.md#algebra-guided-training-tooling)
for the dynamic counterpart that audits a trained module.

## Space

Declare a continuous space:

<!-- compile: false -->
```qvr
space R3 : Euclidean(3)
space R2_bounded : Euclidean(2, low=0.0, high=1.0)
space U : UnitInterval
space P2 : PositiveReals(2)
space S3 : Simplex(3)

# Product space
space RU : R3 * U
```

Spaces resolve to subclasses of
[`ContinuousSpace`](../api/continuous/spaces.md); see the
[continuous spaces guide](continuous-spaces.md) for the full
hierarchy and constraints.

### Type alias

Declare a space alias using `type` (alternative to `space`):

```qvr
# the `space` and `type` forms below are interchangeable
space HiddenSpace : Euclidean(64)

type Hidden = Euclidean 64     # ML-style, parens optional

# product types built from previously-declared spaces
type Output   = Euclidean 32
type Combined = Hidden * Output
```

The `type` keyword provides a more concise, ML-style syntax for
declaring named spaces. Parentheses around arguments are optional.

### Alias (type-level)

`alias` declarations bind a short name to a type-level expression:

<!-- compile: false -->
```qvr
## A short alias for the cartesian product of inputs.
alias Pair = X * Y

## A residuated pattern reused across schemas.
alias Sentence = S \ NP
```

Object-shaped aliases are interchangeable with the underlying
object. Residuated patterns are stored as syntactic aliases and
substituted at schema use-sites; they cannot stand on their own as
morphism domains.

## Kernel

The `kernel` keyword declares a [Markov
kernel](https://ncatlab.org/nlab/show/Markov+kernel) `A -> B`.
Two shapes:

- **Without a `~` clause**: a finite-set lookup-table kernel
  `A -> D(B)`, realized as a learnable matrix of conditional
  probabilities.
- **With a `~ Family [options] [axis_role_clause]` clause**: a
  parametric kernel `A -> G(B)` whose family parameters come from
  the input by a parameter network at sample time. The optional
  [axis-role clause](dsl-programs-and-lets.md#axis-role-clause-over-and-iid-over)
  configures the family's event / batch decomposition over codomain
  factors.

<!-- compile: false -->
```qvr
# Lookup-table kernel on finite sets.
kernel s : X -> Y
kernel cat : X -> Y * Z

# Parametric kernel: input-conditional Normal on R^3.
kernel f : X -> R3 ~ Normal

# Family options control the parameter network.
kernel g : R3 -> R3 ~ Normal [scale=0.5]
kernel k : X -> S3 ~ Dirichlet

# 30+ families are registered; see the families guide.
kernel flow : R3 -> R3 ~ Flow [n_layers=6, hidden_dim=32]
```

The `kernel` keyword unifies the discrete (finite-set lookup) and
continuous / stochastic (parametric Family) cases under one surface;
the runtime branches on whether the codomain is a
[`FinSet`](../api/core/objects.md) or a
[`ContinuousSpace`](../api/continuous/spaces.md).

## Discretize

Convert a continuous space to a finite set via binning:

<!-- compile: false -->
```qvr
discretize d : U -> 20      # discretize UnitInterval into 20 bins
discretize d2 : R3 -> 100   # discretize R^3 into 100 bins
```

The runtime realization is
[`Discretize`](../api/continuous/boundaries.md#quivers.continuous.boundaries.Discretize),
which produces uniform-width bins on bounded supports.

## Embed

Embed a discrete object into a continuous space:

<!-- compile: false -->
```qvr
embed e : X -> R3   # treat X as uniform on R^3
```

The runtime realization is
[`Embed`](../api/continuous/boundaries.md#quivers.continuous.boundaries.Embed).

## Deduction

A `deduction NAME : Domain -> Codomain { ... }` block declares an
agenda-based weighted deduction. See the
[weighted deduction systems guide](deduction.md) for the framework
and the [semiring layer](../api/stochastic/semiring.md) for the
scoring algebra. The seven irreducible parameters of an
agenda-driven deduction (item algebra, rule set, semiring, axiom
source, goal predicate, start symbol, depth bound) become named
fields in the block:

<!-- compile: false -->
```qvr
deduction CCG : Term -> Term {
    atoms {
        NP, S, N, VP, PP,
        Fwd, Bwd,
        span
    }

    rule fwd_app
        : span(I, K, Fwd(X, Y)), span(K, J, Y)
        |- span(I, J, X)

    rule bwd_app
        : span(I, K, Y), span(K, J, Bwd(X, Y))
        |- span(I, J, X)

    semiring  LogProb
    start     S
    depth     6
}
```

- **`atoms { ... }`** declares the closed constructor universe.
  Every identifier appearing in a rule pattern must be either an
  atom or a single-uppercase wildcard variable (`X`, `Y`, `Z`,
  `I`, `J`, `K`, ...).
- **`rule NAME : premises |- conclusion`** is a sequent. Premises
  are comma-separated; arity is arbitrary (unary rules fire on a
  single chart cell, binary on a pair of adjacent cells, etc.).
- **`semiring`** selects the scoring algebra: `LogProb`,
  `Viterbi`, `Boolean`, `Counting`, or `ProductFuzzyAlgebra`.
- **`start`** declares the goal-item predicate (the start atom).
- **`depth`** bounds derivation depth so the agenda terminates on
  any finite input.

Pattern variables are single uppercase identifiers; every other
identifier in a rule pattern must be listed in `atoms`. A variable
appearing more than once in the same rule must unify across
occurrences.

Slash, tensor, and modal type constructors are *user-declared
atoms*, not built-in syntax: `Fwd(X, Y) ≡ X/Y`, `Bwd(X, Y) ≡ X\Y`,
`Tns(X, Y) ≡ X⊗Y`, `Dia(X) ≡ ◇X`, `Box(X) ≡ □X`,
`Cont(X) ≡ continuation-typed X`. The user is free to introduce
additional constructors for any algebra the deduction reasons over.

### Axiom sources

A deduction needs an axiom-injection kernel that maps an input into
initial weighted chart items. The block admits three forms:

<!-- compile: false -->
```qvr
deduction PCFG : Term -> Term {
    atoms { S, NP, VP, Det, N, V, the, cat, sleeps, span, leaf }
    rule branch : span(I, K, B), span(K, J, C) |- span(I, J, A)
    rule anchor : leaf(I, T)                    |- span(I, J, A)

    # Inline lexicon: label-indexed lookup.
    lexicon {
        "the"    : Det = the    @ learnable
        "cat"    : N   = cat    @ learnable
        "sleeps" : V   = sleeps @ learnable
    }

    semiring LogProb
    start    S
}
```

The three alternatives:

| Form | Use when |
|------|----------|
| `lexicon { "word" : Cat = lf @ learnable ... }` | label-indexed lookup table inline in the block |
| `lexicon from "path.tsv" with learnable` | label-indexed lookup loaded from a TSV at compile time |
| `axioms = some_morphism` | general kernel `Input -> List(Item × K)` defined as a declared morphism |

Marking a lexicon entry `@ learnable` allocates an
[`nn.Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Parameter.html)
log-weight initialised to `0.0`.

### Chart-query expressions

The runtime view of a compiled deduction exposes the chart as a
first-class differentiable value with four query methods. Inside a
`program` block (Kleisli-bind sigil `<-`), a deduction call yields
a `chart` value:

<!-- compile: false -->
```qvr
program parse_score : Sentence -> Real ! Sample, Score
    chart <- CCG(input)
    let w = chart.goal_weight()
    observe valid <- Bernoulli(sigmoid(w))
    return w
```

- `chart.weight(item)`: log-weight of a fully-determined item.
- `chart.enumerate(pattern)`: list of `(item, weight)` pairs
  matching a pattern with wildcards.
- `chart.derivations(item)`: derivation forest under the derivation
  semiring.
- `chart.goal_weight()`: log-weight of the goal predicate.

Each returns a
[`torch.Tensor`](https://docs.pytorch.org/docs/stable/tensors.html)
whose gradients flow back through the agenda's semiring operations
to any `learnable` axiom or rule weight.

## Replicated declarations

Declare N independent copies of a morphism. Each copy has
independent parameters; the base name becomes a group that can be
referenced by [`fan`](#fan-out-diagonal-morphism):

<!-- compile: false -->
```qvr
# creates head_0, head_1, head_2, head_3 with independent parameters
kernel head[4] : Latent -> HeadOut ~ Normal [scale=0.1]

# works on every declaration kind that takes a name and a type signature:
kernel T[3] : State -> Obs                       # lookup-table kernel
kernel emit[3] : State -> Obs ~ Normal           # parametric kernel
embed tok[2] : Token -> Hidden                   # finite-to-Euclidean
```

## Combinators on morphisms

These combinators appear inside `let` declarations to assemble
morphisms from simpler pieces. They live at the expression level,
outside `program` blocks.

### Fan-out (diagonal morphism)

Copy a single input to N morphisms and concatenate their outputs.
Accepts explicit morphism names or a group name from a replicated
declaration:

<!-- compile: false -->
```qvr
# explicit: fan-out to three named morphisms
let parallel = fan(f, g, h)

# group expansion: fan(head) expands to fan(head_0, head_1, head_2, head_3)
kernel head[4] : Latent -> HeadOut ~ Normal [scale=0.1]

let multi_head = fan(head)

# commonly followed by a projection to recombine
kernel proj : Combined -> Latent ~ Normal [scale=0.1]

let attention = fan(head) >> proj
```

All component morphisms must have the same domain. The output
dimension is the sum of all component codomain dimensions.

### Repeat (iterated composition)

Compose a morphism (or composed expression) with itself N times.
Two forms are available.

**Static repeat.** Count known at compile time, unrolled into a
fixed composition chain:

<!-- compile: false -->
```qvr
# transition >> transition >> transition
let deep = repeat(transition, 3)

# works with composed expressions too
let layer = attn >> residual >> ffn >> residual
let deep_model = repeat(layer, 6)

# repeat(f, 1) = f
let same = repeat(f, 1)
```

**Runtime-variable repeat.** Count omitted, creates a
[`RepeatMorphism`](../api/core/morphisms.md#quivers.core.morphisms.RepeatMorphism)
whose step count is set via `Program.forward(n_steps=N)`. Uses
repeated squaring for $O(\log n)$ compositions:

<!-- compile: false -->
```qvr
kernel transition : State -> State
kernel emission : State -> Obs

# runtime-variable: no count specified
let n_step = repeat(transition) >> emission

export n_step
```

```python
prog = load("hmm.qvr")
obs_3 = prog(n_steps=3)    # T^3 >> E
obs_50 = prog(n_steps=50)  # T^50 >> E, same model, different length
```

The morphism's codomain must match its domain (endomorphism) for
repeat to work.

### Stack (independent multi-layer)

Create N independent deep copies of a morphism, each with its own
parameters (no weight-tying):

<!-- compile: false -->
```qvr
# stack creates independent parameters per layer
let deep = stack(transition, 3)  # 3 layers, each with own params

# repeat reuses the same parameters (weight-tying)
let tied = repeat(transition, 3)  # 3 iterations, shared params
```

Unlike `repeat`, which composes a morphism with itself using the
same parameters, `stack(f, N)` creates N fresh deep copies of f,
each with independent learnable parameters. This is essential for
deep neural networks where each layer has distinct parameters.

### Scan (temporal recurrence)

Thread hidden state across a sequence using a recurrent cell:

<!-- compile: false -->
```qvr
# Basic syntax: cell has product domain A * H -> H
kernel cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]

let rnn = tok_embed >> scan(cell) >> output_proj

# With learned initial state (default is zeros)
let rnn_learned = tok_embed >> scan(cell, init=learned) >> output_proj
```

The `scan` combinator implements temporal recurrence by threading
hidden state `H` across a sequence:

- **Cell signature.** The morphism passed to `scan` must have a
  product domain `A * H -> H`, where `A` is the input type at each
  step and `H` is the hidden state type. The codomain must equal
  `H` (an endomorphism on the hidden state).
- **Execution.** Given a sequence of inputs `[x_0, x_1, ..., x_T]`
  (implicit in batch-first tensor shape `[batch, seq_len,
  input_dim]`), scan computes:
  - `h_0 = zeros(H)` or a learned initial state (if `init=learned`)
  - `h_t = cell(x_t, h_{t-1})` for `t = 1..T`
  - Returns the final hidden state `h_T`
- **Type.** If `cell : A * H -> H`, then `scan(cell) : A -> H`.
  The sequence dimension is implicit in the tensor's second
  dimension.
- **Works with both forms.** Parametric kernels (`kernel cell : A *
  H -> H ~ Normal`) and `MonadicPrograms` (`program cell(x, h) : A *
  H -> H` with bind / let / return).

**Example: vanilla RNN.**

<!-- compile: false -->
```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 128
type Output = Euclidean 64

embed tok_embed : Token -> Embedded

kernel cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
kernel output_proj : Hidden -> Output ~ Normal [scale=0.1]

let rnn = tok_embed >> scan(cell) >> output_proj

export rnn
```

For deeper temporal models, stack multiple scans:

<!-- compile: false -->
```qvr
let deep_rnn = tok_embed >> scan(cell_1) >> scan(cell_2) >> output_proj
```

Each `scan` threads its own hidden state independently.

### Backward composition

Compose morphisms in reverse order using `<<`, or use [Kleisli
composition](https://ncatlab.org/nlab/show/Kleisli+category) `<=>`:

<!-- compile: false -->
```qvr
# forward composition (both equivalent):
let fg = f >> g
let fg = f >=> g

# backward composition:
let gf = g << f    # equivalent to f >> g
```

`<<` reverses the direction of composition; `>=>` is the Kleisli
composition operator (composes Markov kernels in
[`Stoch`](../api/stochastic/categories.md) or
[`Kern`](../api/stochastic/categories.md)).

### Curry combinators (residuation witnesses)

The `.curry_right` and `.curry_left` postfix combinators witness
the right- and left-[residuation](https://ncatlab.org/nlab/show/residuated+category)
isomorphisms. For an inner morphism `f : X * Y -> Z`:

| Postfix | Result |
|---------|--------|
| `f.curry_right` | morphism `X -> Z/Y` |
| `f.curry_left`  | morphism `Y -> X\Z` |

Underlying tensor data is unchanged; only the domain / codomain
factoring is reinterpreted. Validity requires the inner morphism's
domain to factor as a non-trivial product.

<!-- compile: false -->
```qvr
object X : 3
object Y : 4
object Z : 5

latent f : X * Y -> Z
let g = f.curry_right    # g : X -> Z/Y
export g
```

The [Lambek calculus](https://ncatlab.org/nlab/show/Lambek+calculus)
inference rules (forward / backward application) become *theorems*
derivable from `identity` + `curry`.

## Top-level `let` declarations

Compose morphisms and bind:

<!-- compile: false -->
```qvr
let fg = f >> g
let par = f @ g
let marg = fg.marginalize(Y)
let composed = f >> g >> h
```

For arithmetic and family-argument `let` expressions inside a
`program` block, see
[Programs and Let-Expressions](dsl-programs-and-lets.md#let-expressions-arithmetic-and-primitives).

### `where` clauses

Attach local let-bindings to a top-level let declaration using
`where`:

<!-- compile: false -->
```qvr
let model = embed >> layers >> output_proj

where

    let layers = stack(transition, 3)
    let embed = tok_embed
```

The `where` keyword introduces a block of local definitions scoped
to the parent let binding. This improves readability for complex
nested compositions.

## Structural compression: `signature`, `encoder`, `decoder`, `loss`

The structural-compression surface gives transformers, tree LSTMs,
GNNs, RNNs, autoregressive language models, and the vector
inside-outside parser as instances of one interface. A
`signature { ... }` block declares sorts and constructors over a
free term algebra. An `encoder { ... }` block declares a
per-constructor parametric function realizing the
[F-algebra homomorphism](https://ncatlab.org/nlab/show/algebra+over+an+endofunctor)
from terms to vectors. A `decoder { ... }` block declares the dual
Kleisli arrow from vectors back to terms. A `loss { ... }` block
attaches a training objective at any site in the graph.

The four blocks are detailed in
[Structural Compression: Signatures and Encoders](structural-compression-signatures-and-encoders.md)
and
[Structural Compression: Decoders and Losses](structural-compression-decoders-and-losses.md).

## Export

Export a morphism as a compiled program output. Any number of
`export` declarations may appear per module; each is compiled into
a separate output:

<!-- compile: false -->
```qvr
export f
export fg
export my_prog
```
