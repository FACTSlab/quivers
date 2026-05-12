# The QVR DSL

## Overview

The `.qvr` (quivers) DSL is a declarative language for specifying morphism networks. A `.qvr` file declares objects, spaces, morphisms, and their compositions, then compiles to a trainable `nn.Module` (the `Program`).

The compilation pipeline is:

```
.qvr source
  → panproto tree-sitter parser (qvr grammar)
  → AST (didactic dx.Model nodes)
  → Compiler + resolution lenses
  → Program (nn.Module)
```

Parsing is delegated to [panproto](https://panproto.dev): the QVR tree-sitter grammar at `grammars/qvr/` is registered with the `panproto-grammars-all` distribution, and `quivers.dsl.parser` walks the panproto-produced parse tree, building a tree of `dx.Model` AST nodes (see [`ast_nodes`](../api/dsl/ast_nodes.md)). Resolution from syntactic `TypeExpr` / `SpaceExpr` trees to runtime `SetObject` / `ContinuousSpace` values is expressed as a `dx.Lens` family in [`resolution.py`](../api/dsl/resolution.md). Each compiled program also extracts to a panproto `Schema` via [`program_theory`](../api/dsl/program_theory.md), so diff/migrate/lens-generation tooling applies directly to `.qvr` programs.

Use the high-level API:

```python
from quivers.dsl import loads, load

# Compile from string
prog = loads('''
    object X : 3
    object Y : 4
    latent f : X -> Y
    export f
''')

# Compile from file
prog = load("model.qvr")

# Now a trainable nn.Module
optimizer = torch.optim.Adam(prog.parameters())
```

## Grammar

The authoritative grammar is the tree-sitter source at `grammars/qvr/grammar.js` in the quivers repository. The summary below is a human-readable EBNF view of the same productions; the tree-sitter grammar is the source of truth.

```ebnf
module         := statement*

statement      := quantale_decl
                | deduction_decl
                | object_decl
                | morphism_decl
                | space_decl
                | continuous_decl
                | stochastic_decl
                | discretize_decl
                | embed_decl
                | program_decl
                | let_decl
                | type_decl
                | export_decl

quantale_decl  := 'quantale' ('product_fuzzy' | 'boolean'
                              | 'lukasiewicz' | 'godel' | 'tropical')

# Weighted deduction system: the agenda-based framework subsumes
# CKY, Earley, Viterbi, inside-outside, semi-naïve Datalog, A*,
# Knuth, and bidirectional MLTT proof search.
deduction_decl := 'deduction' IDENT ':' type_expr '->' type_expr
                  '{' deduction_field+ '}'
deduction_field
               := atoms_field | sequent_rule | semiring_field
                | start_field | depth_field
                | lexicon_block | lexicon_from_file
                | axioms_field
                | signature_field | encoder_field
atoms_field    := 'atoms' '{' IDENT (',' IDENT)* '}'
sequent_rule   := 'rule' IDENT ':' term_pattern (',' term_pattern)*
                  ('|-' | '⊢') term_pattern
term_pattern   := IDENT | IDENT '(' term_pattern (',' term_pattern)* ')'
semiring_field := 'semiring' ('LogProb' | 'Viterbi' | 'Boolean'
                              | 'Counting' | 'ProductFuzzy')
start_field    := 'start' IDENT
depth_field    := 'depth' INT
lexicon_block  := 'lexicon' '{' lexicon_entry+ '}'
lexicon_entry  := STRING ':' type_expr '=' let_expr ['@' 'learnable']
lexicon_from_file
               := 'lexicon' 'from' STRING ['with' 'learnable']
axioms_field   := 'axioms' '=' IDENT
signature_field := 'signature' IDENT
encoder_field  := 'encoder' IDENT

# Object declarations come in three forms:
#   object X : 3                                     — anonymous-element FinSet
#   object Atoms = {NP, S, VP}                       — EnumSet
#   object Cat = FreeResiduated(Atoms, depth=4, ops=[slash])  — residuated universe
object_decl    := 'object' IDENT (':' type_expr | '=' object_init)
object_init    := enum_set_literal | free_residuated_expr
enum_set_literal := '{' IDENT (',' IDENT)* '}'
free_residuated_expr := 'FreeResiduated' '(' IDENT
                       (',' free_residuated_arg)* ')'
free_residuated_arg  := 'depth' '=' INT
                      | 'ops' '=' '[' IDENT (',' IDENT)* ']'
free_monoid_expr := 'FreeMonoid' '(' IDENT ',' 'max_length' '=' INT ')'

# TypeExpr is the unified pattern sublanguage. Slash and effect-typed
# forms are legal inside any TypeExpr; the compiler enforces
# residuated-universe constraints on slash patterns at use-site.
type_expr      := type_coproduct
                | type_slash
                | type_product
                | type_effect_apply
                | primary_type
type_coproduct := type_expr '+' type_expr
type_slash     := type_expr ('/' | '\') type_expr
type_product   := type_expr '*' type_expr
type_effect_apply := IDENT '(' type_expr (',' type_expr)* ')'
primary_type   := IDENT | INT | '(' type_expr ')'

morphism_decl  := ('latent' | 'observed') IDENT ':' type_expr '->' type_expr
                  ['[' options ']'] ['=' expr]
options        := IDENT '=' value (',' IDENT '=' value)*
value          := IDENT | INT | FLOAT

space_decl     := 'space' IDENT ':' space_expr
space_expr     := space_product
space_product  := space_primary ('*' space_primary)*
space_primary  := IDENT '(' space_args ')' | IDENT | IDENT INT
space_args     := space_arg (',' space_arg)*
space_arg      := IDENT '=' value | value

type_decl      := 'type' IDENT '=' space_expr

continuous_decl := 'continuous' IDENT ['[' INT ']'] ':' IDENT '->' IDENT
                   '~' IDENT ['[' options ']']

stochastic_decl := 'stochastic' IDENT ['[' INT ']'] ':' type_expr '->' type_expr

discretize_decl := 'discretize' IDENT ':' IDENT '->' INT

embed_decl      := 'embed' IDENT ['[' INT ']'] ':' IDENT '->' IDENT

program_decl   := 'program' IDENT ['(' param_list ')'] ':'
                   type_expr '->' type_expr
                   ['!' effect_set]
                   ['over' IDENT]
                   program_body
param_list     := program_param (',' program_param)*
program_param  := IDENT | IDENT ':' param_kind
param_kind     := 'FinSet' | 'Space' | 'Object'
                | 'Real' | 'Nat'
                | 'Mor' '[' type_expr ',' type_expr ']'
effect_set     := effect (',' effect)*
effect         := 'Sample' | 'Score' | 'Marginal' | 'Pure'

program_body   := program_step+ return_stmt

program_step   := bind_step | observe_step
                | marginalize_step | let_step

# Kleisli bind — the unique sampling step shape.
#   v        <- F(args)              -- scalar draw
#   v : A    <- F(args)              -- A-indexed plate
#   (a, b)   <- F(args)              -- destructuring tuple bind
bind_step      := var_pattern [':' type_expr] '<-' IDENT
                  ['(' draw_arg_list ')']

# Scored bind — same shape as bind_step, prefixed with `observe`.
#   observe v        <- F(args)
#   observe r : N    <- F(theta[N])
observe_step   := 'observe' IDENT [':' type_expr] '<-' IDENT
                  ['(' draw_arg_list ')']

# Scoped marginalisation — coordinate `c` is bound to `F(args)`,
# optionally `A`-indexed; the steps in the `{ … }` body are the
# integration scope. At end of scope the coordinate is pushed
# forward through projection (logsumexp for discrete, fibrewise
# integration for continuous).
marginalize_step := 'marginalize' IDENT [':' type_expr] '<-' IDENT
                    ['(' draw_arg_list ')']
                    'in' '{' program_step* '}'

let_step       := 'let' IDENT '=' let_expr
let_expr       := let_term (('+' | '-') let_term)*
let_term       := let_unary (('*' | '/') let_unary)*
let_unary      := '-' let_atom | let_atom
let_atom       := IDENT '(' let_expr (',' let_expr)* ')'
                | IDENT '[' let_expr (',' let_expr)* ']'
                | IDENT | INT | FLOAT | '(' let_expr ')'
var_pattern    := IDENT | '(' IDENT (',' IDENT)* ')'

# A family argument may be a numeric literal, an identifier, or a
# bracket-indexed family section `theta[N]` denoting a section of
# the N-indexed family `theta : N → P`.
draw_arg_list  := draw_arg (',' draw_arg)*
draw_arg       := IDENT '[' type_expr ']'
                | '-' (INT | FLOAT) | IDENT | INT | FLOAT

return_stmt    := 'return' return_pattern
return_pattern := IDENT | '(' IDENT (',' IDENT)* ')'

let_decl       := 'let' IDENT '=' expr ['where' let_decl+]
expr           := compose_expr
compose_expr   := tensor_expr (('>>' | '>=>' | '<<') tensor_expr)*
tensor_expr    := postfix_expr ('@' postfix_expr)*
postfix_expr   := atom_expr ('.' method_call)*
method_call    := 'marginalize' '(' IDENT (',' IDENT)* ')'
                | 'curry_right'
                | 'curry_left'
atom_expr      := 'identity' '(' IDENT ')'
                | 'fan' '(' expr (',' expr)* ')'
                | 'repeat' '(' expr [',' INT] ')'
                | 'stack' '(' expr ',' INT ')'
                | 'scan' '(' expr [',' scan_init] ')'
                | IDENT
                | '(' expr ')'

scan_init      := 'init' '=' ('zeros' | 'learned')

export_decl    := 'export' expr
```

## Declarations

### Quantale

Choose the enriching quantale (optional, defaults to `product_fuzzy`):

```qvr
quantale product_fuzzy
quantale boolean
quantale lukasiewicz
quantale godel
quantale tropical
```

### Deduction

A `deduction NAME : Domain -> Codomain { … }` block declares an
agenda-based weighted deduction. The seven irreducible parameters
of an agenda-driven deduction — item algebra, rule set, semiring,
axiom source, goal predicate, start symbol, depth bound — become
named fields in the block:

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

- **`atoms { … }`** declares the closed constructor universe.
  Every identifier appearing in a rule pattern must be either an
  atom or a single-uppercase wildcard variable (`X`, `Y`, `Z`,
  `I`, `J`, `K`, …).
- **`rule NAME : premises |- conclusion`** is a sequent. Premises
  are comma-separated; arity is arbitrary (unary rules fire on a
  single chart cell, binary on a pair of adjacent cells, etc.).
- **`semiring`** selects the scoring algebra: `LogProb`,
  `Viterbi`, `Boolean`, `Counting`, or `ProductFuzzy`.
- **`start`** declares the goal-item predicate (the start atom).
- **`depth`** bounds derivation depth so the agenda terminates on
  any finite input.

Pattern variables are single uppercase identifiers; every other
identifier in a rule pattern must be listed in `atoms`. A variable
appearing more than once in the same rule must unify across
occurrences.

Slash, tensor, and modal type constructors are *user-declared
atoms*, not built-in syntax: `Fwd(X, Y) ≡ X/Y`,
`Bwd(X, Y) ≡ X\Y`, `Tns(X, Y) ≡ X⊗Y`, `Dia(X) ≡ ◇X`,
`Box(X) ≡ □X`, `Cont(X) ≡ continuation-typed X`. The user is free
to introduce additional constructors for any algebra the
deduction reasons over.

#### Axiom sources

A deduction needs an axiom-injection kernel that maps an input
into initial weighted chart items. The block admits three forms:

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
| `lexicon { "word" : Cat = lf @ learnable … }` | label-indexed lookup table inline in the block |
| `lexicon from "path.tsv" with learnable` | label-indexed lookup loaded from a TSV at compile time |
| `axioms = some_morphism` | general kernel `Input → List(Item × K)` defined as a declared morphism |

Marking a lexicon entry `@ learnable` allocates an
`nn.Parameter` log-weight initialised to `0.0`.

#### Chart-query expressions

The runtime view of a compiled deduction exposes the chart as a
first-class differentiable value with four query methods. Inside
a `program` block (Kleisli-bind sigil `<-`), a deduction call
yields a `chart` value:

<!-- compile: false -->
```qvr
program parse_score : Sentence -> Real ! Sample, Score
    chart <- CCG(input)
    let w = chart.goal_weight()
    observe valid <- Bernoulli(sigmoid(w))
    return w
```

- `chart.weight(item)` — log-weight of a fully-determined item.
- `chart.enumerate(pattern)` — list of `(item, weight)` pairs
  matching a pattern with wildcards.
- `chart.derivations(item)` — derivation forest under the
  derivation semiring.
- `chart.goal_weight()` — log-weight of the goal predicate.

Each returns a `torch.Tensor` whose gradients flow back through
the agenda's semiring operations to any `learnable` axiom or rule
weight.

### Doc Comments

Lines starting with `##` are *doc comments*: they're attached to the
declaration that immediately follows and surface through the AST,
the panproto schema, and tooling (`qvr check --json`, future LSP
hover). Plain `#` line comments are dropped at parse time.

```qvr
## The terminal vocabulary; cardinality 256 is one byte.
object Token : 256

## Latent token-to-category embedding learned during training.
latent emit : Token -> Token
```

Doc comments are recognised on `object`, `morphism`, `alias`, and
`program` declarations.

### Alias

`alias` declarations bind a short name to a type-level expression:

<!-- compile: false -->
```qvr
## A short alias for the cartesian product of inputs.
alias Pair = X * Y

## A residuated pattern reused across schemas.
alias Sentence = S \ NP
```

Object-shaped aliases (resolvable to a `SetObject`) are interchangeable
with the underlying object — `latent f : Pair -> X` works. Residuated
patterns are stored as syntactic aliases and substituted at schema
use-sites; they cannot stand on their own as morphism domains.

### Object

Three surface forms:

```qvr
# 1. anonymous-element FinSet of given cardinality, or a TypeExpr
object X : 3          # FinSet("X", 3)
object Y : 4
object XY : X * Y     # ProductSet(X, Y)
object Sum : X + Y    # CoproductSet(X, Y)
object Free = FreeMonoid(X, max_length=2)  # FreeMonoid(generators=X, max_length=2)

# 2. FreeMonoid — bounded Kleene closure over a FinSet of generators.
object Strings = FreeMonoid(X, max_length=4)
```

### Morphism

Declare a learnable or fixed morphism:

<!-- compile: false -->
```qvr
# Latent (learnable)
latent f : X -> Y

# With init scale
latent g : Y -> Z [scale=0.3]

# Observed (fixed)
observed h : X -> X = identity(X)

# Observed with explicit tensor (not supported directly; define in compiler)
```

### Space

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

### Continuous Morphism

Declare a conditional distribution:

<!-- compile: false -->
```qvr
# Conditional normal: X → ℝ³
continuous f : X -> R3 ~ Normal

# Conditional with family and options
continuous g : R3 -> R3 ~ Normal [scale=0.5]
continuous k : X -> S3 ~ Dirichlet

# 30+ families supported (see continuous guide)
continuous flow : R3 -> R3 ~ Flow [n_layers=6, hidden_dim=32]
```

### Stochastic Morphism

Declare a Markov kernel (stochastic matrix):

<!-- compile: false -->
```qvr
stochastic s : X -> Y
stochastic cat : X -> (Y * Z)
```

### Discretize

Convert continuous space to finite set via binning:

<!-- compile: false -->
```qvr
discretize d : U -> 20      # discretize UnitInterval into 20 bins
discretize d2 : R3 -> 100   # discretize ℝ³ into 100 bins
```

### Embed

Embed discrete into continuous:

<!-- compile: false -->
```qvr
embed e : X -> R3   # treat X as uniform on ℝ³
```

### Replicated Declarations

Declare N independent copies of a morphism. Each copy has independent parameters; the base name becomes a group that can be referenced by `fan`:

<!-- compile: false -->
```qvr
# creates head_0, head_1, head_2, head_3 with independent parameters
continuous head[4] : Latent -> HeadOut ~ Normal [scale=0.1]

# works with stochastic and embed too
stochastic kernel[3] : State -> Obs

embed tok[2] : Token -> Hidden
```

### Fan-Out (Diagonal Morphism)

Copy a single input to N morphisms and concatenate their outputs. Accepts explicit morphism names or a group name from a replicated declaration:

<!-- compile: false -->
```qvr
# explicit: fan-out to three named morphisms
let parallel = fan(f, g, h)

# group expansion: fan(head) expands to fan(head_0, head_1, head_2, head_3)
continuous head[4] : Latent -> HeadOut ~ Normal [scale=0.1]

let multi_head = fan(head)

# commonly followed by a projection to recombine
continuous proj : Combined -> Latent ~ Normal [scale=0.1]

let attention = fan(head) >> proj
```

All component morphisms must have the same domain. The output dimension is the sum of all component codomain dimensions.

### Repeat (Iterated Composition)

Compose a morphism (or composed expression) with itself N times. Two forms are available:

**Static repeat.** Count known at compile time, unrolled into a fixed composition chain:

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

**Runtime-variable repeat.** Count omitted, creates a `RepeatMorphism` whose step count is set via `Program.forward(n_steps=N)`. Uses repeated squaring for O(log n) compositions:

<!-- compile: false -->
```qvr
stochastic transition : State -> State
stochastic emission : State -> Obs

# runtime-variable: no count specified
let n_step = repeat(transition) >> emission

export n_step
```

```python
prog = load("hmm.qvr")
obs_3 = prog(n_steps=3)    # T^3 >> E
obs_50 = prog(n_steps=50)  # T^50 >> E — same model, different length
```

The morphism's codomain must match its domain (endomorphism) for repeat to work.

### Stack (Independent Multi-Layer)

Create N independent deep copies of a morphism, each with its own parameters (no weight-tying):

<!-- compile: false -->
```qvr
# stack creates independent parameters per layer
let deep = stack(transition, 3)  # 3 layers, each with own params

# repeat reuses the same parameters (weight-tying)
let tied = repeat(transition, 3)  # 3 iterations, shared params
```

Unlike `repeat`, which composes a morphism with itself using the same parameters, `stack(f, N)` creates N fresh deep copies of f, each with independent learnable parameters. This is essential for deep neural networks where each layer has distinct parameters.

### Scan (Temporal Recurrence)

Thread hidden state across a sequence using a recurrent cell:

<!-- compile: false -->
```qvr
# Basic syntax: cell has product domain A * H -> H
continuous cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]

let rnn = tok_embed >> scan(cell) >> output_proj

# With learned initial state (default is zeros)
let rnn_learned = tok_embed >> scan(cell, init=learned) >> output_proj
```

The `scan` combinator implements temporal recurrence by threading hidden state `H` across a sequence:

- **Cell signature:** The morphism passed to `scan` must have a product domain `A * H -> H`, where `A` is the input type at each step and `H` is the hidden state type. The codomain must equal `H` (an endomorphism on the hidden state).

- **Execution:** Given a sequence of inputs `[x_0, x_1, ..., x_T]` (implicit in batch-first tensor shape `[batch, seq_len, input_dim]`), scan computes:
  - `h_0 = zeros(H)` or a learned initial state (if `init=learned`)
  - `h_t = cell(x_t, h_{t-1})` for `t = 1..T`
  - Returns the final hidden state `h_T`

- **Type:** If `cell : A * H -> H`, then `scan(cell) : A -> H`. The sequence dimension is implicit in the tensor's second dimension.

- **Works with both forms:**
  - **ContinuousMorphisms:** `continuous cell : A * H -> H ~ Normal`
  - **MonadicPrograms:** `program cell(x, h) : A * H -> H` with draw/let/return

- **Product domains:** The continuous declaration syntax now supports product types:
  ```qvr
continuous cell : InputType * HiddenType -> HiddenType ~ Normal [scale=0.1]
```

**Example: Vanilla RNN**
```qvr
object Token : 256
type Embedded = Euclidean 64
type Hidden = Euclidean 128
type Output = Euclidean 64

embed tok_embed : Token -> Embedded

continuous cell : Embedded * Hidden -> Hidden ~ Normal [scale=0.1]
continuous output_proj : Hidden -> Output ~ Normal [scale=0.1]

let rnn = tok_embed >> scan(cell) >> output_proj

export rnn
```

For deeper temporal models, stack multiple scans:
<!-- compile: false -->
```qvr
let deep_rnn = tok_embed >> scan(cell_1) >> scan(cell_2) >> output_proj
```

Each `scan` threads its own hidden state independently.

### Kleisli Bind Syntax

The `<-` operator is the unique sampling-step sigil in a `program` body:

<!-- compile: false -->
```qvr
x <- Normal(0.0, 1.0)
```

It introduces `x` as a random variable distributed according to the given family. The same sigil carries every sampling-step variant — scalar draws, indexed plates, scored observes, and scoped marginalisations — distinguished by the surrounding shape (see the program-block section below).

### Backward Composition

Compose morphisms in reverse order using `<<` or use Kleisli composition `<=>`:

<!-- compile: false -->
```qvr
# forward composition (both equivalent):
let fg = f >> g
let fg = f >=> g

# backward composition:
let gf = g << f    # equivalent to f >> g
```

The backward composition operator `<<` reverses the direction of composition, and `<=>` is an alias for `>>` in Kleisli categories (used with stochastic and continuous morphisms).

### Type Alias

Declare a space alias using `type` (alternative to `space`):

```qvr
# the `space` and `type` forms below are interchangeable:
space HiddenSpace : Euclidean(64)

type Hidden = Euclidean 64     # ML-style, parens optional

# product types built from previously-declared spaces
type Output   = Euclidean 32
type Combined = Hidden * Output
```

The `type` keyword provides a more concise, ML-style syntax for declaring named spaces. Parentheses around arguments are optional.

### Where Clauses

Attach local let-bindings to a let declaration using `where`:

<!-- compile: false -->
```qvr
let model = embed >> layers >> output_proj

where

    let layers = stack(transition, 3)
    let embed = tok_embed
```

The `where` keyword introduces a block of local definitions that are scoped to the parent let binding. This improves readability for complex nested compositions.

### Curry Combinators (Residuation Witnesses)

The `.curry_right` and `.curry_left` postfix combinators witness the right- and left-residuation isomorphisms. For an inner morphism `f : X * Y -> Z`:

| Postfix | Result |
|---------|--------|
| `f.curry_right` | morphism `X -> Z/Y` |
| `f.curry_left`  | morphism `Y -> X\Z` |

Underlying tensor data is unchanged; only the domain/codomain factoring is reinterpreted. Validity requires the inner morphism's domain to factor as a non-trivial product.

```qvr
object X : 3
object Y : 4
object Z : 5

latent f : X * Y -> Z
let g = f.curry_right    # g : X -> Z/Y
export g
```

The Lambek calculus inference rules (forward / backward application) become *theorems* derivable from `identity` + `curry`.

### Program

Define a probabilistic program. The body is a sequence of *steps* (bind, observe, let, marginalize) followed by `return`. Each step is a Kleisli arrow on the accumulated random-variable context $\Phi$; the program denotes the composite $\Gamma \to \mathcal{G}(\tau_2)$ in $\mathbf{Kern}$.

<!-- compile: false -->
```qvr
program my_prog : X -> Y
    mu <- LogitNormal(0.0, 1.0)
    x <- Normal(mu, 1.0)

    return x

program with_params(a, b) : (X * Z) -> Y
    let w = a

    x <- f(w)
    y <- g(x, b)
    return y
```

#### Effect Signatures

A program declaration may carry an effect signature after `!`, a comma-separated subset of `{Sample, Score, Marginal, Pure}`. The compiler verifies that the body's actual effects are a subset of the declared set; `! Pure` rejects any sample / score / marginal binds.

<!-- compile: false -->
```qvr
program prior : Unit -> Y ! Sample
    mu <- Normal(0.0, 1.0)
    return mu

program deterministic : X -> X ! Pure
    let y = x
    return y
```

#### Indexed Bind (Plate)

`v : A <- Family(args)` declares `v` as an $A$-indexed family of independent $F$-distributed draws. Categorically `v : A → \mathcal{G}(K)` where `K` is the per-fiber codomain taken from the family; equivalently a single arrow $\mathbf{1} \to \mathcal{G}(K^A)$ via the natural isomorphism $\mathbf{Kern}(\mathbf{1}, K^A) \cong \mathbf{Kern}(A, K)$.

<!-- compile: false -->
```qvr
object Item : 1000

duration_incr : Item <- HalfNormal(1.0)
by_subject    : Subject <- Normal(0.0, sigma)
```

#### Indexed Observe

`observe r : N <- Family(args)` accumulates a batched log-likelihood: a sub-probability kernel $\Phi \to \mathcal{G}_{\le 1}(\Phi)$ with score $\prod_{n \in N} p_F(r_{\mathrm{obs}}(n); \theta(n, \phi))$. The response buffer `r` is supplied at runtime via the `observations` dict passed to `MonadicProgram.rsample` / `log_joint` / `ELBO.forward`. Family arguments may use bracket-indexed sections `theta[N]` to refer to plate variables.

<!-- compile: false -->
```qvr
observe cloze_resp : RespCloze <- Bernoulli(intercept_cloze)
```

#### Scoped Marginalize

`marginalize c : A <- F(args) in { … }` introduces a coordinate `c` bound to a kernel `F(args)`, optionally `A`-indexed, with the `{ … }` block as its integration scope. At the end of the scope the coordinate is pushed forward through the projection $\pi : \Phi \times C \to \Phi$, integrating it out by log-sum-exp on the log-likelihood (discrete) or fibrewise integration (continuous); `c` then falls out of scope.

<!-- compile: false -->
```qvr
marginalize class : Item <- Categorical(class_logits) in {
    observe r : N <- Bernoulli(theta[class[N]])
}
```

#### Indexed Gather in `let`

A `let`-expression of the form `arr[idx]` denotes the Kleisli pullback of a plate variable along a finite fibration. For a plate `v : A -> B` and an index morphism $\iota : N \to A$, the gather $\iota^* v = v \circ \iota$ is itself a $\mathbf{Kern}$-morphism $N \to B$.

<!-- compile: false -->
```qvr
by_verb : Verb <- Normal(0.0, sigma)
let intercept_for_item = by_verb[verb_of_item]
```

#### Parametric Programs

A `program` declaration whose parameter list contains *typed* parameters denotes a dependent family of kernels rather than a single kernel:

$$
\llbracket P \rrbracket \;:\; \prod_{p_1 : P_1} \cdots \prod_{p_k : P_k} \mathbf{Kern}\bigl(\mathrm{dom}(p), \mathrm{cod}(p)\bigr).
$$

Three parameter universes are available:

| Kind | Universe | Quantifies over |
|---|---|---|
| `FinSet`, `Space`, `Object` | object of the relevant subcategory | the carrier of a plate |
| `Real`, `Nat` | hom-object of scalar type | a hyperparameter value |
| `Mor[A, B]` | the hom-set $\mathbf{Kern}(A, B)$ | a kernel passed in by name |

Parametric programs are *not* compiled to runtime `MonadicProgram`s in isolation; the compiler stores them as templates and inlines them at each call site:

<!-- compile: false -->
```qvr
v <- template(arg1, arg2, ...)
```

At each call site the template's body is substituted (formal parameters → actual arguments) and α-renamed (internal latents are prefixed by `v$`, the return variable is renamed to `v` directly). The renamed step list is inlined into the caller, so distinct call sites contribute distinct factors to the parent's joint kernel — fresh latents per use, no inadvertent tying.

```qvr
# Parametric random-intercepts template: one HalfNormal scale and
# a per-level Normal(0, σ) plate, polymorphic over the grouping
# object G and the half-normal hyperparameter scale.
program random_intercepts (G : FinSet, scale : Real) : G -> 1
    sigma <- HalfNormal(scale)
    v : G <- Normal(0.0, sigma)
    return v
```

#### Posterior Blocks

A `program name(latents) : domain -> codomain ! Pure over model` declaration denotes a deterministic post-conditioning kernel. The `over model` modifier marks the program as consuming the named model's latents; the consumed latents appear as data parameters in the parameter list. The `! Pure` effect signature rejects any sample, score, or marginal binds — the body is restricted to `let` (and `marginalize` over its own scope). Categorically it is a $\mathbf{Kern}$-morphism $\text{Latents} \to \tau_{\mathrm{out}}$ that lifts to $\text{Data} \to \mathcal{G}(\tau_{\mathrm{out}})$ by post-composition with the model's posterior kernel $q(\theta \mid \mathrm{data})$.

<!-- compile: false -->
```qvr
type Logits4 = Euclidean 4

program scored : Item -> Logits4
    raw_logits <- Normal(0.0, 1.0)
    return raw_logits

program class_probs(raw_logits) : Item -> Logits4 ! Pure over scored
    let probs = softmax(raw_logits)
    return probs
```

The data parameter `raw_logits` names the model latent the body consumes — a per-sample snapshot of the model's trace.

### Hierarchical Bayesian Models

The plate-draw, vectorised-observe, parametric-program, and `marginalize` constructs compose into idiomatic hierarchical Bayesian models. The pattern below shows crossed random intercepts on two grouping factors, both reusing a single parametric template:

```qvr
object Subject : 200
object Verb : 100
object Resp : 5000

program random_intercepts (G : FinSet, scale : Real) : G -> 1
    sigma <- HalfNormal(scale)
    v : G <- Normal(0.0, sigma)
    return v

program crossed : Resp -> Resp
    intercept <- Normal(0.0, 1.0)

    by_subject <- random_intercepts(Subject, 1.0)
    by_verb    <- random_intercepts(Verb,    1.0)

    observe response : Resp <- Bernoulli(intercept)
    return intercept

export crossed
```

Each call to `random_intercepts` inlines a fresh `sigma` and a fresh per-level plate `v` under α-renamed names (`by_subject$sigma`, `by_subject$v`, …), so the two grouping factors share *structure* but not *latents*. Monotone ordinal-spline coefficients are expressed as `cumsum` of `HalfNormal` increments; categorical latent classes are marginalised with a scoped `marginalize … in { … }` block.

### Let Expressions (Arithmetic)

Inside a `program` block, `let` bindings support full arithmetic with standard operator precedence, unary negation, and built-in functions:

<!-- compile: false -->
```qvr
# arithmetic: +, -, *, /
let eta = mu + sigma * z_raw + lambda * shared_factor
let adjusted = (1.0 - lapse) * p_raw + 0.5 * lapse
let mean = (x + y + z) / 3.0
let negated = -raw_score

# built-in functions: sigmoid, exp, log, abs, softplus,
# cumsum, softmax, cholesky_quad_form
let prob = sigmoid(eta)
let positive = softplus(raw)
let log_rate = log(rate)
let magnitude = abs(x - 0.5)
let monotone = cumsum(increments)
let weights = softmax(logits)
```

Each `let`-builtin denotes a deterministic measurable map, lifted into the Kleisli category as a Dirac kernel. `cumsum` realises the partial-sum endomorphism over a plate; `softmax` is the standard simplex map; `cholesky_quad_form(L, x)` computes $x^\top L L^\top x$ for a lower-triangular Cholesky factor `L`.

### Inline Distributions

Bind and observe steps support inline distribution construction with any mix of literal and variable arguments. All 11 distribution families support arbitrary combinations:

<!-- compile: false -->
```qvr
# all-literal (fixed): Unit -> codomain
x <- Normal(0.0, 1.0)
p <- Beta(2.0, 5.0)

# all-variable (direct): variables -> codomain
y <- Normal(mu, sigma)
b <- Bernoulli(theta)

# mixed literal/variable: any combination works
h_cand <- Normal(reset_hidden, 0.5)
z <- Normal(0.0, learned_scale)
r <- TruncatedNormal(mu, sigma, 0.0, 1.0)

# negative literals
z <- Normal(-1.5, 0.3)
```

The supported inline distribution families are:

| Family | Parameters | Codomain |
|---|---|---|
| `Normal` | `loc`, `scale` | Euclidean |
| `LogitNormal` | `mu`, `sigma` | UnitInterval |
| `Uniform` | `low`, `high` | UnitInterval / Euclidean |
| `Bernoulli` | `probs` | FinSet(2) |
| `Beta` | `concentration1`, `concentration0` | UnitInterval |
| `Exponential` | `rate` | PositiveReals |
| `HalfCauchy` | `scale` | PositiveReals |
| `HalfNormal` | `scale` | PositiveReals |
| `LogNormal` | `loc`, `scale` | PositiveReals |
| `Gamma` | `concentration`, `rate` | PositiveReals |
| `TruncatedNormal` | `mu`, `sigma`, `low`, `high` | Euclidean (bounded) |

Every parameter position in every family accepts either a literal value or a previously-bound variable. When all arguments are literals, a fixed distribution is created; when any argument is a variable, the general `MixedInlineDistribution` mechanism handles parameter resolution at runtime.

For conditional distributions (learned neural-network parameterization), use the `continuous` declaration instead.

### Let (Top-Level)

Compose morphisms and bind:

<!-- compile: false -->
```qvr
let fg = f >> g
let par = f @ g
let marg = fg.marginalize(Y)
let composed = f >> g >> h
```

### Export

Export a morphism as a compiled program output. Any number of `export` declarations may appear per module; each is compiled into a separate output:

<!-- compile: false -->
```qvr
export f
export fg
export my_prog
```

## Examples

### Simple Discrete Model

```qvr
object X : 3
object Y : 4

latent f : X -> Y
latent g : Y -> Y

let fg = f >> g

export fg
```

### Continuous Conditional Model

```qvr
object Cond : 2

space Latent : Euclidean(3)
space Obs : Euclidean(5)

continuous prior : Cond -> Latent ~ Normal
continuous likelihood : Latent -> Obs ~ Normal [scale=0.1]

let posterior = prior >> likelihood

export posterior
```

### Probabilistic Program with Observations

```qvr
object Data : 1

space Y : Euclidean(2)

program regression : Data -> Y
    theta <- LogitNormal(0.0, 1.0)
    y <- Normal(theta, 0.5)

    observe _ <- Normal(y, 0.1)

    return y
```

### Factivity Model (from examples)

```qvr
object Entity : 1
object Truth : 2
object Resp : 1

program factivity : Entity -> Truth * Truth * Truth * Resp
    theta_know <- LogitNormal(0.0, 1.0)
    theta_cg <- LogitNormal(0.0, 1.0)

    let cg_complement = 1.0 - theta_cg

    tau_know <- Bernoulli(theta_know)
    cg_matrix <- Bernoulli(theta_cg)
    sigma <- Uniform(0.0, 1.0)
    observe response <- TruncatedNormal(theta_know, sigma, 0.0, 1.0)
    return (tau_know, cg_complement, cg_matrix, response)
```

For more examples, see the [Examples Gallery](../examples/index.md). For a formal account of what `.qvr` programs *mean*, see the [Denotational Semantics](../semantics/index.md).

## Compilation Process

The `Compiler` transforms the AST to a `Program`:

1. **Resolve declarations**: collect all objects, spaces, morphisms. Type and space resolution is delegated to the lens family in `quivers.dsl.resolution` — `TypeExprToSetObject` (parameterized by the object inventory) and `SpaceExprToContinuousSpace` (parameterized by the space and object inventories). Each lens is `dx.Lens[<AST>, <runtime value>, <AST>]`; round-trip laws hold by construction.
2. **Type check**: ensure domains/codomains match in compositions.
3. **Build morphism DAG**: construct morphism modules.
4. **Wrap in Program**: create an `nn.Module` that manages all parameters.

```python
from quivers.dsl import Compiler, parse

source = "object X : 3\nlatent f : X -> X\nexport f"
ast = parse(source)
compiler = Compiler(ast)
program = compiler.compile()
```

### Programs as panproto schemas

After compilation, the resolved environment can be exported as a panproto `Schema` over `QVR_PROGRAM_PROTOCOL`:

```python
from quivers.dsl import loads, extract_program_schema

program = loads(source)
schema = extract_program_schema(program._compiler)  # Schema instance
```

The schema's vertices enumerate every declared object, space, and morphism (with kinds drawn from `finset`, `product_set`, `coproduct_set`, `free_monoid`, `empty_set`, `euclidean`, `simplex`, `positive_reals`, `product_space`, plus the declaration variants). This makes `panproto schema diff`, `panproto lens generate`, and the rest of the panproto toolbox available on `.qvr` programs without further work.

## Error Handling

The DSL provides two error types:

- `ParseError`: syntactic error (the tree-sitter grammar reported an error node, or a required field was missing in the parse tree)
- `CompileError`: semantic error (type mismatch, undefined name, ill-formed program structure)

```python
from quivers.dsl import loads, ParseError, CompileError

try:
    prog = loads(bad_source)
except ParseError as e:
    print(f"Parse error: {e}")
except CompileError as e:
    print(f"Compilation error: {e}")
```

Tree-sitter's lexer is integrated with the grammar, so lexical errors surface as `ParseError`.

## Comments

Lines starting with `#` are ignored:

```qvr
# This is a comment
object X : 3  # inline comment

# Define morphisms
latent f : X -> X
```

## Tips

1. **Always declare objects before using them** in morphisms.
2. **Quantale must come first** (if specified).
3. **Use let to name complex compositions** for clarity.
4. **Programs are the main output**: use them for inference (see inference guide).
5. **Type errors in composition** happen at compile time, not runtime.
