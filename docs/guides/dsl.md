# The QVR DSL

## Overview

The `.qvr` (quivers) DSL is a declarative language for specifying morphism networks. A `.qvr` file declares objects, spaces, morphisms, and their compositions, then compiles to a trainable `nn.Module` (the `Program`).

The compilation pipeline is:

```
.qvr source → Lexer → Tokens → Parser → AST → Compiler → Program (nn.Module)
```

Use the high-level API:

```python
from quivers.dsl import loads, load

# Compile from string
prog = loads('''
    object X : 3
    object Y : 4
    latent f : X -> Y
    output f
''')

# Compile from file
prog = load("model.qvr")

# Now a trainable nn.Module
optimizer = torch.optim.Adam(prog.parameters())
```

## Grammar

The full grammar is:

```
module         := statement*

statement      := quantale_decl
                | category_decl
                | rule_decl
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
                | output_decl

quantale_decl  := 'quantale' ('product_fuzzy' | 'boolean'
                              | 'lukasiewicz' | 'godel' | 'tropical')

category_decl  := 'category' IDENT (',' IDENT)*

rule_decl      := 'rule' IDENT '(' IDENT (',' IDENT)* ')' ':'
                  cat_pattern (',' cat_pattern)* '=>' cat_pattern
cat_pattern    := cat_slash
cat_slash      := cat_product (('/' | '\') cat_product)*
cat_product    := cat_primary ('*' cat_primary)*
cat_primary    := IDENT | '(' cat_pattern ')'

object_decl    := 'object' IDENT ':' type_expr
type_expr      := product_type ('+' product_type)*
product_type   := primary_type ('*' primary_type)*
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
                   program_body
param_list     := IDENT (',' IDENT)*

program_body   := program_step+ return_stmt

program_step   := draw_step | observe_step | let_step
draw_step      := 'draw' var_pattern '~' IDENT ['(' arg_list ')']
                | IDENT '<-' IDENT ['(' arg_list ')']
observe_step   := 'observe' var_pattern '~' IDENT ['(' arg_list ')']
let_step       := 'let' IDENT '=' let_expr
let_expr       := let_term (('+' | '-') let_term)*
let_term       := let_unary (('*' | '/') let_unary)*
let_unary      := '-' let_atom | let_atom
let_atom       := IDENT '(' let_expr (',' let_expr)* ')'
                | IDENT | INT | FLOAT | '(' let_expr ')'
var_pattern    := IDENT | '(' IDENT (',' IDENT)* ')'
arg_list       := arg (',' arg)*
arg            := '-' (INT | FLOAT) | IDENT | INT | FLOAT

return_stmt    := 'return' return_pattern
return_pattern := IDENT | '(' return_entry (',' return_entry)* ')'
return_entry   := IDENT ':' IDENT | IDENT

let_decl       := 'let' IDENT '=' expr ['where' let_decl+]
expr           := compose_expr
compose_expr   := tensor_expr (('>>' | '>=>' | '<<') tensor_expr)*
tensor_expr    := postfix_expr ('@' postfix_expr)*
postfix_expr   := atom_expr ('.' method_call)*
method_call    := 'marginalize' '(' IDENT (',' IDENT)* ')'
atom_expr      := 'identity' '(' IDENT ')'
                | 'fan' '(' expr (',' expr)* ')'
                | 'repeat' '(' expr [',' INT] ')'
                | 'stack' '(' expr ',' INT ')'
                | 'scan' '(' expr [',' scan_init] ')'
                | 'parser' '(' parser_args ')'
                | 'ccg' '(' parser_args ')'
                | 'lambek' '(' parser_args ')'
                | IDENT
                | '(' expr ')'

scan_init      := 'init' '=' ('zeros' | 'learned')
parser_args    := 'rules' '=' '[' IDENT (',' IDENT)* ']'
                  [',' 'categories' '=' '[' IDENT (',' IDENT)* ']']
                  [',' 'start' '=' (IDENT | INT)]
                  [',' 'depth' '=' INT]
                  [',' 'constructors' '=' '[' IDENT (',' IDENT)* ']']

output_decl    := 'output' expr
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

### Category

Declare category atoms, generators for a free categorical structure used by grammar-based parsers. These are distinct from `object` declarations (which define finite sets with cardinality).

```qvr
# single declaration
category S

# comma-separated (equivalent to three separate declarations)
category S, NP, N, VP, PP
```

Category atoms are used by `parser()` to build a `CategorySystem` from which complex categories (slash types, products, etc.) are enumerated.

### Rule (Rule of Inference)

Declare a structural rule of inference using sequent-style notation. Rules are universally quantified over pattern variables and can appear in `parser(rules=[...])` alongside built-in schema primitives.

```qvr
# binary rule: forward application
rule forward_app(X, Y) : X/Y, Y => X

# binary rule: backward application
rule backward_app(X, Y) : Y, X\Y => X

# binary rule: forward composition
rule forward_comp(X, Y, Z) : X/Y, Y/Z => X/Z

# unary rule: left projection
rule left_proj(A, B) : A * B => A
```

Each rule takes a parenthesized list of universally quantified variables, a colon, a comma-separated list of premise patterns, a `=>` sequent arrow, and a conclusion pattern. Category patterns support slash types (`X/Y`, `X\Y`), product types (`A * B`), and parenthesized grouping.

Rules with two premises compile to `PatternBinarySchema`; rules with one premise compile to `PatternUnarySchema`. Both are `RuleSchema` functors that compose with the built-in schemas via `|` (union).

```qvr
# use DSL-declared rules in a parser
category S, NP, N, VP, PP
object Token : 256

rule fwd(X, Y) : X/Y, Y => X
rule bwd(X, Y) : Y, X\Y => X
rule fwd_comp(X, Y, Z) : X/Y, Y/Z => X/Z

let grammar = parser(
    rules=[fwd, bwd, fwd_comp],
    terminal=Token,
    start=S
)

output grammar
```

DSL-declared rules and built-in schema primitives can be freely mixed:

```qvr
category S, NP, N
object Token : 256

# custom binary rule alongside a built-in schema
rule my_rule(X, Y) : X/Y, Y => X

let grammar = parser(
    rules=[my_rule, harmonic_composition],
    terminal=Token,
    start=S
)
```

### Object

Declare a finite set:

```qvr
object X : 3          # FinSet("X", 3)
object Y : 4
object XY : X * Y     # ProductSet(X, Y)
object Sum : X + Y    # CoproductSet(X, Y)
object Free : FreeMonoid(X, max_length=2)  # (not currently supported in DSL)
```

### Morphism

Declare a learnable or fixed morphism:

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

```qvr
stochastic s : X -> Y
stochastic cat : X -> (Y * Z)
```

### Discretize

Convert continuous space to finite set via binning:

```qvr
discretize d : U -> 20      # discretize UnitInterval into 20 bins
discretize d2 : R3 -> 100   # discretize ℝ³ into 100 bins
```

### Embed

Embed discrete into continuous:

```qvr
embed e : X -> R3   # treat X as uniform on ℝ³
```

### Replicated Declarations

Declare N independent copies of a morphism. Each copy has independent parameters; the base name becomes a group that can be referenced by `fan`:

```qvr
# creates head_0, head_1, head_2, head_3 with independent parameters
continuous head[4] : Latent -> HeadOut ~ Normal [scale=0.1]

# works with stochastic and embed too
stochastic kernel[3] : State -> Obs

embed tok[2] : Token -> Hidden
```

### Fan-Out (Diagonal Morphism)

Copy a single input to N morphisms and concatenate their outputs. Accepts explicit morphism names or a group name from a replicated declaration:

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

```qvr
stochastic transition : State -> State
stochastic emission : State -> Obs

# runtime-variable: no count specified
let n_step = repeat(transition) >> emission

output n_step
```

```python
prog = load("hmm.qvr")
obs_3 = prog(n_steps=3)    # T^3 >> E
obs_50 = prog(n_steps=50)  # T^50 >> E — same model, different length
```

The morphism's codomain must match its domain (endomorphism) for repeat to work.

### Stack (Independent Multi-Layer)

Create N independent deep copies of a morphism, each with its own parameters (no weight-tying):

```qvr
# stack creates independent parameters per layer
let deep = stack(transition, 3)  # 3 layers, each with own params

# repeat reuses the same parameters (weight-tying)
let tied = repeat(transition, 3)  # 3 iterations, shared params
```

Unlike `repeat`, which composes a morphism with itself using the same parameters, `stack(f, N)` creates N fresh deep copies of f, each with independent learnable parameters. This is essential for deep neural networks where each layer has distinct parameters.

### Scan (Temporal Recurrence)

Thread hidden state across a sequence using a recurrent cell:

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

output rnn
```

For deeper temporal models, stack multiple scans:
```qvr
let deep_rnn = tok_embed >> scan(cell_1) >> scan(cell_2) >> output_proj
```

Each `scan` threads its own hidden state independently.

### Arrow Bind Syntax

Alternative syntax for draw steps using the `<-` operator:

```qvr
# these are equivalent:
draw x ~ Normal(0.0, 1.0)
x <- Normal(0.0, 1.0)
```

Both forms introduce a variable x in a probabilistic program. The `<-` notation is more concise and aligns with functional programming conventions.

### Backward Composition

Compose morphisms in reverse order using `<<` or use Kleisli composition `<=>`:

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
# these are equivalent:
space Hidden : Euclidean(64)

type Hidden = Euclidean 64     # ML-style, parens optional

# product types
type Combined = Hidden * Output
```

The `type` keyword provides a more concise, ML-style syntax for declaring named spaces. Parentheses around arguments are optional.

### Where Clauses

Attach local let-bindings to a let declaration using `where`:

```qvr
let model = embed >> layers >> output_proj

where

    let layers = stack(transition, 3)
    let embed = tok_embed
```

The `where` keyword introduces a block of local definitions that are scoped to the parent let binding. This improves readability for complex nested compositions.

### Grammar Constructs

The DSL provides keywords for differentiable parsing over formal grammars. All compile to `DeductiveSystem` subclasses (as `nn.Module`) that accept tokenized sentences and return log-probabilities.

#### Deductive Parser

The `parser` keyword creates a differentiable deductive parser from a `rules=[...]` list. Each entry in `rules` is resolved at compile time, either as a **schema functor** (a registered `CategorySystem → RuleSystem` natural transformation) or as a **declared morphism** (whose type signature determines its deductive role). The same uniform `parser()` interface handles categorial grammars, PCFGs, and anything between.

**Schema rules.** When entries resolve to registered schema primitives, the compiler composes them and applies the resulting functor to a category system. Category atoms are declared with the `category` keyword (these are generators for a free categorical structure, distinct from `object` declarations which define finite sets with cardinality):

```qvr
# category atoms: generators for the free categorical structure
category S, NP, N, VP, PP

# terminal vocabulary: a finite set with cardinality
object Token : 256

let grammar = parser(
    rules=[evaluation, harmonic_composition, crossed_composition],
    terminal=Token,
    start=S
)

output grammar
```

Alternatively, categories can be listed inline via `categories=[...]` for concise one-off definitions:

```qvr
object Token : 256

let grammar = parser(
    categories=[S, NP, N, VP, PP],
    rules=[evaluation, harmonic_composition, crossed_composition],
    terminal=Token,
    start=S
)

output grammar
```

When neither `categories=[...]` nor `category` declarations are present, the compiler raises an error; there is no implicit inference. Similarly, `terminal=` is required for schema-based parsers: it explicitly names the declared `object` serving as the terminal vocabulary.

```qvr
# Lambek calculus: evaluation + adjunction units + tensor operations
category S, NP, N
object Token : 256

let grammar = parser(
    rules=[evaluation, adjunction_units, tensor_introduction, tensor_projection],
    terminal=Token,
    start=S
)
```

```qvr
# novel grammar: evaluation + harmonic composition + tensor (no crossed composition)
category S, NP, N
object Token : 256

let hybrid = parser(
    rules=[evaluation, harmonic_composition, tensor_introduction],
    terminal=Token,
    start=S
)
```

The `rules` parameter lists names of rule schema primitives (resolved via `SCHEMA_REGISTRY`). The `terminal` parameter names the declared object whose cardinality gives the terminal vocabulary size. The `start` parameter (default `S`) selects the start category. Two additional optional parameters control the category inventory:

- **`depth`** (default 1): Maximum nesting depth for generated complex categories. `depth=2` generates categories like `(S/NP)/VP` in addition to `S/NP`.
- **`constructors`** (default `[slash]`): Which type constructors to use when enumerating categories. Available constructors are `slash`, `product`, `unit`, `diamond`, and `box`. When omitted, only slash categories (X/Y, X\Y) are generated.

```qvr
category S, NP, N
object Token : 256

# depth=2 for deeper slash nesting
let grammar = parser(
    rules=[evaluation, harmonic_composition],
    terminal=Token,
    depth=2,
    start=S
)

# multimodal type-logical grammar with diamond modalities
category VP, PP

let tlg = parser(
    rules=[evaluation, adjunction_units, modal_introduction, modal_elimination],
    terminal=Token,
    constructors=[slash, diamond],
    depth=1,
    start=S
)

# full multimodal with unit type
let mtlg = parser(
    rules=[
        evaluation, adjunction_units,
        unit_introduction, unit_elimination,
        modal_introduction, modal_elimination,
        modal_application
    ],
    terminal=Token,
    constructors=[slash, unit, diamond],
    depth=1,
    start=S
)
```

The available rule schema primitives are:

| Primitive | Categorical operation | Example rules |
|---|---|---|
| `evaluation` | Counit of hom-tensor adjunction | X/Y Y → X, Y X\Y → X |
| `harmonic_composition` | Composition of same-direction homs | X/Y Y/Z → X/Z |
| `crossed_composition` | Composition mixing slash directions | X/Y Y\Z → X\Z |
| `generalized_composition` | Higher-order composition (B^n) | X/Y Y\|Z₁...\|Zₙ → X\|Z₁...\|Zₙ |
| `adjunction_units` | Units of the hom-tensor adjunction | A → B/(A\B), A → (B/A)\B |
| `tensor_introduction` | Product formation | A, B → A⊗B |
| `tensor_projection` | Product elimination | A⊗B → A, A⊗B → B |
| `commutative_evaluation` | Evaluation with reversed argument order | B X/B → X |
| `unit_introduction` | Monoidal unit laws | I⊗A → A, A⊗I → A |
| `unit_elimination` | Unit coercion | A → I |
| `modal_introduction` | Modal injection | A → ◇A (or □A) |
| `modal_elimination` | Modal projection | ◇A → A (or □A → A) |
| `modal_application` | Modal function application | ◇(C/B) ⊗ ◇B → ◇C |

**Convenience aliases:** `ccg(...)` is shorthand for `parser(... rules=[evaluation, harmonic_composition, crossed_composition])`. `lambek(...)` is shorthand for `parser(... rules=[evaluation, adjunction_units, tensor_introduction, tensor_projection])`. Both accept an optional `rules=` override.

**Morphism rules.** When entries resolve to declared morphisms, the compiler inspects their type signatures to determine their deductive role. A morphism `N → N ⊗ N` (codomain is a product of the domain with itself) contributes binary deductions; a morphism `N → T` (codomain differs from the domain) contributes lexical axioms. The deductive system is derived entirely from the types.

```qvr
object N : 10   # nonterminals
object T : 64   # terminals

# Kleisli morphisms: branching and lexicalization
stochastic binary_rules : N -> N * N
stochastic lexical_rules : N -> T

let pcfg = parser(
    rules=[binary_rules, lexical_rules],
    start=0
)

output pcfg
```

No special keywords distinguish "binary" from "lexical"; the compiler reads the types. The `start` parameter (default `0`) selects the start nonterminal index.

The same primitives are available in Python for programmatic use via composable `RuleSchema` objects:

```python
from quivers.stochastic import (
    CategorySystem, ChartParser, VITERBI,
    EVALUATION, HARMONIC_COMPOSITION, ADJUNCTION_UNITS,
    MODAL_INTRODUCTION, MODAL_ELIMINATION,
    CCG, LAMBEK,
)

# compose schemas with | (union)
my_schema = EVALUATION | HARMONIC_COMPOSITION | ADJUNCTION_UNITS

# instantiate over a category system
cs = CategorySystem.from_atoms_and_slash_depth(["S", "NP", "N"], max_depth=1)
parser = ChartParser.from_schema(my_schema, cs, n_terminals=100, start="S")

# with semiring selection (Viterbi for best-parse decoding)
parser = ChartParser.from_schema(
    my_schema, cs, n_terminals=100, start="S", semiring=VITERBI,
)

# use a grammar preset directly
parser = ChartParser.from_schema(CCG, cs, n_terminals=100, start="S")

# multimodal category system with diamond constructor
cs = CategorySystem.from_generators(
    atoms=["S", "NP", "N"],
    constructors=["slash", "diamond"],
    max_depth=1,
)
modal_schema = EVALUATION | ADJUNCTION_UNITS | MODAL_INTRODUCTION | MODAL_ELIMINATION
parser = ChartParser.from_schema(modal_schema, cs, n_terminals=100, start="S")
```

**Semiring parameterization:** The deductive parser is parameterized by a `ChartSemiring` (Goodman, 1999), which determines the scoring algebra. Different semirings yield different parsing algorithms from the same CKY skeleton:

| Semiring | ⊕ (plus) | ⊗ (times) | Use case |
|---|---|---|---|
| `LOG_PROB` (default) | logsumexp | + | Marginal log-probability |
| `VITERBI` | max | + | Best-parse decoding |
| `BOOLEAN` | or | and | Recognition (yes/no) |
| `COUNTING` | + | × | Derivation counting |

**Learnable rule weights:** Each structural rule carries a learnable log-weight that biases the deduction scores. These are registered as `nn.Parameter` by default. To fix rule weights, pass `learnable_rule_weights=False` to the parser constructor.

### Program

Define a probabilistic program:

```qvr
program my_prog : X -> Y
    draw mu ~ LogitNormal(0.0, 1.0)
    draw x ~ Normal(mu, 1.0)

    return x

program with_params(a, b) : (X * Z) -> Y
    let w = a

    draw x ~ f(w)
    draw y ~ g(x, b)
    return y
```

### Let Expressions (Arithmetic)

Inside a `program` block, `let` bindings support full arithmetic with standard operator precedence, unary negation, and built-in functions:

```qvr
# arithmetic: +, -, *, /
let eta = mu + sigma * z_raw + lambda * shared_factor
let adjusted = (1.0 - lapse) * p_raw + 0.5 * lapse
let mean = (x + y + z) / 3.0
let negated = -raw_score

# built-in functions: sigmoid, exp, log, abs, softplus
let prob = sigmoid(eta)
let positive = softplus(raw)
let log_rate = log(rate)
let magnitude = abs(x - 0.5)
```

### Inline Distributions

Draw and observe steps support inline distribution construction with any mix of literal and variable arguments. All 11 distribution families support arbitrary combinations:

```qvr
# all-literal (fixed): Unit -> codomain
draw x ~ Normal(0.0, 1.0)
draw p ~ Beta(2.0, 5.0)

# all-variable (direct): variables -> codomain
draw y ~ Normal(mu, sigma)
draw b ~ Bernoulli(theta)

# mixed literal/variable: any combination works
draw h_cand ~ Normal(reset_hidden, 0.5)
draw z ~ Normal(0.0, learned_scale)
draw r ~ TruncatedNormal(mu, sigma, 0.0, 1.0)

# negative literals
draw z ~ Normal(-1.5, 0.3)
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

```qvr
let fg = f >> g
let par = f @ g
let marg = fg.marginalize(Y)
let composed = f >> g >> h
```

### Output

Export a morphism as the program output:

```qvr
output f
output fg
output my_prog
```

## Examples

### Simple Discrete Model

```qvr
object X : 3
object Y : 4

latent f : X -> Y
latent g : Y -> Y

let fg = f >> g

output fg
```

### Continuous Conditional Model

```qvr
object Cond : 2

space Latent : Euclidean(3)
space Obs : Euclidean(5)

continuous prior : Cond -> Latent ~ Normal
continuous likelihood : Latent -> Obs ~ Normal [scale=0.1]

let posterior = prior >> likelihood

output posterior
```

### Probabilistic Program with Observations

```qvr
object Data : 1

space Y : Euclidean(2)

program regression : Data -> Y
    draw theta ~ LogitNormal(0.0, 1.0)
    draw y ~ Normal(theta, 0.5)

    observe _ ~ Normal(y, 0.1)

    return y
```

### Factivity Model (from examples)

```qvr
object Entity : 1
object Truth : 2
object Resp : 1

program factivity : Entity -> Truth * Truth * Truth * Resp
    draw theta_know ~ LogitNormal(0.0, 1.0)
    draw theta_cg ~ LogitNormal(0.0, 1.0)

    let cg_complement = 1.0 - theta_cg

    draw tau_know ~ Bernoulli(theta_know)
    draw cg_matrix ~ Bernoulli(theta_cg)
    draw sigma ~ Uniform(0.0, 1.0)
    observe response ~ TruncatedNormal(theta_know, sigma, 0.0, 1.0)
    return (tau_know: tau_know, cg_complement: cg_complement, cg_matrix: cg_matrix, response: response)
```

For more examples, see the [Examples Gallery](../examples/index.md).

## Compilation Process

The `Compiler` transforms the AST to a `Program`:

1. **Resolve declarations**: collect all objects, spaces, morphisms
2. **Type check**: ensure domains/codomains match in compositions
3. **Build morphism DAG**: construct morphism modules
4. **Wrap in Program**: create an `nn.Module` that manages all parameters

```python
from quivers.dsl.compiler import Compiler
from quivers.dsl.parser import parse

source = "object X : 3\nlatent f : X -> X\noutput f"
ast = parse(source)
compiler = Compiler(ast)
program = compiler.compile()
```

## Error Handling

The DSL provides three error types:

- `LexError`: invalid tokens or characters
- `ParseError`: syntax error (wrong token sequence)
- `CompileError`: semantic error (type mismatch, undefined name)

```python
from quivers.dsl import loads, LexError, ParseError, CompileError

try:
    prog = loads(bad_source)
except LexError as e:
    print(f"Lexical error: {e}")
except ParseError as e:
    print(f"Parse error: {e}")
except CompileError as e:
    print(f"Compilation error: {e}")
```

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
