# Quivers Indexed Effect Core

The Quivers Indexed Effect Core (QIEC) is the typed integration boundary for
indexed families and algebraic effects. QVR v0.19 parses these constructs into
a distinct source AST and projects indexed declarations through Didactic
0.15's public `GADT` API. It then lowers them through the exact
`qvr-source/v0.19` to `qiec-core/v1alpha1` route. This **QIEC route** checks the
result against both Didactic's compiled first-order theory and the Python
reference kernel's deterministic serialization contract. It does not
reinterpret the independent representations used by probabilistic `program`
declarations.

QIEC makes two boundaries explicit. First, the **stable core boundary** contains
only typed, deterministic data: kinds, telescopes, indexed-family declarations,
effect interfaces, rows, handler signatures, value and computation terms,
source provenance, and equality evidence. Second, the **runtime attachment
boundary** associates stable identifiers with process-local values and handler
implementations. Host callables and mutable resources cannot cross the stable
core boundary. This note first describes the static kernel; it then turns to
serialization and evaluation, gives four complete v0.19 examples, and closes
with the typed IR and target-capability contract used by the transpilers.

## Static kernel

The kernel separates family parameters from refinable indices. A
`FamilyDecl` fixes both telescopes, while each `ConstructorDecl` specifies its
fields and result indices. Case analysis uses an explicit `CaseMotive`; branch
refinement introduces rigid skolems and local equality evidence that cannot
escape the branch in which it was established.

Coverage is conservative. QIEC omits a constructor only when distinct closed
constructors certify that the branch is impossible. If refinement is unknown,
the branch remains part of both the coverage obligation and the inferred effect
row. This policy may reject some valid programs until a stronger solver is
available, but it does not erase effects on the basis of an unproved
contradiction.

Effects are parameterized interfaces. Thus `State[Int]` and `State[String]`
share a declaration but remain distinct applied interfaces. Each lexical
instance also receives its own stable `EffectInstanceId`, so two state cells of
the same type remain distinguishable in an effect row. Operations may bind
additional static arguments, and request checking substitutes both the
interface and operation telescopes before checking argument and result types.

Handlers state their coverage and resumption contracts in serializable data.
The kernel records coverage (`total` or `partial`) separately from
unknown-operation forwarding, as well as each clause's resumption grade `0`,
`aff`, `1`, or `omega`; the reference evaluator enforces these grades
dynamically. A total handler removes exactly the matched lexical instance from
the row; a partial handler retains it; and an explicitly forwarding partial
handler passes structurally uncovered operations to an outer handler.

## Stable serialization

`quivers.qiec.dumps` and `quivers.qiec.loads` encode the stable core using the
versioned `qiec-json/v1` envelope and the `qiec-core/v1alpha1` ABI identifier.
The codec is an allowlist over QIEC records, identifiers, enums, tuples, bytes,
and JSON scalars. It rejects unknown node tags, malformed records, non-finite
floats, runtime attachments, and host callables. Encoding is deterministic, so
the same core graph yields the same serialized JSON text; UTF-8 encoding thus
yields the same bytes.

These format and ABI identifiers belong to the serialization boundary. An
authored effect declaration contains only its name, static telescope, and
operation signatures.

This format is intentionally independent of Didactic's internal model and
Panproto's generic carrier. Didactic checks an ephemeral GADT projection and
negotiates the exact source and target versions before QVR lowering completes,
while Panproto can carry the canonical envelope without inspecting its nodes.
QIEC retains ownership of stable identities, its uniform-parameter/refinable-
index distinction, the typed allowlist, malformed-node rejection, and reference
semantics.

## Reference evaluator

The evaluator gives `Return`, `Bind`, `Perform`, `Handle`, and indexed `Case` a
small CEK-style operational semantics. Handlers are lexical and deep: a resumed
continuation reinstalls the matching handler, while effects performed by the
clause body itself are offered only to outer handlers. Dispatch checks the exact
applied interface as well as the lexical instance and operation identifier.

Resumption grades are enforced dynamically at the attachment boundary. A
grade-`0` clause cannot resume; an affine clause may resume at most once; a
linear clause must resume exactly once; and an unrestricted clause may resume
multiple times only when the captured context is declared duplicable. Each
multi-shot branch extends the dynamic address with a resumption path, preserving
the identity needed by traces and replay.

The built-in prelude currently supplies declarations and reference handlers for
the following interfaces:

- `Random` and `Score`, including drawing, conditioning, intervention/replay,
  and trace forwarding;
- `State[S]` and `Abort[E]`;
- `Choose` for multi-shot nondeterminism; and
- `Weight[K]` for caller-supplied accumulation operations.

These handlers are executable specifications, not the optimized production
implementations of a backend. Runtime samplers, observation tables, semiring
operations, state cells, and trace recorders remain process-local attachments.

A distribution construction evaluates to a
[`RuntimeDistribution`](../api/qiec/distributions.md), which samples and
scores through an installed
[`DistributionBackend`](../api/qiec/distributions.md). The reference backend
implements every family of the semantic registry in plain Python: the scalar
core beside the backend and the structured and compositional families in
[`quivers.qiec.reference_families`](../api/qiec/reference_families.md), with
the small linear algebra the matrix families need written out rather than
taken from a host library. Each density is held to the torch definition the
transpile probes use, or to the torch runtime's own definition where torch has
no class, so the reference machine is an oracle a host can be checked against.
A family whose draw has a size no parameter fixes, the LKJ families, reads it
from the constructed type's event extents. The `Transformed` chain names its
transforms as a comma-separated string over `exp`, `log`, `sigmoid`, `logit`,
`softplus`, `tanh`, and `neg`, applied left to right.

### Named execution boundary

`qvr run FILE COMPUTATION [JSON ...]` selects one checked
`NamedComputation`, specializes its static telescope, validates its value
arguments, and evaluates it with a fresh attachment table. `--static
NAME=TERM` supplies a closed type, index, or effect application for each static
binder. `--runtime FILE.json` selects the built-in `core` runtime or installed
`quivers.qiec_runtime` providers without placing executable objects in the
configuration file. `--fuel STEPS` bounds the number of evaluation steps, since
a recursive computation may otherwise never return; exhaustion is the
`qiec-run-fuel` diagnostic. `--trace` writes stable events in plain mode, while
`--json` includes the result, specialized result type, runtime label, and trace
in one document.

The REPL exposes the same boundary through `:runtime`, `:run`, and `:detach`.
The Textual status bar shows the active runtime and the most recent result.
Argument, specialization, provider, validator, evaluation, and result failures
retain their `qiec-run-*` diagnostic codes across the CLI, REPL, and TUI.

## QVR surface

The following examples use the normative grammar. Each block parses,
lowers through the QIEC route, validates against the kernel, and reaches a
canonical parse–emit fixed point in the documentation tests.

A length-indexed vector illustrates constructor refinement:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

define head[A : Type, n : Nat](xs : Vec[A](S(n))) : A !{} =
    case xs motive (m : Nat) => A
        Cons[n](head, tail) =>
            return head
```

The scrutinee type `Vec[A](S(n))` rules out `Nil`; matching `Cons` then
introduces a rigid local predecessor index. This refinement makes the single
branch exhaustive, but the local index cannot escape into the type of the
complete case.

Computations call one another by ordinary application, recursion included,
and branch on Booleans with `if`. Pure expressions share the `program` let
grammar: operators resolve to primitives of a closed registry by operand
type, an integer operand beside a real one is converted first (the kernel
term carries the `int_to_real`), tuples build finite products, and `t[i]`
projects a component.

<!-- compile: qiec -->
```qvr
define triangle(n : Int) : Int !{} =
    if n <= 0 then
        return 0
    else
        let rest <- triangle(n - 1)
        return n + rest

define split(x : Int, y : Real) : Real !{} =
    let pair = (real(x / 2) * y, x % 3)
    let scaled = pair[0] + real(pair[1])
    return max(scaled, 0.5)
```

Recursion runs on the reference machine's explicit stack, and on a
trampoline in every generated host runtime, so its depth is bounded by memory
rather than the host call stack. A run may be given a step budget, which
turns divergence into the stable `qiec-run-fuel` diagnostic.

A parameterized state effect separates an interface from its lexical
instances. The `left` and `right` declarations below have the same applied
interface but different lexical identities:

<!-- compile: qiec -->
```qvr
effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

instance left : State[Int]
instance right : State[Int]

handler run_state[S : Type, A : Type] for State[S] : A -> A [coverage=total, implementation=foreign]
    get resumes 1
    put resumes 0

define exchange(next : Int) : Int !{left, right} =
    let previous <- perform left.get()
    perform right.put(next)
    return previous

define read_left() : Int !{} =
    handle left with run_state[Int, Int] in
        let current <- perform left.get()
        return current
```

The row on `exchange` retains both instances. By contrast, the total
`run_state` handler removes `left` from the row of `read_left`. The handler
declaration records a stable signature and resumption grades; its executable
clauses remain runtime attachments keyed by the derived handler identifier.

Probabilistic operations become instances of the same effect calculus:

<!-- compile: qiec -->
```qvr
effect Random
    draw : Real -> Real

effect Score
    score : Real -> Unit

instance random : Random
instance score : Score

handler replay for Random : Real -> Real [coverage=total, implementation=foreign]
    draw resumes 1

define replayed_model() : Real !{score} =
    handle random with replay in
        let x <- perform random.draw(1.0)
        perform score.score(x)
        return x
```

`replay` is total and handles the exact `random` instance. The remaining
`score` entry makes the residual effect explicit. Runtime replay tables,
samplers, and trace recorders do not enter the stable module.

Distributions are values of the same calculus. Applying a family of the
semantic registry, such as `Normal(mu, 1.0)` or `Dirichlet([1.0, 2.0, 3.0])`,
builds a `Sampleable[A]` whose element type the registry fixes; a list
literal is a `Tensor` whose leading dimension is its entry count, so a
nested literal is a matrix; `site("x")` names a sample site at the `Site[A]`
type its position expects; and `log_prob(d, x)` evaluates a distribution's
log density as a `LogWeight`. The prelude's `Random` and `Score` interfaces
need no declaration:

<!-- compile: qiec -->
```qvr
instance random : Random
instance score : Score

handler draw for Random : Real -> Real [coverage=total, implementation=foreign]
    sample[A : Type] resumes 1

handler accumulate for Score : Real -> (Real * LogWeight) [coverage=total, implementation=foreign]
    add resumes 1

define model(mu : Real) : Real !{random, score} =
    let x <- perform random.sample[Real](site("x"), Normal(mu, 1.0))
    perform score.add(log_prob(Categorical([0.2, 0.8]), 1))
    return x

define run(mu : Real) : Real * LogWeight !{} =
    handle score with accumulate in
        handle random with draw in
            model(mu)
```

On the reference machine a distribution value is a
[`RuntimeDistribution`][quivers.qiec.RuntimeDistribution] that samples and
scores through the installed
[`DistributionBackend`][quivers.qiec.DistributionBackend]; the core runtime
provider's `draw` and `score` handler kinds implement `draw` and `accumulate`
above. Each dynamic target spells the construction in its own library through
[`spell_distribution`][quivers.transpile.family_spelling.spell_distribution],
applying the target's parameterization conventions (a rate against a scale,
a complemented probability, a shifted support, a folded half-line family) so
that `log_prob` agrees with the registry's density on every host.

A `program` is domain-specific notation for a named computation, and
elaborates to one. Its data, observations, and fibrations become the
computation's typed parameters; a `sample` step performs `Random.sample`
on the module's canonical `random` instance, an `observe` step scores
`log_prob` of its family at the observation on the canonical `score`
instance, a `score` step adds its value as a weight, a `let` step binds a
pure expression, and a `marginalize` block becomes a helper computation
that allocates a `Random` instance for the latent, handles it with the
enumeration handler, collects the scope's weights under a `Weight`
instance, and answers the log marginal, which the enclosing scope scores.
Plates are typed: `xs : N <- Normal(0.0, 1.0)` samples a
`Tensor[Real]([|N|])` from a construction plated over `N`, and a grouped
marginalization's `via` fibration re-indexes the latent's arguments to the
observation rows and segments the weights back to the groups. A product
group `[over=[G, H]]` is one flat axis of extent `|G| * |H|`, and the
product fibration `[via=[g, h]]` its row-major flattening, the last
factor varying fastest. The block's `reduction` selects the enumeration
handler: `enumerate_marginal` answers the log-sum-exp of the weighted
shots, `enumerate_marginal_sum` their sum, and `enumerate_marginal_mean`
their average; each has an `enumerate_grouped_` counterpart that answers
one aggregate per group position, which is what a block nested inside a
grouped block adds to the enclosing group's weights. The nested block's
group must be the enclosing group's extent, identified with it position
by position, or a product with the enclosing axis as a factor, whose
per-position marginals are segment-summed onto the enclosing positions
and whose arguments shaped by the enclosing group are gathered by that
projection. A `sample` inside a block that reads nothing the block binds
is drawn once, before the block, since every shot shares it; a draw that
reads the latent is refused. A `sample` of a vector family written with
its entries spread, `Dirichlet(1.0, 2.0, 3.0)`, gathers them into one
concentration, and a single literal is the symmetric concentration at the
dimension the step's plate or the program's codomain fixes. Programs and
computations call each other by ordinary application: a program body
binds a call with `let x <- helper(args)`, and a computation calls a
program by its name.

<!-- compile: qiec -->
```qvr
object Obs : FinSet 4

define noisy(x : Real) : Real !{random} =
    let y <- perform random.sample[Real](site("noise"), Normal(x, 0.1))
    return y

program prog : Obs -> Obs
    sample a <- Normal(0.0, 1.0)
    let c <- noisy(a)
    observe y <- Normal(c, 0.5)
    return c
export prog

define twice(y : Real) : Real !{random, score} =
    let first <- prog(y)
    let second <- prog(y)
    return first + second
```

The elaborated module records each program as a
[`ProgramEntry`][quivers.qiec.ProgramEntry]: the roles of its parameters,
its sites, and its return names, beside the computation itself.
[`run_program`][quivers.qiec.program_runtime.run_program] wraps the entry
point in a scoring replay of `random` and an accumulating `score`, so the
reference machine applied to data and a value for every site yields the
program's value and its log joint density; the tests hold that number to
the torch runtime's trace on the same models.

An input whose shape nothing in the program fixes, such as the
concentration vector of `sample probs <- Dirichlet(alpha)` written without
a plate, is typed over an index variable the program's computation binds:
the computation gains one `Nat` binder per open extent, recorded on the
entry's `telescope`, and its parameter reads `Tensor[Real]([alpha_extent])`.
A run reads each extent off the data, a call step from another program
takes the extent as its own, and an authored computation names it with an
index literal, `prog[3](alpha)`.

A draw through a declared kernel morphism reads the family's parameters
off the morphism's parameter map, whose learned tensors are typed
program inputs with the roles `weight`, `bias`, and `table`: a kernel
with a real domain applies an affine map to its conditioning row, the
step's arguments or the program's domain inputs, and the heads of the
family read their row blocks; a kernel over a finite domain reads a row
of a table at the element the step names. A morphism carrying
`[param_source=mlp]` interposes hidden layers, each an affine map
followed by `tanh`, with the widths the option spells (`mlp(a, b)`, or
`hidden_dim`, or two layers of sixty-four), and its inputs
`<m>_param_layer<i>_weight` and `<m>_param_layer<i>_bias` match the shapes
of the torch runtime's `MLPSource` layer by layer. An embedding
morphism, `[role=embed]`, is a Normal kernel over its finite domain whose
table concatenates the centres and log scales the torch runtime learns.
Under a plate the map applies at every row: an unbound conditioning name
of `observe y : Resp <- net(x)` becomes a `Tensor[Real]([|Resp|, width])`
data input, an unbound index of an embedding a `Tensor[Int]([|Resp|])`,
and each head is a comprehension over the rows. A composite drawn from
in one step, `sample h <- embed >> fan(head) >> out`, expands into one
site per factor before elaboration: `fan` over a `[replicate=k]`
morphism draws through each replica `head_0` through `head_{k-1}` and
bundles the draws, a kernel conditioned on a bundle reads its rows in
order, `stack(block, n)` composes `n` copies whose morphisms past the
first are declared afresh under `_copy<i>` names with parameters of
their own, and the branches of a tensor product `encoder @ decoder` at
the head of a chain each read their own factor of the program's domain.
The expanded module declares every replica and copy, so the entry's
inputs name each one.

Logic programming uses the same handler and resumption rules:

<!-- compile: qiec -->
```qvr
effect Choose
    choose[A : Type] : A -> A

effect Weight[K : Type]
    add : K -> Unit

instance choice : Choose
instance weight : Weight[Real]

handler depth_first[A : Type] for Choose : A -> A [coverage=total, implementation=foreign]
    choose resumes omega

define searched() : Int !{weight} =
    handle choice with depth_first[Int] in
        let selected <- perform choice.choose[Int](1)
        perform weight.add(0.0)
        return selected
```

The `Choose` clause has an unrestricted resumption because a runtime attachment
may explore several alternatives. Handling `choice` leaves the parameterized
`weight` instance in the row. This factorization lets a search attachment and a
weight attachment compose without introducing another source calculus.

A `deduction` declaration is domain-specific notation for exactly this
factorization, and elaborates to it. Its items become a closed family
`<name>__Item` with one nullary constructor per atom and one constructor
per applied symbol the rules and lexicon mention, a slot typed `Int` where
the system uses it as a position and `<name>__Item` otherwise; a lexicon's
logical forms number their bound variables canonically, so alpha-equivalent
forms are one item. The module gains a `Search` effect, `choose[a : Type, n
: Nat] : Tensor[a]([n]) -> a`, a `<name>__choice` instance of it, a
`<name>__weight` instance of `Weight[K]` for the carrier `K` of the declared
semiring (`LogWeight` for `LogProb` and `Viterbi`, `Bool` for `Boolean`,
`Int` for `Counting`), and a `params` instance of `Param` through which
learned weights are read under the names the agenda engine keys its
parameters by: `<name>.lex.<index>` for a lexicon entry and
`<name>.rule.<rule>:<bindings>` for a rule, the bindings rendered in name
order by the `<name>__show` computation. The recursive `<name>__derive`
computation chooses an axiom or a group of rules with premises of one
shape, derives each premise by recursion, matches it against the rule's
pattern by `case` analysis (a shared pattern variable is checked by the
family's `<name>__eq`), instantiates the conclusion, and adds the rule's
weight; a failed match performs an empty choice. A system whose rules all
relate `span` items derives by span, choosing split positions for the
premises and deriving each over its own span; any other system takes its
axioms and their weights as input. The entry `<name>__run` handles the
weight instance with a collecting handler and the search instance with a
handler that resumes once per alternative and combines the shots by the
semiring's addition, so its answer is the semiring sum over derivations of
the goal of the product of the weights along each: the inside weight the
agenda engine tabulates, to the depth the `depth` option bounds. A program
calls the deduction with `let chart = parse(D, sentence)`, which types
`sentence` as a tensor of tokens over one of the program's open extents,
and reads `chart.goal_weight()` as a real to score.

<!-- compile: qiec -->
```qvr
object Term : FinSet 16
object LogWeight : Real 1

deduction AB : Term -> Term [semiring=LogProb, start=S, depth=6]
    atoms S, NP, N, Fwd, Bwd, span, the, dog, runs
    rule fwd_app : span(I, K, Fwd(X, Y)), span(K, J, Y) |- span(I, J, X) #[learnable]
    rule bwd_app : span(I, K, Y), span(K, J, Bwd(X, Y)) |- span(I, J, X) #[learnable]
    lexicon
        "the"  : Fwd(NP, N) = the  #[learnable]
        "dog"  : N          = dog  #[learnable]
        "runs" : Bwd(S, NP) = runs #[learnable]

program fit : Term -> LogWeight
    let chart = parse(AB, sentence)
    score log_Z = chart.goal_weight()
    return log_Z

export fit
```

Two runtime facilities make the search compose with the weights. A foreign
clause may answer with `TailResume(value)`, which resumes its continuation
on the machine's own stack, as an authored `resume` in tail position does,
rather than in a nested host call that stays pending; the collecting
`Weight` handler answers this way, so a multi-shot resumption captured
later inside its scope may run the continuation any number of times. And
a collecting handler configured `forkable` gives each shot its own copy of
the total so far, which is the context cloning an unrestricted resumption
needs.

## Integration contract

The v0.19 integration establishes five results. First, QVR has one typed
surface for indexed families, effect interfaces and instances, handler
signatures, and stable computation terms. Second, Didactic and Panproto check
the first-order indexed-family projection, after which the Quivers kernel checks
branch refinement, effect rows, handler coverage, and resumption contracts and
lowers the surface into a serializable `QiecModule`. Third, the compiler
attaches that checked module to its environment and produces a `Program` without
mixing it into probabilistic elaboration. Fourth, `Lower` projects the complete
module into `IRQiecModule`, whose typed nodes retain declarations, rows,
provenance, values, evidence, computations, and resumption grades. Fifth, the
CLI, TUI, REPL, language server, Pygments and tree-sitter highlighters, TextMate
grammar, Panproto migration assets, and transpilers share the v0.19 surface and
diagnostic vocabulary.

A potential worry is that a common IR implies identical target capabilities.
The **QIEC capability boundary** blocks that inference. Pyro, NumPyro, PyMC,
Edward2, Turing, Gen, WebPPL, and Church lower the complete stable computation
graph through corresponding target-language implementations of the QIEC
runtime ABI. This route retains stable operation and instance identifiers,
dynamic addresses, lexical handler scope, indexed constructors, equality
transport, and the four resumption grades.

Stan, BUGS, and JAGS instead accept a checked first-order subset. A computation
must have a closed, empty effect row and an empty static telescope, return a
scalar, and be composed only of scalar `Return` and `Bind` forms over literal
or variable values. An empty static telescope does not prohibit ordinary value
parameters: Stan may bind named `Bool`, `Int`, and `Real` parameters. BUGS and
JAGS currently require a parameterless entry point because their emitted graph
has no callable parameter ABI. During rendering, `graft_qiec_dynamic` or
`graft_qiec_static` runs the analyzer immediately before the QIEC definitions
are grafted into the target schema. It reports each missing feature as
`qiec:capability:<feature>:<computation>` at that boundary. Thus the static
targets reject only the construct they cannot preserve.

### Target runtime ABI

Each host-language renderer exports a QVR computation named `f` as
`qiec_f`. Its source parameters come first. The entry point also accepts a
static-specialization argument and three invocation-local mappings: `qiec_attachments`,
`qiec_handlers`, and `qiec_operations`. The specialization argument instantiates
the checked type, index, and effect telescope; a polymorphic entry point rejects
an omitted argument, a wrong arity, or a static argument of the wrong kind.
Python and Julia expose this ABI through optional or keyword arguments.
JavaScript uses trailing positional arguments, and Church accepts an optional
rest list. The generated `model` or `build_model` entry point is unchanged.

The renderer tests check the corresponding implementations rather than assume
their equivalence. Every target's emitted QIEC source is reparsed, external
syntax checks run when the target toolchain is available, and runtime-backed
tests exercise Python, Julia, JavaScript, and Scheme entry points. These tests
cover stable-ID dispatch, indexed cases, attachments, resumption grades, and
handler lifecycle behavior. They do not constitute a backend-level adequacy
proof against the reference evaluator.

Attachment entries are keyed by their `qiec:attachment:*` identifier and must
be typed binding descriptors. The runtime checks the descriptor's structural
type against the checked reference and invokes its validator before exposing
its value; a raw host value is not an attachment. A mutable duplicable binding
also declares a `fork` callback, which produces a fresh value from the pristine
capture seed for each unrestricted shot.

Handler entries are keyed by `qiec:handler:*`. The generated manifest, rather
than the host attachment, fixes the handler's effect, coverage, forwarding
policy, clause set, and resumption grades. The attachment's `operations` table
is keyed by `qiec:operation:*`, and each clause receives `(request, resume,
context)`. The request contains the stable instance, effect, operation, static
arguments, value arguments, result type, source origin, and dynamic address.
The context contains the specialized handler manifest, static arguments, and
the current resumption-use count. A return clause receives
`(value, context)`. Result validators check values passed into `resume`, while
the handler output validator checks the clause or return-clause answer.

A stateful handler attachment uses `context_factory` to construct one context
per installation. Optional `on_enter`, `on_exit`, and `on_drop` callbacks form
an exact-once lifecycle. An unrestricted resumption may capture a handler only
when it declares `duplicable_context`; a mutable captured context must also
provide `fork_context`. Each shot, including the first, forks from an untouched
seed. Partial handlers forward structurally uncovered operations to an outer
handler without changing the checked forwarding policy.

Unhandled operations are keyed by the pair of `qiec:effect-instance:*` and
`qiec:operation:*` identifiers and receive the complete `request`. Thus neither
dispatch path assigns semantics to display names such as `Random`, `Score`, or
`Choose`. An address contains the stable source-site identifier, dynamic
frames, and the resumption path; each unrestricted shot appends its branch
number before evaluating the captured continuation. Python stores active
handler state in an invocation-local context, Julia stores it per task, and
Scheme uses a dynamically scoped parameter. JavaScript evaluation is
synchronous, so its scoped stack cannot overlap independent invocations.

This boundary represents QIEC-only and mixed modules without requiring a
synthetic probabilistic `program`; a mixed module retains both its probabilistic
nodes and its typed `IRQiecModule`. Acceptance still depends on the selected
target. Declaration-only modules remain valid on every target. The eight
host-language targets accept executable QIEC-only modules, while the three
static targets accept them only when every computation satisfies the subset
above. Mixed modules must additionally satisfy the selected target's ordinary
probabilistic support boundary. A renderer may omit declarations that have no
runtime effect, but it cannot drop a computation body or erase an unsupported
effect to make the module fit.

For current purposes, QIEC also has no source term for recursive calls, scoped
instance allocation, or authored handler-clause bodies. Handler bodies remain
runtime attachments, and `NamedComputation` values are checked entry points
rather than mutually recursive functions. These limits leave two live
possibilities: target runtimes may acquire native, optimized implementations of
particular handlers, and the static target subset may expand when a
semantics-preserving encoding exists for additional value or control forms.

Panproto 0.74.2 restores version-aware validation for persisted objects written
by earlier releases. The migration gate reads the original QVR history without
rewriting its content-addressed object IDs, so those fixtures continue to test
the compatibility contract rather than a regenerated approximation of it.
