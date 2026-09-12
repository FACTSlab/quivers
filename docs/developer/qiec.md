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
with the boundary that the existing probabilistic transpilers enforce.

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
the row; a partial handler retains it; and a handler may forward future
operations only when its interface declares that evolution policy.

## Stable serialization

`quivers.qiec.dumps` and `quivers.qiec.loads` encode the stable core using the
versioned `qiec-json/v1` envelope and the `qiec-core/v1alpha1` ABI identifier.
The codec is an allowlist over QIEC records, identifiers, enums, tuples, bytes,
and JSON scalars. It rejects unknown node tags, malformed records, non-finite
floats, runtime attachments, and host callables. Encoding is deterministic, so
the same core graph yields the same serialized JSON text; UTF-8 encoding thus
yields the same bytes.

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

## QVR v0.19 surface

The following examples use the normative v0.19 grammar. Each block parses,
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

A parameterized state effect separates an interface from its lexical
instances. The `left` and `right` declarations below have the same applied
interface but different lexical identities:

<!-- compile: qiec -->
```qvr
effect State[S : Type] [version=1, evolution=sealed]
    get : Unit -> S
    put : S -> Unit

instance left : State[Int]
instance right : State[Int]

handler run_state[S : Type, A : Type] for State[S] : A -> A [coverage=total]
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
effect Random [version=1, evolution=forwarding]
    draw : Real -> Real

effect Score [version=1, evolution=sealed]
    score : Real -> Unit

instance random : Random
instance score : Score

handler replay for Random : Real -> Real [coverage=total]
    draw resumes 1

define replayed_model() : Real !{score} =
    handle random with replay in
        let x <- perform random.draw(1.0)
        perform score.score(x)
        return x
```

`Random` permits interface evolution, but `replay` is total for version 1 and
handles the exact `random` instance. The remaining `score` entry makes the
residual effect explicit. Runtime replay tables, samplers, and trace recorders
do not enter the stable module.

Logic programming uses the same handler and resumption rules:

<!-- compile: qiec -->
```qvr
effect Choose [version=1, evolution=sealed]
    choose[A : Type] : A -> A

effect Weight[K : Type] [version=1, evolution=sealed]
    add : K -> Unit

instance choice : Choose
instance weight : Weight[Real]

handler depth_first[A : Type] for Choose : A -> A [coverage=total]
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

## Integration contract

The v0.19 integration establishes five results. First, QVR has one typed
surface for indexed families, effect interfaces and instances, handler
signatures, and stable computation terms. Second, Didactic and Panproto check
the first-order indexed-family projection, after which the Quivers kernel checks
branch refinement, effect rows, handler coverage, and resumption contracts and
lowers the surface into a serializable `QiecModule`. Third, the compiler
attaches that checked module to its environment and produces a `Program` without
mixing it into probabilistic elaboration. Fourth, the structural IR preserves
declaration metadata as canonical JSON, while one central boundary gives every
transpiler the same refusal for computation bodies. Fifth, the CLI, TUI, REPL,
language server, Pygments and tree-sitter highlighters, TextMate grammar, and
Panproto migration assets share the v0.19 surface and diagnostic vocabulary.

A potential worry is that the probabilistic compiler and QIEC route now sit
beside one another. This is an intentional boundary, but it is also a concrete
limit. A mixed module may attach checked, declaration-only QIEC metadata to an
ordinary probabilistic `program`; the shared IR retains that metadata as a
canonical `qiec-json/v1` envelope, though the eleven target renderers do not
interpret it. If the module contains a QIEC computation body, every target
instead issues the same central `qiec:computation-body:<name>` refusal because
the probabilistic IR cannot represent `perform`, `handle`, indexed-case
evidence, or resumption grades. It never erases the body.

For current purposes, QIEC also has no source term for recursive calls, scoped
instance allocation, or authored handler-clause bodies. Handler bodies remain
runtime attachments, and `NamedComputation` values are checked entry points
rather than mutually recursive functions. These limits leave two live
possibilities: a QIEC-aware backend may consume the stable module directly, or
a later IR extension may preserve the missing control and evidence forms for
the existing backend family.

Panproto 0.74.2 restores version-aware validation for persisted objects written
by earlier releases. The migration gate reads the original QVR history without
rewriting its content-addressed object IDs, so those fixtures continue to test
the compatibility contract rather than a regenerated approximation of it.
