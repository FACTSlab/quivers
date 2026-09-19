# Effects and handlers

An effect declaration gives names and types to requests. An instance gives one
application of that interface a lexical identity. A computation's effect row
then records the instances it may request from, and a handler interprets one
instance. This separation is the **lexical-instance discipline (LID)**: two
instances of `State[Int]` have the same interface but remain distinct effects.

## Declare an interface

<!-- compile: qiec -->
```qvr
effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

instance left : State[Int]
instance right : State[Int]

define exchange(next : Int) : Int !{left, right} =
    let previous <- perform left.get()
    perform right.put(next)
    return previous
```

Square brackets bind static parameters on the interface or an operation. An
operation's argument types precede `->`; an operation with no arguments may
write either `Unit -> T` or the corresponding nullary form admitted by the
grammar. A request is always qualified by an instance:
`perform left.get()`. This qualification is what keeps `left` and `right`
separate in the row.

## Prelude interfaces

These interfaces are available without a source declaration:

| Interface | Principal operation | Purpose |
| --- | --- | --- |
| `Random` | `sample[A](Site[A], Sampleable[A]) -> A` | draw or replay a named random value |
| `Score` | `add(LogWeight) -> Unit` | add a density or factor |
| `State[S]` | `get`, `put` | lexical mutable state under a handler |
| `Abort[E]` | `abort[A](E) -> A` | typed early exit |
| `Choose` | `choose[A](Tensor[A]) -> A` | nondeterministic search |
| `Weight[K]` | `add(K) -> Unit` | semiring-parametric accumulation |
| `Compute[X, A]` | `apply(X) -> A` | process-local host computation |
| `Param` | `get[A](String) -> A` | named learned-parameter lookup |

`Search` is generated for a deduction or schema-backed parser because its
choice arity depends on the elaborated system. Structural declarations create
`Compute` instances; a caller normally encounters them through
`Program.run`, not by writing the requests manually.

## Effect rows

A computation signature ends in an effect row:

| Row | Meaning |
| --- | --- |
| `!{}` | closed and pure |
| `!{left, right}` | exactly these lexical instances |
| `!{robust | rho}` | `robust` plus an open tail `rho` |
| `!{robust | rho lacks robust}` | open tail that cannot already contain `robust` |

Rows contain instances, not interface names. A call contributes its callee's
row after static substitution. A `perform` contributes the named instance. An
`if` or `case` joins the rows of every reachable branch. Conservative case
coverage also means that an unproved-impossible branch contributes its
effects.

`lacks` constraints prevent accidental duplication when a row-polymorphic
computation adds an instance of its own. Use them when an abstraction handles
or introduces the same lexical name that a caller's tail might otherwise
contain.

## Handler declarations

A handler declares the interface it handles, an input and answer type, options,
one return clause, and operation clauses:

<!-- compile: qiec -->
```qvr
effect Robust
    shrink : Real -> Real

instance robust : Robust

handler half_weight for Robust : Real -> Real [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    shrink(x : Real) resumes 1 =>
        resume(0.5 * x)

define robustify(x : Real) : Real !{} =
    handle robust with half_weight in
        let adjusted <- perform robust.shrink(x)
        return adjusted
```

The `return` clause handles a computation that finishes without another
request. Each operation clause receives the request's fields and may call
`resume`. Handlers are deep: a resumed continuation runs beneath the same
handler. Effects performed by the clause body itself are offered only to
outer handlers.

### Handler options

| Option | Values | Effect |
| --- | --- | --- |
| `coverage` | `total`, `partial` | whether every operation is handled |
| `forwards` | `none`, `unknown` | whether a partial handler passes uncovered requests outward |
| `implementation` | `authored`, `foreign` | whether clauses have QVR bodies or runtime attachments |
| `introduces` | instance names | additional effects a handler body may perform |

A total handler removes the matched instance from the body's row. A partial
handler retains it. Forwarding and coverage are separate: an uncovered
operation in a non-forwarding partial handler is an error rather than an
implicit outer request.

### Resumption grades

| Grade | Contract |
| --- | --- |
| `0` | the clause must not resume |
| `aff` | at most once |
| `1` | exactly once |
| `omega` | any number of times when the continuation is duplicable |

The checker records the grade, and the evaluator enforces it. `omega` is used
by search handlers that explore several alternatives; a state update that
continues once normally uses `1`; an abort clause uses `0`.

## Scoped instances

`with instance` creates a fresh lexical identity inside a computation:

<!-- compile: qiec -->
```qvr
effect State[S : Type]
    get : Unit -> S
    put : S -> Unit

handler run_state for State[Int] : Int -> Int [coverage=total, forwards=none, implementation=authored]
    return x =>
        return x
    get resumes 1 =>
        resume(7)
    put(s : Int) resumes 1 =>
        resume(unit)

define local_read() : Int !{} =
    with instance cell : State[Int] in
        handle cell with run_state in
            perform cell.put(3)
            let value <- perform cell.get()
            return value
```

The instance does not exist outside its block. A total handler must eliminate
it before control leaves; otherwise the checker reports
`qiec-instance-escape`. Nesting two `with instance cell ...` blocks creates two
identities even though the source spelling is shadowed.

## Foreign handlers and providers

A clause with no `=>` body is a signature for a process-local implementation,
and the declaration must say `implementation=foreign`. The stable module
contains the handler type, operation coverage, and grades, but no callable.
At invocation, the selected runtime provider binds the handler's stable ID to
an implementation. Runtime configuration files contain provider names and
data only; they cannot inject code.

The built-in `core` provider supplies the reference implementations used by
program execution, deductions, and the standard prelude. Structural models
compose `core` with a structural provider holding their compiled PyTorch
modules. See [Execution and tooling](execution-and-tooling.md#runtime-providers).

## Common failures

- **Request absent from the row.** Add the lexical instance or make the row
  polymorphic when the abstraction is intended to preserve caller effects.
- **Wrong applied interface.** `State[Int]` and `State[String]` are distinct;
  a handler must match the instance's exact application.
- **Instance escape.** Handle a scoped instance before leaving its block.
- **Grade violation.** Make the number of `resume` calls match the declared
  grade.
- **Missing provider.** Attach a provider that implements the foreign handler,
  or use an authored handler.

The [effects tutorial](../../tutorials/qvr/10-effects-and-handlers.md) builds
and traces an authored handler end to end.
