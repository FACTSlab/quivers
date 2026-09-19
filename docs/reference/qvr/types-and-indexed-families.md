# Types and indexed families

Indexed families make an invariant part of a value's type. A value of
`Measurement(Observed)` and one of `Measurement(Missing)` may share a family,
but a checked computation cannot confuse them. This property is the
**index-carrying invariant (ICI)**: the constructor determines the index, and
case analysis refines it locally.

## Kinds and binders

QVR uses square and round telescopes for different phases:

| Form | Meaning | Used by |
| --- | --- | --- |
| `[A : Type]` | static type binder | families, constructors, effects, handlers, computations |
| `[n : Nat]` | static index binder | constructors, operations, computations |
| `[E : Effect]` | static effect-interface binder | declarations that abstract over an effect |
| `(n : Nat)` | refinable family index | family declarations and case motives |
| `(x : T)` | runtime value parameter | computations and handler clauses |

Static arguments are supplied in brackets, as in `length[Real, S(Z)](xs)`.
Runtime values are supplied in parentheses. An integer literal may fill a
`Nat` static binder, so `window[32](xs)` is valid when `window` binds
`[n : Nat]`.

The built-in value types are `Unit`, `Bool`, `Int`, `Real`, `String`,
`LogWeight`, `Tensor[A]([d1, ...])`, `Sampleable[A]`, and `Site[A]`. A product
type is written `A * B`; a tuple value is written `(a, b)` and projected as
`pair[0]` or `pair[1]`.

## Closed index sorts

An `index` declaration introduces a closed sort and all of its constructors:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)
index Availability = Observed | Missing
```

Index constructors may be nullary or recursive. Because the sort is closed,
the checker knows every constructor that a `case` must cover. Index values are
static terms rather than runtime integers; use a `Nat` literal only when a
declaration explicitly binds a `Nat`.

## Family declarations

A family may have uniform static parameters in brackets and refinable indices
in parentheses:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))
```

`A` is uniform: a case branch cannot change which element type the vector
uses. `n` is refinable: matching `Nil` proves that the branch index is `Z`,
while matching `Cons[m]` proves that it is `S(m)`. Constructor fields precede
`->`; a nullary constructor omits them.

Family and constructor names occupy distinct namespaces. A constructor's
static arguments include the family's uniform parameters followed by its own
telescope. Thus a three-element real vector can be built as follows:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

define triple(a : Real, b : Real, c : Real) : Vec[Real](S(S(S(Z)))) !{} =
    let last = construct Cons[Real, Z](c, construct Nil[Real]() as Vec[Real](Z)) as Vec[Real](S(Z))
    let pair = construct Cons[Real, S(Z)](b, last) as Vec[Real](S(S(Z)))
    return construct Cons[Real, S(S(Z))](a, pair) as Vec[Real](S(S(S(Z))))
```

`construct` and `as` are both required. The annotation closes the expected
family application and prevents an ambiguous constructor from acquiring a
type from unrelated context.

## Indexed case analysis

A case expression states a motive, that is, the result type as a function of
the scrutinee's indices:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

define length[A : Type, n : Nat](xs : Vec[A](n)) : Int !{} =
    case xs motive (m : Nat) => Int
        Nil =>
            return 0
        Cons[k](head, tail) =>
            let rest <- length[A, k](tail)
            return 1 + rest
```

The branch binder `k` is a rigid local index. It may determine the type of
`tail` and the static argument of the recursive call, but it cannot escape the
branch in the result type. Coverage is conservative: a constructor may be
omitted only when closed-constructor equality proves that branch impossible.

This rule permits a total `head` without a `Nil` branch:

<!-- compile: qiec -->
```qvr
index Nat = Z | S(Nat)

family Vec[A : Type](n : Nat) : Type
    constructor Nil : Vec[A](Z)
    constructor Cons[m : Nat] : A * Vec[A](m) -> Vec[A](S(m))

define head[A : Type, n : Nat](xs : Vec[A](S(n))) : A !{} =
    case xs motive (m : Nat) => A
        Cons[k](first, rest) =>
            return first
```

`Vec[A](S(n))` cannot have been built by `Nil`, whose result index is `Z`.
The omitted branch is thus impossible, rather than a partial match.

## Pure expressions

Computation values and `program` expressions use one primitive registry.
Operators associate left within each precedence level:

| Precedence, low to high | Forms |
| --- | --- |
| Boolean disjunction | `||` |
| Boolean conjunction | `&&` |
| comparison | `==`, `!=`, `<`, `<=`, `>`, `>=` |
| additive | `+`, `-` |
| multiplicative | `*`, `/`, `%` |
| unary | `-`, `not` |

The built-in calls are `real`, `int`, `exp`, `log`, `sqrt`, `pow`, `abs`,
`min`, and `max`. Integer division and remainder truncate toward zero on every
target. Mixed integer-real arithmetic inserts an explicit conversion when the
surrounding form permits it. Write a comparison with a negative literal as
`x < (-1)` because `<-` is the effectful binding token.

List literals build tensors. `[1, 2, 3]` has integer elements;
`[1.0, 2.0, 3.0]` has real elements; nested lists add dimensions. Distribution
family applications build `Sampleable` values, `site("x")` builds a typed site,
and `log_prob(distribution, value)` returns a `LogWeight`.

## Common failures

- **Missing branch.** Add the constructor unless its result indices are
  provably incompatible with the scrutinee.
- **Escaped index.** Return a type independent of a branch-local static binder,
  or package the witness inside a family whose outer type is fixed.
- **Wrong constructor telescope.** Supply uniform family parameters first,
  then constructor-local parameters.
- **Value/static confusion.** Put types and indices in `[...]`; put runtime
  values in `(...)`.
- **Shadowed primitive type.** A telescope binder named `Real` is a local type
  variable, not the primitive. Rename it unless that abstraction is intended.

Continue with [Computations](computations.md) for recursion and binding, or
with [Effects and handlers](effects-and-handlers.md) for indexed computations
that perform requests.
