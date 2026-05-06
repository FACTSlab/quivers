# Compositional Effects

The Phase 6/7 categorial-effects integration adds two parallel
typeclass towers — Haskell-style monads and Hughes-style arrows — and
layers a class-driven schema-lifting machinery on top, so that
linguistic effects (scope-taking, anaphora, focus / alternatives,
presupposition, supplements, plurality) compose with the residuated
category universe through the same surface as ordinary categorial
grammar rules.

This guide walks through the framework. Formal denotations are in the
[Denotational Semantics](../semantics/effects.md) section.

## Typeclass hierarchy

Both towers live as Python ABCs in `quivers.monadic.typeclasses` and
`quivers.arrows.typeclasses`. Effects (instances) implement the
operations a typeclass requires; the lifting machinery dispatches on
which classes an effect inhabits, never on the effect's identity.

The monad-side hierarchy:

| Class | Adds | Linguistic use |
|-------|------|----------------|
| `Functor` | `fmap` | structure-preserving lifts |
| `Applicative` | `pure`, `apply` | uniform-lift of base rules through an effect |
| `Monad` | `join` (and derived `bind`) | scope-extruding lifts (Charlow) |
| `Alternative` | `empty`, `alt` | non-deterministic / Hamblin-style branches |
| `MonadPlus` | combines `Monad` and `Alternative` | branching effects with substitution |
| `Foldable` / `Traversable` | `foldr`, `traverse` | distribute Applicative actions through a structure |
| `MonadTrans` | `lift` | stack one monad on top of another |

The arrow-side hierarchy (Hughes 2000, [doi:10.1016/S0167-6423(99)00023-4](https://doi.org/10.1016/S0167-6423(99)00023-4)):

| Class | Adds | Use |
|-------|------|-----|
| `Category_` | `id_arr`, `compose` | the bare composition law |
| `Arrow` | `arr`, `first` | symmetric-monoidal arrow shape |
| `ArrowChoice` | `left_arr` | sum-elimination |
| `ArrowApply` | `app` | equivalent power to `Monad` via `arrow_monad` |
| `ArrowLoop` | `loop_arr` | feedback / fixed-point (the chart-fold) |
| `ArrowZero` / `ArrowPlus` | `zero_arr`, `alt_arr` | pointwise alternatives on hom-sets |

The `kleisli` and `arrow_monad` bridges in
`quivers.monadic.bridges` translate between the two towers freely;
ArrowApply and Monad are interconvertible.

## Stdlib effect instances

Eight stdlib effects ship in `quivers.monadic.instances`:

| Effect | Class instances | Linguistic use |
|--------|------------------|----------------|
| `Identity` | `Monad`, `Functor` | trivial / no-effect base case |
| `Maybe` | `MonadPlus`, `Functor` | partiality / presupposition failure |
| `Alternative_` | `MonadPlus`, `Foldable`, `Traversable` | Hamblin alternatives, focus, questions |
| `Continuation(answer)` | `Monad`, `Functor` | scope-taking, generalised quantifiers |
| `State(state)` | `Monad`, `Functor` | dynamic anaphora, discourse referents |
| `Reader(env)` | `Monad`, `Functor` | assignment functions, indexicality |
| `Writer(monoid)` | `Monad`, `Functor` | supplements, nonrestrictive content |
| `List(max_length)` | `MonadPlus`, `Foldable`, `Traversable` | bag-of-readings parsers |

Each effect is a `dx.Model` carrying the relevant parameters (e.g.
`Continuation(answer=S)` is parameterised by the answer type), and
each registers against its appropriate ABC(s) via `ABC.register(...)`.

## Class-driven schema lifting

The `class_directed_lifts(base_schema, effect)` function in
`quivers.stochastic.effect_lifts` introspects which typeclasses an
effect inhabits and emits one lifted schema per class:

| Effect class | Lifts emitted |
|--------------|---------------|
| `Applicative` | `pure_T`, `apply_T` |
| `Monad` | adds `bind_T` (scope-extruding) |
| `Alternative` | adds `alt_T` |
| `MonadPlus` | union of Monad + Alternative |

Each lift is a real `SchemaDecl` and feeds into the existing
`PatternBinarySchema` / `PatternUnarySchema` runtime; the chart parser
consumes lifted and base schemas uniformly.

```python
from quivers.dsl.ast_nodes import SchemaDecl, TypeName, TypeProduct, TypeSlash
from quivers.monadic.instances import Continuation, Alternative_
from quivers.stochastic.effect_lifts import class_directed_lifts
from quivers.core.objects import FinSet

S = FinSet(name="S", cardinality=2)

forward_app = SchemaDecl(
    name="forward_app",
    parameter_names=(("X", "Y"),),
    parameter_types=(TypeName(name="Cat"),),
    domain=TypeProduct(components=(
        TypeSlash(result=TypeName(name="X"), argument=TypeName(name="Y"), direction="/"),
        TypeName(name="Y"),
    )),
    codomain=TypeName(name="X"),
)

cont_lifts = class_directed_lifts(forward_app, Continuation(answer=S))
# -> 3 lifts: pure_Continuation, apply_Continuation, bind_Continuation

alt_lifts = class_directed_lifts(forward_app, Alternative_())
# -> 4 lifts: pure_Alternative, apply_Alternative, bind_Alternative, alt_Alternative
```

## Algebraic effects + handlers

For effects that don't fit a closed-form typeclass instance,
`quivers.monadic.algebraic` provides a free-monad-over-signature
construction:

- `Operation(name, parameter, result)` — one operation in a signature.
- `EffectSignature(name, operations)` — a signature; lifts to a
  panproto theory via `to_theory()`.
- `FreeMonad(signature)` — the free monad over the signature; a
  `Monad` instance automatically.
- `Handler(signature, target, return_clause, operation_clauses)` —
  interprets a `FreeMonad`-valued computation in a target monad.
  Equivalently: a panproto theory morphism from
  `signature.to_theory()` into the target monad's theory.

Handlers compose with `chart_fold` to produce parsers that interpret
their effect-typed denotation through registered handlers, ending in
an effect-pure target.

## Bridges between the two towers

`quivers.monadic.bridges` contains:

- `kleisli(monad)` — wraps a `Monad` as a `Kleisli` arrow registered
  against `Arrow` and `ArrowApply`.
- `arrow_monad(arrow)` — wraps an `ArrowApply` as an `ArrowMonad`
  registered against `Monad`.

The pair gives a free choice of presentation: write effects as monads
or as arrows; the framework treats them uniformly.

## Joint type-and-effect dispatch in the parser

When the chart parser fires at span `(i, j)`, each cell carries a
distribution over `(Cat, EffectStack)` pairs (the residuated universe
×️ a stack of declared effects). At each cell the parser considers:

1. **Base firings** — when both children are effect-pure, the base
   schema's pattern unifies on the type coordinate as in classical
   Lambek.
2. **Lift firings** — when child effect-stacks differ, the
   class-driven lifts of `class_directed_lifts` interpose `pure_T` /
   `fmap_T` / `apply_T` / `bind_T` etc. as appropriate.
3. **Handler firings** — when the cell holds a `T(α)` distribution and
   a `Handler` from `T` is registered, applying the handler produces
   a fresh cell at a strictly-shorter effect-stack.
4. **Commutation firings** — when a `DistributiveLaw(T, U)` (or arrow
   analogue) is registered, the `swap_TU` schema exchanges sibling
   effect orderings.

The four kinds compose freely; the chart's CKY enumeration explores
the joint search space.

## Worked example: scope-taking via Continuation

The example file `src/quivers/dsl/examples/quantifier_scope.qvr`
illustrates a Charlow-style scope-taking grammar:

```qvr
object Atoms = {NP, S, VP, N, PP}
object Cat = FreeResiduated(Atoms, depth=2, ops=[slash])
object Token : 256

# Base schemas (the Lambek calculus core).
schema forward_app[X, Y : Cat] : (X/Y) * Y -> X
schema backward_app[X, Y : Cat] : Y * (X\Y) -> X

# Lifted schemas under the Continuation effect.
schema apply_Cont_fwd[X, Y : Cat] : Cont_S(X/Y) * Cont_S(Y) -> Cont_S(X)
schema scope_take[X, Y : Cat] : Cont_S(X) * (X\Y) -> Cont_S(Y)

latent lex : Token -> Cat

let grammar = parser(
    rules=[forward_app, backward_app, apply_Cont_fwd, scope_take],
    terminal=Token, start=S
)
output grammar
```

In production code, the lifted schemas (`apply_Cont_fwd`,
`scope_take`) are auto-generated by `class_directed_lifts`; the
explicit declaration above demonstrates the *resulting surface* a
user would see if they enumerated the lifts manually.

## References

- Hughes, J. (2000). *Generalising Monads to Arrows*. [doi:10.1016/S0167-6423(99)00023-4](https://doi.org/10.1016/S0167-6423(99)00023-4)
- Bauer, A. and Pretnar, M. (2015). *Programming with Algebraic Effects and Handlers*. [doi:10.1016/j.jlamp.2014.02.001](https://doi.org/10.1016/j.jlamp.2014.02.001)
- Bumford, D. and Charlow, S. (2014). *Making distinctions: linguistic effects and their interactions*. [doi:10.1007/s10988-014-9167-3](https://doi.org/10.1007/s10988-014-9167-3)
- Charlow, S. (2020). *Static and dynamic exceptional scope*. [doi:10.3765/sp.13.16](https://doi.org/10.3765/sp.13.16)
- Bumford, D. (2017). *Split-scope effects*. [doi:10.1007/s10988-017-9216-9](https://doi.org/10.1007/s10988-017-9216-9)
