# Compositional Effects

QVR exposes two parallel typeclass towers, Haskell-style monads and
Hughes-style arrows, together with a class-driven schema-lifting
machinery, so that linguistic effects (scope-taking, anaphora, focus
/ alternatives, presupposition, supplements, plurality) compose with
the residuated category universe through the same surface as ordinary
categorial grammar rules.

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
| `Monad` | `join` (and derived `bind`) | scope-extruding lifts (Bumford & Charlow 2026) |
| `Alternative` | `empty`, `alt` | non-deterministic / Hamblin-style branches |
| `MonadPlus` | combines `Monad` and `Alternative` | branching effects with substitution |
| `Foldable` / `Traversable` | `foldr`, `traverse` | distribute Applicative actions through a structure |
| `MonadTrans` | `lift` | stack one monad on top of another |

The arrow-side hierarchy (Hughes 2000, *Generalising monads to arrows*, [doi:10.1016/S0167-6423(99)00023-4](https://doi.org/10.1016/S0167-6423(99)00023-4)):

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

- `Operation(name, parameter, result)`: one operation in a signature.
- `EffectSignature(name, operations)`: a signature; lifts to a
  panproto theory via `to_theory()`.
- `FreeMonad(signature)`: the free monad over the signature; a
  `Monad` instance automatically.
- `Handler(signature, target, return_clause, operation_clauses)` :
  interprets a `FreeMonad`-valued computation in a target monad.
  Equivalently: a panproto theory morphism from
  `signature.to_theory()` into the target monad's theory.

Handlers compose with a `deduction { … }` block to produce parsers
that interpret their effect-typed denotation through registered
handlers, ending in an effect-pure target.

## Bridges between the two towers

`quivers.monadic.bridges` contains:

- `kleisli(monad)`: wraps a `Monad` as a `Kleisli` arrow registered
  against `Arrow` and `ArrowApply`.
- `arrow_monad(arrow)`: wraps an `ArrowApply` as an `ArrowMonad`
  registered against `Monad`.

The pair gives a free choice of presentation: write effects as monads
or as arrows; the framework treats them uniformly.

## Joint type-and-effect dispatch in the parser

When the chart parser fires at span `(i, j)`, each cell carries a
distribution over `(Cat, EffectStack)` pairs (the residuated universe
×️ a stack of declared effects). At each cell the parser considers:

1. **Base firings**, when both children are effect-pure, the base
   schema's pattern unifies on the type coordinate as in classical
   Lambek.
2. **Lift firings**, when child effect-stacks differ, the
   class-driven lifts of `class_directed_lifts` interpose `pure_T` /
   `fmap_T` / `apply_T` / `bind_T` etc. as appropriate.
3. **Handler firings**, when the cell holds a `T(α)` distribution and
   a `Handler` from `T` is registered, applying the handler produces
   a fresh cell at a strictly-shorter effect-stack.
4. **Commutation firings**, when a `DistributiveLaw(T, U)` (or arrow
   analogue) is registered, the `swap_TU` schema exchanges sibling
   effect orderings.

The four kinds compose freely; the chart's CKY enumeration explores
the joint search space.

## Worked example: scope-taking via Continuation

The example file `src/quivers/dsl/examples/quantifier_scope.qvr`
illustrates a Bumford & Charlow-style scope-taking grammar:

```qvr
object Term : 16

deduction QScope : Term -> Term {
    atoms {
        S, NP, N, VP, PP,
        Fwd, Bwd, Cont,
        span
    }

    # Base forward / backward application.
    rule fwd_app
        : span(I, K, Fwd(A, B)), span(K, J, B)
        |- span(I, J, A)
    rule bwd_app
        : span(I, K, B), span(K, J, Bwd(A, B))
        |- span(I, J, A)

    # Applicative lift of forward application under Cont:
    #   Cont(A/B) * Cont(B) ⊢ Cont(A)
    rule fwd_app_cont
        : span(I, K, Cont(Fwd(A, B))), span(K, J, Cont(B))
        |- span(I, J, Cont(A))

    # Pure: A ⊢ Cont(A)
    rule pure_cont
        : span(I, J, A) |- span(I, J, Cont(A))

    # Scope-extruding bind: a Cont-typed expression takes scope
    # over an inner argument by absorbing the surrounding context.
    #   Cont(A) * (A\B) ⊢ Cont(B)
    rule scope_take
        : span(I, K, Cont(A)), span(K, J, Bwd(B, A))
        |- span(I, J, Cont(B))

    semiring  LogProb
    start     S
    depth     6
}
```

In production code, the lifted schemas (`apply_Cont_fwd`,
`scope_take`) are auto-generated by `class_directed_lifts`; the
explicit declaration above demonstrates the *resulting surface* a
user would see if they enumerated the lifts manually.

## References

- Bumford, D. and Charlow, S. (2026). [*Effect-Driven Interpretation: Functors for Natural Language Composition*](https://www.cambridge.org/core/elements/abs/effectdriven-interpretation/56671E539160AAA1DACF8555B82A2FE4). Cambridge Elements in Semantics. Cambridge University Press. Online ISBN 9781009285377; preprint [arXiv:2504.00316](https://arxiv.org/abs/2504.00316), draft at [simoncharlow.com/papers/cup-effects.pdf](https://simoncharlow.com/papers/cup-effects.pdf).
- Hughes, J. (2000). [*Generalising monads to arrows*](https://doi.org/10.1016/S0167-6423(99)00023-4). Science of Computer Programming, 37(1–3), 67–111.
- Plotkin, G. and Power, J. (2003). [*Algebraic operations and generic effects*](https://doi.org/10.1023/A:1023064908962). Applied Categorical Structures, 11(1), 69–94.
- Bauer, A. and Pretnar, M. (2015). [*Programming with algebraic effects and handlers*](https://doi.org/10.1016/j.jlamp.2014.02.001). Journal of Logical and Algebraic Methods in Programming, 84(1), 108–123.
- Charlow, S. (2025). [*Static and dynamic exceptional scope*](https://doi.org/10.1093/jos/ffad012). Journal of Semantics (advance article).
- McBride, C. and Paterson, R. (2008). [*Applicative programming with effects*](https://doi.org/10.1017/S0956796807006326). Journal of Functional Programming, 18(1), 1–13.
