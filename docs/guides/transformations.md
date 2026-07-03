# Transformations and Composition Rules

This guide covers two related surfaces: *transformations* (the first-class value-level surface for change-of-base) and the *composition-rule hierarchy* (the algebraic surface beneath `>>`). Both are essential vocabulary for working with morphisms that cross enrichment algebras.

## First-class transformations

A *transformation* turns a morphism in one enrichment algebra into a morphism in another. Two kinds exist:

- **Algebra homomorphisms** $\varphi : \mathcal{V} \to \mathcal{W}$ act pointwise: every entry of the morphism's tensor is sent through $\varphi$.
- **Morphism transformations** act on the whole tensor, consuming axis information; softmax row-normalization, L1/L2 normalization, and Bayes inversion under a prior all live here.

The library treats both as first-class DSL values, living in a transformation namespace disjoint from the morphism namespace. You can let-bind them, compose them with `>>>`, and pass either kind to `change_base`.

### Singletons

Bare-name transformations carry a fixed `(source, target)` pair. The shipped catalogue:

| Name | `source` | `target` |
|---|---|---|
| `expectation` | `Markov` | `ProductFuzzyAlgebra` |
| `material_implication` | `ProductFuzzyAlgebra` | `Godel` |
| `log_prob` | `ProductFuzzyAlgebra` | `LogProb` |
| `max_plus` | `LogProb` | `MaxPlus` |
| `threshold` | `ProductFuzzyAlgebra` | `Boolean` |
| `boolean_embedding` | `Boolean` | `ProductFuzzyAlgebra` |
| `probability_clamp` | `Real` | `Probability` |
| `probability_to_real` | `Probability` | `Real` |
| `counting_from_real` | `Real` | `Counting` |
| `counting_to_real` | `Counting` | `Real` |

### Constructors

Parametric transformations take one argument (an object or a morphism) and produce a transformation value:

| Constructor | Argument | `source` | `target` |
|---|---|---|---|
| `softmax(B)` | object | `ProductFuzzyAlgebra` | `Markov` |
| `l1_normalize(B)` | object | `Real` | `Markov` |
| `l2_normalize(B)` | object | `Real` | `Real` |
| `bayes_invert(prior)` | morphism | `Markov` | `Markov` |

### Let-binding and composition

Both forms can be let-bound and composed with `>>>`. Composition checks the seam at compile time: `t1 >>> t2` requires `target(t1) = source(t2)`.

<!-- compile: false -->
```qvr
composition product_fuzzy [level=algebra]
object A : FinSet 3
object B : FinSet 4
morphism f : A -> B [role=latent]

define s    = softmax(B)
define pipe = s >>> expectation
define g    = f.change_base(pipe)
export g
```

`softmax(B)` produces a transformation $\mathcal{V}_\mathrm{pf} \to \mathcal{V}_\mathrm{M}$; `expectation` is $\mathcal{V}_\mathrm{M} \to \mathcal{V}_\mathrm{pf}$; their composition round-trips and gives a row-normalized morphism back in $\mathcal{V}_\mathrm{pf}$.

The `change_base` postfix accepts any expression that denotes a transformation: a bare singleton, a let-bound name, a constructor call, or a `>>>` chain.

### Python API

The same surface is exposed in the Python core at `quivers.core.trans`:

<!-- python: skip -->
```python
from quivers.core.trans import compose_trans
from quivers.core.morphism_transformations import softmax
from quivers.core.algebra_morphisms import EXPECTATION

pipe = compose_trans(softmax(B), EXPECTATION)
g    = f.change_base(pipe)
```

The same flat-sequence representation underpins both surfaces; the DSL's `>>>` desugars to `compose_trans`.

## The composition-rule hierarchy

Underneath `>>` sits the *composition rule* of the enclosing module. The library's hierarchy:

```
CompositionRule
    ├── BilinearForm     no associativity promise
    └── Semigroupoid     associative ⊗, no identity
            └── Algebra  associative + identity + meet + negate
```

The four `.qvr` level keywords select the level:

| Declaration | Required level | Available operations |
|---|---|---|
| `composition X [level=algebra]` | `Algebra` | `>>`, `@`, `identity(A)`, `f.dagger`, `f.trace(A)`, `cup(A)`, `cap(A)`, all compact-closed surface |
| `composition X [level=semigroupoid]` | `Semigroupoid` | `>>`, `@`, no identity / compact-closed surface |
| `composition X [level=bilinear_form]` | `BilinearForm` | `>>`, `@`, no associativity guarantee |
| `composition X [level=rule]` | `CompositionRule` | permissive; accepts any rule |

A typed `CompileError` flags every Algebra-only operation used inside a non-Algebra module. The diagnostic names the operation and the offending rule's level.

### User-defined inline rules

A composition rule can be defined inline. The body is a sequence of entries whose RHS is a `let`-arithmetic expression:

<!-- compile: false -->
```qvr
composition my_godel [level=algebra]
    tensor_op(a, b) = a * b
    join(t) = sum(t)
    unit = 1.0
    zero = 0.0

composition my_semi [level=semigroupoid]
    tensor_op(a, b) = (1 - a + a * b)
    join(t) = prod(t)

composition signed_dot [level=bilinear_form]
    tensor_op(a, b) = (a + b) * 0.5
    join(t) = sum(t)
```

Required entries per level:

| Level | Required | Optional |
|---|---|---|
| `algebra` | `tensor_op`, `join`, `unit`, `zero` | `negation`, `meet` |
| `semigroupoid` | `tensor_op`, `join` | |
| `bilinear_form` | `tensor_op`, `join` | |

`CustomAlgebra` runs a small sanity check at construction; `CustomSemigroupoid` runs an associativity smoke check on a fixed sample (skippable via `verify_associative=False` in the Python API). `CustomBilinearForm` runs no check, since the type opts out of the associativity promise.

### Material implication

The shipped `material_implication` rule is a `CustomSemigroupoid` with tensor $a \otimes b = 1 - a + a \cdot b$ (Reichenbach implication) and join $\bigvee = \prod$ (product reduction). Use it with the DSL keyword:

<!-- compile: false -->
```qvr
composition material_impl [level=semigroupoid]
object A : FinSet 3
object B : FinSet 3
morphism f : A -> B [role=latent]
morphism g : B -> B [role=latent]
define composed = f >> g
export composed
```

The compact-closed surface is not available; the module is otherwise normal.

## Operadic contractions

Binary composition `f >> g` contracts two morphisms along a single shared axis. Many tensor-network patterns contract three or more inputs at a shared reduction; the `contraction` declaration is the surface for this.

### Wiring spec

The wiring spec is an einsum-style string with comma-separated input letter-tuples and an arrow to the output letters:

```
"sp, sq, pqo -> so"
```

Each input lists its axes; the arrow's right-hand side names the surviving output letters. Letters present in inputs but not in the output are contracted under the rule's `join`. Letters in the output must appear in at least one input.

### Declaration

<!-- compile: false -->
```qvr
contraction op_apply (
    arg1 : A -> B,
    arg2 : A -> C,
    kernel : B -> D
) : A -> D
    rule product_fuzzy
    wiring "sp, sq, pqd -> sd"
```

The declared `op_apply` becomes callable from any expression site: `let combined = op_apply(arg1_morph, arg2_morph, kernel_morph)`. Each call checks the argument count against the declared input arity, and each argument's domain / codomain shape (by numel) against the declared per-input signature.

### Validation

The wiring spec is validated at declaration time:

- Each input must be non-empty.
- No input may repeat a letter; diagonal / trace contractions are out of scope at this stratum.
- The output spec may not repeat a letter.
- Every output letter must appear in at least one input.

A spec violating any of these constraints raises `CompileError` at the declaration, not at the call site.

### Python API

`quivers.core.wiring` exposes the same surface:

<!-- python: skip -->
```python
from quivers.core.wiring import einsum_wiring, contract

wiring = einsum_wiring(PRODUCT_FUZZY, "sp, sq, pqo -> so")
out    = contract(wiring, arg1, arg2, kernel)
```

`EinsumWiring` works against any `CompositionRule` instance, not just algebras:

<!-- python: skip -->
```python
from quivers.core.algebras import material_implication

mi = material_implication()
wiring = einsum_wiring(mi, "ij, jk -> ik")
out    = wiring.apply(f_tensor, g_tensor)
```

## Further reading

- The [QVR tutorial chapter 7](../tutorials/qvr/07-categorical.md) gives a user-friendly introduction to the categorical surface, including transformations and composition rules.
- The [Python tutorial chapters 6 and 7](../tutorials/python/06-first-class-trans.md) cover the Python API in detail.
- [Composition Rules](../semantics/composition-rules.md) is the formal denotational treatment.
- [Algebras and Base Change](../semantics/algebras.md) covers the eleven shipped algebras and the homomorphism registry.
