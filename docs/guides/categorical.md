# Categorical Structures

## Functors

A functor $F: \mathcal{C} \to \mathcal{D}$ between $\mathcal{V}$-enriched categories maps objects to objects and morphisms to morphisms, preserving composition and identity.

In quivers, [`Functor`](../api/categorical/functors.md) is an abstract base class with three abstract methods:

- `map_object(obj)`: the object map $F_\text{obj}: \text{Ob}(\mathcal{C}) \to \text{Ob}(\mathcal{D})$
- `map_morphism(morph)`: the morphism map, returning a [`FunctorMorphism`](../api/core/morphisms.md)
- `map_tensor(tensor, algebra)`: the tensor-level action

Concrete subclasses ship with the library:

```python
from quivers.categorical.functors import IdentityFunctor, ComposedFunctor, FreeMonoidFunctor
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# Identity endofunctor
I = IdentityFunctor()
assert I.map_object(X) == X

# Free monoid functor on FinSet
F = FreeMonoidFunctor(max_length=3)
FX = F.map_object(X)            # FreeMonoid(generators=X, max_length=3)

f = morphism(X, Y)
Ff = F.map_morphism(f)          # FunctorMorphism with domain F(X), codomain F(Y)

# Composition of functors
FF = ComposedFunctor(F, F)      # F ∘ F applied as outer ∘ inner
```

To define your own functor, subclass `Functor` and implement the three abstract methods.

## Natural Transformations

A natural transformation $\alpha: F \Rightarrow G$ between functors $F, G: \mathcal{C} \to \mathcal{D}$ is a family of morphisms $\{\alpha_X: F(X) \to G(X)\}$ indexed by objects of $\mathcal{C}$, such that the naturality square commutes:

$$
\begin{array}{cccc}
F(X) \xrightarrow{\alpha_X} & G(X) \\
\Big\downarrow F(f) & & \Big\downarrow G(f) \\
F(Y) \xrightarrow{\alpha_Y} & G(Y)
\end{array}
$$

[`NaturalTransformation`](../api/categorical/natural_transformations.md) is an ABC; use [`ComponentwiseNT`](../api/categorical/natural_transformations.md) to construct one from a callable that produces each component:

```python
from quivers.categorical.functors import IdentityFunctor, FreeMonoidFunctor
from quivers.categorical.natural_transformations import ComponentwiseNT
from quivers.categorical.adjunctions import FreeForgetfulAdjunction

F = IdentityFunctor()
G = FreeMonoidFunctor(max_length=3)

# unit η: Id ⇒ Free (length-1 embedding) reuses the adjunction's unit
adj = FreeForgetfulAdjunction(max_length=3)
eta = ComponentwiseNT(F, G, adj.unit_component)

# verify naturality on a chosen morphism
f = morphism(X, Y)
assert eta.verify_naturality(f)
```

## Adjunctions

An adjunction $F \dashv G$ is a pair of functors with morphisms $\eta: \text{id}_\mathcal{C} \Rightarrow GF$ (unit) and $\varepsilon: FG \Rightarrow \text{id}_\mathcal{D}$ (counit) satisfying triangle identities.

The free-forgetful adjunction $\text{Free} \dashv \text{Forget}$ is shipped as [`FreeForgetfulAdjunction`](../api/categorical/adjunctions.md):

```python
from quivers.categorical.adjunctions import FreeForgetfulAdjunction
from quivers.core.objects import FinSet, FreeMonoid

A = FinSet(name="A", cardinality=5)
adjunction = FreeForgetfulAdjunction(max_length=3)

# Left and right functors
F = adjunction.left        # FreeMonoidFunctor
U = adjunction.right       # ForgetfulFunctor

# Monad image T(A) = U(F(A)) is the free monoid on A
T_A = U.map_object(F.map_object(A))
assert isinstance(T_A, FreeMonoid)

# Unit and counit components
eta_A = adjunction.unit_component(A)        # A → A*
eps_A = adjunction.counit_component(A)      # A* → A

# Triangle identities
assert adjunction.verify_triangle_right(A)
```

## Monoidal Structures

A monoidal category $(\mathcal{C}, \otimes, I)$ has:

- A bifunctor $\otimes: \mathcal{C} \times \mathcal{C} \to \mathcal{C}$ (the product)
- A unit object $I$
- Natural associativity, left identity, right identity (with coherence)

### Cartesian Monoidal

The standard monoidal structure on finite sets. [`CartesianMonoidal`](../api/categorical/monoidal.md) exposes the object-level `product`, the `unit` (terminal object), and the coherence morphisms `associator`, `left_unitor`, `right_unitor`, `braiding`:

```python
from quivers.categorical.monoidal import CartesianMonoidal
from quivers.core.objects import FinSet, Unit

monoidal = CartesianMonoidal()

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

# product on objects
XY = monoidal.product(X, Y)
assert XY == X * Y
assert monoidal.unit == Unit

# coherence morphisms
swap = monoidal.braiding(X, Y)              # X × Y → Y × X
assoc = monoidal.associator(X, Y, X)        # (X×Y)×X → X×(Y×X)
```

Morphism-level tensor product is provided directly on `Morphism` via the `@` operator:

```python
f = morphism(X, Y)
g = morphism(X, Y)
fg = f @ g                                   # ProductMorphism: X×X → Y×Y
```

### Coproduct Monoidal

Coproduct as the monoidal operation:

```python
from quivers.categorical.monoidal import CoproductMonoidal

monoidal = CoproductMonoidal()

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)

XY = monoidal.product(X, Y)
assert XY == X + Y
```

The unit here is the initial object (empty set), exposed as `EMPTY` from `quivers.categorical.monoidal`.

## Base Change

Change the enriching algebra without changing the structure of morphisms. [`BaseChange`](../api/categorical/base_change.md) is an ABC; concrete instances are `BoolToFuzzy` and `FuzzyToBool`:

```python
from quivers.categorical.base_change import BoolToFuzzy, FuzzyToBool
from quivers.core.algebras import BOOLEAN, PRODUCT_FUZZY

# fuzzy → boolean via thresholding
fuzzy_morph = morphism(X, Y, algebra=PRODUCT_FUZZY)
to_bool = FuzzyToBool(threshold=0.5)
bool_morph = to_bool.apply_to_morphism(fuzzy_morph)

assert bool_morph.algebra == BOOLEAN
assert bool_morph.domain == fuzzy_morph.domain
assert bool_morph.codomain == fuzzy_morph.codomain

# boolean → fuzzy via inclusion
to_fuzzy = BoolToFuzzy()
fuzzy_again = to_fuzzy.apply_to_morphism(bool_morph)
```

## Traced Monoidal Structure

A traced monoidal category adds a trace operator:

$$
\text{tr}^X_{Y,Z}: \mathcal{C}(Y \otimes X, Z \otimes X) \to \mathcal{C}(Y, Z)
$$

This is useful for feedback and recursive definitions. The free function [`trace`](../api/categorical/traced.md) wraps a [`CartesianTrace`](../api/categorical/traced.md):

```python
from quivers.categorical.traced import trace

X = FinSet(name="X", cardinality=2)
Y = FinSet(name="Y", cardinality=3)
Z = FinSet(name="Z", cardinality=4)

# f: (Y × X) → (Z × X)
f = morphism(Y * X, Z * X)

# Trace over X: returns A → B with the feedback loop closed
traced_f = trace(f, feedback=X, domain=Y, codomain=Z)
assert traced_f.domain == Y
assert traced_f.codomain == Z
```

The trace contracts the matching $X$ wire, leaving the external domain $Y$ and codomain $Z$.

## Summary

| Structure | Concept | Syntax |
|-----------|---------|--------|
| Functor | Object and morphism map | `F: C → D` |
| Natural Transform | Family of morphisms | `α: F ⇒ G` |
| Adjunction | Inverse functors up to natural isos | `F ⊣ G` |
| Monoidal | Product with coherence | `(C, ⊗, I)` |
| Base Change | Enrichment transformation | `FuzzyToBool().apply_to_morphism(f)` |
| Traced | Feedback/iteration | `trace(f, feedback=X, domain=Y, codomain=Z)` |
