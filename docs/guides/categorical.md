# Categorical Structures

## Functors

A functor $F: \mathcal{C} \to \mathcal{D}$ between $\mathcal{V}$-enriched categories maps objects to objects and morphisms to morphisms, preserving composition and identity.

In quivers, a `Functor` is defined by:
- An object map $F_\text{obj}: \text{Ob}(\mathcal{C}) \to \text{Ob}(\mathcal{D})$
- A morphism map preserving composition: $F(g \circ f) = F(g) \circ F(f)$

```python
from quivers.categorical.functors import Functor
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism, identity

# Define a functor by providing object and morphism maps
X = FinSet("X", 3)
Y = FinSet("Y", 4)
Z = FinSet("Z", 2)

# Object map: X -> X, Y -> Z
def obj_map(obj):
    if obj == X:
        return X
    elif obj == Y:
        return Z
    else:
        raise ValueError(f"unknown object {obj}")

# Morphism map: given f: A -> B, return F(f): F(A) -> F(B)
def morph_map(morph):
    if morph.domain == X and morph.codomain == Y:
        return morphism(X, Z)  # F(f)
    elif morph == identity(X):
        return identity(X)
    else:
        raise NotImplementedError()

F = Functor(obj_map, morph_map)

# Apply to objects and morphisms
f = morphism(X, Y)
Ff = F(f)
assert Ff.domain == X and Ff.codomain == Z
```

## Natural Transformations

A natural transformation $\alpha: F \Rightarrow G$ between functors $F, G: \mathcal{C} \to \mathcal{D}$ is a family of morphisms $\{\alpha_X: F(X) \to G(X)\}$ indexed by objects of $\mathcal{C}$, such that the naturality square commutes:

$$
\begin{array}{cccc}
F(X) \xrightarrow{\alpha_X} & G(X) \\
\Big\downarrow F(f) & & \Big\downarrow G(f) \\
F(Y) \xrightarrow{\alpha_Y} & G(Y)
\end{array}
$$

In code:

```python
from quivers.categorical.natural_transformations import NaturalTransformation

# Define the component morphisms
alpha_X = morphism(F(X), G(X))
alpha_Y = morphism(F(Y), G(Y))

components = {X: alpha_X, Y: alpha_Y}

# Create the natural transformation
alpha = NaturalTransformation(F, G, components)

# Verify naturality (composition commutes)
f = morphism(X, Y)
# G(f) ∘ α_X == α_Y ∘ F(f)
assert alpha.is_natural(f)
```

## Adjunctions

An adjunction $F \dashv G$ is a pair of functors with morphisms $\eta: \text{id}_\mathcal{C} \Rightarrow GF$ (unit) and $\varepsilon: FG \Rightarrow \text{id}_\mathcal{D}$ (counit) satisfying triangle identities.

A classical example: the free-forgetful adjunction. `FreeMonoidFunctor` constructs free monoids, and `ForgetfulFunctor` forgets structure:

```python
from quivers.categorical.adjunctions import FreeMonoidAdjunction
from quivers.core.objects import FinSet, FreeMonoid

G = FinSet("generators", 5)

# Free monoid monad: T(X) = Free Monoid on X
# Forgetful: F.Alg -> Set forgets the monoid structure

adjunction = FreeMonoidAdjunction(G, max_length=3)

# The monad T = GF acts on sets:
# T(X) = Free(X, max_length=3)
T_X = adjunction.monad(X)
assert isinstance(T_X, FreeMonoid)
```

## Monoidal Structures

A monoidal category $(\mathcal{C}, \otimes, I)$ has:
- A bifunctor $\otimes: \mathcal{C} \times \mathcal{C} \to \mathcal{C}$ (the product)
- A unit object $I$
- Natural associativity, left identity, right identity (with coherence)

### Cartesian Monoidal

The standard monoidal structure on finite sets:

```python
from quivers.categorical.monoidal import CartesianMonoidal

monoidal = CartesianMonoidal()

X = FinSet("X", 3)
Y = FinSet("Y", 4)

# The tensor is Cartesian product
XY = monoidal.tensor(X, Y)
assert XY == X * Y
assert monoidal.unit == Unit

# Lifts to morphisms
f = morphism(X, Y)
g = morphism(X, Y)

fg = monoidal.tensor(f, g)
assert fg == f @ g
```

### Coproduct Monoidal

Coproduct as the monoidal operation:

```python
from quivers.categorical.monoidal import CoproductMonoidal

monoidal = CoproductMonoidal()

X = FinSet("X", 3)
Y = FinSet("Y", 4)

# The tensor is coproduct
XY = monoidal.tensor(X, Y)
assert XY == X + Y
```

## Base Change

Change the enriching quantale without changing morphisms themselves. `BaseChange` provides:

```python
from quivers.categorical.base_change import BaseChange
from quivers.core.quantales import BOOLEAN, PRODUCT_FUZZY

# Change from fuzzy to boolean
fuzzy_morph = morphism(X, Y, quantale=PRODUCT_FUZZY)
bool_morph = BaseChange.bool_from_fuzzy(fuzzy_morph)

assert fuzzy_morph.codomain == bool_morph.codomain
assert bool_morph.quantale == BOOLEAN

# Change from boolean to fuzzy (with scaling)
bool_to_fuzzy = BaseChange.fuzzy_from_bool(bool_morph, scale=0.5)
```

## Traced Monoidal Structure

A traced monoidal category adds a trace operator:

$$
\text{tr}^X_{Y,Z}: \mathcal{C}(Y \otimes X, Z \otimes X) \to \mathcal{C}(Y, Z)
$$

This is useful for feedback and recursive definitions:

```python
from quivers.categorical.traced import trace_morphism

X = FinSet("X", 2)
Y = FinSet("Y", 3)
Z = FinSet("Z", 4)

# f: (Y ⊗ X) → (Z ⊗ X)
f = morphism(Y * X, Z * X)

# Remove the feedback loop (trace over X)
traced_f = trace_morphism(f, X)
assert traced_f.domain == Y
assert traced_f.codomain == Z
```

The trace contracts the last components, leaving the first components as the domain and codomain.

## Summary

| Structure | Concept | Syntax |
|-----------|---------|--------|
| Functor | Object and morphism map | `F: C → D` |
| Natural Transform | Family of morphisms | `α: F ⇒ G` |
| Adjunction | Inverse functors up to natural isos | `F ⊣ G` |
| Monoidal | Product with coherence | `(C, ⊗, I)` |
| Base Change | Enrichment transformation | `BaseChange.foo_from_bar()` |
| Traced | Feedback/iteration | `trace(f, X)` |
