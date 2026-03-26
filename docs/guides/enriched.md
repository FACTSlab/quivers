# Enriched Category Theory

## Ends and Coends

An end $\int_X F(X, X)$ is a generalized intersection: the limit over the diagonal. Dually, a coend $\int^X F(X, Y)$ is a generalized union: the colimit over the diagonal.

Formally, an end is an object $E$ equipped with projections $\pi_X: E \to F(X, X)$ such that for all $f: X \to Y$, the diagram

$$
E \xrightarrow{\pi_X} F(X, X) \xrightarrow{F(f, \text{id})} F(Y, X) \\
E \xrightarrow{\pi_Y} F(Y, Y) \xrightarrow{F(\text{id}, f)} F(Y, X)
$$

commutes (naturality in the diagonals).

In quivers:

```python
from quivers.enriched.ends_coends import end, coend
from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism

# A diagram F: C^op × C → Set
# For finite sets, we compute the end as an intersection of fibers

X = FinSet("X", 3)
Y = FinSet("Y", 4)

# Functor F(A, B) that takes two objects and returns a morphism
def F(A, B):
    return morphism(A, B)

# End: ∫_Z F(Z, Z)
e = end({X, Y}, F)

# Coend: ∫^Z F(Z, Z) (geometric realization / coequalizer)
ce = coend({X, Y}, F)
```

### Yoneda Lemma (via Ends)

The Yoneda lemma states:

$$
[\mathcal{C}^\text{op}, \mathcal{V}] (y_A, F) \cong F(A)
$$

where $y_A$ is the representable presheaf and $F$ is any presheaf. This is an isomorphism of natural transformations and elements.

```python
from quivers.enriched.yoneda import yoneda_lemma

# Compute Yoneda isomorphism
A = FinSet("A", 3)
F = ...  # presheaf

# Naturals from representable to F correspond to elements of F(A)
iso = yoneda_lemma(A, F)
assert iso.shape == F(A).shape
```

## Kan Extensions

A Kan extension is a universal way to extend a functor $F: \mathcal{A} \to \mathcal{B}$ along an inclusion $i: \mathcal{A} \to \mathcal{C}$.

The left Kan extension $\text{Lan}_i F: \mathcal{C} \to \mathcal{B}$ is the initial such extension. The right Kan extension $\text{Ran}_i F$ is terminal.

Left Kan extension is computed via a colimit:

$$
\text{Lan}_i F(C) = \text{colim}_{(A, f: C \to i(A))} F(A)
$$

In quivers:

```python
from quivers.enriched.kan_extensions import left_kan, right_kan, Projection, Inclusion

# Define the inclusion i: A → C
A = [FinSet("A", 2), FinSet("B", 3)]
C = [A[0], A[1], FinSet("C", 4)]

inclusion = Inclusion(A, C)

# Define functor F: A → D
def F(obj):
    return morphism(obj, FinSet("target", 5))

# Left Kan extension: extends F along inclusion
left_ext = left_kan(F, inclusion)

# Right Kan extension
right_ext = right_kan(F, inclusion)

# Query the extension
ext_C = left_ext(C[2])  # Value at new object
```

## Weighted Limits and Colimits

A weighted limit is a limit parameterized by a weight object (a presheaf or profunctor). If $W: \mathcal{C}^\text{op} \to \mathcal{V}$ is a weight and $F: \mathcal{C} \to \mathcal{D}$ is a diagram, the weighted limit $\{W, F\}$ is defined by the universal property:

$$
[\mathcal{D}](X, \{W, F\}) \cong [\mathcal{C}^\text{op}, \mathcal{V}](W, [\mathcal{D}](X, F(-)))
$$

```python
from quivers.enriched.weighted_limits import (
    weighted_limit, Weight, Diagram, representable_weight
)

# Weight: W: C^op → V
W = representable_weight(C, X)  # representable by object X

# Diagram: F: C → D
F = Diagram({obj: morphism(obj, D) for obj in C})

# Weighted limit
limit = weighted_limit(W, F)

# Projections to diagram objects
for obj in C:
    proj = limit.projection(obj)
    assert proj.codomain == F[obj].codomain
```

Weighted colimits are the dual. Weighted limits generalize:
- **Ordinary limits** (weight = terminal presheaf)
- **Cotensor products** (weight = hom functor)
- **Powers and copowers** (weight = constant presheaf)

## Profunctors

A profunctor (or distributor) $P: \mathcal{A} \nrightarrow \mathcal{B}$ is a $\mathcal{V}$-valued bimodule, i.e., a functor $P: \mathcal{B}^\text{op} \times \mathcal{A} \to \mathcal{V}$.

```python
from quivers.enriched.profunctors import Profunctor

# Profunctor P: A ↛ B
# For objects a ∈ A, b ∈ B, provides a value P(b, a) ∈ V

class MyProfileunctor(Profunctor):
    def __call__(self, b, a):
        """P(b, a) ∈ V"""
        pass

# Composition of profunctors (Kleisli category of *-)
# (Q ∘ P)(c, a) = ∫^b P(b, a) ⊗ Q(c, b)
```

Morphisms in $\mathcal{C}$ embed as hom-profunctors: $\hom_\mathcal{C}(-,-): \mathcal{C}^\text{op} \times \mathcal{C} \to \mathcal{V}$.

## Yoneda Embedding and Representability

The Yoneda embedding $y: \mathcal{C} \to [\mathcal{C}^\text{op}, \mathcal{V}]$ sends an object $A$ to the representable presheaf $y_A = \hom(-,A)$.

A presheaf $F$ is representable if $F \cong y_A$ for some object $A$.

```python
from quivers.enriched.yoneda import (
    yoneda_embedding, representable_profunctor, verify_yoneda_fully_faithful
)

# Yoneda embedding of an object
A = FinSet("A", 3)
presheaf = yoneda_embedding(A)

# Check that yoneda_embedding is fully faithful
is_ff = verify_yoneda_fully_faithful(C)
assert is_ff
```

## Day Convolution

The Day convolution $F *_\otimes G$ on presheaves (with monoidal codomain) is defined by:

$$(F *_\otimes G)(X) = \int^{Y, Z} F(Y) \otimes G(Z) \otimes \hom(X, Y \otimes Z)$$

Day convolution lifts the monoidal structure from the base category to presheaves.

```python
from quivers.enriched.day_convolution import (
    day_convolution, day_unit, day_convolution_profunctors
)

# Presheaves F, G on a monoidal category
F = ...  # presheaf
G = ...  # presheaf

# Day convolution
FG = day_convolution(F, G)

# The unit object for Day convolution
unit = day_unit()
```

## Optics

An optic $p: (S, T) \to (A, B)$ is a pair of morphisms satisfying:

$$
p = (s_p, t_p): \exists C, . \; (S \to C \to A) \times (B \to C \to T)
$$

Optics compose, and they generalize lenses, prisms, adapters, and other bidirectional morphisms.

### Lens

A lens focuses on a component of a product:

```python
from quivers.enriched.optics import Lens

# Lens: (S, T) → (A, B)
# A lens picks out a part A of a whole S and shows how to update
# the whole T given a new value B.

S = FinSet("S", 3)
A = FinSet("A", 2)

# Lens: focuses on first component of a product
lens = Lens.get_set(
    get=morphism(S * A, A),  # project A from S
    set=morphism(S * A, S),  # update S with new A
)
```

### Prism

A prism focuses on a case of a coproduct:

```python
from quivers.enriched.optics import Prism

# Prism: (S, T) → (A, B)
# A prism tries to match a case and extract the value

S = FinSet("S", 5)  # sum type
A = FinSet("A", 2)  # matched case

prism = Prism.match_build(
    match=morphism(S, A),        # extract case
    build=morphism(A, S),        # rebuild
)
```

### Adapter

An adapter is a bijection between types:

```python
from quivers.enriched.optics import Adapter

# Adapter: isomorphism between S and A
fwd = morphism(S, A)
rev = morphism(A, S)

adapter = Adapter(fwd, rev)
```

### Grate

A grate distributes an update function:

```python
from quivers.enriched.optics import Grate

# Grate: entire structure is "writable"
grate = Grate(morphism(...), ...)
```

Optics compose with the `∘` operator:

```python
from quivers.enriched.optics import compose_optics

p1 = lens  # S ↔ A
p2 = prism  # A ↔ C

composed = compose_optics(p1, p2)  # S ↔ C
```
