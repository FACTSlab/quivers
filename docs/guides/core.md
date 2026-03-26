# Core Types & Quantales

## Objects: Finite Sets

In quivers, all morphisms relate finite sets. The type hierarchy is:

```
SetObject (abstract)
├── FinSet              — atomic named set with fixed cardinality
├── ProductSet          — Cartesian product A × B × C ...
├── CoproductSet        — tagged union (coproduct) A + B + C ...
└── FreeMonoid          — free monoid on generators, truncated by length
```

### FinSet

A `FinSet` is the most basic object:

```python
from quivers.core.objects import FinSet

X = FinSet("phoneme", 40)
print(X.size)   # 40
print(X.shape)  # (40,)
```

The terminal object is provided as a singleton:

```python
from quivers.core.objects import Unit
print(Unit.size)  # 1
```

### ProductSet and CoproductSet

Products and coproducts are formed with the `*` and `+` operators, which automatically flatten nesting:

```python
X = FinSet("X", 3)
Y = FinSet("Y", 4)

P = X * Y          # ProductSet(X, Y)
print(P.size)      # 12
print(P.shape)     # (3, 4)

C = X + Y          # CoproductSet(X, Y)
print(C.size)      # 7
print(C.shape)     # (7,) — flat representation
```

Products represent the Cartesian product: elements are pairs. Coproducts are tagged unions: an element belongs to exactly one component, and `CoproductSet` tracks offsets for indexing.

### FreeMonoid

A free monoid on a generator set $G$ (truncated by length) represents all strings of length 0 through $n$:

```python
from quivers.core.objects import FreeMonoid

G = FinSet("letter", 26)
FM = FreeMonoid(G, max_length=3)

# Components: Unit + G + G^2 + G^3
print(FM.size)  # 1 + 26 + 676 + 17576 = 18279

# Encode a word
idx = FM.encode((0, 1, 2))    # first three letters
word = FM.decode(idx)          # back to tuple
```

## Quantales: Enrichment Algebras

A quantale $(\mathcal{L}, \otimes, \bigvee, \bigwedge, \neg, I, \perp)$ is a complete lattice with a monoidal operation. It defines the algebra in which morphism values live and how morphisms compose.

The six primitive operations:

- $\otimes$ (tensor): monoidal product (associative, unital with $I$)
- $\bigvee$ (join): least upper bound; in composition, aggregates paths
- $\bigwedge$ (meet): greatest lower bound
- $\neg$ (negation): complement
- $I$ (unit): identity for $\otimes$
- $\perp$ (zero): identity for $\bigvee$ (bottom element)

Composition in a $\mathcal{V}$-enriched category uses the quantale's operations:

$$
(g \circ f)(a, c) = \bigvee_b f(a, b) \otimes g(b, c)
$$

### ProductFuzzy

The enrichment for fuzzy relations with product t-norm:

$$
\begin{align}
\mathcal{L} &= [0, 1] \\
a \otimes b &= a \cdot b \\
\bigvee_i x_i &= 1 - \prod_i (1 - x_i) \quad \text{(noisy-OR)} \\
\bigwedge_i x_i &= \prod_i x_i \\
\neg a &= 1 - a \\
I &= 1, \quad \perp = 0
\end{align}
$$

This is the default enrichment in quivers, suitable for probabilistic reasoning and soft constraints.

### BooleanQuantale

Crisp binary relations:

$$
\begin{align}
\mathcal{L} &= \{0, 1\} \\
a \otimes b &= a \land b \\
\bigvee_i x_i &= \max_i x_i \quad \text{(OR)} \\
\bigwedge_i x_i &= \min_i x_i \quad \text{(AND)} \\
\neg a &= 1 - a \\
I &= 1, \quad \perp = 0
\end{align}
$$

### LukasiewiczQuantale

Łukasiewicz t-norm (strongest continuous t-norm):

$$
\begin{align}
a \otimes b &= \max(a + b - 1, 0) \\
\bigvee_i x_i &= \min(1, \sum_i x_i) \quad \text{(bounded sum)} \\
\bigwedge_i x_i &= \min_i x_i \\
\neg a &= 1 - a
\end{align}
$$

Useful for resource-sensitive reasoning where evidence can "cancel out."

### GodelQuantale

Gödel (min) t-norm (weakest continuous t-norm):

$$
\begin{align}
a \otimes b &= \min(a, b) \\
\bigvee_i x_i &= \max_i x_i \\
\bigwedge_i x_i &= \min_i x_i \\
\neg a &= \begin{cases} 1 & \text{if } a = 0 \\ 0 & \text{otherwise} \end{cases}
\end{align}
$$

Composition computes minimax paths (the "best worst-case").

### TropicalQuantale

The tropical semiring enrichment for generalized metric spaces:

$$
\begin{align}
\mathcal{L} &= [0, \infty] \\
a \otimes b &= a + b \\
\bigvee_i x_i &= \inf_i x_i \quad \text{(shortest path)} \\
\bigwedge_i x_i &= \sup_i x_i \\
I &= 0, \quad \perp = \infty
\end{align}
$$

The identity tensor has $0$ on the diagonal and $\infty$ elsewhere. Composition computes shortest paths via $(g \circ f)(a, c) = \inf_b [f(a, b) + g(b, c)]$.

## Creating and Using Quantales

```python
from quivers.core.quantales import (
    PRODUCT_FUZZY, BOOLEAN,
    LukasiewiczQuantale, GodelQuantale, TropicalQuantale
)

# Use singletons
qnt = PRODUCT_FUZZY

# Or instantiate custom quantales
luka = LukasiewiczQuantale()
godel = GodelQuantale()
tropical = TropicalQuantale()
```

Pass a quantale to a morphism to set its enrichment:

```python
from quivers.core.morphisms import morphism

X = FinSet("X", 3)
Y = FinSet("Y", 4)

# default: PRODUCT_FUZZY
f_fuzzy = morphism(X, Y)

# explicit quantale
f_bool = morphism(X, Y, quantale=BOOLEAN)
f_godel = morphism(X, Y, quantale=GodelQuantale())
```

Once set, all operations on the morphism (composition, marginalization, etc.) use that quantale's operations.
