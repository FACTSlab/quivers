# Monads & Comonads

## Monads

A monad $T: \mathcal{C} \to \mathcal{C}$ is an endofunctor with two natural transformations:

- **Unit** $\eta: \text{id} \Rightarrow T$ (embedding into the monad)
- **Multiply** $\mu: T \circ T \Rightarrow T$ (flattening nested applications)

satisfying associativity and unitality (monad laws):

$$
\mu \circ \mu T = \mu \circ T \mu, \quad \mu \circ \eta T = \text{id}, \quad \mu \circ T \eta = \text{id}
$$

[`Monad`](../api/monadic/typeclasses.md) (in `quivers.monadic.typeclasses`) is an abstract base class. Concrete implementations must provide `endofunctor`, `fmap_obj`, `fmap`, `pure`, and `join`; the Eilenberg-Moore aliases `unit` / `multiply` are usually exposed too.

## Kleisli Category

Every monad $T$ induces a Kleisli category $\mathcal{C}_T$ where:

- Objects are the same as $\mathcal{C}$
- Morphisms $X \rightsquigarrow Y$ are morphisms $X \to T(Y)$ in $\mathcal{C}$
- Composition is Kleisli composition: $(g \circ_T f)(x) = \mu(T(g)(f(x)))$

In quivers, [`KleisliCategory`](../api/monadic/monads.md) wraps any `Monad` and exposes `identity` and `compose`:

```python
from quivers.monadic.monads import KleisliCategory, FuzzyPowersetMonad
from quivers.core.morphisms import morphism
from quivers.core.objects import FinSet

monad = FuzzyPowersetMonad()
kleisli = KleisliCategory(monad)

X = FinSet(name="X", cardinality=3)
Y = FinSet(name="Y", cardinality=4)
Z = FinSet(name="Z", cardinality=2)

f = morphism(X, Y)              # interpreted as Kleisli arrow X ⇝ Y
g = morphism(Y, Z)

h = kleisli.compose(f, g)       # X ⇝ Z via μ ∘ T(g) ∘ f
```

## Fuzzy Powerset Monad

The fuzzy powerset monad on the V-enriched category of finite sets:

```python
from quivers.monadic.monads import FuzzyPowersetMonad
from quivers.core.objects import FinSet
from quivers.core.algebras import PRODUCT_FUZZY

X = FinSet(name="X", cardinality=3)

monad = FuzzyPowersetMonad(algebra=PRODUCT_FUZZY)

# At the set level, T(X) = X
TX = monad.fmap_obj(X)
assert TX == X

# η_X : X → T(X) is the identity morphism with PRODUCT_FUZZY enrichment
eta_X = monad.unit(X)
```

The Kleisli category of the fuzzy powerset monad is the V-enriched relation category; composition uses the algebra's join (noisy-OR for `PRODUCT_FUZZY`).

## Free Monoid Monad

The free monoid monad on finite sets, truncated to a maximum word length:

```python
from quivers.monadic.monads import FreeMonoidMonad
from quivers.core.objects import FinSet, FreeMonoid

monad = FreeMonoidMonad(max_length=3)

X = FinSet(name="X", cardinality=5)

# T(X) = FreeMonoid(generators=X, max_length=3)
TX = monad.fmap_obj(X)
assert isinstance(TX, FreeMonoid)
assert TX.generators == X
assert TX.max_length == 3
```

## Comonads

A comonad $W: \mathcal{C} \to \mathcal{C}$ is a dual monad with:

- **Counit** $\varepsilon: W \Rightarrow \text{id}$ (projection from comonad)
- **Comultiply** $\delta: W \Rightarrow W \circ W$ (extend / duplicate)

[`Comonad`](../api/monadic/comonads.md) is an ABC; subclasses provide `endofunctor`, `counit(obj)`, and `comultiply(obj)`.

## CoKleisli Category

The coKleisli category $\mathcal{C}^W$ has:

- Objects as in $\mathcal{C}$
- Morphisms $X \rightleftharpoons Y$ are morphisms $W(X) \to Y$ in $\mathcal{C}$
- Composition $f \,=\!\!>\!\!>\, g = g \circ W(f) \circ \delta$

```python
from quivers.monadic.comonads import CoKleisliCategory, DiagonalComonad

comonad = DiagonalComonad()
cokleisli = CoKleisliCategory(comonad)
```

## Diagonal Comonad

The diagonal comonad $W(A) = A \times A$ has the first projection as counit and the diagonal-on-pairs as comultiplication:

```python
from quivers.monadic.comonads import DiagonalComonad
from quivers.core.objects import FinSet

X = FinSet(name="X", cardinality=3)
comonad = DiagonalComonad()

# Endofunctor action: W(X) = X × X
WX = comonad.endofunctor.map_object(X)
assert WX == X * X

# Counit ε_X : X × X → X (first projection)
eps = comonad.counit(X)

# Comultiplication δ_X : X × X → (X × X) × (X × X)
delta = comonad.comultiply(X)
```

## Cofree Comonad

The cofree comonad on a store object:

```python
from quivers.monadic.comonads import CofreeComonad
from quivers.core.objects import FinSet

store = FinSet(name="S", cardinality=4)
comonad = CofreeComonad(store=store)
```

## Algebras and Coalgebras

A $T$-algebra is a pair $(A, \alpha: T(A) \to A)$ satisfying:

$$
\alpha \circ \eta_A = \text{id}_A, \quad \alpha \circ \mu_A = \alpha \circ T(\alpha)
$$

```python
from quivers.monadic.algebras import FreeAlgebra, ObservedAlgebra
import torch

# Free algebra: carrier is T(A), structure map is μ_A
free_alg = FreeAlgebra(monad, X)
assert free_alg.carrier == monad.fmap_obj(X)

# Verify the algebra laws
assert free_alg.verify_unit_law()
assert free_alg.verify_associativity()

# Algebra from an explicit structure tensor α : T(A) → A
TA = monad.fmap_obj(X)
alpha_tensor = torch.eye(TA.size, X.size)[: TA.size, : X.size]  # any tensor of shape (TA.size, X.size)
explicit_alg = ObservedAlgebra(monad, X, alpha_tensor)
```

A $W$-coalgebra is the dual notion $(A, \gamma: A \to W(A))$.

## Eilenberg-Moore Category

The Eilenberg-Moore category of $T$ has $T$-algebras as objects and algebra homomorphisms as morphisms:

```python
from quivers.monadic.algebras import EilenbergMooreCategory

em = EilenbergMooreCategory(monad)

# Free algebra on X
free_alg = em.free_algebra(X)

# Check whether a base morphism f : A → B is a homomorphism between two algebras
f = morphism(X, X)
is_hom = em.is_homomorphism(f, free_alg, free_alg)
```

## Distributive Laws

A distributive law $\lambda: T \circ S \Rightarrow S \circ T$ between monads allows lifting one through the other:

```python
from quivers.monadic.distributive_laws import FreeMonoidPowersetLaw

law = FreeMonoidPowersetLaw(max_length=3)
```

The lifted monad combines word-formation with fuzzy aggregation; see [`distributive_laws`](../api/monadic/distributive_laws.md) for the precise construction.
