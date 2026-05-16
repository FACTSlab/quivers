# Types and Spaces

This page gives the denotation of the syntactic categories `TypeExpr` (finite-set types) and `SpaceExpr` (continuous spaces). Both are interpreted compositionally by structural recursion on the syntactic constructors of [`quivers.dsl.ast_nodes`](../api/dsl/ast_nodes.md), and both are implemented as the `forward` direction of the `dx.Lens` family in [`quivers.dsl.resolution`](../api/dsl/resolution.md).

## 1. Syntactic categories

We assume the following grammar fragment, whose constructors are realized one-to-one by the AST classes named in parentheses.

$$
\begin{array}{rcl}
\tau & ::= & n \;\big|\; X \;\big|\; \tau_1 \times \tau_2 \;\big|\; \tau_1 + \tau_2 \;\big|\; \tau_1 / \tau_2 \;\big|\; \tau_1 \backslash \tau_2 \;\big|\; T(\bar \tau)
\\[2pt]
& & \quad (\textsf{TypeName} \mid \textsf{TypeProduct} \mid \textsf{TypeCoproduct} \mid \textsf{TypeSlash} \mid \textsf{TypeEffectApply})
\\[6pt]
\sigma & ::= & C(\bar a; \bar k) \;\big|\; S \;\big|\; \sigma_1 \times \sigma_2
\\[2pt]
& & \quad (\textsf{SpaceConstructor} \mid \textsf{SpaceName} \mid \textsf{SpaceProduct})
\end{array}
$$

Here $n \in \mathbb{N}$ is an integer literal, $X$ ranges over object names, $S$ over space names, $T$ over declared effect identifiers, and $C \in \{\mathrm{Euclidean}, \mathrm{Simplex}, \mathrm{PositiveReals}, \mathrm{UnitInterval}\}$ over constructor names. The notation $C(\bar a; \bar k)$ stands for a constructor invocation with positional arguments $\bar a$ and keyword arguments $\bar k$. The two slash directions (`/` and `\`) and the effect-apply form are residuated / effect-extended formers whose denotations live in a residuated-monoidal universe ([Schemas §3](schemas.md#3-residuated-type-formers)); they have no denotation in the bare $\mathbf{FinSet}$ stratum, and well-typedness restricts them to expressions whose every named atom resolves in a residuated universe such as `FreeResiduated`.

## 2. Denotation of types

Type denotation is a function

$$
\llbracket \cdot \rrbracket_{\mathrm{Ty}} : \textsf{TypeExpr} \times \mathrm{Env}_{\mathrm{obj}} \to \mathrm{Ob}(\mathbf{FinSet}).
$$

We omit the environment argument when it is clear.

$$
\begin{array}{rcl}
\llbracket n \rrbracket
& = & \{0, 1, \dots, n - 1\}
\\[2pt]
\llbracket X \rrbracket_{\rho}
& = & \rho_{\mathrm{obj}}(X) \quad \text{if } X \in \mathrm{dom}(\rho_{\mathrm{obj}})
\\[2pt]
\llbracket \tau_1 \times \tau_2 \rrbracket_{\rho}
& = & \llbracket \tau_1 \rrbracket_{\rho} \times \llbracket \tau_2 \rrbracket_{\rho}
\\[2pt]
\llbracket \tau_1 + \tau_2 \rrbracket_{\rho}
& = & \llbracket \tau_1 \rrbracket_{\rho} \sqcup \llbracket \tau_2 \rrbracket_{\rho}
\\[2pt]
\llbracket \tau_1 / \tau_2 \rrbracket_{\rho}
& = & \llbracket \tau_1 \rrbracket_{\rho} \,/\, \llbracket \tau_2 \rrbracket_{\rho}
\quad \text{(right residual; see [Schemas §3.1](schemas.md#31-residuation-slashes))}
\\[2pt]
\llbracket \tau_1 \backslash \tau_2 \rrbracket_{\rho}
& = & \llbracket \tau_1 \rrbracket_{\rho} \,\backslash\, \llbracket \tau_2 \rrbracket_{\rho}
\quad \text{(left residual; see [Schemas §3.1](schemas.md#31-residuation-slashes))}
\\[2pt]
\llbracket T(\tau_1, \dots, \tau_k) \rrbracket_{\rho}
& = & T\bigl(\llbracket \tau_1 \rrbracket_{\rho}, \dots, \llbracket \tau_k \rrbracket_{\rho}\bigr)
\quad \text{(effect application; see [Schemas §3.2](schemas.md#32-effect-application))}
\end{array}
$$

The last three cases require the surrounding object universe to be residuated and/or effect-extended; otherwise the denotation is undefined and well-typedness rejects the expression.

If $X$ is not bound in $\rho_{\mathrm{obj}}$ but parses as a non-negative integer literal, the literal rule is used: this allows ad-hoc cardinalities such as `f : 3 -> 4` without prior `object` declarations.

The cartesian product $\times$ in $\mathbf{FinSet}$ is associative and commutative up to canonical isomorphism. The implementation chooses the right-associated, flattened representative; this is enforced by the `_flatten_products` converter on `ProductSet.components` and the analogous flattener on `CoproductSet.components`. The chosen representatives are *equal* (not merely isomorphic) under the `dx.Model` structural equality used throughout the codebase.

## 2a. Object initializers

The surface form `object X = …` admits three *initializer* shapes that bind $X$ to a concrete finite-set object built from explicit data rather than from a syntactic `TypeExpr`. Each is interpreted at the value layer and contributes its denotation directly to $\rho_{\mathrm{obj}}(X)$.

### 2a.1 Enum sets

```
object Atoms = {NP, S, VP}
```

denotes the finite set whose elements are exactly the named labels, ordered by declaration position:

$$
\llbracket \{e_1, \dots, e_n\} \rrbracket \;=\; \{e_1, \dots, e_n\},
\qquad |{\cdot}| = n.
$$

The label identity is part of the object: two enum sets with the same cardinality but different labels are *different* objects of $\mathbf{FinSet}$ under the `dx.Model` structural equality used throughout the codebase.

A `category C_1, …, C_n` declaration is the singleton-cardinality special case ([Schemas §1](schemas.md#1-category-atoms)).

### 2a.2 Free monoids

The `FreeMonoid(X, max_length = n)` initialiser binds $X$ to the bounded Kleene closure of the generator set. See §3 below for the denotation.

### 2a.3 Free residuated categories

The `FreeResiduated(G, depth = d, ops = O)` initialiser binds $X$ to a finite enumeration of category expressions over the generators, closed under the chosen residuation operations up to a depth bound. The denotation is given in [Schemas §4](schemas.md#4-the-free-residuated-universe).

## 3. Free monoids

The runtime value layer exposes a `FreeMonoid` object class, which arises in the deduction fragment as the carrier of strings over an alphabet (see [Weighted Deduction Fragment](grammar.md)). At the surface, `FreeMonoid(X, max_length = n)` appears only as an *object initializer* (§2a.2 above), not as a `TypeExpr` constructor: a free monoid must be bound to an `object` name via `object Words = FreeMonoid(...)` before it can be referenced as a type.

The denotation of `FreeMonoid(generators = X, max_length = n)` is the bounded Kleene star

$$
\llbracket \mathrm{FreeMonoid}(X, n) \rrbracket \;=\; \coprod_{k = 0}^{n} \rho_{\mathrm{obj}}(X)^{k},
$$

a finite set of size $\sum_{k=0}^{n} |\rho_{\mathrm{obj}}(X)|^{k}$. The bound $n$ is supplied by the grammar's depth parameter; the unbounded free monoid $\rho_{\mathrm{obj}}(X)^{*}$ is countably infinite and is not realized as a `SetObject`.

## 4. Denotation of spaces

Space denotation is a function

$$
\llbracket \cdot \rrbracket_{\mathrm{Sp}} : \textsf{SpaceExpr} \times \mathrm{Env}_{\mathrm{spc}} \times \mathrm{Env}_{\mathrm{obj}} \to \mathrm{Ob}(\mathbf{SBor}).
$$

For brevity write $\rho = (\rho_{\mathrm{spc}}, \rho_{\mathrm{obj}})$. The constructor cases are:

$$
\begin{array}{rcl}
\llbracket \mathrm{Euclidean}(d) \rrbracket
& = & \mathbb{R}^{d}
\\[2pt]
\llbracket \mathrm{Euclidean}(d; \mathrm{low} = \ell, \mathrm{high} = h) \rrbracket
& = & \prod_{i=1}^{d} [\ell, h]
\\[2pt]
\llbracket \mathrm{Simplex}(d) \rrbracket
& = & \Delta^{d-1} \;=\; \Bigl\{ x \in \mathbb{R}^{d}_{\ge 0} \,\Big|\, \textstyle\sum_i x_i = 1 \Bigr\}
\\[2pt]
\llbracket \mathrm{PositiveReals}(d) \rrbracket
& = & (0, +\infty)^{d}
\\[2pt]
\llbracket \mathrm{UnitInterval}(d) \rrbracket
& = & [0, 1]^{d}
\end{array}
$$

each carrying its standard Borel $\sigma$-algebra. The remaining cases are:

$$
\begin{array}{rcl}
\llbracket S \rrbracket_{\rho}
& = & \rho_{\mathrm{spc}}(S) \quad \text{if } S \in \mathrm{dom}(\rho_{\mathrm{spc}})
\\[2pt]
\llbracket S \rrbracket_{\rho}
& = & \iota\bigl(\rho_{\mathrm{obj}}(S)\bigr) \quad \text{otherwise, if } S \in \mathrm{dom}(\rho_{\mathrm{obj}})
\\[2pt]
\llbracket \sigma_1 \times \sigma_2 \rrbracket_{\rho}
& = & \llbracket \sigma_1 \rrbracket_{\rho} \times \llbracket \sigma_2 \rrbracket_{\rho}
\end{array}
$$

where $\iota : \mathbf{FinSet} \hookrightarrow \mathbf{SBor}$ is the canonical inclusion (every finite set as a discrete standard Borel space). The fallback rule for `SpaceName` allows mixed-domain `ProductSpace` instances such as `Euclidean(3) * Token` where `Token : 256` is a previously declared object. The denotation is the product in $\mathbf{SBor}$, which is well-defined precisely because $\iota$ is a faithful functor preserving finite products.

## 5. Coherence

The two type-formers $\times$ and $+$ on `TypeExpr` are interpreted by the cartesian product and disjoint union in $\mathbf{FinSet}$, both of which are associative, commutative, and unital up to canonical isomorphism (with units the singleton $\mathbf{1}$ and the empty set $\emptyset$ respectively). Concretely:

- The flattening converters on `ProductSet` and `CoproductSet` realize the *coherence isomorphisms* $((\tau_1 \times \tau_2) \times \tau_3) \cong (\tau_1 \times (\tau_2 \times \tau_3))$ as identities on the chosen representative.
- Empty products denote $\mathbf{1}$ (a singleton; `EmptySet` for coproducts denotes $\emptyset$).
- The `EmptySet` constructor in `categorical.monoidal` is the unit for $+$.

Mac Lane's coherence theorem guarantees that any two parses of the same `TypeExpr` denote *equal* finite sets in $\mathbf{FinSet}$, not merely isomorphic ones, when the implementation uses the canonical flattened normal form.

## 6. Resolution as a lens

The map $\llbracket \cdot \rrbracket_{\mathrm{Ty}}$ is realized in code as the `forward` component of `TypeExprToSetObject(env)`, an instance of `dx.Lens[TypeExpr, SetObject, TypeExpr]`. The complement is the original `TypeExpr` AST node, so the `backward` direction recovers the unresolved syntax verbatim. Both round-trip laws hold by construction:

$$
\mathrm{backward}(\mathrm{forward}(t)) = t,
\qquad
\mathrm{forward}(\mathrm{backward}(s, c)) = (s, c).
$$

The same structure holds for `SpaceExprToContinuousSpace`, with codomain $\mathbf{SBor}$ (which contains $\iota(\mathbf{FinSet})$ as the discrete-$\sigma$-algebra full subcategory used by the mixed-domain fallback).
