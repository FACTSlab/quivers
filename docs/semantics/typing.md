# A Type Theory for QVR

This chapter develops the **type theory** of the QVR DSL as a formal proof system, paired with the denotational interpretation already given in the surrounding chapters. The denotational semantics (chapters [Setting](setting.md) through [Adequacy](adequacy.md)) tells us *what* a well-typed QVR phrase means: an object of $\mathbf{FinSet}$ or $\mathbf{SBor}$, a morphism in $\mathbf{Stoch}$, a Markov kernel in $\mathbf{Kern}$, a Π-indexed family of Kleisli arrows, and so on. The present chapter gives the **type system** that picks out *which* phrases are well-typed in the first place: judgments, contexts, inference rules, and the soundness theorem that ties the two layers together.

The development is organized so that:

* every judgment form is given with its denotational interpretation, so the proof theory and the model theory stay synchronised;
* every inference rule is justified by an appeal to the categorical structure already exhibited in the denotational chapters;
* the soundness theorem (Theorem [§9.1](#91-soundness)) is the precise statement under which the REPL's `:type` and `:kind` commands are guaranteed to report mathematically meaningful answers.

## 0. Notation

We collect the notation used throughout, with cross-references to the other chapters that introduce each symbol. Every chapter of the semantics development uses the same conventions; deviations are flagged in the chapter where they occur.

| Symbol | Meaning | Defined in |
|---|---|---|
| $\mathcal{V}$ | A QVR algebra (one of eleven, plus de Morgan duals) | [Algebras §1](algebras.md#1-the-eleven-algebras) |
| $\otimes, \mathbf{1}$ | Monoidal product and unit of $\mathcal{V}$ | [Setting §1](setting.md#1-algebras-as-enrichment-bases) |
| $\bigoplus$ | Join of $\mathcal{V}$ | [Setting §1](setting.md#1-algebras-as-enrichment-bases) |
| $\mathbf{FinSet}$, $\mathbf{SBor}$ | Finite sets / standard Borel spaces | [Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels) |
| $\mathcal{V}\text{-}\mathbf{Rel}$ | $\mathcal{V}$-enriched relations on $\mathbf{FinSet}$ | [Setting §2](setting.md#2-mathcalv-enriched-relations) |
| $\mathbf{Stoch}, \mathbf{Kern}$ | Kleisli categories of the finitary / measure-theoretic Giry monad | [Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels) |
| $\mathcal{G}$ | Giry monad (finitary or measure-theoretic, disambiguated by domain) | [Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels) |
| $\diamond$ | Kleisli composition in $\mathbf{Kern}$ (or $\mathbf{Stoch}$) | [Programs §1](programs.md#1-the-giry-monad-as-semantic-substrate) |
| $\diamond_\alpha$ | Kleisli composition keyed by the algebra $\alpha$ | [Composition rules](composition-rules.md) |
| $\eta, \mu$ | Unit and multiplication of $\mathcal{G}$ | [Programs §1](programs.md#1-the-giry-monad-as-semantic-substrate) |
| $\llbracket \cdot \rrbracket$ | Denotation function | this chapter §[8](#8-denotational-interpretation) |
| $\Gamma$ | Module-level value context (typing) | §[2.1](#21-the-value-context-gamma) |
| $\Delta$ | Π-bound parameter context (parametric programs) | §[2.2](#22-the-parameter-context-delta) |
| $\Phi$ | Trace context (random variables inside a program body) | §[2.3](#23-the-trace-context-phi) |
| $\rho$ | Semantic environment (assignment of denotations to free names) | [Setting §5](setting.md#5-environments) |
| $A \rightsquigarrow B$ | Kleisli signature (Markov kernel $A \to \mathcal{G}(B)$) | this chapter §[5](#5-inference-rules-for-morphism-expressions) |
| $(\Delta) \Rightarrow A \rightsquigarrow B$ | Π-quantified Kleisli signature (parametric program) | this chapter §[7](#7-inference-rules-for-programs) |
| $\dashv$ | Trace-context updater (statement typing) | this chapter §[6](#6-inference-rules-for-statements) |
| $\Gamma \vdash \mathcal{J}$ | Generic typing judgment | this chapter §[3](#3-inference-rules-for-types-and-kinds) |
| $\phi \equiv \psi$ | Denotational equivalence | §[9.5](#95-equivalence-and-conservativity) |

Composition of morphisms uses $\diamond$ throughout (with the algebra subscript $\alpha$ omitted when it is the module's declared default); the surface operator family (`>>`, `<<`, `>=>`, `*>`, `~>`, `||>`, `?>`, `&&>`, `+>`, `$>`, `%>`) all denote $\diamond_\alpha$ for the appropriate $\alpha$ ([Composition rules](composition-rules.md), [Expressions §2.8](expressions.md#28-the-eleven-composition-operators)). Tensor product uses $\otimes$ throughout (surface operator `@`).

## 1. Syntactic categories

We work over four disjoint syntactic universes. Each is given by a context-free grammar over a fixed alphabet of names (identifiers).

### 1.1 Kinds

$$
\kappa \;::=\;
   \ast_{\mathrm{FinSet}}
\;\mid\; \ast_{\mathrm{Space}}
\;\mid\; \ast_{\mathrm{Sort}}
\;\mid\; \ast_{\mathrm{Atom}}
\;\mid\; \mathsf{Family}[\Theta,\, B]
\;\mid\; \mathsf{Mor}[A,\, B]
\;\mid\; \mathsf{Scalar}_{R}
$$

Kinds classify type-level constructions. The seven primitive kinds are:

| Kind | Inhabitants | Denotational target |
|---|---|---|
| $\ast_{\mathrm{FinSet}}$ | finite-set objects | objects of $\mathbf{FinSet}$ |
| $\ast_{\mathrm{Space}}$ | standard-Borel spaces | objects of $\mathbf{SBor}$ |
| $\ast_{\mathrm{Sort}}$ | item sorts of a deduction signature | small fibre of a generalized algebraic theory |
| $\ast_{\mathrm{Atom}}$ | abstract category atoms | hom-objects of a residuated category universe |
| $\mathsf{Family}[\Theta, B]$ | stochastic families | $\mathrm{Hom}_{\mathbf{Kern}}(\Theta, B)$ |
| $\mathsf{Mor}[A, B]$ | morphism-valued names | $\mathrm{Hom}_{\mathbf{Kern}}(A, B)$ |
| $\mathsf{Scalar}_R$ | scalar hyperparameters | elements of the rig $R$ (typically $\mathbb{R}$ or $\mathbb{N}$) |

We write $\ast$ for the disjoint union $\ast_{\mathrm{FinSet}} + \ast_{\mathrm{Space}}$ when the discrete/continuous distinction is immaterial, and call inhabitants of $\ast$ *types* (in the QVR sense, following [Types and spaces §1](types-and-spaces.md)).

### 1.2 Type expressions

$$
\begin{aligned}
\tau \;::=\;& X
\;\mid\; \mathsf{FinSet}\,n
\;\mid\; \mathsf{Real}\,n
\;\mid\; \mathsf{Simplex}\,n
\;\mid\; \mathsf{Sphere}\,n
\;\mid\; \mathsf{Ball}\,n
\;\mid\; \cdots \\
&\;\mid\; \tau_1 \times \tau_2
\;\mid\; \tau_1 + \tau_2
\;\mid\; \tau_1 / \tau_2
\;\mid\; \tau_1 \backslash \tau_2
\;\mid\; T(\tau)
\end{aligned}
$$

Type variables $X$ range over names introduced by `object`, `space`, `atom`, `sort`, `vertex_kind`, `edge_kind`, and `binder` declarations. The constructors $\mathsf{FinSet}, \mathsf{Real}, \ldots$ are the discrete and continuous primitive constructors of [Types and spaces §2–§3](types-and-spaces.md). Products $\times$, coproducts $+$, residuated formers $/, \backslash, T(\cdot)$ obey their usual category-theoretic universal properties; the residuated forms are well-typed only in atoms-kind context.

### 1.3 Family expressions

Distribution families occupy a separate syntactic class because their typing is parametric over an explicit parameter object:

$$
F \;::=\;
   \mathsf{Normal}
\;\mid\; \mathsf{Bernoulli}
\;\mid\; \mathsf{Categorical}
\;\mid\; \mathsf{Dirichlet}
\;\mid\; \mathsf{Beta}
\;\mid\; \cdots
\;\mid\; F_{\text{user}}
$$

The family registry assigns each $F$ a parameter object $\Theta_F$ and a value space $B_F$ (see [Programs §1](programs.md)), so that $F$ denotes a kernel $\Theta_F \to \mathcal{G}(B_F)$.

### 1.4 Term expressions

Morphism-level expressions appear in `morphism`, `let`, statement initialisers, and inline composition. Their grammar is:

$$
\begin{aligned}
e \;::=\;& x
\;\mid\; f
\;\mid\; e_1 \mathbin{\diamond_\alpha} e_2
\;\mid\; e_1 \mathbin{@} e_2
\;\mid\; c(\bar y) \\
&\;\mid\; \mathsf{id}_\tau
\;\mid\; \mathsf{id}
\;\mid\; e^\dagger
\;\mid\; \mathsf{trace}(e)
\;\mid\; e\,.\mathsf{change\_base}(\varphi)
\;\mid\; e_1 \mathrel{*} e_2 \\
&\;\mid\; \mathsf{cup}(\tau)
\;\mid\; \mathsf{cap}(\tau)
\;\mid\; e\,.\mathsf{marginalize}(v_1, \ldots, v_m)
\;\mid\; \mathsf{fan}(e_1, \ldots, e_n) \\
&\;\mid\; \mathsf{repeat}(e, n)
\;\mid\; \mathsf{stack}(e, n)
\;\mid\; \mathsf{scan}(e)
\;\mid\; \mathsf{freeze}(e)
\;\mid\; \mathsf{from\_data}(\bar a) \\
&\;\mid\; \mathsf{parser}(\bar r;\; \bar c)
\;\mid\; \mathsf{chart\_fold}(\ldots)
\;\mid\; \mathsf{curry}(e, k)
\end{aligned}
$$

The variable cases distinguish a bound name $x$ (introduced inside a program body by a bind / let statement) from a free morphism name $f$ (introduced at module scope by a `morphism`, `program`, `let`, or `export` declaration). The contraction-call form $c(\bar y)$ (`ExprMorphismCall`) applies a registered $n$-ary contraction $c$ to morphism-scope names $\bar y$ inside a `let`-binding initializer (see [Composition rules §4](composition-rules.md)).

**Program instantiation is not an expression form.** Surface program calls $P(\bar a)$ live in the program-body sub-language as the family slot of a `DrawStep`: a statement $v \leftarrow P(\bar a)$ with $P$ a program template is interpreted by inlining the template's body, as described in [Programs §3a](programs.md#3a-parametric-programs). Consequently the typing rule for parametric instantiation is presented at the statement level (§[6.8](#68-template-inlining)), not as a morphism-expression rule.

Sequential composition $\mathbin{\diamond_\alpha}$ is parameterized by a choice of enrichment algebra $\alpha$: the QVR surface syntax exposes one operator per algebra. The eleven-operator family is `>>` (`ProductFuzzy` noisy-or, the default), `<<` (reversed `ProductFuzzy`), `>=>` (Kleisli composition for the operands' shared algebra), `*>` (Markov sum-product), `~>` (`LogProb`), `||>` (Gödel), `?>` (Viterbi / Max-plus), `&&>` (Boolean), `+>` (Łukasiewicz), `$>` (`Real` sum-product), and `%>` (`Probability` saturated sum); see [Composition rules](composition-rules.md). The tensor `@` denotes the symmetric monoidal product $\otimes$ ([Morphisms §2](morphisms.md)). The dagger $e^\dagger$ is the compact-closed dual, $\mathsf{trace}$ is the categorical trace, $\mathsf{cup}/\mathsf{cap}$ are the unit / counit of the compact-closed structure, and $.\mathsf{change\_base}(\varphi)$ is the base-change functor between algebras. The remaining combinators $\mathsf{fan}, \mathsf{repeat}, \mathsf{stack}, \mathsf{scan}, \mathsf{freeze}, \mathsf{from\_data}, \mathsf{parser}, \mathsf{chart\_fold}, \mathsf{curry}$ are explained in [Expressions](expressions.md); we present typing for the core fragment ($\diamond$, $@$, $\mathsf{id}$, $\mathsf{fan}$, $\mathsf{repeat}$, and program instantiation) in §[5](#5-inference-rules-for-morphism-expressions) below and refer to [Expressions](expressions.md) for the derived combinators.

Notably absent from the expression sub-language are first-class projections $\pi_i$ and injections $\mathsf{inl}, \mathsf{inr}$: products are introduced and eliminated implicitly through tuple-pattern bindings in the program-body sub-language (§[1.5](#15-statements-program-body-sub-language)), and the discrete coproduct is reached through declared `morphism` arrows initialized from data rather than through dedicated combinators. Projections appear in the *meta-language* of the denotation (e.g. $\pi_i : \llbracket A_1 \times \cdots \times A_k \rrbracket \to \llbracket A_i \rrbracket$) but not in the surface syntax.

### 1.5 Statements (program-body sub-language)

Program bodies are sequences of statements; statements have side-effect-like typing because each one extends the *trace context* $\Phi$ with newly-bound random variables. The grammar is:

$$
\begin{aligned}
s \;::=\;& v \leftarrow F(\bar a)
\;\mid\; (v_1, \ldots, v_m) \leftarrow F(\bar a)\\
&\;\mid\; \mathsf{observe}\ v \leftarrow F(\bar a)\\
&\;\mid\; \mathsf{marginalize}\ v \leftarrow F(\bar a)\,\{\, s_1; \ldots; s_n\,\}\\
&\;\mid\; \mathsf{let}\ v = e\\
&\;\mid\; \mathsf{score}\ e
\end{aligned}
$$

The trailing `return e` clause is not a statement; it is the program's *exit* and is typed at the program level (§[7](#7-inference-rules-for-programs)).

### 1.6 Programs and declarations

A program declaration has the shape

$$
\mathsf{program}\ P\ (\Delta)\ :\ A\ \to\ B\ \{\ s_1;\ \ldots;\ s_n;\ \mathsf{return}\ e\ \}
$$

where $\Delta$ is a parameter context (§[2](#2-contexts)), $A, B$ are types, and the body is a sequence of statements terminated by a `return`. Module-level declarations (`object`, `space`, `morphism`, `program`, `let`, `export`, `signature`, `encoder`, `decoder`, `loss`, `bundle`, `composition`, `deduction`, `category`, `schema`, `rule`) extend the value context $\Gamma$ (§[2](#2-contexts)).

## 2. Contexts

The type theory uses four kinds of context, each tracking a different layer of binding. We write $\varepsilon$ for the empty context, $\Gamma, x : \tau$ for context extension, and $\Gamma_1, \Gamma_2$ for concatenation when the two are compatible (disjoint domains).

### 2.1 The value context $\Gamma$

$\Gamma$ tracks every name in scope at module level:

$$
\Gamma \;::=\; \varepsilon
\;\mid\; \Gamma,\ X : \kappa
\;\mid\; \Gamma,\ f : A \rightsquigarrow B
\;\mid\; \Gamma,\ F : \mathsf{Family}[\Theta, B]
\;\mid\; \Gamma,\ P : (\Delta) \Rightarrow A \rightsquigarrow B
$$

Module-level type names $X$ carry a kind (typically $\ast_{\mathrm{FinSet}}$ or $\ast_{\mathrm{Space}}$); morphism names $f$ carry a Kleisli signature $A \rightsquigarrow B$; family names $F$ carry a $\mathsf{Family}$ kind; program names $P$ carry both a parameter context $\Delta$ and a Kleisli signature.

### 2.2 The parameter context $\Delta$

The parameter context tracks the dependent Π-binders introduced by a parametric program declaration:

$$
\Delta \;::=\; \varepsilon \;\mid\; \Delta,\ p : P
$$

where $p$ is a parameter name and $P$ ranges over the parameter universes listed in [Programs §3a](programs.md#3a-parametric-programs). The implementation in [`ProgramParam`](../api/dsl/ast_nodes.md#quivers.dsl.ast_nodes.programparam) records three concrete variants:

| Variant | Surface | Universe |
|---|---|---|
| [`ObjectParam`](../api/dsl/ast_nodes.md#quivers.dsl.ast_nodes.objectparam) | `G : FinSet` / `G : Space` / `G : Object` | an object of $\mathbf{FinSet}$ / $\mathbf{SBor}$ / either |
| [`ScalarParam`](../api/dsl/ast_nodes.md#quivers.dsl.ast_nodes.scalarparam) | `α : Real` / `α : Nat` | the underlying set of the rig $\mathbb{R}$ or $\mathbb{N}$ |
| [`MorphismParam`](../api/dsl/ast_nodes.md#quivers.dsl.ast_nodes.morphismparam) | `f : Mor[A, B]` | a kernel $A \to \mathcal{G}(B)$ in $\mathrm{Hom}_{\mathbf{Kern}}(A, B)$ |

A bare-identifier parameter list $(q_1, \ldots, q_k)$ is the special case of a $\Delta$ all of whose entries are *projection binders*: they do not contribute a Π-quantifier (their denotation is identity), they only name the components of the program's domain.

### 2.3 The trace context $\Phi$

Inside a program body, the trace context records the random variables bound by previous statements:

$$
\Phi \;::=\; \varepsilon \;\mid\; \Phi,\ v : \tau
$$

We treat $\Phi$ as a list rather than a multiset because the body's denotation depends on the *order* of binding (sample sites composed left-to-right by Kleisli composition $\diamond$). The denotation of a $\Phi$ is the product object

$$
\llbracket \Phi \rrbracket \;=\; \llbracket \tau_1 \rrbracket \times \cdots \times \llbracket \tau_n \rrbracket
\quad\text{when}\quad \Phi = v_1 : \tau_1, \ldots, v_n : \tau_n
$$

### 2.4 Well-formed contexts

The judgments $\vdash \Gamma\ \mathsf{ok}$, $\Gamma \vdash \Delta\ \mathsf{ok}$, and $\Gamma; \Delta \vdash \Phi\ \mathsf{ok}$ are defined by mutual induction over the rules in §[3](#3-inference-rules-for-types-and-kinds) and §[6](#6-inference-rules-for-statements). They are bookkeeping and admit standard structural rules (weakening, contraction, exchange) modulo the order-sensitivity of $\Phi$ noted above.

## 3. Inference rules for types and kinds

The kinding judgment is $\Gamma \vdash \tau : \kappa$. We give the rules grouped by syntactic form.

### 3.1 Type variables

$$
\frac{X : \kappa \in \Gamma}{\Gamma \vdash X : \kappa}\ \textsc{TyVar}
$$

### 3.2 Discrete primitives

$$
\frac{n \in \mathbb{N}}{\Gamma \vdash \mathsf{FinSet}\,n : \ast_{\mathrm{FinSet}}}\ \textsc{FinSet}
\qquad
\frac{\Gamma \vdash X : \ast_{\mathrm{FinSet}}}{\Gamma \vdash \mathsf{FinSet}\,X : \ast_{\mathrm{FinSet}}}\ \textsc{FinSetVar}
$$

The second rule covers the `FinSet X` shape, where $X$ is a previously-declared finite-set object: its denotation is the cardinality of $X$.

### 3.3 Continuous primitives

$$
\frac{n \in \mathbb{N}}{\Gamma \vdash \mathsf{Real}\,n : \ast_{\mathrm{Space}}}\ \textsc{Real}
\qquad
\frac{n \in \mathbb{N}}{\Gamma \vdash \mathsf{Simplex}\,n : \ast_{\mathrm{Space}}}\ \textsc{Simplex}
$$

with analogous rules for $\mathsf{Sphere}, \mathsf{Ball}, \mathsf{CholeskyFactor}, \mathsf{Covariance}, \mathsf{Correlation}, \mathsf{Orthogonal}, \mathsf{Stiefel}, \mathsf{LowerTriangular}, \mathsf{Diagonal}$, each documented in [Types and spaces §3](types-and-spaces.md).

### 3.4 Products and coproducts

$$
\frac{\Gamma \vdash \tau_1 : \kappa_1 \quad \Gamma \vdash \tau_2 : \kappa_2 \quad \kappa_1 \sqcup \kappa_2\ \text{defined}}{\Gamma \vdash \tau_1 \times \tau_2 : \kappa_1 \sqcup \kappa_2}\ \textsc{TyProd}
\qquad
\frac{\Gamma \vdash \tau_1 : \ast_{\mathrm{FinSet}} \quad \Gamma \vdash \tau_2 : \ast_{\mathrm{FinSet}}}{\Gamma \vdash \tau_1 + \tau_2 : \ast_{\mathrm{FinSet}}}\ \textsc{TySum}
$$

The kind-join $\sqcup$ is the partial function

$$
\sqcup\ :\ \{\ast_{\mathrm{FinSet}}, \ast_{\mathrm{Space}}\}^2 \rightharpoonup \{\ast_{\mathrm{FinSet}}, \ast_{\mathrm{Space}}\}
$$

defined exactly by the four cases

$$
\ast_{\mathrm{FinSet}} \sqcup \ast_{\mathrm{FinSet}} = \ast_{\mathrm{FinSet}},
\qquad
\ast_{\mathrm{FinSet}} \sqcup \ast_{\mathrm{Space}}
= \ast_{\mathrm{Space}} \sqcup \ast_{\mathrm{FinSet}}
= \ast_{\mathrm{Space}},
\qquad
\ast_{\mathrm{Space}} \sqcup \ast_{\mathrm{Space}} = \ast_{\mathrm{Space}}.
$$

$\sqcup$ is undefined on $\ast_{\mathrm{Sort}}$, $\ast_{\mathrm{Atom}}$, and the parametric kinds $\mathsf{Family}[\Theta, B]$, $\mathsf{Mor}[A, B]$, $\mathsf{Scalar}_R$: products of items at those kinds are not legal QVR objects. $\textsc{TyProd}$'s third premise "$\kappa_1 \sqcup \kappa_2$ defined" is the well-typedness restriction that rules out $\mathsf{FinSet}\,3 \times \mathsf{Sort}_T$ etc. The defined cases implement the discrete-component absorption discussed in [Types and spaces §6](types-and-spaces.md#6-resolution-in-code).

Coproducts are restricted to the discrete sub-language because $\mathbf{SBor}$ has no general finite coproducts that play well with the Giry monad ([Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels)).

### 3.5 Residuated formers

$$
\frac{\Gamma \vdash \tau_1 : \ast_{\mathrm{Atom}} \quad \Gamma \vdash \tau_2 : \ast_{\mathrm{Atom}}}{\Gamma \vdash \tau_1 / \tau_2 : \ast_{\mathrm{Atom}}}\ \textsc{TySlashR}
\qquad
\frac{\Gamma \vdash \tau_1 : \ast_{\mathrm{Atom}} \quad \Gamma \vdash \tau_2 : \ast_{\mathrm{Atom}}}{\Gamma \vdash \tau_1 \backslash \tau_2 : \ast_{\mathrm{Atom}}}\ \textsc{TySlashL}
$$

$$
\frac{\Gamma \vdash \tau : \ast_{\mathrm{Atom}}}{\Gamma \vdash T(\tau) : \ast_{\mathrm{Atom}}}\ \textsc{TyEff}
$$

The residuated formers and the effect type-constructor $T(\cdot)$ are typed only over $\ast_{\mathrm{Atom}}$; see [Schemas §3](schemas.md) for the categorical setting.

### 3.6 Kind subsumption

There is no general kind subsumption rule. The only implicit coercion is the absorption built into $\sqcup$: in a mixed product $\sigma_{\mathrm{FinSet}} \times \tau_{\mathrm{Space}}$, the discrete factor is implicitly indicator-embedded into the ambient continuous space ($[n] \hookrightarrow \mathbb{R}^n$), so that the product lands in $\mathbf{SBor}$. The compiler realizes this through the resolution dispatch in [`_resolve_any_space`](../api/dsl/resolution.md): a `ProductSet` lifts to a `ProductSpace` whenever any component is a `ContinuousSpace`. No standalone embedding combinator exists in the surface syntax.

## 4. Inference rules for families

The family-typing judgment is $\Gamma \vdash F : \mathsf{Family}[\Theta, B]$. Every family in the registry comes with a fixed pair $(\Theta_F, B_F)$, so the rule is uniform:

$$
\frac{F \in \mathsf{FamReg}\ \text{with parameter}\ \Theta_F\ \text{and value space}\ B_F}{\Gamma \vdash F : \mathsf{Family}[\Theta_F, B_F]}\ \textsc{FamReg}
$$

A *family application* $F(a_1, \ldots, a_k)$ is typed by checking that the actual argument tuple $\bar a$ matches the parameter object $\Theta_F$ that the registry assigns to $F$. The parameter object is not necessarily a simple product of scalar slots: a `Normal(mu, sigma)` family has $\Theta_{\mathrm{Normal}} = \mathbb{R} \times \mathbb{R}_{> 0}$ (mean and a positive scale), whereas `MultivariateNormal(loc, scale_tril)` has $\Theta = \mathbb{R}^d \times \mathrm{CholeskyFactor}(d)$ (a vector and a Cholesky factor at dimension $d$). The general typing rule abstracts over this with an arity-$k$ argument map $\mathrm{params}_F : \Theta_F \to A_1 \times \cdots \times A_k$ that specifies, per family, what each surface argument slot inhabits:

$$
\frac{\Gamma \vdash F : \mathsf{Family}[\Theta_F,\, B]
       \qquad \mathrm{params}_F = (A_1, \ldots, A_k)
       \qquad \Gamma; \Phi \vdash a_i : A_i\ \text{(each}\ a_i\ \text{a morphism}\ \Phi \rightsquigarrow A_i)\ \text{for}\ i = 1, \ldots, k}
      {\Gamma; \Phi \vdash F(\bar a) : \mathsf{Kernel}[\Phi, B]}\ \textsc{FamApp}
$$

Here $\mathsf{Kernel}[\Phi, B]$ is the kind of Kleisli arrows $\Phi \to \mathcal{G}(B)$. Each argument $a_i$ is itself a morphism from the trace $\Phi$ into its declared slot $A_i$: this is what allows a family parameter to *depend on* previously-bound random variables (e.g. `Categorical(theta)` where `theta` is a Dirichlet-distributed simplex value bound earlier in the body). The composite $F \circ \mathrm{params}_F^{-1} \circ \langle a_1, \ldots, a_k \rangle$ is the resulting kernel on $\Phi$. The implementation realizes $\mathrm{params}_F$ as the `FamilySpec.param_specs` list in `src/quivers/continuous/families.py`.

## 5. Inference rules for morphism expressions

The morphism-typing judgment is $\Gamma; \Phi \vdash e : A \rightsquigarrow B$, read "in value context $\Gamma$ and trace context $\Phi$, the expression $e$ is a Kleisli arrow from $A$ to $B$". When the trace context is empty (module-level expression, not inside a program body) we elide it: $\Gamma \vdash e : A \rightsquigarrow B$.

### 5.1 Variables

$$
\frac{x : \tau \in \Phi}{\Gamma; \Phi \vdash x : \Phi \rightsquigarrow \tau}\ \textsc{TraceVar}
\qquad
\frac{f : A \rightsquigarrow B \in \Gamma}{\Gamma; \Phi \vdash f : A \rightsquigarrow B}\ \textsc{ModuleVar}
$$

The two rules separate trace-bound names (which project from the current $\Phi$) from module-bound names (which carry their declared Kleisli signature).

### 5.2 Composition

Sequential composition splits into two rules by the algebra-dispatch convention of [Expressions §2.8](expressions.md#28-the-eleven-composition-operators).

For the *algebra-polymorphic* operators (`>>`, `<<`, `>=>`), the composition fires when both operands inhabit the *same* algebra; the module's declared algebra $\alpha_{\text{mod}}$ supplies $\otimes$ and $\bigoplus$:

$$
\frac{\Gamma; \Phi \vdash e_1 : A \rightsquigarrow B \qquad \Gamma; \Phi \vdash e_2 : B \rightsquigarrow C \qquad \mathrm{alg}(e_1) = \mathrm{alg}(e_2) = \alpha}
     {\Gamma; \Phi \vdash e_1 \mathbin{\diamond_\alpha^{\mathrm{poly}}} e_2 : A \rightsquigarrow C}\ \textsc{ComposePoly}_\alpha
$$

For each *algebra-tagged* operator $\circ_\beta$ in the set $\{\mathbin{*>} : \beta = \mathcal{V}_{\mathrm{M}},\ \mathbin{\sim>} : \beta = \mathcal{V}_{\mathrm{LP}},\ \mathbin{||>} : \beta = \mathcal{V}_{\mathrm{G}},\ \mathbin{?>} : \beta = \mathcal{V}_{\mathrm{MP}},\ \mathbin{\&\&>} : \beta = \mathcal{V}_{\mathbb{B}},\ \mathbin{+>} : \beta = \mathcal{V}_{\mathrm{L}},\ \mathbin{\$>} : \beta = \mathcal{V}_{\mathbb{R}},\ \mathbin{\%>} : \beta = \mathcal{V}_{[0,1]}\}$, the composition fixes its target algebra $\beta$ and requires both operands to inhabit it:

$$
\frac{\Gamma; \Phi \vdash e_1 : A \rightsquigarrow B \qquad \Gamma; \Phi \vdash e_2 : B \rightsquigarrow C \qquad \mathrm{alg}(e_1) = \mathrm{alg}(e_2) = \beta}
     {\Gamma; \Phi \vdash e_1 \mathbin{\diamond_\beta^{\mathrm{tag}}} e_2 : A \rightsquigarrow C}\ \textsc{ComposeTag}_\beta
$$

The function $\mathrm{alg}(\cdot)$ is the synthesized algebra of a morphism expression, computed inductively from the algebras of declared morphisms ([Composition rules §3](composition-rules.md#3-user-defined-composition-rules)). When operands disagree, the typechecker rejects rather than auto-coercing; explicit base change is the surface syntax `.change_base(φ)` ([Expressions §4](expressions.md), `ExprChangeBase`). Soundness (Theorem [§9.1](#91-soundness)) collapses both syntactic shapes onto Kleisli composition $\diamond$ in $\mathbf{Kern}$ at the algebra $\alpha$ (resp. $\beta$).

### 5.3 Tensor product

$$
\frac{\Gamma; \Phi \vdash e_1 : A_1 \rightsquigarrow B_1 \qquad \Gamma; \Phi \vdash e_2 : A_2 \rightsquigarrow B_2}
     {\Gamma; \Phi \vdash e_1 \otimes e_2 : A_1 \times A_2 \rightsquigarrow B_1 \times B_2}\ \textsc{Tensor}
$$

The tensor uses the symmetric monoidal structure of $\mathbf{Kern}$ inherited from $\mathbf{SBor}$; see [Morphisms §2](morphisms.md).

### 5.4 Identity

$$
\frac{\Gamma \vdash \tau : \kappa}{\Gamma; \Phi \vdash \mathsf{id}_\tau : \tau \rightsquigarrow \tau}\ \textsc{Id}
$$

Identity is the only "structural" combinator with a dedicated expression form (`ExprIdentity` for a typed identity at a specific object, plus the un-annotated `id` form whose target is synthesized from context). Products and coproducts have no first-class projection or injection combinators; their introduction and elimination is handled by:

* the program-body sub-language's tuple-pattern bind $(v_1, \ldots, v_m) \leftarrow F(\bar a)$ (§[6.2](#62-destructuring-bind)), which both introduces a product (the family's codomain) and eliminates it (extending the trace by all $m$ component variables);
* bare-identifier program headers `program P (q₁, …, qₖ) : A → B`, which add the projections $q_i = \pi_i^A$ to the body's initial trace context $\Phi_0$ (§[7.2](#72-bare-identifier-projection-programs));
* the data-initialized morphism declaration `morphism f : A → B ~ from_data(...)`, which compiles to a concrete tensor / kernel realizing whatever projection or injection the user encoded in the data.

### 5.5 Higher combinators

$\mathsf{fan}, \mathsf{repeat}, \mathsf{stack}, \mathsf{scan}$ are first-class Expr variants (`ExprFan`, `ExprRepeat`, `ExprStack`, `ExprScan`) realising the higher-arity combinators described in [Expressions §3](expressions.md). Their typing rules are:

$$
\frac{\Gamma; \Phi \vdash e_i : A \rightsquigarrow B_i \quad (1 \le i \le n)}
     {\Gamma; \Phi \vdash \mathsf{fan}(e_1, \ldots, e_n) : A \rightsquigarrow B_1 \times \cdots \times B_n}\ \textsc{Fan}
$$

$$
\frac{\Gamma; \Phi \vdash e : A \rightsquigarrow A \quad n \in \mathbb{N}_{\ge 1}}
     {\Gamma; \Phi \vdash \mathsf{repeat}(e, n) : A \rightsquigarrow A}\ \textsc{Repeat}
$$

(per [Expressions §3.3](expressions.md#33-repeat), $\mathsf{repeat}$ is the $n$-fold Kleisli composition of $e$ with itself, requiring $e$ to be an endomorphism so the composition typechecks).

$$
\frac{\Gamma; \Phi \vdash e : A \rightsquigarrow B \quad n \in \mathbb{N}_{\ge 1}}
     {\Gamma; \Phi \vdash \mathsf{stack}(e, n) : A^n \rightsquigarrow B^n}\ \textsc{Stack}
$$

(per [Expressions §3.2](expressions.md#32-stack), $\mathsf{stack}$ is the $n$-fold tensor product of $e$ with itself; the compiler additionally clones $e$'s parameters per replica via $\mathsf{copy.deepcopy}$ so each layer is independently learnable). $\mathsf{scan}$ has the shape of a categorical trace ([Expressions §3.4](expressions.md#34-scan)):

$$
\frac{\Gamma; \Phi \vdash e : A \times S \rightsquigarrow B \times S}
     {\Gamma; \Phi \vdash \mathsf{scan}(e) : A \rightsquigarrow B}\ \textsc{Scan}
$$

### 5.6 Contraction application

A contraction $c$ registered in the value context with arity $k$ and inputs $A_1, \ldots, A_k$ producing a codomain $B$ may be applied to morphism-scope names $y_1, \ldots, y_k$ in a `let`-binding initializer:

$$
\frac{c \in \mathrm{ContractionReg}\ \text{with arity}\ k\ \text{and signature}\ (A_1, \ldots, A_k) \to B
       \qquad y_i : \tau_i \in \Gamma\ \text{and}\ \tau_i\ \text{matches}\ A_i\ (1 \le i \le k)}
      {\Gamma; \Phi \vdash c(y_1, \ldots, y_k) : \mathbf{1} \rightsquigarrow B}\ \textsc{ContractApp}
$$

See [Composition rules §4](composition-rules.md) for the operadic semantics of contractions as flat wirings.

**Note on program instantiation.** A program template $P$ with signature $(\Delta) \Rightarrow A \rightsquigarrow B$ is not first-class as an expression: there is no rule of the form "$\Gamma; \Phi \vdash P\langle \bar a \rangle : \cdots$". Instead, $P$ is instantiated inside a program body by the statement $v \leftarrow P(\bar a)$, whose typing rule is given in §[6.8](#68-template-inlining). This design choice reflects the implementation strategy of inline expansion by substitution and avoids a higher-order Π-application form that would not contribute to the surface DSL's expressive power.

## 6. Inference rules for statements

The statement-typing judgment is $\Gamma; \Phi \vdash s \dashv \Phi'$, read "in trace context $\Phi$, statement $s$ produces the extended trace context $\Phi'$".

### 6.1 Bind

$$
\frac{\Gamma; \Phi \vdash F(\bar a) : \mathsf{Kernel}[\Phi, B] \quad v\ \text{fresh in}\ \Phi}
     {\Gamma; \Phi \vdash v \leftarrow F(\bar a) \dashv \Phi, v : B}\ \textsc{Bind}
$$

The trace context $\Phi$ becomes the kernel's input space; the bound variable $v$ adds a coordinate of type $B$ to the trace. This is the typing analogue of the Kleisli arrow

$$
\mathcal{S}\llbracket v \leftarrow F(\bar a) \rrbracket : \Phi \to \mathcal{G}(\Phi \times B)
$$

from [Programs §2.1](programs.md#21-bind).

### 6.2 Destructuring bind

$$
\frac{\Gamma; \Phi \vdash F(\bar a) : \mathsf{Kernel}[\Phi, B_1 \times \cdots \times B_m]
       \quad v_1, \ldots, v_m\ \text{fresh and distinct}}
     {\Gamma; \Phi \vdash (v_1, \ldots, v_m) \leftarrow F(\bar a) \dashv \Phi, v_1 : B_1, \ldots, v_m : B_m}\ \textsc{BindTuple}
$$

### 6.3 Observe

$$
\frac{\Gamma; \Phi \vdash F(\bar a) : \mathsf{Kernel}[\Phi, B] \quad v \in \mathrm{dom}(\Phi)\ \text{or}\ v\ \text{is the observed name}}
     {\Gamma; \Phi \vdash \mathsf{observe}\ v \leftarrow F(\bar a) \dashv \Phi}\ \textsc{Observe}
$$

The trace context is unchanged because the observed value is externally supplied; the statement contributes only a score factor. The denotation lives in the unnormalized Giry sub-monad $\mathcal{G}_{\le 1}$ (see [Programs §2.2](programs.md#22-observe)).

### 6.4 Marginalize

$$
\frac{\Gamma; \Phi \vdash F(\bar a) : \mathsf{Kernel}[\Phi, B]
       \qquad \Gamma; \Phi, v : B \vdash s_1; \ldots; s_n \dashv \Phi'}
     {\Gamma; \Phi \vdash \mathsf{marginalize}\ v \leftarrow F(\bar a)\,\{\, s_1; \ldots; s_n\,\} \dashv \Phi'\setminus v}\ \textsc{Marginalize}
$$

The variable $v$ is bound *inside* the marginalize body but does not appear in the resulting trace context: marginalization is precisely the pushforward $\pi_{\Phi *}$ that eliminates $v$ from the joint kernel (see [Programs §2.6](programs.md#26-marginalize)).

### 6.5 Let

$$
\frac{\Gamma; \Phi \vdash e : \Phi \rightsquigarrow \tau \quad v\ \text{fresh in}\ \Phi}
     {\Gamma; \Phi \vdash \mathsf{let}\ v = e \dashv \Phi, v : \tau}\ \textsc{Let}
$$

A let statement is the deterministic special case of bind: the kernel is a Dirac.

### 6.6 Score

$$
\frac{\Gamma; \Phi \vdash e : \Phi \rightsquigarrow \mathsf{Real}\,1}
     {\Gamma; \Phi \vdash \mathsf{score}\ e \dashv \Phi}\ \textsc{Score}
$$

A score statement contributes a log-density factor; the trace context is preserved because no new random variable is introduced.

### 6.7 Statement sequencing

Statement sequencing is the structural rule that *defines* what a statement list means:

$$
\frac{\Gamma; \Phi_0 \vdash s_1 \dashv \Phi_1 \qquad \Gamma; \Phi_1 \vdash s_2; \ldots; s_n \dashv \Phi_n}
     {\Gamma; \Phi_0 \vdash s_1; s_2; \ldots; s_n \dashv \Phi_n}\ \textsc{Seq}
$$

The empty sequence is typed by $\Gamma; \Phi \vdash \varepsilon \dashv \Phi$.

### 6.8 Template inlining

When the family slot of a draw statement refers to a program template $P$ rather than a registered distribution family, the statement is interpreted by inline expansion of the template's body. The typing rule is:

$$
\frac{P : (\Delta) \Rightarrow A \rightsquigarrow B \in \Gamma
       \qquad \Delta = p_1 : P_1, \ldots, p_k : P_k
       \qquad \Gamma; \Phi \vdash a_i : P_i \quad (1 \le i \le k)
       \qquad \Phi \vdash A[\bar a / \bar p]\ \text{matches the actual input shape}}
     {\Gamma; \Phi \vdash v \leftarrow P(a_1, \ldots, a_k) \dashv \Phi, v : B[\bar a / \bar p]}\ \textsc{Inline}
$$

The substitution $[\bar a / \bar p]$ replaces each formal parameter $p_i$ in the declared $A$ and $B$ with the actual argument $a_i$. The denotation is exactly the inline-expansion semantics of [Programs §3a](programs.md#3a-parametric-programs): the template's body steps are α-renamed under a fresh prefix $v\$$, the return-variable is renamed to $v$, and the renamed statement list replaces the call site. Cycle detection in the compiler rejects template self-application and any mutually recursive template clique, so the inlining procedure terminates on every well-typed module.

## 7. Inference rules for programs

The program-typing judgment is $\Gamma \vdash p : (\Delta) \Rightarrow A \rightsquigarrow B$, read "program $p$ is parametric over $\Delta$ and denotes a Kleisli arrow $A \rightsquigarrow B$".

### 7.1 Program declaration

$$
\frac{\Gamma \uplus \Delta \vdash A : \kappa_A
       \qquad \Gamma \uplus \Delta \vdash B : \kappa_B
       \qquad \Gamma \uplus \Delta; A \vdash s_1; \ldots; s_n \dashv \Phi_n
       \qquad \Gamma \uplus \Delta; \Phi_n \vdash e : \Phi_n \rightsquigarrow B}
     {\Gamma \vdash \mathsf{program}\ P\ (\Delta)\ :\ A \to B\ \{\,s_1; \ldots; s_n; \mathsf{return}\ e\,\} : (\Delta) \Rightarrow A \rightsquigarrow B}\ \textsc{Prog}
$$

The operator $\uplus$ is *parameter folding*: $\Gamma \uplus \Delta$ extends $\Gamma$ with one new entry per $\Delta$-binder, treating each ScalarParam $p : R$ as a fresh module-level binding of $p$ at the scalar kind $\mathsf{Scalar}_R$, each ObjectParam $X : \mathrm{U}$ as a fresh type-level binding at the universe kind $\mathrm{U} \in \{\ast_{\mathrm{FinSet}}, \ast_{\mathrm{Space}}, \mathrm{either}\}$, and each MorphismParam $f : \mathsf{Mor}[A, B]$ as a fresh morphism-level binding at signature $A \rightsquigarrow B$. The fold is well-defined because $\Delta$'s binders are syntactically disjoint from $\Gamma$ (any clash is a compile error). $\uplus$ is the syntactic device that fits a Π-binder list into a flat value context for the body's typing; the Π-structure resurfaces in the conclusion as the $(\Delta) \Rightarrow$ prefix.

The rule reads bottom-up: a program is well-typed when (i) the declared signature $A \to B$ is kind-correct in the parameter-extended context, (ii) the body, threaded through the trace contexts $\Phi_0 = A$ through $\Phi_n$, is a valid statement sequence, and (iii) the return expression projects $\Phi_n$ onto $B$.

The dependent-product structure of the conclusion's denotation,

$$
\llbracket P \rrbracket : \prod_{p_1 : P_1} \cdots \prod_{p_k : P_k} \mathbf{Kern}(\llbracket A \rrbracket, \llbracket B \rrbracket),
$$

reflects exactly the parameter context $\Delta$: when $\Delta = \varepsilon$, the product is trivial and $\llbracket P \rrbracket$ is a plain kernel; when $\Delta$ is non-empty, $\llbracket P \rrbracket$ is the dependent Kleisli family of [Programs §3a](programs.md#3a-parametric-programs). This is the categorical reason why the REPL renders typed parameters as a constraint context

$$
P :: (p_1 : P_1, \ldots, p_k : P_k) \Rightarrow A \to B
$$

rather than as a curried arrow chain: the parameters live in a Π, not in the kernel's domain.

### 7.2 Bare-identifier projection programs

The special case $\Delta = q_1, \ldots, q_k$ of bare-identifier parameters is governed by:

$$
\frac{\Gamma \vdash A = \sigma_1 \times \cdots \times \sigma_k : \kappa
       \qquad \Gamma; q_1 : \sigma_1, \ldots, q_k : \sigma_k \vdash s_1; \ldots; s_n \dashv \Phi_n
       \qquad \cdots}
     {\Gamma \vdash \mathsf{program}\ P\ (q_1, \ldots, q_k)\ :\ A \to B\ \{\,\ldots\,\} : A \rightsquigarrow B}\ \textsc{ProgProj}
$$

The resulting Kleisli signature has *no parameter context* because the $q_i$ are not Π-binders but projections: their denotation is $q_i = \pi_i : A \to \sigma_i$. The body sees the projections as already-bound variables in its initial trace $\Phi_0$.

### 7.3 Module-level declarations

Module-level declarations extend $\Gamma$ through the rules:

$$
\frac{\Gamma \vdash \tau : \kappa}
     {\Gamma \vdash \mathsf{object}\ X : \tau \;\Rightarrow\; \Gamma, X : \kappa}\ \textsc{DeclObject}
\qquad
\frac{\Gamma; \varepsilon \vdash e : A \rightsquigarrow B}
     {\Gamma \vdash \mathsf{morphism}\ f : A \to B\ \sim\ e \;\Rightarrow\; \Gamma, f : A \rightsquigarrow B}\ \textsc{DeclMorph}
$$

with analogous rules for `space`, `let`, `export`, `signature`, `encoder`, `decoder`, `loss`, `bundle`, `composition`, `deduction`, `category`, `schema`, `rule`. We omit the full enumeration; each rule has the form "premises check the body of the declaration in $\Gamma$; conclusion extends $\Gamma$ with the declared name at its declared kind/signature".

## 8. Denotational interpretation

A denotation is a partial function on syntax that is total on well-typed phrases. We summarise the structure of $\llbracket \cdot \rrbracket$ here; full definitions appear in the surrounding chapters. The key claim is that *every* judgment of §[3](#3-inference-rules-for-types-and-kinds)–§[7](#7-inference-rules-for-programs) has a denotational counterpart, and the inference rules are exactly the conditions under which the denotation is defined.

### 8.1 Kind denotation

A kind $\kappa$ denotes a (large) category $\llbracket \kappa \rrbracket$:

| $\kappa$ | $\llbracket \kappa \rrbracket$ |
|---|---|
| $\ast_{\mathrm{FinSet}}$ | $\mathbf{FinSet}$ |
| $\ast_{\mathrm{Space}}$ | $\mathbf{SBor}$ |
| $\ast_{\mathrm{Sort}}$ | fibre of the signature's generalized algebraic theory |
| $\ast_{\mathrm{Atom}}$ | hom-objects of a free residuated category universe |
| $\mathsf{Family}[\Theta, B]$ | $\mathrm{Hom}_{\mathbf{Kern}}(\Theta, B)$ (a set, not a category) |
| $\mathsf{Mor}[A, B]$ | $\mathrm{Hom}_{\mathbf{Kern}}(\llbracket A \rrbracket, \llbracket B \rrbracket)$ |
| $\mathsf{Scalar}_R$ | the underlying set of the rig $R$ |

### 8.2 Type denotation

A type $\tau$ with $\Gamma \vdash \tau : \kappa$ denotes an object $\llbracket \tau \rrbracket \in \llbracket \kappa \rrbracket$. The interpretation is compositional:

$$
\begin{aligned}
\llbracket \mathsf{FinSet}\,n \rrbracket &= [n] \in \mathbf{FinSet} \\
\llbracket \mathsf{Real}\,n \rrbracket &= \mathbb{R}^n \in \mathbf{SBor} \\
\llbracket \tau_1 \times \tau_2 \rrbracket &= \llbracket \tau_1 \rrbracket \times \llbracket \tau_2 \rrbracket \\
\llbracket \tau_1 + \tau_2 \rrbracket &= \llbracket \tau_1 \rrbracket \sqcup \llbracket \tau_2 \rrbracket
\end{aligned}
$$

and so on for the remaining constructors; see [Types and spaces §2–§3](types-and-spaces.md).

### 8.3 Morphism denotation

A judgment $\Gamma; \Phi \vdash e : A \rightsquigarrow B$ denotes a Kleisli arrow

$$
\llbracket e \rrbracket : \llbracket \Phi \times A \rrbracket \to \mathcal{G}(\llbracket B \rrbracket)
$$

(in the empty-$\Phi$ case the input simplifies to $\llbracket A \rrbracket$). The composition, tensor, and combinator rules of §[5](#5-inference-rules-for-morphism-expressions) interpret to the corresponding categorical operations: Kleisli composition $\diamond$, the symmetric monoidal product $\otimes$, the structural projections / injections of $\mathbf{Kern}$.

### 8.4 Statement denotation

A judgment $\Gamma; \Phi \vdash s \dashv \Phi'$ denotes a Kleisli arrow

$$
\llbracket s \rrbracket : \llbracket \Phi \rrbracket \to \mathcal{G}(\llbracket \Phi' \rrbracket).
$$

The rules of §[6](#6-inference-rules-for-statements) thread $\Phi$ through the body, building the Kleisli composite of [Programs §2 equation (3)](programs.md#2-statements):

$$
\mathcal{B}\llbracket s_1; \ldots; s_n; \mathsf{return}\ e \rrbracket
\;=\;
\llbracket s_1 \rrbracket \diamond \cdots \diamond \llbracket s_n \rrbracket \diamond (\eta \circ \pi_e).
$$

### 8.5 Program denotation

A program judgment $\Gamma \vdash p : (\Delta) \Rightarrow A \rightsquigarrow B$ denotes a dependent family

$$
\llbracket p \rrbracket \;\in\; \prod_{\delta : \llbracket \Delta \rrbracket} \mathrm{Hom}_{\mathbf{Kern}}(\llbracket A \rrbracket, \llbracket B \rrbracket),
$$

which collapses to a plain element of $\mathrm{Hom}_{\mathbf{Kern}}(\llbracket A \rrbracket, \llbracket B \rrbracket)$ when $\Delta = \varepsilon$.

## 9. Soundness

### 9.1 Soundness

**Theorem (Soundness).** *Suppose $\Gamma$ is well-formed and the following hold:*

* *if $\Gamma \vdash \tau : \kappa$ then $\llbracket \tau \rrbracket$ is defined and $\llbracket \tau \rrbracket \in \llbracket \kappa \rrbracket$;*
* *if $\Gamma; \Phi \vdash e : A \rightsquigarrow B$ then $\llbracket e \rrbracket$ is defined and $\llbracket e \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\llbracket A \rrbracket,\, \mathcal{G}(\llbracket B \rrbracket))$;*
* *if $\Gamma; \Phi \vdash F(\bar a) : \mathsf{Kernel}[\Phi, B]$ then $\llbracket F(\bar a) \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\llbracket \Phi \rrbracket,\, \mathcal{G}(\llbracket B \rrbracket))$;*
* *if $\Gamma; \Phi \vdash s \dashv \Phi'$ then $\llbracket s \rrbracket$ is defined and $\llbracket s \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\llbracket \Phi \rrbracket,\, \mathcal{G}(\llbracket \Phi' \rrbracket))$;*
* *if $\Gamma \vdash p : (\Delta) \Rightarrow A \rightsquigarrow B$ then $\llbracket p \rrbracket$ is defined and $\llbracket p \rrbracket \in \prod_{\delta : \llbracket \Delta \rrbracket} \mathrm{Hom}_{\mathbf{Kern}}(\llbracket A[\delta] \rrbracket,\, \mathcal{G}(\llbracket B[\delta] \rrbracket))$.*

The trace context $\Phi$ enters as the domain of the *statement* and *family-application* judgments; in the *morphism* judgment it is a scope for resolving free names (via $\textsc{TraceVar}$, where the morphism's domain $A$ may *equal* $\Phi$, but is not multiplied with it). The asymmetry tracks the actual roles of $\Phi$ in the rules of §[5](#5-inference-rules-for-morphism-expressions), §[4](#4-inference-rules-for-families), and §[6](#6-inference-rules-for-statements).

**Proof.** By mutual induction on the derivation. The structure is:

* the *type-formation* clauses ($\textsc{FinSet}, \textsc{Real}, \textsc{TyProd}, \textsc{TySum}, \textsc{TySlashR}, \textsc{TySlashL}, \textsc{TyEff}, \textsc{FinSetVar}, \textsc{TyVar}$) each appeal to the corresponding constructor in $\mathbf{FinSet}$ / $\mathbf{SBor}$ / the residuated atom universe ([Types and spaces §2–§3](types-and-spaces.md)), totality of which is a standing assumption of the categorical setting;
* the *morphism* clauses ($\textsc{TraceVar}, \textsc{ModuleVar}, \textsc{Compose}, \textsc{Tensor}, \textsc{Id}, \textsc{Fan}, \textsc{Repeat}, \textsc{ContractApp}$, plus the derived combinators $\mathsf{stack}, \mathsf{scan}, \mathsf{cup}, \mathsf{cap}, \mathsf{dagger}, \mathsf{trace}, \mathsf{change\_base}$) each appeal to a categorical operation in $\mathbf{Kern}$ ([Morphisms](morphisms.md) and [Expressions](expressions.md)), whose totality on the hom-set of the premises is the conclusion;
* the *family* clauses ($\textsc{FamReg}, \textsc{FamApp}$) appeal to the registered family's measurable parameter map ([Programs §2.1](programs.md#21-bind));
* the *statement* clauses ($\textsc{Bind}, \textsc{BindTuple}, \textsc{Observe}, \textsc{Marginalize}, \textsc{Let}, \textsc{Score}, \textsc{Seq}, \textsc{Inline}$) appeal to the explicit Kleisli arrows of [Programs §2](programs.md#2-statements);
* the *program* clause ($\textsc{Prog}, \textsc{ProgProj}$) appeals to the dependent-product structure of [Programs §3a](programs.md#3a-parametric-programs) plus the iterated $\diamond$-composition of the body.

We give four representative cases ($\textsc{Compose}, \textsc{Bind}, \textsc{Marginalize}, \textsc{Prog}$) in full; every other case follows the same template, substituting the relevant categorical operation for $\diamond$.

*Case $\textsc{ComposePoly}_\alpha$* (the $\textsc{ComposeTag}_\beta$ case is identical with $\beta$ in place of $\alpha$).
By induction, $\llbracket e_1 \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\llbracket A \rrbracket,\, \mathcal{G}(\llbracket B \rrbracket))$ and $\llbracket e_2 \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\llbracket B \rrbracket,\, \mathcal{G}(\llbracket C \rrbracket))$. Using the Kleisli-composition convention of [Setting §3](setting.md#3-standard-borel-spaces-and-markov-kernels) ($k_1 \diamond k_2 = \mu \circ \mathcal{G}(k_2) \circ k_1$, *first* $k_1$ *then* $k_2$), the denotation
$$
\llbracket e_1 \mathbin{\diamond_\alpha} e_2 \rrbracket \;=\; \llbracket e_1 \rrbracket \diamond \llbracket e_2 \rrbracket
\;=\; \mu \circ \mathcal{G}(\llbracket e_2 \rrbracket) \circ \llbracket e_1 \rrbracket
\;:\; \llbracket A \rrbracket \;\to\; \mathcal{G}(\llbracket C \rrbracket)
$$
is Kleisli composition in $\mathbf{Kern}$ at the algebra $\alpha$, well-defined and total on hom-sets by the categorical structure of $\mathbf{Kern}$. The trace context $\Phi$ does not enter the signature: it is the scope in which $\textsc{TraceVar}$ resolves any free names appearing in $e_1$ or $e_2$, and the IH gives both operand denotations relative to that scope.

*Case $\textsc{Bind}$.* By induction (on the family-application clause), $\llbracket F(\bar a) \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\llbracket \Phi \rrbracket,\, \mathcal{G}(\llbracket B \rrbracket))$. To produce the bind statement's required kernel $\llbracket \Phi \rrbracket \to \mathcal{G}(\llbracket \Phi \rrbracket \times \llbracket B \rrbracket)$ — note that the conclusion's trace context $\Phi, v : B$ denotes $\Phi \times B$ — we use the strength $\mathrm{str}_{\Phi, B} : \llbracket \Phi \rrbracket \times \mathcal{G}(\llbracket B \rrbracket) \to \mathcal{G}(\llbracket \Phi \rrbracket \times \llbracket B \rrbracket)$ of $\mathcal{G}$ on $\mathbf{SBor}$ ([Programs §5](programs.md#5-soundness-of-monadic-semantics)) to bundle the unchanged trace with the freshly-sampled value:
$$
\llbracket v \leftarrow F(\bar a) \rrbracket
\;=\; \mathrm{str}_{\Phi, B} \circ \langle \mathrm{id}_\Phi,\ \llbracket F(\bar a) \rrbracket \rangle.
$$
Concretely, the kernel acts on measurable rectangles $E \times C$ with $E \in \Sigma_\Phi$, $C \in \Sigma_B$ as
$$
\llbracket v \leftarrow F(\bar a) \rrbracket(\phi,\, E \times C)
\;=\;
\mathbf{1}_{E}(\phi) \cdot \llbracket F(\bar a) \rrbracket(\phi)(C),
$$
extended to all of $\Sigma_{\Phi \times B}$ by Caratheodory's theorem — the rectangles generate the product $\sigma$-algebra, and the map $\phi \mapsto \mathbf{1}_E(\phi) \cdot \llbracket F(\bar a) \rrbracket(\phi)(C)$ is measurable in $\phi$ because $\llbracket F(\bar a) \rrbracket$ is a Markov kernel.

*Case $\textsc{Marginalize}$.* By IH on the family premise, $\llbracket F(\bar a) \rrbracket : \llbracket \Phi \rrbracket \to \mathcal{G}(\llbracket B \rrbracket)$; by IH on the body premise, $k_{\mathrm{body}} : \llbracket \Phi \rrbracket \times \llbracket B \rrbracket \to \mathcal{G}(\llbracket \Phi' \rrbracket)$ (the body's trace context is the extended $\Phi, v : B$, denoting $\Phi \times B$). Build the marginalize statement's denotation in three composable Kleisli stages:

1. *Bind*: $\widetilde{F} \;=\; \mathrm{str}_{\Phi, B} \circ \langle \mathrm{id}_\Phi, \llbracket F(\bar a) \rrbracket \rangle : \llbracket \Phi \rrbracket \to \mathcal{G}(\llbracket \Phi \rrbracket \times \llbracket B \rrbracket)$, as in the $\textsc{Bind}$ case;
2. *Body*: Kleisli-compose with $k_{\mathrm{body}}$ (with the $\diamond$ convention "first left then right") to obtain $\widetilde{F} \diamond k_{\mathrm{body}} \;=\; \mu \circ \mathcal{G}(k_{\mathrm{body}}) \circ \widetilde{F} : \llbracket \Phi \rrbracket \to \mathcal{G}(\llbracket \Phi' \rrbracket)$;
3. *Pushforward*: let $\pi : \llbracket \Phi' \rrbracket \to \llbracket \Phi' \setminus v \rrbracket$ be the measurable projection that drops the $v$-coordinate. The Giry monad's functorial action sends $\pi$ to the pushforward kernel $\mathcal{G}(\pi) : \mathcal{G}(\llbracket \Phi' \rrbracket) \to \mathcal{G}(\llbracket \Phi' \setminus v \rrbracket)$, defined on measures by $\mathcal{G}(\pi)(\mu)(F) = \mu(\pi^{-1}(F))$ for $F \in \Sigma_{\Phi' \setminus v}$. Compose: $\mathcal{G}(\pi) \circ (\widetilde{F} \diamond k_{\mathrm{body}}) : \llbracket \Phi \rrbracket \to \mathcal{G}(\llbracket \Phi' \setminus v \rrbracket)$.

The pushforward is the measure-theoretic content of marginalization: integrating out the $v$-coordinate of the joint distribution, equivalently applying Fubini–Tonelli on the product $\sigma$-algebra $\Sigma_{\Phi' \setminus v} \otimes \Sigma_{\{v\}}$.

*Case $\textsc{Prog}$.* Let $\Delta = p_1 : P_1, \ldots, p_k : P_k$. The body's denotation is computed in the parameter-extended value context $\Gamma \uplus \Delta$, so it is naturally a *function of the parameter environment*. Concretely, by IH applied to the premise $\Gamma \uplus \Delta; A \vdash s_1; \ldots; s_n \dashv \Phi_n$, the statement-typing soundness clause yields a measurable map

$$
\Lambda \;:\; \prod_{i = 1}^{k} \llbracket P_i \rrbracket \;\longrightarrow\; \mathrm{Hom}_{\mathbf{Kern}}\bigl(\llbracket A[\delta] \rrbracket,\, \mathcal{G}(\llbracket \Phi_n[\delta] \rrbracket)\bigr)
$$

defined by interpreting the body's denotation function at each $\delta \in \prod_i \llbracket P_i \rrbracket = \llbracket \Delta \rrbracket$; measurability in $\delta$ follows from the fact that each clause of the body's denotation function (§[8.4](#84-statement-denotation)) is built from measurable categorical operations ($\diamond$, $\mathrm{str}$, $\pi_*$, the family's parameter map), and a composite of measurable maps is measurable. Pre- and post-composing with the return denotation $\llbracket e[\delta] \rrbracket : \llbracket \Phi_n[\delta] \rrbracket \to \mathcal{G}(\llbracket B[\delta] \rrbracket)$ (a Dirac kernel induced by the return-coordinate projection, also measurable in $\delta$) gives a measurable map $\delta \mapsto k_\delta : \llbracket A[\delta] \rrbracket \to \mathcal{G}(\llbracket B[\delta] \rrbracket)$. This *is* the dependent product

$$
\llbracket p \rrbracket \;\in\; \prod_{\delta : \llbracket \Delta \rrbracket} \mathrm{Hom}_{\mathbf{Kern}}\bigl(\llbracket A[\delta] \rrbracket,\, \mathcal{G}(\llbracket B[\delta] \rrbracket)\bigr).
$$

Note that the $\delta$-indexing is *denotational*, not syntactic: each $\delta \in \llbracket \Delta \rrbracket$ ranges over the full categorical universe (e.g. every kernel in $\mathrm{Hom}_{\mathbf{Kern}}(\llbracket A_i \rrbracket, \llbracket B_i \rrbracket)$ for a $\mathsf{Mor}[A_i, B_i]$-typed parameter), not just those expressible by a QVR expression. The Substitution Lemma of §[9.3](#93-structural-lemmas) gives the *syntactic* counterpart used in §[9.2](#92-subject-reduction-parametric-instantiation), and the two agree on the expressible sub-product $\{(\llbracket a_1 \rrbracket, \ldots, \llbracket a_k \rrbracket) : \Gamma \vdash a_i : P_i\} \subseteq \llbracket \Delta \rrbracket$ by the Semantic Substitution Corollary.

$\square$

### 9.2 Subject reduction (parametric instantiation)

**Theorem (Subject reduction).** *Let $P$ be a program with $\Gamma \vdash P : (\Delta) \Rightarrow A \rightsquigarrow B$ and body $\{s_1; \ldots; s_n; \mathsf{return}\,e\}$, and let $\bar a = a_1, \ldots, a_k$ be a tuple of actual arguments inhabiting $\Delta = p_1 : P_1, \ldots, p_k : P_k$ in $\Gamma$ (per the per-parameter clauses of the Substitution Lemma, §[9.3](#93-structural-lemmas)). Then the substituted body*

$$
\{s_1[\bar a / \bar p];\ \ldots;\ s_n[\bar a / \bar p];\ \mathsf{return}\,e[\bar a / \bar p]\}
$$

*is well-typed under $\Gamma$ as a morphism $A[\bar a / \bar p] \rightsquigarrow B[\bar a / \bar p]$, in the sense that*

$$
\Gamma;\ A[\bar a / \bar p] \;\vdash\; s_1[\bar a / \bar p];\ \ldots;\ s_n[\bar a / \bar p] \;\dashv\; \Phi_n[\bar a / \bar p],
\qquad
\Gamma;\ \Phi_n[\bar a / \bar p] \;\vdash\; e[\bar a / \bar p] : \Phi_n[\bar a / \bar p] \rightsquigarrow B[\bar a / \bar p],
$$

*and the corresponding Kleisli composite agrees with the program's $\Pi$-fibre value at $\llbracket \bar a \rrbracket$:*

$$
\mathcal{B}\bigl\llbracket s_1[\bar a / \bar p];\ \ldots;\ s_n[\bar a / \bar p];\ \mathsf{return}\,e[\bar a / \bar p] \bigr\rrbracket
\;=\;
\llbracket P \rrbracket (\llbracket a_1 \rrbracket, \ldots, \llbracket a_k \rrbracket)
$$

*in $\mathrm{Hom}_{\mathbf{Kern}}(\llbracket A[\bar a / \bar p] \rrbracket, \mathcal{G}(\llbracket B[\bar a / \bar p] \rrbracket))$.*

**Proof.** Apply the Substitution Lemma of §[9.3](#93-structural-lemmas) at each $a_i$ for $p_i$ in turn — using whichever per-parameter clause of the lemma matches the variant of $p_i$ ($P_i = \mathsf{Scalar}_R$ uses the kinding-judgment clause; $P_i = \mathsf{Mor}[A_i, B_i]$ uses the morphism-judgment clause; object-parameter clauses use the kinding-judgment clause for $X : \kappa$ entries) — to the body's typing derivations

$$
\Gamma \uplus \Delta;\ A \;\vdash\; s_1; \ldots; s_n \;\dashv\; \Phi_n,
\qquad
\Gamma \uplus \Delta;\ \Phi_n \;\vdash\; e : \Phi_n \rightsquigarrow B,
$$

obtaining the substituted versions stated above. The Semantic Substitution Corollary then identifies the Kleisli composite of the substituted body with the $\Pi$-fibre evaluation $\llbracket P \rrbracket(\llbracket a_1 \rrbracket, \ldots, \llbracket a_k \rrbracket)$, because each clause of $\mathcal{B}\llbracket \cdot \rrbracket$ ([Programs §2](programs.md#2-statements)) commutes with substitution by naturality of the categorical operations involved.

The α-renaming step of inline expansion ([Programs §3a](programs.md#3a-parametric-programs)) prefixes every internal latent name by $v\$$ to avoid capture in the caller's context. *Soundness of α-renaming*: typing is invariant under simultaneous renaming of bound and binding occurrences by any injective name permutation. Formally, if $\rho$ is an injection on the name space that fixes every name occurring in $\Gamma$, then $\Gamma; \Phi \vdash \mathcal{J}$ is derivable iff $\Gamma; \rho(\Phi) \vdash \rho(\mathcal{J})$ is derivable (by induction on the derivation, each rule's premise pattern is closed under $\rho$ because every consulted entry's lookup goes to the renamed name in the renamed context). Splicing the renamed substituted body into a caller's statement list then composes with Weakening (§[9.3](#93-structural-lemmas)) and the Seq rule: each spliced statement's typing premise holds in the caller's trace context because (i) the body's initial trace $A[\bar a / \bar p]$ matches the caller's $\Phi$ by the Inline rule's input-shape side condition, and (ii) each subsequent statement extends $\Phi$ inside the caller exactly as it extended its own trace inside the body. $\square$

### 9.3 Structural lemmas

The Soundness theorem (§[9.1](#91-soundness)) and the Subject-Reduction theorem (§[9.2](#92-subject-reduction-parametric-instantiation)) appeal to three structural facts about the type system that are themselves worth stating as lemmas. Each is proved by structural induction on the derivation; the inductive cases follow the corresponding clause of the relevant judgment.

**Lemma (Weakening).** *Let $\mathcal{J}$ be any judgment form. If $\Gamma \vdash \mathcal{J}$ and $\Gamma'$ extends $\Gamma$ with entries whose names are fresh with respect to $\Gamma$ and to every name appearing in $\mathcal{J}$, then $\Gamma' \vdash \mathcal{J}$. The analogous statements hold for the parameter context $\Delta$ and for the trace context $\Phi$ under the same freshness condition.*

*Proof.* Induction on the derivation. Each leaf rule ($\textsc{TyVar}, \textsc{FinSet}, \textsc{TraceVar}, \textsc{ModuleVar}, \textsc{FamReg}$) requires only that a specific entry appear in the context; extension preserves that entry. Each inductive rule's premises are themselves judgments derivable under the same context, so the IH applies. The freshness side-conditions of $\textsc{Bind}$, $\textsc{BindTuple}$, $\textsc{Let}$ — "$v$ fresh in $\Phi$" — are preserved by $\Gamma$-weakening trivially ($\Gamma$-extension does not touch $\Phi$) and by $\Phi$-weakening because the lemma's hypothesis demands that the extension's names be fresh with respect to every name in $\mathcal{J}$, including the bound $v$. The kind-join $\sqcup$ side-condition of $\textsc{TyProd}$ is unchanged by context extension because $\sqcup$ depends only on the kinds, which the IH supplies unchanged. $\square$

**Lemma (Substitution).** *Suppose the parameter universe $P$ and the hypothesis "$a$ inhabits $P$" take one of three forms, matching the $\mathsf{ProgramParam}$ variants:*

* *$P$ is a kind ($\ast_{\mathrm{FinSet}}, \ast_{\mathrm{Space}}, \ast_{\mathrm{Object}}, \mathsf{Scalar}_R$, $\ldots$) and the hypothesis is the kinding judgment $\Gamma \vdash a : P$;*
* *$P = \mathsf{Mor}[A, B]$ and the hypothesis is the morphism judgment $\Gamma; \Phi \vdash a : A \rightsquigarrow B$.*

*Then for every judgment $\mathcal{J}$ derivable in $\Gamma \uplus (p : P); \Phi$, the substituted judgment $\mathcal{J}[a/p]$ is derivable in $\Gamma; \Phi[a/p]$.*

*Proof (syntactic).* Induction on the derivation. The substitution $[a/p]$ replaces every occurrence of $p$ in $\Phi$ and in the conclusion's subject and type with $a$. Cases:

* $\textsc{TyVar}$ at $p$ (when $P$ is a kind): the conclusion was $\Gamma \uplus (p : P) \vdash p : P$; substituted conclusion is $\Gamma \vdash a : P$, which is the hypothesis.
* $\textsc{TyVar}$ at $X \neq p$: $X : \kappa \in \Gamma$ survives unchanged.
* $\textsc{ModuleVar}$ at $p$ (when $P = \mathsf{Mor}[A, B]$): the conclusion was $\Gamma \uplus (p : A \rightsquigarrow B); \Phi \vdash p : A \rightsquigarrow B$; substituted conclusion is $\Gamma; \Phi[a/p] \vdash a : A \rightsquigarrow B$. The hypothesis $\Gamma; \Phi \vdash a : A \rightsquigarrow B$ gives this directly (and $\Phi[a/p] = \Phi$ when no entries of $\Phi$ mention $p$, which is the case here because morphism-level $p$ cannot appear inside a type expression).
* $\textsc{ModuleVar}$ at $f \neq p$, $\textsc{TraceVar}$ at any $x$, $\textsc{FamReg}$ at any $F$: the consulted entry is in $\Phi$ or in the $\Gamma$ part of $\Gamma \uplus (p : P)$, not in the $p$ part, so the rule re-applies under $\Gamma$ unchanged.
* $\textsc{TyVar}$ at a $p$-typed object-parameter (when $P \in \{\ast_{\mathrm{FinSet}}, \ast_{\mathrm{Space}}\}$): the $p$-occurrences may appear inside compound type expressions in $\Phi$. The substitution replaces them with $a$ (an object name); each occurrence becomes a $\textsc{TyVar}$ at $a$, which fires because $a$ is declared in $\Gamma$.
* All other rules: the premises mention strictly smaller derivations; the IH gives them substituted; the rule re-fires because its premise pattern is closed under substitution (each premise's syntactic shape maps to a permissible substituted shape, the rule's well-formedness side-conditions like "$\sqcup$ defined" or "$v$ fresh" being preserved by substitution).
$\square$

**Corollary (Semantic substitution).** *Under the hypothesis of the Substitution Lemma, $\llbracket \mathcal{J}[a/p] \rrbracket = \llbracket \mathcal{J} \rrbracket\bigl[\llbracket a \rrbracket / p\bigr]$ in the appropriate $\Pi$-fibre, where the right side denotes evaluation of the $\Pi$-indexed family at $\llbracket a \rrbracket$.*

*Proof.* By induction on the derivation. Each rule's denotation is compositional in its premises' denotations, and the categorical operations used in §[8](#8-denotational-interpretation) ($\diamond$, $\otimes$, $\pi_*$, $\mathrm{str}$) are themselves natural in their arguments, so they commute with substitution by naturality. $\square$

**Lemma (Type uniqueness).** *Suppose $\Gamma; \Phi \vdash e : A_1 \rightsquigarrow B_1$ and $\Gamma; \Phi \vdash e : A_2 \rightsquigarrow B_2$. Then $A_1 = A_2$ and $B_1 = B_2$. The analogous statement holds for the statement, family, type-formation, and program judgments.*

*Proof.* Induction on $e$. Each Expr AST variant has a unique introduction rule, modulo two syntactic discriminators:

* $\textsf{ExprIdent}\,x$ dispatches to $\textsc{TraceVar}$ if $x \in \mathrm{dom}(\Phi)$ and to $\textsc{ModuleVar}$ if $x \in \mathrm{dom}(\Gamma)$. The two domains are *disjoint*: every rule that extends $\Phi$ ($\textsc{Bind}$, $\textsc{BindTuple}$, $\textsc{Let}$, ...) carries a freshness side-condition "$v$ fresh in $\Phi$" *and* the implicit invariant that program-body-scope names never collide with module-level names ($\Gamma$ entries are declared once at module level before any program body is typed). The two cases therefore cannot simultaneously apply.
* $\textsf{ExprCompose}$ dispatches to $\textsc{ComposePoly}_\alpha$ or $\textsc{ComposeTag}_\beta$ by its `op` field; the `op` value determines the rule and hence the algebra ($\alpha$ from the module's declared algebra for polymorphic ops; $\beta$ fixed by the tagged op's column in the table of §[5.2](#52-composition)).

Modulo these dispatchers, each rule determines the conclusion's $A$ and $B$ as functions of the premises' types: e.g. $\textsc{ComposePoly}_\alpha$ fixes the conclusion's $A$ to be the first premise's $A$ and the conclusion's $B$ to be the second premise's $C$. The premises themselves have unique types by induction. Base cases: $\textsc{ModuleVar}$ reads $A \rightsquigarrow B$ uniquely from $\Gamma$ by the disjointness of context entries; $\textsc{TraceVar}$ reads $\tau$ uniquely from $\Phi$ by the same disjointness on $\Phi$. $\square$

**Lemma (Inversion).** *Suppose $\Gamma \vdash \mathcal{J}$ is derivable. Then there is exactly one rule whose conclusion matches the shape of $\mathcal{J}$, and that rule's premises are necessarily derivable in $\Gamma$.*

*Proof.* By inspection of the rule set, no two rules share a conclusion shape (up to the syntactic discriminators noted below). Specifically:

* Type-formation rules: the head constructor of $\tau$ partitions the rules into disjoint classes — $\textsf{TypeName}$ goes to $\textsc{TyVar}$, an integer-literal $\textsf{DiscreteConstructor}$ to $\textsc{FinSet}$, a continuous constructor $C$ to the corresponding $\textsc{Real} / \textsc{Simplex} / \ldots$ rule, $\textsf{ObjectProduct}$ to $\textsc{TyProd}$, $\textsf{ObjectCoproduct}$ to $\textsc{TySum}$, $\textsf{ObjectSlash}$ to $\textsc{TySlashR}$ or $\textsc{TySlashL}$ (the AST's `direction` field distinguishes), $\textsf{ObjectEffectApply}$ to $\textsc{TyEff}$.
* Morphism-expression rules: the AST variant of $e$ (one of $\textsf{ExprIdent}$, $\textsf{ExprIdentity}$, $\textsf{ExprCompose}$, $\textsf{ExprTensorProduct}$, $\textsf{ExprFan}$, $\ldots$) partitions them. Two refinements:
  - $\textsf{ExprIdent}\,x$ splits on whether $x \in \mathrm{dom}(\Phi)$ ($\to \textsc{TraceVar}$) or $x \in \mathrm{dom}(\Gamma)$ ($\to \textsc{ModuleVar}$); the two domains are disjoint by the program-body freshness invariant.
  - $\textsf{ExprCompose}$ is further split on its `op` field: the polymorphic `op` $\in \{\mathtt{>>}, \mathtt{<<}, \mathtt{>=>}\}$ selects $\textsc{ComposePoly}_\alpha$ (with $\alpha$ the module's declared algebra); the tagged `op` $\in \{\mathtt{*>}, \mathtt{\sim>}, \mathtt{||>}, \mathtt{?>}, \mathtt{\&\&>}, \mathtt{+>}, \mathtt{\$>}, \mathtt{\%>}\}$ selects the corresponding $\textsc{ComposeTag}_\beta$.
* Statement rules: the AST variant of $s$ (one of $\textsf{DrawStep}$, $\textsf{ObserveStep}$, $\textsf{MarginalizeStep}$, $\textsf{LetStep}$, $\textsf{ScoreStep}$) partitions them. $\textsf{DrawStep}$ is further split on whether the family slot resolves to a $\textsf{FamilySpec}$ (giving $\textsc{Bind}$ / $\textsc{BindTuple}$) or to a $\textsf{ProgramDecl}$ (giving $\textsc{Inline}$).
* Program rules: the $\textsf{ProgramParam}$ tagged-union discriminator on the parameter list selects $\textsc{Prog}$ (any typed parameter) or $\textsc{ProgProj}$ (all bare-identifier).

Once the rule is determined, its premises are exactly the conditions under which the derivation could have ended in that rule; they must therefore be derivable. $\square$

The four lemmas together underwrite every "by induction" step in §[9.1](#91-soundness) and §[9.2](#92-subject-reduction-parametric-instantiation): Weakening discharges the cases where a premise's context differs from the conclusion's; Substitution underwrites the $\textsc{Prog}$ case's claim that each fibre is well-defined; Type Uniqueness ensures the algorithm of §[10](#10-algorithmic-typechecking) is well-defined (each synthesis returns *the* type, not a non-empty family); Inversion is the syntactic-direction property the bidirectional algorithm of §[10](#10-algorithmic-typechecking) exploits to drive its check / synth dispatch.

### 9.4 Completeness

Soundness (§[9.1](#91-soundness)) says every well-typed phrase has a denotation. The converse — that no semantically-meaningful phrase is rejected by the type system — requires a denotation function that is defined independently of any typing derivation.

**Definition (Partial raw denotation).** *Extend the clauses of §[8](#8-denotational-interpretation) to raw syntax by setting $\llbracket \phi \rrbracket_{\mathrm{raw}}$ undefined whenever:*

* *a leaf identifier is absent from the surrounding environment $\rho$;*
* *a primitive constructor is invoked outside its domain (e.g. $\mathsf{FinSet}(n)$ at $n < 0$);*
* *a binary node receives operands whose recursively-computed denotations are incompatible with the node's categorical operation (e.g. $\textsc{Compose}$ at $\llbracket e_1 \rrbracket_{\mathrm{raw}} \in \mathrm{Hom}(A, B)$ and $\llbracket e_2 \rrbracket_{\mathrm{raw}} \in \mathrm{Hom}(B', C)$ with $B \neq B'$);*
* *any recursive call returns undefined.*

Otherwise $\llbracket \phi \rrbracket_{\mathrm{raw}}$ is the value of the structural recursion in §[8](#8-denotational-interpretation).

**Theorem (Completeness).** *For every phrase $\phi$ and every environment $\rho$ over which $\llbracket \phi \rrbracket_{\mathrm{raw}}$ is defined, there exist a typing context $\Gamma$ (with the same identifier bindings as $\rho$ but tracking sorts rather than denotations) and a derivation $\Gamma \vdash \mathcal{J}(\phi)$ such that the §[8](#8-denotational-interpretation) denotation of that derivation equals $\llbracket \phi \rrbracket_{\mathrm{raw}}$.*

**Proof.** By structural induction on $\phi$ with mutual cases for each syntactic category. We give type expressions in full; the morphism, family, statement, and program cases follow the same template.

*Type expressions.* By the definition above, $\llbracket \tau \rrbracket_{\mathrm{raw}}$ is defined exactly when (a) every $\textsf{TypeName}$ leaf $X$ has $\rho_{\mathrm{obj}}(X)$ or $\rho_{\mathrm{spc}}(X)$ defined, (b) every $\textsf{DiscreteConstructor}$ / $\textsf{ContinuousConstructor}$ leaf has a defined primitive interpretation ([Types and spaces §2–§4](types-and-spaces.md)), and (c) every product / coproduct / residuated / effect-apply node has all children defined. Construct $\Gamma$ by declaring, for each leaf $X$ in $\tau$, the entry $X : \kappa(X)$ where

$$
\kappa(X) \;=\;
\begin{cases}
\ast_{\mathrm{FinSet}} & \text{if } X \in \mathrm{dom}(\rho_{\mathrm{obj}})\\
\ast_{\mathrm{Space}} & \text{if } X \in \mathrm{dom}(\rho_{\mathrm{spc}})\\
\ast_{\mathrm{Sort}} & \text{if } X \in \mathrm{dom}(\rho_{\mathrm{sort}})\\
\ast_{\mathrm{Atom}} & \text{if } X \in \mathrm{dom}(\rho_{\mathrm{cat}})
\end{cases}
$$

(the cases are exhaustive and disjoint by the well-formedness of $\rho$; cf. [Setting §5](setting.md#5-environments)). For each clause of "$\llbracket \tau \rrbracket_{\mathrm{raw}}$ defined", the corresponding kinding rule's premise is exactly this condition: $\textsc{TyVar}$ requires $X : \kappa \in \Gamma$ which we just supplied; $\textsc{FinSet}$ requires $n \in \mathbb{N}$ which is the definedness of the primitive at $n$; $\textsc{TyProd}$ requires the partial $\sqcup$ to be defined at the children's kinds, which is precisely the well-formedness premise on the raw side. The inductive hypothesis discharges the premises about children. The constructed derivation satisfies $\llbracket \mathrm{derivation} \rrbracket = \llbracket \tau \rrbracket_{\mathrm{raw}}$ because both sides apply the same structural-recursive clauses to the same inputs.

*Morphism expressions, families, statements, programs.* The same template: the partial raw denotation's defined-ness condition at each node is, by the definition above, exactly the premise of the rule whose conclusion matches that node's syntactic shape. The compositionality of $\llbracket \cdot \rrbracket$ ensures the constructed-derivation denotation agrees with the raw denotation.

$\square$

**Corollary (Decidable type membership, decidable fragment).** *For every phrase $\phi$ free of the residuated formers and the effect-typed nodes ($\textsf{ObjectSlash}$, $\textsf{ObjectEffectApply}$) and every candidate type $\tau$, the question "is $\phi$ of type $\tau$?" is decidable.*

This follows from Soundness + Completeness + Type Uniqueness (§[9.3](#93-structural-lemmas)) restricted to the decidable fragment of §[10.1](#101-decidability): synthesize $\tau' = \mathrm{type}(\phi)$ by the bidirectional algorithm of §[10](#10-algorithmic-typechecking) (which terminates on this fragment by the proposition there), check $\tau' = \tau$ structurally. By Soundness the synthesis returns a valid type or fails; by Completeness it succeeds whenever $\llbracket \phi \rrbracket_{\mathrm{raw}}$ is defined; by Type Uniqueness the synthesized type is unique. For the residuated / effect-typed extension, decidability is conditional on the underlying solver as noted in §[10.1](#101-decidability).

### 9.5 Equivalence and conservativity

Two well-typed phrases $\phi_1, \phi_2$ at the same judgment are *denotationally equivalent*, written $\phi_1 \equiv \phi_2$, when $\llbracket \phi_1 \rrbracket = \llbracket \phi_2 \rrbracket$. The QVR type system is *conservative* in the following sense: for any judgment form $\mathcal{J}$, if $\phi_1$ and $\phi_2$ are $\mathcal{J}$-derivable and denotationally equal, the type system does not assign them distinguishable types (no judgment $\mathcal{J}'$ separates them). This conservativity is the categorical content of the "every well-typed phrase has a unique denotation" reading of [Adequacy](adequacy.md).

## 10. Algorithmic typechecking

The inference rules of §[3](#3-inference-rules-for-types-and-kinds)–§[7](#7-inference-rules-for-programs) are *declarative*: they specify which judgments are derivable, not how to derive them. The implementation in `src/quivers/dsl/compiler/` realizes a *bidirectional* algorithm with two modes:

| Mode | Read as |
|---|---|
| **Check** $\Gamma; \Phi \vdash e \mathrel{\Leftarrow} A \rightsquigarrow B$ | given $e$ and the expected signature $A \rightsquigarrow B$, verify the rule's premises |
| **Synth** $\Gamma; \Phi \vdash e \mathrel{\Rightarrow} A \rightsquigarrow B$ | given $e$, compute the (unique) signature $A \rightsquigarrow B$ |

The compiler synthesizes from leaves upward: variables and module-level names look up declared signatures; composition and tensor synthesize from their operands; program instantiation synthesizes through substitution. The compiler checks at boundaries where the expected signature is fixed by the surrounding context: morphism declarations (the right-hand side of `~` must check against the declared $A \to B$), program declarations (the return expression must check against the declared codomain), and statement composition (each statement's $\Phi'$ must match the next statement's input $\Phi$).

The relevant entry points in the implementation are:

* [`_resolve_any_space`](../api/dsl/resolution.md) — type-formation judgment $\Gamma \vdash \tau : \kappa$, returning the denotational object.
* [`_compile_expr`](../api/dsl/compiler.md) — morphism judgment $\Gamma; \Phi \vdash e \mathrel{\Rightarrow} A \rightsquigarrow B$, returning a record carrying the resolved domain and codomain.
* [`_compile_program`](../api/dsl/compiler.md) — program judgment, including the per-statement $\Phi$ chain tracked by `ChainShape`.

### 10.1 Decidability

**Proposition.** *The fragment of the type system excluding the residuated formers and effect-typed expressions ($\mathsf{TySlashR}, \mathsf{TySlashL}, \mathsf{TyEff}$) is decidable: there is an algorithm that, given $\Gamma$ and a phrase $\phi$, returns either a derivation of $\Gamma \vdash \phi : \cdot$ or a proof that no such derivation exists.*

**Proof sketch.** Type formation is structurally recursive over the syntax of $\tau$ and bottoms out at primitive constructors or context lookup. Morphism and statement typing are bidirectional and also structurally recursive over the term / statement, with no rule that introduces a metavariable requiring search. Program-typing's $\textsc{Prog}$ rule reads the body bottom-up, threading $\Phi$ through $\textsc{Seq}$; this is finite recursion on the body's statement count. Type-equality at the conclusion ($A_1 = A_2$, etc.) is decidable because the kinds of §[1.1](#11-kinds) and the type expressions of §[1.2](#12-type-expressions) are first-order finite trees. (The compiler additionally rejects cyclic *template-call graphs* at the inline-expansion stage of §[6.8](#68-template-inlining), which is a separate semantic check needed for instantiation but not for typechecking itself; typechecking terminates regardless of the call graph because it is structural recursion on the AST.) $\square$

The residuated and effect-typed fragments require additional side conditions ([Schemas §3](schemas.md), [Effects §4](effects.md)); their decidability is conditional on the underlying constraint solver.

## 11. Relation to other chapters

* [Setting](setting.md) fixes the meta-theoretic universe ($\mathcal{V}$-enriched symmetric monoidal categories) in which the kinds of §[8.1](#81-kind-denotation) live.
* [Types and spaces](types-and-spaces.md) gives the object-level denotation $\llbracket \tau \rrbracket$ used in §[8.2](#82-type-denotation).
* [Morphisms](morphisms.md) and [Expressions](expressions.md) give the morphism-level denotations used in §[8.3](#83-morphism-denotation) and provide the categorical structure (composition, tensor, marginalization) appealed to in the soundness proofs.
* [Programs](programs.md) gives the statement and program-level denotations used in §[8.4](#84-statement-denotation)–§[8.5](#85-program-denotation), with full equational detail on the Kleisli composites.
* [Effects](effects.md) and [Schemas](schemas.md) cover the residuated and effect-typed fragments mentioned in §[3.5](#35-residuated-formers), §[10.1](#101-decidability).
* [Adequacy](adequacy.md) states the implementation-correctness theorem that subsumes §[9.1](#91-soundness): the compiler agrees with the denotation on every well-typed module.

## 12. Worked example

To anchor the rules in a concrete derivation, consider the LDA program of [`docs/examples/source/lda.qvr`](../examples/source/lda.qvr):

```
program lda (alpha : Real, beta : Real) : Word -> Word { ... }
```

The judgment we want to derive is

$$
\Gamma \;\vdash\; \mathsf{lda} \,:\, (\alpha : \mathsf{Real},\, \beta : \mathsf{Real}) \Rightarrow \mathsf{Word} \rightsquigarrow \mathsf{Word}.
$$

Working bottom-up from $\textsc{Prog}$:

1. The parameter context $\Delta = (\alpha : \mathsf{Real},\, \beta : \mathsf{Real})$ is well-formed because $\mathsf{Real}$ is a scalar universe.
2. $\Gamma \vdash \mathsf{Word} : \ast_{\mathrm{FinSet}}$ by $\textsc{TyVar}$ (assuming $\mathsf{Word}$ is a previously-declared finite-set object).
3. The body's statements are typed by $\textsc{Bind}$ (the `sample theta`, `sample phi` sites), $\textsc{Marginalize}$ (the `marginalize z { observe w }` block), and the trailing $\textsc{Seq}$ rule. The trace context starts at $\Phi_0 = \mathsf{Word}$ (the program's declared input) and grows through the successive sample sites: $\Phi_1 = \Phi_0, \theta : \mathrm{Doc \to Topic}$ after the first sample; $\Phi_2 = \Phi_1, \phi : \mathrm{Topic \to Word}$ after the second; the marginalize block's body extends $\Phi_2$ with $z : \mathrm{Topic}$ internally, then $w$ is observed (no trace extension) and $z$ is integrated out, leaving the post-marginalize trace at $\Phi_2$ again.
4. The return expression `return theta` checks against the declared codomain $\mathsf{Word}$ by $\textsc{TraceVar}$ followed by the appropriate projection.

The conclusion $\Gamma \vdash \mathsf{lda} : (\alpha : \mathsf{Real}, \beta : \mathsf{Real}) \Rightarrow \mathsf{Word} \rightsquigarrow \mathsf{Word}$ is the judgment derived by the typechecker. The REPL's `:type` prints the *type* of `lda` in the GHCi convention, where the constraint context names only the universes (not the binders, which are implementation detail):

```
:type lda
lda :: (Real, Real) => Word -> Word
```

The bidirectional algorithm of §[10](#10-algorithmic-typechecking) carries this derivation out automatically. The dependent Π over $\Delta$ surfaces in the context to the left of the constraint arrow $\Rightarrow$; the parameter *names* are accessible through `:info lda` (and through the `ProgramDecl`'s `type_params` AST field) but are absent from the rendered type because they are bound by the Π and so are not free in the surface type.

## Bibliography

The categorical apparatus is standard; in addition to the references in [Setting](setting.md), the type-theoretic presentation here draws on:

* Pierce, B. *Types and Programming Languages*. MIT Press, 2002. (For bidirectional typechecking and decidability arguments.)
* Hofmann, M., Streicher, T. "The groupoid interpretation of type theory." *Twenty-five years of constructive type theory*, 1998. (For the categorical semantics of dependent products.)
* Staton, S., Yang, H., Heunen, C., Kammar, O., Wood, F. "Semantics for probabilistic programming: higher-order functions, continuous distributions, and soft constraints." *LICS 2016*. [https://doi.org/10.1145/2933575.2935313](https://doi.org/10.1145/2933575.2935313). (For the Kleisli-arrow semantics of bind, observe, and score statements.)
* Vákár, M., Kammar, O., Staton, S. "A domain theory for statistical probabilistic programming." *POPL 2019*. [https://doi.org/10.1145/3290349](https://doi.org/10.1145/3290349). (For the categorical setting of measure-theoretic probability with score.)
* Cho, K., Jacobs, B. "Disintegration and Bayesian inversion via string diagrams." *Mathematical Structures in Computer Science*, 2019. [https://doi.org/10.1017/S0960129518000488](https://doi.org/10.1017/S0960129518000488). (For the Markov-category-with-conditioning setting of $\textsc{Observe}$.)
* Fritz, T. "A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics." *Advances in Mathematics*, 2020. [https://doi.org/10.1016/j.aim.2020.107239](https://doi.org/10.1016/j.aim.2020.107239). (For the axiomatic-Markov-category framework underlying the $\mathbf{Kern}$ denotation.)
