# A Type Theory for QVR

This chapter develops the **type theory** of the QVR DSL as a formal proof system, paired with the denotational interpretation already given in the surrounding chapters. The denotational semantics (chapters [Setting](setting.md) through [Adequacy](adequacy.md)) tells us *what* a well-typed QVR phrase means: an object of $\mathbf{FinSet}$ or $\mathbf{SBor}$, a morphism in $\mathbf{Stoch}$, a Markov kernel in $\mathbf{Kern}$, a Π-indexed family of Kleisli arrows, and so on. The present chapter gives the **type system** that picks out *which* phrases are well-typed in the first place: judgments, contexts, inference rules, and the soundness theorem that ties the two layers together.

The development is organised so that:

* every judgment form is given with its denotational interpretation, so the proof theory and the model theory stay synchronised;
* every inference rule is justified by an appeal to the categorical structure already exhibited in the denotational chapters;
* the soundness theorem (Theorem [§7.1](#71-soundness)) is the precise statement under which the REPL's `:type` and `:kind` commands are guaranteed to report mathematically meaningful answers.

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
| $\ast_{\mathrm{Sort}}$ | item sorts of a deduction signature | small fibre of a generalised algebraic theory |
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

The variable cases distinguish a bound name $x$ (introduced inside a program body by a bind / let statement) from a free morphism name $f$ (introduced at module scope by a `morphism`, `program`, `let`, or `export` declaration). The contraction-call form $c(\bar y)$ (`ExprMorphismCall`) applies a registered $n$-ary contraction $c$ to morphism-scope names $\bar y$ inside a `let`-binding initialiser (see [Composition rules §4](composition-rules.md)).

**Program instantiation is not an expression form.** Surface program calls $P(\bar a)$ live in the program-body sub-language as the family slot of a `DrawStep`: a statement $v \leftarrow P(\bar a)$ with $P$ a program template is interpreted by inlining the template's body, as described in [Programs §3a](programs.md#3a-parametric-programs). Consequently the typing rule for parametric instantiation is presented at the statement level (§[6.8](#68-template-inlining)), not as a morphism-expression rule.

Sequential composition $\mathbin{\diamond_\alpha}$ is parameterised by a choice of enrichment algebra $\alpha$: the QVR surface syntax exposes one operator per algebra, including `>>` (`ProductFuzzy` noisy-or, the default), `<<` (reversed `ProductFuzzy`), `>=>` (Kleisli composition for the operands' shared algebra), `*>` (Markov sum-product), `~>` (`LogProb`), `||>` (Gödel), `?>` (Viterbi), `&&>` (Boolean), and `+>` (Łukasiewicz); see [Composition rules](composition-rules.md). The tensor `@` denotes the symmetric monoidal product $\otimes$ ([Morphisms §2](morphisms.md)). The dagger $e^\dagger$ is the compact-closed dual, $\mathsf{trace}$ is the categorical trace, $\mathsf{cup}/\mathsf{cap}$ are the unit / counit of the compact-closed structure, and $.\mathsf{change\_base}(\varphi)$ is the base-change functor between algebras. The remaining combinators $\mathsf{fan}, \mathsf{repeat}, \mathsf{stack}, \mathsf{scan}, \mathsf{freeze}, \mathsf{from\_data}, \mathsf{parser}, \mathsf{chart\_fold}, \mathsf{curry}$ are explained in [Expressions](expressions.md); we present typing for the core fragment ($\diamond$, $@$, $\mathsf{id}$, $\mathsf{fan}$, $\mathsf{repeat}$, and program instantiation) in §[5](#5-inference-rules-for-morphism-expressions) below and refer to [Expressions](expressions.md) for the derived combinators.

Notably absent from the expression sub-language are first-class projections $\pi_i$ and injections $\mathsf{inl}, \mathsf{inr}$: products are introduced and eliminated implicitly through tuple-pattern bindings in the program-body sub-language (§[1.5](#15-statements-program-body-sub-language)), and the discrete coproduct is reached through declared `morphism` arrows initialised from data rather than through dedicated combinators. Projections appear in the *meta-language* of the denotation (e.g. $\pi_i : \llbracket A_1 \times \cdots \times A_k \rrbracket \to \llbracket A_i \rrbracket$) but not in the surface syntax.

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

where $p$ is a parameter name and $P$ ranges over the parameter universes listed in [Programs §3a](programs.md#3a-parametric-programs):

| $P$ | Universe |
|---|---|
| $\ast_{\mathrm{FinSet}}$, $\ast_{\mathrm{Space}}$, $\ast_{\mathrm{Atom}}$ | an object of the relevant sub-category |
| $\mathsf{Scalar}_\mathbb{R}$, $\mathsf{Scalar}_\mathbb{N}$ | a scalar hyperparameter |
| $\mathsf{Mor}[A, B]$ | a kernel $A \to \mathcal{G}(B)$ |

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
\frac{\Gamma \vdash \tau_1 : \kappa_1 \quad \Gamma \vdash \tau_2 : \kappa_2}{\Gamma \vdash \tau_1 \times \tau_2 : \kappa_1 \sqcup \kappa_2}\ \textsc{TyProd}
\qquad
\frac{\Gamma \vdash \tau_1 : \ast_{\mathrm{FinSet}} \quad \Gamma \vdash \tau_2 : \ast_{\mathrm{FinSet}}}{\Gamma \vdash \tau_1 + \tau_2 : \ast_{\mathrm{FinSet}}}\ \textsc{TySum}
$$

The product rule uses the kind-join $\sqcup$:

$$
\ast_{\mathrm{FinSet}} \sqcup \ast_{\mathrm{FinSet}} = \ast_{\mathrm{FinSet}}
\qquad
\ast_{\mathrm{FinSet}} \sqcup \ast_{\mathrm{Space}}
= \ast_{\mathrm{Space}} \sqcup \ast_{\mathrm{FinSet}}
= \ast_{\mathrm{Space}}
\qquad
\ast_{\mathrm{Space}} \sqcup \ast_{\mathrm{Space}} = \ast_{\mathrm{Space}}
$$

implementing the discrete-component absorption discussed in [Resolution §1.3](types-and-spaces.md#13-mixed-products). Coproducts are restricted to the discrete sub-language because $\mathbf{SBor}$ has no general finite coproducts that play well with the Giry monad.

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

There is no general kind subsumption rule. The only implicit coercion is the absorption built into $\sqcup$: in a mixed product $\sigma_{\mathrm{FinSet}} \times \tau_{\mathrm{Space}}$, the discrete factor is implicitly indicator-embedded into the ambient continuous space ($[n] \hookrightarrow \mathbb{R}^n$), so that the product lands in $\mathbf{SBor}$. The compiler realises this through the resolution dispatch in [`_resolve_any_space`](../api/dsl/compiler/resolution.md): a `ProductSet` lifts to a `ProductSpace` whenever any component is a `ContinuousSpace`. No standalone embedding combinator exists in the surface syntax.

## 4. Inference rules for families

The family-typing judgment is $\Gamma \vdash F : \mathsf{Family}[\Theta, B]$. Every family in the registry comes with a fixed pair $(\Theta_F, B_F)$, so the rule is uniform:

$$
\frac{F \in \mathsf{FamReg}\ \text{with parameter}\ \Theta_F\ \text{and value space}\ B_F}{\Gamma \vdash F : \mathsf{Family}[\Theta_F, B_F]}\ \textsc{FamReg}
$$

A *family application* $F(a_1, \ldots, a_k)$ is typed by checking that the actual argument tuple $\bar a$ matches the family's parameter shape: the parameters are usually nested in a tuple structure $\Theta_F = R_1 \times \cdots \times R_k$ over a rig $R_i$, so each $a_i$ must inhabit the corresponding scalar slot.

$$
\frac{\Gamma \vdash F : \mathsf{Family}[R_1 \times \cdots \times R_k,\, B]
       \qquad \Gamma; \Phi \vdash a_i : R_i \quad (1 \le i \le k)}
      {\Gamma; \Phi \vdash F(\bar a) : \mathsf{Kernel}[\Phi, B]}\ \textsc{FamApp}
$$

Here $\mathsf{Kernel}[\Phi, B]$ is the kind of Kleisli arrows $\Phi \to \mathcal{G}(B)$, and the parameter arguments may depend on the trace context $\Phi$ (the family becomes a kernel parameterised by the current trace).

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

$$
\frac{\Gamma; \Phi \vdash e_1 : A \rightsquigarrow B \qquad \Gamma; \Phi \vdash e_2 : B \rightsquigarrow C}
     {\Gamma; \Phi \vdash e_1\,{>\!\!>}\,e_2 : A \rightsquigarrow C}\ \textsc{Compose}
$$

This is Kleisli composition $\diamond$ in the underlying Markov category. Soundness (Theorem [§7.1](#71-soundness)) collapses the syntactic $>\!\!>$ onto $\diamond$.

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

Identity is the only "structural" combinator with a dedicated expression form (`ExprIdentity` for a typed identity at a specific object, plus the un-annotated `id` form whose target is synthesised from context). Products and coproducts have no first-class projection or injection combinators; their introduction and elimination is handled by:

* the program-body sub-language's tuple-pattern bind $(v_1, \ldots, v_m) \leftarrow F(\bar a)$ (§[6.2](#62-destructuring-bind)), which both introduces a product (the family's codomain) and eliminates it (extending the trace by all $m$ component variables);
* bare-identifier program headers `program P (q₁, …, qₖ) : A → B`, which add the projections $q_i = \pi_i^A$ to the body's initial trace context $\Phi_0$ (§[7.2](#72-bare-identifier-projection-programs));
* the data-initialised morphism declaration `morphism f : A → B ~ from_data(...)`, which compiles to a concrete tensor / kernel realising whatever projection or injection the user encoded in the data.

### 5.5 Higher combinators

$\mathsf{fan}, \mathsf{repeat}, \mathsf{stack}, \mathsf{scan}$ are syntactic sugar for derived expressions; their typing is determined by their unfolding in [Expressions §3–§5](expressions.md). For example:

$$
\frac{\Gamma; \Phi \vdash e_i : A \rightsquigarrow B_i \quad (1 \le i \le n)}
     {\Gamma; \Phi \vdash \mathsf{fan}(e_1, \ldots, e_n) : A \rightsquigarrow B_1 \times \cdots \times B_n}\ \textsc{Fan}
$$

$$
\frac{\Gamma; \Phi \vdash e : A \rightsquigarrow B \quad n \in \mathbb{N}_{\ge 1}}
     {\Gamma; \Phi \vdash \mathsf{repeat}(e, n) : A^n \rightsquigarrow B^n}\ \textsc{Repeat}
$$

with similar shapes for $\mathsf{stack}$ (axis-1 broadcast) and $\mathsf{scan}$ (sequential fold).

### 5.6 Contraction application

A contraction $c$ registered in the value context with arity $k$ and inputs $A_1, \ldots, A_k$ producing a codomain $B$ may be applied to morphism-scope names $y_1, \ldots, y_k$ in a `let`-binding initialiser:

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

The trace context is unchanged because the observed value is externally supplied; the statement contributes only a score factor. The denotation lives in the unnormalised Giry sub-monad $\mathcal{G}_{\le 1}$ (see [Programs §2.2](programs.md#22-observe)).

### 6.4 Marginalize

$$
\frac{\Gamma; \Phi \vdash F(\bar a) : \mathsf{Kernel}[\Phi, B]
       \qquad \Gamma; \Phi, v : B \vdash s_1; \ldots; s_n \dashv \Phi'}
     {\Gamma; \Phi \vdash \mathsf{marginalize}\ v \leftarrow F(\bar a)\,\{\, s_1; \ldots; s_n\,\} \dashv \Phi'\setminus v}\ \textsc{Marginalize}
$$

The variable $v$ is bound *inside* the marginalize body but does not appear in the resulting trace context: marginalization is precisely the pushforward $\pi_{\Phi *}$ that eliminates $v$ from the joint kernel (see [Programs §2.5](programs.md#25-marginalize)).

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
\frac{\Gamma \vdash A : \kappa_A
       \qquad \Gamma \vdash B : \kappa_B
       \qquad \Gamma; \Delta \vdash A\ \mathsf{ok}\ \text{and}\ B\ \mathsf{ok}
       \qquad \Gamma, \Delta; A \vdash s_1; \ldots; s_n \dashv \Phi_n
       \qquad \Gamma, \Delta; \Phi_n \vdash e : \Phi_n \rightsquigarrow B}
     {\Gamma \vdash \mathsf{program}\ P\ (\Delta)\ :\ A \to B\ \{\,s_1; \ldots; s_n; \mathsf{return}\ e\,\} : (\Delta) \Rightarrow A \rightsquigarrow B}\ \textsc{Prog}
$$

The rule reads bottom-up: a program is well-typed when (i) the declared signature $A \to B$ is kind-correct in any extension of $\Gamma$ by $\Delta$, (ii) the body, threaded through the trace contexts $\Phi_0 = A$ through $\Phi_n$, is a valid statement sequence, and (iii) the return expression projects $\Phi_n$ onto $B$.

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
| $\ast_{\mathrm{Sort}}$ | fibre of the signature's generalised algebraic theory |
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
* *if $\Gamma; \Phi \vdash e : A \rightsquigarrow B$ then $\llbracket e \rrbracket$ is defined and $\llbracket e \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\llbracket \Phi \rrbracket \times \llbracket A \rrbracket,\, \mathcal{G}(\llbracket B \rrbracket))$;*
* *if $\Gamma; \Phi \vdash s \dashv \Phi'$ then $\llbracket s \rrbracket$ is defined and $\llbracket s \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\llbracket \Phi \rrbracket,\, \mathcal{G}(\llbracket \Phi' \rrbracket))$;*
* *if $\Gamma \vdash p : (\Delta) \Rightarrow A \rightsquigarrow B$ then $\llbracket p \rrbracket$ is defined and $\llbracket p \rrbracket \in \prod_{\delta : \llbracket \Delta \rrbracket} \mathrm{Hom}_{\mathbf{Kern}}(\llbracket A \rrbracket,\, \mathcal{G}(\llbracket B \rrbracket))$.*

**Proof.** By mutual induction on the derivation. We give the four key cases; the remaining cases are similar.

*Case $\textsc{Compose}$.* By induction, $\llbracket e_1 \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\Phi \times A,\, \mathcal{G}(B))$ and $\llbracket e_2 \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\Phi \times B,\, \mathcal{G}(C))$. Kleisli composition $\diamond$ in $\mathbf{Kern}$ (extended over the trace argument by the obvious pre-composition with the duplication map $\Delta_\Phi$) lands in $\mathrm{Hom}_{\mathbf{Kern}}(\Phi \times A,\, \mathcal{G}(C))$, which is the required signature. Total composition is well-defined because $\mathbf{Kern}$ is a category, established in [Morphisms §2](morphisms.md).

*Case $\textsc{Bind}$.* By induction, $\llbracket F(\bar a) \rrbracket \in \mathrm{Hom}_{\mathbf{Kern}}(\Phi,\, \mathcal{G}(B))$. The denotation of the bind statement is the kernel

$$
\mathcal{S}\llbracket v \leftarrow F(\bar a) \rrbracket(\phi, B' \times C)
\;=\;
\mathbf{1}_{B'}(\phi) \cdot \llbracket F(\bar a) \rrbracket(\phi)(C),
$$

a measurable map $\Phi \to \mathcal{G}(\Phi \times B)$ on measurable rectangles (extended to the full $\sigma$-algebra by Caratheodory). This is exactly the conclusion of $\textsc{Bind}$.

*Case $\textsc{Marginalize}$.* By induction, the body denotes a Kleisli arrow $\llbracket \Phi, v : B \rrbracket \to \mathcal{G}(\llbracket \Phi' \rrbracket)$. Pre-composing with $\llbracket F(\bar a) \rrbracket : \Phi \to \mathcal{G}(B)$ and post-composing with the projection $\pi_{\Phi' \setminus v}$ (which integrates out the $v$ coordinate) gives a kernel $\Phi \to \mathcal{G}(\Phi' \setminus v)$, which is the statement's required signature. The categorical content is that $\mathbf{Kern}$ has all (countable) coproducts and that $\pi_*$ is the corresponding fold.

*Case $\textsc{Prog}$.* Let $\Delta = p_1 : P_1, \ldots, p_k : P_k$. For each $\delta \in \llbracket \Delta \rrbracket$, the assumption "$\Gamma, \Delta; A \vdash s_1; \ldots; s_n \dashv \Phi_n$" produces (by induction, weakening $\delta$ into $\Gamma$) a kernel $\llbracket \Phi_0[\delta] \rrbracket \to \mathcal{G}(\llbracket \Phi_n[\delta] \rrbracket)$. Composing with the return arrow $\Phi_n[\delta] \to \mathcal{G}(B[\delta])$ gives the desired fibrewise kernel. The dependent product over $\delta$ is total because each fibre is well-defined.

$\square$

### 9.2 Subject reduction (parametric instantiation)

**Theorem (Subject reduction).** *Let $P : (\Delta) \Rightarrow A \rightsquigarrow B$ with $\Delta = p_1 : P_1, \ldots, p_k : P_k$, and let $\bar a = a_1, \ldots, a_k$ be a tuple of expressions with $\Gamma; \Phi \vdash a_i : P_i$. Then the inline-expanded program*

$$
P[\bar a / \bar p]
$$

*satisfies $\Gamma; \Phi \vdash P[\bar a / \bar p] : A[\bar a / \bar p] \rightsquigarrow B[\bar a / \bar p]$, and*

$$
\llbracket P[\bar a / \bar p] \rrbracket = \llbracket P \rrbracket (\llbracket \bar a \rrbracket).
$$

**Proof sketch.** Substitution commutes with the body's denotation function $\mathcal{B}\llbracket \cdot \rrbracket$ because $\mathcal{B}\llbracket \cdot \rrbracket$ is defined compositionally on the syntactic structure of the body, and each clause is closed under syntactic substitution of free parameters. The α-renaming step (used by inline expansion to give fresh names to internal latents) is sound because $\mathcal{B}\llbracket \cdot \rrbracket$ depends only on the multiset of bound-variable types, not on the names. The detailed argument is the substitution lemma of [Programs §3a](programs.md#3a-parametric-programs). $\square$

### 9.3 Equivalence and conservativity

Two well-typed phrases $\phi_1, \phi_2$ at the same judgment are *denotationally equivalent*, written $\phi_1 \equiv \phi_2$, when $\llbracket \phi_1 \rrbracket = \llbracket \phi_2 \rrbracket$. The QVR type system is *conservative* in the following sense: for any judgment form $\mathcal{J}$, if $\phi_1$ and $\phi_2$ are $\mathcal{J}$-derivable and denotationally equal, the type system does not assign them distinguishable types (no judgment $\mathcal{J}'$ separates them). This conservativity is the categorical content of the "every well-typed phrase has a unique denotation" reading of [Adequacy](adequacy.md).

## 10. Algorithmic typechecking

The inference rules of §[3](#3-inference-rules-for-types-and-kinds)–§[7](#7-inference-rules-for-programs) are *declarative*: they specify which judgments are derivable, not how to derive them. The implementation in `src/quivers/dsl/compiler/` realises a *bidirectional* algorithm with two modes:

| Mode | Read as |
|---|---|
| **Check** $\Gamma; \Phi \vdash e \mathrel{\Leftarrow} A \rightsquigarrow B$ | given $e$ and the expected signature $A \rightsquigarrow B$, verify the rule's premises |
| **Synth** $\Gamma; \Phi \vdash e \mathrel{\Rightarrow} A \rightsquigarrow B$ | given $e$, compute the (unique) signature $A \rightsquigarrow B$ |

The compiler synthesises from leaves upward: variables and module-level names look up declared signatures; composition and tensor synthesise from their operands; program instantiation synthesises through substitution. The compiler checks at boundaries where the expected signature is fixed by the surrounding context: morphism declarations (the right-hand side of `~` must check against the declared $A \to B$), program declarations (the return expression must check against the declared codomain), and statement composition (each statement's $\Phi'$ must match the next statement's input $\Phi$).

The relevant entry points in the implementation are:

* [`_resolve_any_space`](../api/dsl/compiler/resolution.md) — type-formation judgment $\Gamma \vdash \tau : \kappa$, returning the denotational object.
* [`_compile_expr`](../api/dsl/compiler/expressions.md) — morphism judgment $\Gamma; \Phi \vdash e \mathrel{\Rightarrow} A \rightsquigarrow B$, returning a record carrying the resolved domain and codomain.
* [`_compile_program`](../api/dsl/compiler/programs.md) — program judgment, including the per-statement $\Phi$ chain tracked by `ChainShape`.

### 10.1 Decidability

**Proposition.** *The fragment of the type system excluding the residuated formers and effect-typed expressions ($\mathsf{TySlashR}, \mathsf{TySlashL}, \mathsf{TyEff}$) is decidable: there is an algorithm that, given $\Gamma$ and a phrase $\phi$, returns either a derivation of $\Gamma \vdash \phi : \cdot$ or a proof that no such derivation exists.*

**Proof sketch.** Type formation is structurally recursive over the syntax of $\tau$ and bottoms out at primitive constructors or context lookup. Morphism and statement typing are bidirectional and also structurally recursive over the term / statement, with no rule that introduces a metavariable requiring search. Program-instantiation substitution is straightforward syntactic substitution. The only source of unbounded computation would be a recursive program definition; QVR forbids recursive `program` declarations, so all derivations are finite. $\square$

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
3. The body's statements are typed by $\textsc{Bind}$ (the `sample theta`, `sample phi` sites), $\textsc{Marginalize}$ (the `marginalize z { observe w }` block), and the trailing $\textsc{Seq}$ rule. The trace context grows from $\Phi_0 = \mathsf{Doc}$ (the bare-identifier projection of the input) through the successive sample sites.
4. The return expression `return theta` checks against the declared codomain $\mathsf{Word}$ by $\textsc{TraceVar}$ followed by the appropriate projection.

The conclusion $\Gamma \vdash \mathsf{lda} : (\alpha : \mathsf{Real}, \beta : \mathsf{Real}) \Rightarrow \mathsf{Word} \rightsquigarrow \mathsf{Word}$ is exactly the string the REPL prints for `:type lda`:

```
:type lda
lda :: (alpha : Real, beta : Real) => Word -> Word
```

The bidirectional algorithm of §[10](#10-algorithmic-typechecking) carries this derivation out automatically, with the dependent Π over $\Delta$ surfacing in the context to the left of the constraint arrow $\Rightarrow$ and the kernel signature surfacing to the right.

## Bibliography

The categorical apparatus is standard; in addition to the references in [Setting](setting.md), the type-theoretic presentation here draws on:

* Pierce, B. *Types and Programming Languages*. MIT Press, 2002. (For bidirectional typechecking and decidability arguments.)
* Hofmann, M., Streicher, T. "The groupoid interpretation of type theory." *Twenty-five years of constructive type theory*, 1998. (For the categorical semantics of dependent products.)
* Staton, S., Yang, H., Heunen, C., Kammar, O., Wood, F. "Semantics for probabilistic programming: higher-order functions, continuous distributions, and soft constraints." *LICS 2016*. [https://doi.org/10.1145/2933575.2935313](https://doi.org/10.1145/2933575.2935313). (For the Kleisli-arrow semantics of bind, observe, and score statements.)
* Vákár, M., Kammar, O., Staton, S. "A domain theory for statistical probabilistic programming." *POPL 2019*. [https://doi.org/10.1145/3290349](https://doi.org/10.1145/3290349). (For the categorical setting of measure-theoretic probability with score.)
* Cho, K., Jacobs, B. "Disintegration and Bayesian inversion via string diagrams." *Mathematical Structures in Computer Science*, 2019. [https://doi.org/10.1017/S0960129518000488](https://doi.org/10.1017/S0960129518000488). (For the Markov-category-with-conditioning setting of $\textsc{Observe}$.)
* Fritz, T. "A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics." *Advances in Mathematics*, 2020. [https://doi.org/10.1016/j.aim.2020.107239](https://doi.org/10.1016/j.aim.2020.107239). (For the axiomatic-Markov-category framework underlying the $\mathbf{Kern}$ denotation.)
