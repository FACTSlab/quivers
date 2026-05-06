# Adequacy

The implementation in `quivers.dsl.compiler.Compiler` is a *concrete realisation* of the denotational semantics. This page states and sketches a proof of the adequacy theorem connecting the two.

## 1. The compiler as a function

The compiler is a function

$$
\mathcal{C} : \mathrm{AST} \to \mathrm{Program} \cup \{\bot\},
$$

producing a `quivers.Program` (an `nn.Module`) on success and raising `CompileError` (denoted $\bot$) on type-error or undefined-name failure. A `quivers.Program` is in turn a *runnable representation* of a kernel: invoking $\mathrm{Program}.\mathrm{forward}$ produces, for each input, a value or a sample distribution.

We define the *operational denotation* of a successfully-compiled module $M$ as

$$
\mathcal{C}\llbracket M \rrbracket : \llbracket \tau_{\mathrm{in}} \rrbracket \to \mathcal{G}\bigl(\llbracket \tau_{\mathrm{out}} \rrbracket\bigr),
\qquad
\mathcal{C}\llbracket M \rrbracket(x) \;=\; \text{distribution induced by } \mathrm{Program}.\mathrm{forward}(x).
$$

For deterministic (purely $\mathcal{V}$-enriched) modules this collapses to a Dirac kernel.

## 2. Statement of adequacy

**Theorem (Adequacy).** Let $M$ be a well-typed QVR module and let $\theta_M$ be the parameter assignment realised in the compiled program. Then

$$
\mathcal{C}\llbracket M \rrbracket \;=\; \llbracket M \rrbracket
\quad \text{in } \mathbf{Kern},
$$

where $\llbracket M \rrbracket$ is the denotation of [Setting](setting.md), [Morphisms](morphisms.md), [Expressions](expressions.md), and [Programs](programs.md), instantiated at the parameters $\theta_M$.

Equality is in $\mathbf{Kern}$ — i.e.\ as Markov kernels — and is therefore equality of measures on the $\sigma$-algebra of the codomain. In the implementation this equality holds *up to the floating-point precision* of the underlying tensor library; the test suite asserts the equality to within numerical tolerance on a representative set of modules.

## 3. Proof sketch

The proof proceeds by structural induction on the module $M$. We outline the induction at each layer.

### 3.1 Types and spaces

For every `TypeExpr` $\tau$ and every well-formed environment $\rho_{\mathrm{obj}}$, the lens `TypeExprToSetObject(env)` satisfies

$$
\mathrm{forward}(\tau) \;=\; \llbracket \tau \rrbracket_{\rho_{\mathrm{obj}}}
$$

by definition: the recursion in `_resolve` is term-by-term equal to the recursion in [Types §2](types-and-spaces.md#2-denotation-of-types). The $\mathrm{backward}$ direction recovers the original AST verbatim, so the GetPut and PutGet laws hold by construction. The same argument applies to `SpaceExprToContinuousSpace`.

### 3.2 Discrete morphisms

A `latent` declaration creates a learnable PyTorch parameter $W \in \mathbb{R}^{|X| \times |Y|}$ and applies the constraint map $\sigma : \mathbb{R} \to V$ pointwise. The denotation $\llbracket f \rrbracket(x, y) = \sigma(W[x, y])$ is exactly the value computed by `LatentMorphism.tensor`. An `observed` declaration's denotation is the user-supplied tensor verbatim.

### 3.3 Composition

The `>>` operator in PyTorch is realised as $V$-quantale matrix multiplication (`noisy_or_contract` for $\mathcal{V}_{\mathrm{pf}}$, ordinary matrix multiplication for $\mathcal{V}_{\mathbb{B}}$ over `BoolTensor`, etc.). The composition of two tensors $(M_1, M_2)$ at indices $(x, z)$ is

$$
\bigoplus_{y} \sigma(M_1[x, y]) \otimes \sigma(M_2[y, z]),
$$

term-by-term equal to the categorical composition of [Morphisms §1.1](morphisms.md#11-composition-tensor-identity).

### 3.4 Tensor product, marginalisation, fan, stack, repeat, scan

Each combinator is implemented as a tensor-level operation whose definition is the term-by-term unfolding of its denotation. The proofs are straightforward: `@` is `torch.einsum` of the appropriate shape; `marginalize` is `torch.sum` (or quantale-join) along the marginalised axes; `fan` is `torch.stack`; `stack` is `kron`; `repeat` is repeated composition; `scan` is a Python-level fold realising the trace of [Expressions §3.4](expressions.md#34-scan).

### 3.5 Stochastic morphisms

Softmax along the codomain axis realises the parameter map $\mathbb{R}^{|Y|} \to \Delta^{|Y| - 1}$. The composition of two stochastic morphisms is implemented as ordinary matrix multiplication, which coincides with Kleisli composition in $\mathbf{Stoch}$ ([Morphisms §2](morphisms.md#2-stochastic-morphisms)).

### 3.6 Continuous morphisms

A `continuous f : σ -> σ' ~ Family` declaration is realised as a PyTorch `Distribution` parameterised by a neural network mapping $\llbracket \sigma \rrbracket$ to $\Theta_{\mathrm{Family}}$. Sampling and density evaluation are exposed by `rsample` and `log_prob`. Composition is realised by *sampled-composition* (Monte-Carlo integration) or, for closed-form families, by the appropriate analytical convolution.

The equality

$$
\int_{\llbracket \sigma' \rrbracket} g_2(t, C)\, g_1(s, \mathrm{d}t) \;=\; \mathbb{E}_{T \sim g_1(s, \cdot)}\bigl[g_2(T, C)\bigr]
$$

between the Chapman–Kolmogorov composition and its Monte-Carlo realisation holds in expectation; the `quivers` runtime uses reparameterisation to obtain unbiased gradient estimates.

### 3.7 Programs

The body-interpreter `_compile_program_body` of [`quivers.dsl.compiler`](../api/dsl/compiler.md) realises the Kleisli chain of [Programs §2](programs.md#2-statements). Each statement is interpreted as a Python operation that:

- *Draw* — calls `family.rsample(theta(context))` and appends the result to the trace;
- *Observe* — calls `family.log_prob(value, theta(context))` and accumulates the score;
- *Let* — evaluates the arithmetic expression and binds the result;
- *Return* — projects the trace onto the named coordinates.

The categorical equations of [Programs §5](programs.md#5-soundness-of-monadic-semantics) are inherited from PyTorch's distribution / random-variable algebra, which is well-known to satisfy the Giry-monad axioms.

### 3.8 Schema extraction

The schema extractor `extract_program_schema` walks the same resolved environment used by the compiler. Its denotation is exactly the schema-level interpretation of [The Program Theory §2](program-theory.md#2-the-extraction-functor); the equality $\mathcal{S}(M) = \mathrm{extract\_program\_schema}(\mathrm{Compiler}(M))$ holds by construction.

## 4. Tests as adequacy witnesses

The test suite at `tests/test_resolution_lenses.py` and `tests/test_program_theory.py` asserts the following propositions, each a special case of the adequacy theorem.

| Test file | Assertion |
|-----------|-----------|
| `test_resolution_lenses.py` | $\mathrm{forward}(\tau) = \llbracket \tau \rrbracket$ and $\mathrm{forward}(\sigma) = \llbracket \sigma \rrbracket$ on every example program. |
| `test_resolution_lenses.py` | GetPut and PutGet laws hold on every example. |
| `test_resolution_lenses.py` | The lens-driven resolution agrees with the legacy compiler dispatch on every example. |
| `test_program_theory.py` | $\mathcal{S}(M)$ validates against $\mathsf{QVR}$ on every example. |
| `test_program_theory.py` | $\mathcal{S}(M_1) \neq \mathcal{S}(M_2)$ when $M_1, M_2$ differ in declarations. |
| `test_program_theory.py` | Idempotence: $\mathcal{S}(M)$ recomputed twice yields equal schemas. |
| `test_model_roundtrips.py` | Every `dx.Model` instance round-trips through `model_dump_json` / `model_validate_json`. |

The tests are not a substitute for a fully formal proof, but they constitute a finite set of adequacy witnesses across the language's surface syntax, verified on every commit.

## 5. Limitations

The adequacy theorem holds *up to floating-point precision* and *modulo the chosen Monte-Carlo seeds* for sampled compositions in $\mathbf{Kern}$. The semantic equality is between *kernels* (probability distributions); the implementation realises samples and gradient estimates of those kernels, with an error that decays with the number of Monte-Carlo samples and is irrelevant in the limit.

The four tensor-bearing accumulators (`Presheaf`, `Weight`, `SampleSite`, `Trace`) carry mutable runtime state and are not part of the value layer; the adequacy theorem does not assign them denotations. They are scratch space for inference algorithms; their semantics is given operationally by the inference layer and is outside the scope of this document.

## References

- Joyal, A., Street, R., and Verity, D. (1996). *Traced Monoidal Categories*. Mathematical Proceedings of the Cambridge Philosophical Society, 119(3), 447–468.
- Giry, M. (1982). *A Categorical Approach to Probability Theory*. Lecture Notes in Mathematics, 915, 68–85.
- Cho, K. and Jacobs, B. (2019). *Disintegration and Bayesian inversion via string diagrams*. Mathematical Structures in Computer Science, 29(7), 938–971.
- Fritz, T. (2020). *A synthetic approach to Markov kernels, conditional independence and theorems on sufficient statistics*. Advances in Mathematics, 370, 107239.
- Kelly, G. M. (1982). *Basic Concepts of Enriched Category Theory*. Cambridge University Press.
- Lawvere, F. W. (1973). *Metric Spaces, Generalized Logic, and Closed Categories*. Rendiconti del Seminario Matematico e Fisico di Milano, 43(1), 135–166.
