# The Program Theory

The denotational semantics of the previous pages assigns each well-typed QVR module a morphism in $\mathbf{Kern}$ (or one of its sub-strata). This page introduces a *second*, schema-level denotation: every compiled module also lifts to a panproto `Schema` over a fixed protocol $\mathsf{QVR}$. The two interpretations are connected by a structure-preserving map, and the schema-level interpretation is what powers diff/migrate/lens-generation tooling on `.qvr` programs.

## 1. The protocol $\mathsf{QVR}$

The constant `QVR_PROGRAM_PROTOCOL` defined in [`quivers.dsl.program_theory`](../api/dsl/program_theory.md) is a panproto protocol: a pair of generalized algebraic theories $(\mathcal{T}_{\mathrm{schema}}, \mathcal{T}_{\mathrm{instance}})$ together with vertex- and edge-kind declarations. Its content is summarized below.

### 1.1 Vertex kinds

Vertex kinds enumerate the runtime *value* layer:

| Group | Vertex kinds |
|-------|--------------|
| Discrete objects | `finset`, `product_set`, `coproduct_set`, `free_monoid`, `empty_set`, `enum_set`, `free_residuated` |
| Continuous spaces | `euclidean`, `simplex`, `positive_reals`, `product_space` |
| Declarations | `object_decl`, `space_decl`, `morphism_decl`, `kernel_decl`, `discretize_decl`, `embed_decl`, `output_decl`, `schema_decl` |
| Module root | `program` |

Each vertex carries a string label (typically the declared identifier or a synthesized key) and the kind-specific payload (cardinality, dimension, family name, …).

### 1.2 Edge kinds

Edges encode structural relations:

| Edge kind | Source | Target | Meaning |
|-----------|--------|--------|---------|
| `decl` | `program` | any `*_decl` | Declaration is part of the module |
| `binds_to` | `object_decl`, `space_decl` | object / space vertex | Declared name binds to its semantic value |
| `component` | `product_set`, `coproduct_set`, `product_space` | object / space vertex | Position in a finite tuple |
| `generators` | `free_monoid`, `free_residuated` | `finset`, `enum_set` | Underlying alphabet |
| `domain` | morphism / kernel / discretize / embed `_decl` | object / space vertex | Domain of the declaration |
| `codomain` | morphism / kernel / discretize / embed `_decl` | object / space vertex | Codomain of the declaration |
| `output` | `program` | `output_decl` | The module's public export |

The schema and instance theories of $\mathsf{QVR}$ are panproto's `ThBratSchema` and `ThBratInstance` (the shape-graph theories used by the brat protocol family); the QVR protocol reuses them and specialises via the kind enumerations and edge rules above.

## 2. The extraction functor

The function `extract_program_schema` in [`quivers.dsl.program_theory`](../api/dsl/program_theory.md) realizes a *functor*

$$
\mathcal{S} : \mathrm{Modules}_{\mathrm{compiled}} \to \mathrm{Schema}(\mathsf{QVR}),
$$

from the category of well-typed, compiled QVR modules (with module morphisms given by environment-preserving renamings) to the category of schemas over $\mathsf{QVR}$.

Concretely, $\mathcal{S}$ walks the resolved environment $\rho_{\mathrm{obj}} \cup \rho_{\mathrm{spc}} \cup \rho_{\mathrm{mor}}$ produced by the compiler and emits one vertex per binding. The walk is *id-based*: two structurally-equal `dx.Model` instances declared under different identifiers produce two distinct vertices (sharing structural payload but with distinct labels), reflecting the user's intent that distinct declarations are distinct schema elements.

## 3. Two denotations, one module

For a compiled module $M$ we therefore have two denotations:

$$
\llbracket M \rrbracket_{\mathrm{kern}} \in \mathbf{Kern}\bigl(\llbracket \tau_{\mathrm{in}} \rrbracket, \llbracket \tau_{\mathrm{out}} \rrbracket\bigr),
\qquad
\mathcal{S}(M) \in \mathrm{Schema}(\mathsf{QVR}).
$$

These are related by a *forgetful* projection: $\mathcal{S}(M)$ enumerates the *generators* of $\llbracket M \rrbracket_{\mathrm{kern}}$, but does not record the numerical content of any morphism (no tensor data, no learned parameters). Two modules $M_1, M_2$ with $\mathcal{S}(M_1) = \mathcal{S}(M_2)$ have *isomorphic shape*, but their kernel denotations may differ on the parameter values.

## 4. Diff, migrate, and lens generation

Because $\mathcal{S}(M) \in \mathrm{Schema}(\mathsf{QVR})$, every panproto operation on schemas is automatically available on `.qvr` modules:

- $\mathrm{diff} : \mathrm{Schema}(\mathsf{QVR})^2 \to \mathrm{Patch}(\mathsf{QVR})$: produce a structural diff between two modules;
- $\mathrm{auto\_lens} : \mathrm{Schema}(\mathsf{QVR})^2 \to \mathrm{Lens}(\mathsf{QVR})$: derive a bidirectional migration lens between two module versions;
- $\mathrm{check} : \mathrm{Schema}(\mathsf{QVR})^2 \times \mathrm{Lens}(\mathsf{QVR}) \to \mathrm{Bool}$: check that a candidate lens satisfies the GetPut/PutGet round-trip laws.

These operate on the *shape* of a module (its declarations and the relations between them) and not on its parameter values. Numerical-content migration, e.g.\ initialising the parameters of a renamed `latent` morphism from those of the original, is a separate concern, handled by panproto's *field-transform* layer ([Field Transforms](https://panproto.dev/skills/panproto-field-transforms.html)) using `compute-field` / `coerce-type` rules.

## 5. Naturality

The extraction $\mathcal{S}$ is *natural* in the following sense. Let $\phi : \rho \to \rho'$ be a renaming of the resolved environment that preserves structural payload (changes only identifiers). Then there is an induced renaming $\phi^{*} : \mathcal{S}(M_{\rho}) \to \mathcal{S}(M_{\rho'})$ on the level of schemas, and $\mathcal{S}$ commutes with this action:

$$
\mathcal{S}(M_{\phi(\rho)}) \;=\; \phi^{*}\bigl(\mathcal{S}(M_{\rho})\bigr).
$$

This naturality is what licenses the use of panproto's auto-lens generation on QVR modules: the lens search may freely rename vertices, knowing that the underlying denotation is preserved up to the canonical relabelling.

## 6. Connection to the kernel denotation

There is an *evaluation functor*

$$
\mathrm{eval} : \mathrm{Schema}(\mathsf{QVR}) \times \Theta \to \mathbf{Kern}
$$

that takes a schema and a parameter assignment $\theta \in \Theta$ (where $\Theta$ collects the parameters of all `latent` and `kernel` declarations in the schema) and produces the corresponding kernel. The denotation $\llbracket M \rrbracket_{\mathrm{kern}}$ is recovered by

$$
\llbracket M \rrbracket_{\mathrm{kern}} \;=\; \mathrm{eval}\bigl(\mathcal{S}(M),\ \theta_M\bigr),
$$

where $\theta_M$ is the parameter assignment realized in the compiled module's PyTorch state. This factorization is the formal statement that the schema layer captures the *type structure* and the kernel layer captures the *parameter content*.

The implementation does not currently expose `eval` as a stand-alone operation; it is implicit in the compiler's tensor-construction pass.
