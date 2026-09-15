# The Program-Shape Protocol

`quivers.dsl.program_theory` provides a schema-level view of a compiled QVR environment. The word *theory* here names a panproto protocol, not a proof that schema equality captures program behavior.

## 1. `QVR_PROGRAM_PROTOCOL`

`QVR_PROGRAM_PROTOCOL` reuses panproto's `ThBratSchema` and `ThBratInstance` shape-graph theories with QVR-specific kinds, constraints, and edge rules.

The principal vertex groups are:

| Group | Representative kinds |
|---|---|
| Discrete values | `finset`, `product_set`, `coproduct_set`, `free_monoid`, `empty_set`, `enum_set`, `free_residuated` |
| Continuous values | `euclidean`, `simplex`, `positive_reals`, `product_space` |
| Declarations | `object_decl`, `space_decl`, morphism-role declarations, `output_decl`, `schema_decl` |
| Kernel declarations | `effect_decl`, `operation`, `instance_decl`, `handler_decl`, `handler_clause`, `computation_decl`, `computation_parameter` |
| Entry points | `program_entry`, `program_parameter`, `program_site` |
| Root | `program` |

Edges record membership in the module, declaration-to-value bindings, product components, generators, morphism domains and codomains, and the selected output. Constraints record metadata such as names, cardinalities, dimensions, families, roles, and bounds.

The kernel group mirrors the checked [`QiecModule`](../api/qiec/module.md) the compiler holds as [`Compiler.qiec_module`](../api/dsl/compiler.md): one vertex per effect interface and its operations, per lexical instance, per handler and its clauses, and per named computation with its parameters. Edges record which interface an instance applies (`instance_of`), which interface a handler handles (`handles`), which instances a computation's declared row names (`row`), and what a body does: the computations it calls (`calls`), the instances it performs requests on (`performs`), and the handlers it installs (`installs`). A `program` declaration elaborates to a computation plus a `program_entry` vertex whose `body` edge names it, whose `parameter` children carry each parameter's role (domain, scalar, data, observation, fibration, weight, bias, kernel-input), whose `site` children carry each probabilistic step's kind, family, and plate axes, and whose `random` and `score` edges name the canonical instances its samples and scores address. A module whose programs the elaboration cannot yet lower carries the diagnostic on the root as a `gap` constraint.

Because a program and a computation that calls it are vertices of the same kind joined by the same `calls` edge, a mixed module yields one call graph rather than a morphism graph beside a computation graph.

## 2. Extraction

`extract_program_schema(compiler)` reads a populated `Compiler` environment and returns a `panproto.Schema` with protocol name `qvr_program`. `extract_deduction_schema(compiler)` performs the analogous extraction for deduction structures.

The writer caches emitted runtime objects by Python identity. This avoids collapsing two equal-looking component occurrences into one vertex when repeated edges would otherwise be lost under panproto's edge-set semantics.

Extraction is deterministic for the cases covered by `tests/test_program_theory.py`, and every example schema in that test validates against `QVR_PROGRAM_PROTOCOL`.

## 3. What a schema records

The schema records static shape, selected declaration metadata, and the call, request, and handler structure of every checked body. It does not contain learned tensors, observations, optimizer state, distribution objects, or the term-level bodies needed to reconstruct a kernel; those live in the checked module's serialization.

Thus two practical implications follow.

First, a nonempty `panproto.diff_schemas(a, b)` identifies a structural difference between extracted environments. The test suite checks this on distinct example programs.

Second, equal extracted schemas do not imply equal program behavior. Two compilations may share all recorded vertices, edges, and constraints while carrying different parameter values or executable functions.

## 4. Migration scope

Panproto schema operations may consume the extracted shape, but this module does not itself define `auto_lens`, prove lens laws, migrate `.qvr` source, or implement an evaluator from `(Schema, parameters)` back to a QVR kernel. Those are separate operations and require their own validation.

In particular, renaming a schema vertex does not by itself rename every reference in source text or preserve learned state. Source migration is handled by the grammar migration tooling, while parameter migration needs an explicit value-level policy.

## 5. Evidence

`tests/test_program_theory.py` checks that:

1. every current example produces a validating `qvr_program` schema;
2. extracted schemas contain a `program` root;
3. selected object and output metadata are recorded;
4. structurally distinct examples produce a nonempty diff;
5. recompiling the same example produces schemas with the same recorded structure;
6. a module holding only a `program` yields its computation, entry point, sites, and canonical instances;
7. computations and programs calling one another form a single call graph;
8. changing one site's family changes the extracted schema.

These checks establish the extractor's current shape contract. They do not establish functoriality, naturality, or behavioral equivalence.
