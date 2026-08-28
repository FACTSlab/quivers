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
| Root | `program` |

Edges record membership in the module, declaration-to-value bindings, product components, generators, morphism domains and codomains, and the selected output. Constraints record metadata such as names, cardinalities, dimensions, families, roles, and bounds.

## 2. Extraction

`extract_program_schema(compiler)` reads a populated `Compiler` environment and returns a `panproto.Schema` with protocol name `qvr_program`. `extract_deduction_schema(compiler)` performs the analogous extraction for deduction structures.

The writer caches emitted runtime objects by Python identity. This avoids collapsing two equal-looking component occurrences into one vertex when repeated edges would otherwise be lost under panproto's edge-set semantics.

Extraction is deterministic for the cases covered by `tests/test_program_theory.py`, and every example schema in that test validates against `QVR_PROGRAM_PROTOCOL`.

## 3. What a schema records

The schema records static shape and selected declaration metadata. It does not contain learned tensors, observations, optimizer state, distribution objects, or the executable bodies needed to reconstruct a kernel.

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
5. recompiling the same example produces schemas with the same recorded structure.

These checks establish the extractor's current shape contract. They do not establish functoriality, naturality, or behavioral equivalence.
