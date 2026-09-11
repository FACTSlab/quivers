# Implementation Correspondence and Limits

This page records the **implementation-correspondence claim** (ICC): the compiler should construct the runtime object described by the semantics pages for each supported QVR phrase. The ICC is a specification tested on representative cases. It is not a mechanized adequacy theorem for the whole language.

## 1. What the compiler returns

`Compiler` resolves a parsed module into an environment of objects, morphisms, programs, deductions, structural components, and exports. A successful compilation does not always return one `Program`, and `Program.forward` does not in general return a probability distribution. Depending on the compiled artifact, evaluation may return a tensor, a sample, a dictionary of returned values, or another runtime wrapper.

Thus correspondence must be stated per artifact:

| Phrase | Runtime object | Evidence-bearing operation |
|---|---|---|
| `object` | `SetObject` or `ContinuousSpace` | structural equality, shape, membership where implemented |
| discrete `morphism` | `Morphism` | `.tensor` |
| family morphism | `ContinuousMorphism` | `.rsample`, `.log_prob` |
| `program` | `MonadicProgram` or compiled program wrapper | sampling and `.log_joint` |
| `deduction` | `DeductionSystem` | agenda/chart evaluation |
| structural declaration | encoder, decoder, signature, or loss object | component-specific method |

## 2. Direct correspondences

Several clauses are close enough to their implementations to check pointwise.

First, object resolution is a structural walk over `ObjectExpr`. Nested products and coproducts are flattened without reordering components. Continuous constructors dispatch through the compiler's constructor table.

Second, a discrete latent morphism stores a raw parameter and exposes `sigmoid(raw)` as its tensor. An observed morphism exposes its supplied tensor. Composition calls the active composition rule's `tensor_op` and `join`; tensor product applies `tensor_op` pointwise.

Third, the expression combinators have distinct operational contracts:

| Combinator | Current implementation |
|---|---|
| `.marginalize(X, ...)` | reduces named **codomain** axes with the active algebra's `join` |
| `fan(f, g, ...)` | feeds one input to each component and concatenates outputs |
| `repeat(f, n)` | sequentially composes the same instance, sharing parameters |
| `stack(f, n)` | sequentially composes deep copies, giving each layer independent parameters |
| `scan(cell)` | iterates a continuous cell `A * H -> H` over an implicit time axis and returns the final state |

These last two distinctions matter: `stack` is not tensor power, and `scan` is not `.trace`.

## 3. Probabilistic composition

Sampling from `f >> g` is ancestral: sample the intermediate value with `f.rsample`, then pass it to `g.rsample`. This produces samples from the composite kernel when the component samplers implement their declared kernels.

Density evaluation is different. `SampledComposition.log_prob` draws a fixed number of intermediate samples, evaluates the second density, and combines the values with `logsumexp - log(n)`. The density-scale average is a Monte Carlo estimator of the integral; taking its logarithm introduces finite-sample bias. The default sample count is 100. Thus the returned log density is an approximation, not pointwise equality with the Chapman–Kolmogorov integral.

`ScanMorphism.log_prob` currently returns zeros for the final state, while `log_joint(x, hidden_states)` scores a supplied full hidden trajectory. A semantics that requires the marginal density of the final recurrent state is thus not implemented by `log_prob`.

## 4. Programs and data

Program statements preserve their leading-keyword distinction:

- `sample` introduces a random value;
- `observe` adds a family log density at supplied data;
- `let` binds deterministic tensor arithmetic;
- `score` adds an explicitly computed scalar to the log joint;
- `marginalize` introduces a scoped latent and removes it from the outer scope;
- `return` selects the program result.

`from_data("key")` is resolved through data bound to the compiler before expression compilation. It should not be described as a learnable value or as an unbound runtime lookup.

## 5. What the tests establish

The repository contains focused evidence rather than one exhaustive theorem:

- compiler and DSL tests exercise parsing, resolution, typing errors, expression composition, contractions, programs, and structural declarations;
- `tests/test_program_theory.py` checks extraction and validation of program-shape schemas;
- `tests/test_model_roundtrips.py` checks JSON round trips for `dx.Model` values;
- `tests/transpile/test_structural.py`, `test_lens_laws.py`, `test_external_syntax.py`, and `test_numeric_equivalence.py` test the separate transpilation pipeline described in [Transpilation correctness](transpile-correctness/index.md).

Passing these tests supports the covered cases. It does not establish equality for every phrase, algebra, distribution parameterization, backend, input, or floating-point value.

## 6. Conditions and open gaps

The ICC must be qualified in five places.

1. Categorical equations that use distributivity, arbitrary joins, or compact closure require those laws of the active algebra. The `Algebra` class does not prove them.
2. Floating-point operations may differ from exact real arithmetic, especially near support boundaries and saturated reductions.
3. Monte Carlo density calculations are approximate at finite sample counts.
4. Family registries state support and parameter maps operationally; registry membership alone does not prove measurability, normalization, or equivalence to an external library.
5. Schema extraction records program shape. Equality of extracted schemas is not equality of program behavior or parameter values.

These limits identify concrete work: add property tests for each claimed algebraic law, compare probabilistic compositions against analytic cases, and connect each declarative typing rule to a compiler test. Until then, the semantics pages should distinguish implemented behavior, test evidence, and conditional mathematical interpretation.
