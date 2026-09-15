# QVR indexed-effects unification backlog

## Purpose

This is the implementation backlog for completing the indexed-family and
algebraic-effect work on `feat/qiec-language-integration`. It is a handoff
document, not a description of released behavior. The branch is not complete
while probabilistic `program` declarations and QIEC computations lower through
parallel semantic paths.

The target is the **single-core invariant**:

> Every executable QVR declaration is checked as a typed QIEC computation, and
> every backend receives that computation through one canonical compiler
> boundary. Surface forms such as `sample`, `observe`, `score`, and
> `marginalize` remain concise model notation, but elaborate to the same effect
> operations, handler applications, calls, and pure terms that users may write
> directly.

The inference engine remains a backend concern. QVR describes models and
model transformations; Pyro, Stan, PyMC, and the other targets continue to run
their own sampling, optimization, or exact-inference machinery. Algebraic
effects must organize the model definition and its interpretations. They must
not move Monte Carlo loops into authored QVR.

## Handoff snapshot

- Branch: `feat/qiec-language-integration`.
- Last committed revision: `0e2770a` (`docs(changelog): describe final 0.19
  behavior`).
- Pull request: #59.
- The green checks attached to `0e2770a` certify only that committed revision.
  They do not certify the uncommitted working tree or the work in this backlog.
- Project version: `0.19.0` in `pyproject.toml` and `uv.lock`.
- Required upstream packages are present: Didactic `0.15.0`, Panproto `0.74.2`,
  and `panproto-grammars-all` `0.74.2` are locked. Didactic's public indexed-
  family API is already used by `src/quivers/dsl/qiec_lowering.py`. Panproto
  `0.74.2` contains the historical-object validation correction requested for
  the migration gate.
- The working tree is intentionally dirty. An interrupted cleanup removed the
  discarded qualifiers from parts of the grammar, QVR AST, parser, emitter,
  QIEC kernel, serialization, IR, tooling tests, documentation, and changelog.
  Preserve and finish those edits. Do not reset, stash-pop, or regenerate
  historical object hashes.
- The current tree has no QIEC computation-call term, no recursion, no scoped
  instance allocation, and no authored handler-clause body.
- The existing `program` compiler does not elaborate `sample`, `observe`,
  `score`, or `marginalize` to QIEC. `IRProgram.body` and `IRProgram.qiec` are
  parallel payloads.
- The dynamic-target QIEC renderer grafts standalone `qiec_*` entry points onto
  ordinary model output. That is a useful prototype of the runtime ABI, but it
  is not the final unified lowering.
- The current built-in `Random` and `Score` definitions are closer to the
  desired core than the first source examples suggest: `Random.sample[A]`
  accepts a typed site and sampleable, and `Score.add` accepts a log weight.
  The missing work is to make the QVR model surface construct those values and
  elaborate through those operations.

Before modifying anything, run:

```bash
git status --short --branch
git diff --check
git diff --stat
```

Read the full uncommitted diff. The edits belong to this branch and must not be
discarded merely because they are unfinished.

## Non-negotiable completion criteria

The implementation is complete only when every item below is true.

1. There is one executable typed core. An ordinary probabilistic program and a
   directly authored effectful computation become the same core declaration
   before target-specific lowering.
2. `IRProgram.qiec` is gone. No renderer walks an ordinary model body and then
   independently grafts a second computation module onto the result.
3. Named computations can call other named computations with checked static
   and value arguments. Direct and mutual recursion are supported by the
   checker, reference evaluator, serializer, and every admitting backend.
4. Handler clauses can have authored bodies. Their operation arguments,
   continuation, handler parameters, local values, and return clause are typed.
   Runtime attachments remain available for foreign or optimized handlers, but
   they are no longer the only place executable handler behavior can exist.
5. Lexical effect instances can be allocated inside a computation. Their stable
   identities are scope-derived, and neither the instance nor a row containing
   it may escape its scope.
6. Existing `program` syntax stays useful. Its probabilistic steps elaborate to
   the canonical `Random` and `Score` interfaces and standard handlers. Users
   are not required to spell a free-monad encoding to define a model.
7. Indexed values and GADT case refinement may occur inside real model code,
   including computations that sample, score, call helpers, and invoke
   handlers. They are not confined to pure demonstration entry points.
8. The Python compiler and reference runtime execute the unified model. A QVR
   program can call an effectful helper, and an effectful helper can invoke a
   model computation when its declared type permits it.
9. All eleven transpilers consume the same checked core input. A target either
   emits semantics-preserving code or produces a source-located capability
   diagnostic. It never erases a call, effect, handler, recursion edge, index
   refinement, or program step.
10. The CLI, REPL, TUI, language server, formatter, Pygments lexer, tree-sitter
    query, VS Code extension, and Zed extension understand all new syntax and
    all new declaration/reference relationships.
11. The QVR grammar history and packaged parser artifacts are regenerated. The
    `v0.18.0 -> HEAD` Panproto migration succeeds without changing any
    historical content-addressed object.
12. The discarded effect-declaration qualifiers have no parser rule, AST field,
    core field, IR field, serialized field, highlighter entry, example,
    compatibility reader, deprecation path, or changelog narrative. This was
    unreleased work, so there is nothing to preserve.
13. The version and changelog describe the final `0.19.0` behavior. The
    changelog does not list corrections to intermediate branch implementations
    as user-visible fixes.
14. Local gates and every required PR check are green on the final pushed
    commit. A green run on an earlier SHA does not satisfy this criterion.

## Required architecture

### One core, two lowering stages

Use two stages with a strict ownership boundary:

1. **QVR elaboration** converts every executable source form to checked QIEC.
   This stage owns name resolution, kinding, GADT refinement, effect-row
   inference, handler typing, call typing, lexical instance identity, source
   provenance, and desugaring of model notation.
2. **Target lowering** converts checked QIEC to a backend-specific execution
   plan. A PPL target may recover optimized model nodes for canonical
   `Random.sample` and `Score.add` operations. A general host-language target
   may use the free-computation runtime for user-defined effects. Static graph
   targets may monomorphize or reject constructs according to explicit
   capabilities.

The existing probabilistic IR may remain as a target plan during the
transition, but it must be produced *from checked QIEC*. It must not remain an
independent source lowering. A useful final split is:

- `IRQiecModule`: canonical, typed, target-independent module IR;
- `IRPplPlan`: optional backend-oriented plan derived from one selected entry
  computation; and
- renderer schema: target syntax emitted from the target plan and residual
  host-runtime terms.

Do not keep the current meaning of `IRProgram`, in which `body` is one program
and `qiec` is unrelated executable material.

### Surface design

Keep the existing declaration families, but give them one meaning.

- `define NAME[...] (...) : RESULT !ROW =` declares a typed computation.
- A computation application is an inline computation form, not a host-language
  call. Prefer ordinary `NAME[STATIC...](VALUE...)` syntax if tree-sitter can
  distinguish it from an effect request and a constructor. If an explicit
  keyword is needed for an unambiguous first implementation, use one spelling
  everywhere and migrate to ordinary application before release. Do not ship
  two equivalent call syntaxes.
- All top-level computation signatures are collected before bodies are
  checked. This makes forward calls and mutual recursion module-scoped by
  construction; no special `rec` declaration is required.
- `program NAME ...` remains domain-specific notation for a named computation
  with generated typed parameters, generated lexical instances, and a
  generated standard-handler boundary. Its body elaborates rather than
  compiling through `MonadicProgram` first.
- A pure `let name = EXPRESSION` binds a serializable pure expression. A
  monadic `let name <- COMPUTATION` binds a computation result. They must have
  separate AST nodes and typing rules.
- Handler declarations acquire a return clause and operation clauses with
  bound operation parameters and a typed continuation. The resumption grade is
  retained as part of each clause contract and checked against the clause body.
- Add a scoped instance form, for instance `with instance cell : State[Int] in
  ...`. The final spelling must be fixed in the grammar corpus and language
  reference before implementation fans out to tooling.
- GADT constructor patterns continue to introduce rigid branch variables and
  equality evidence. Pattern-bound values and static variables must be visible
  inside calls, pure expressions, effect requests, and handler bodies.

### Canonical probabilistic meaning

The built-in probabilistic interfaces are the canonical meaning of model
steps:

- `Random.sample[A](site : Site[A], sampleable : Sampleable[A]) : A` denotes a
  named latent request.
- `Score.add(weight : LogWeight) : Unit` denotes addition to the current model
  score.
- `sample x <- Family(args)` elaborates to construction of a typed site and
  sampleable followed by `perform random.sample[...]`.
- `observe y <- Family(args)` binds `y` from the model's typed observation/data
  parameters and performs `Score.add(log_prob(Family(args), y))`. If a backend
  has a native observed-site construct, target lowering may fuse this canonical
  pair only after checking.
- `score name = expr` evaluates the pure expression, binds `name`, and performs
  `Score.add(name)`.
- `marginalize` elaborates to the standard enumeration handler over the scoped
  random request and score accumulation. Grouped/fibred marginalization must
  preserve its `over`, `via`, plate, and reduction data in typed core terms or
  typed primitive arguments. It must not remain an unrelated `IRMarginalize`
  source path.
- Existing effect combinators in `src/quivers/effects/` become standard
  handlers or handler combinators over the same requests. There must not be a
  second event/request representation for tracing, conditioning, replay,
  intervention, blocking, masking, scaling, reparameterization, or collapse.

`Site[A]`, `Sampleable[A]`, and `LogWeight` need source-constructible,
serializable representations. A Python distribution object may still appear at
the runtime attachment boundary, but it cannot be the canonical compiler IR.
Represent a distribution structurally with its family identifier, typed
arguments, event type, support constraint, plate shape, and provenance. Target
lowering turns that structure into `Normal(...)`, `dist.Normal(...)`, or the
corresponding backend form.

### Pure expression layer

Real models need more values than variables, scalars, and GADT constructors.
Unify the current `LetExprNode` functionality with QIEC instead of embedding
Python callables.

The stable pure layer must cover, at minimum:

- numeric, Boolean, string, unit, tuple, and list literals;
- variables and typed data references;
- tuple construction and projection;
- tensor indexing and named-axis gathering;
- arithmetic, comparison, Boolean, and supported tensor primitives;
- family/distribution construction;
- `log_prob` as a typed primitive used by observation elaboration;
- plate/shape-aware broadcast and reduction;
- GADT construction and case scrutinees;
- pure named-function calls, represented through the same stable computation
  call when the callee has an empty row; and
- attachment references only at an explicit foreign-value boundary.

Every pure primitive needs a stable identifier, an arity/type rule, an IR node,
a serializer entry, a reference evaluator implementation, a capability tag,
and renderer support or refusal. Do not serialize arbitrary callee strings and
resolve them dynamically in a backend.

### Calls and recursion

Add stable `ComputationId` values and make `NamedComputation` carry one. A call
node contains:

- the callee ID and display name;
- closed or lexically scoped static arguments;
- checked value arguments;
- the instantiated result type;
- the instantiated effect row; and
- source provenance for diagnostics and traces.

The checker must build a signature table before checking bodies. Compute the
call graph and its strongly connected components. Within an SCC:

- every computation has an explicit result type and effect row;
- calls instantiate the callee telescope capture-free;
- arguments are checked against instantiated parameter types;
- the caller's inferred row includes the instantiated callee row;
- branch-local skolems may be passed to a callee while in scope but may not
  appear in an escaping result or row;
- scoped instances may be used by a callee only while their binder is live;
- duplicate stable IDs and ambiguous display names are rejected; and
- general recursion is allowed to diverge. Do not claim termination.

The reference evaluator should execute calls with an explicit call frame in
the CEK machine. Avoid relying on unbounded Python recursion. Trace events must
include the stable call site and a dynamic call frame so two recursive
iterations do not collide. Add an optional execution-step/fuel limit for the
CLI and tests; exhaustion is a stable execution diagnostic, not a Python
`RecursionError`.

### Authored handlers

Replace signature-only source handlers with stable handler definitions that
can carry bodies. A handler needs:

- static parameters and ordinary value parameters;
- the handled interface and input/output computation types;
- coverage, introduced row, and unknown-operation behavior;
- a return clause;
- one typed operation clause per authored operation;
- binders for operation static arguments and value arguments;
- a typed resumption local whose function type includes the operation result,
  handler answer type, and residual row;
- an optional explicit state/context value represented in stable terms; and
- source provenance for the declaration, return clause, operation clauses, and
  resumption calls.

Add core terms for resumption invocation and any handler-local state access
that the chosen surface requires. Check grades statically by counting paths
through the clause body where possible:

- grade `0`: no invocation;
- affine: at most one invocation on every path;
- linear: exactly one invocation on every terminating path;
- unrestricted: any number of invocations, subject to duplicability rules.

Keep the dynamic grade guard in the evaluator and generated host runtimes. The
static analysis is a rejection boundary; the runtime guard protects foreign
handlers and compiler bugs.

Foreign handlers remain explicit runtime-provider entries keyed by stable
handler ID. A declaration must choose authored or foreign implementation; an
absent authored body must not silently mean “look up an arbitrary callback.”
Provider manifests must validate that their attached handler signature equals
the checked declaration.

### Scoped instances

Top-level `instance` declarations remain module-scoped lexical instances. Add a
computation term for local allocation. Its binder carries a stable
`EffectInstanceId` derived from module, computation, lexical path, and applied
interface. The checker must:

- extend the row/name environment only within the body;
- reject shadowing that makes qualified operation resolution ambiguous;
- prevent the instance ID from escaping in the result type, a returned value,
  a row variable constraint, or a stored closure;
- preserve distinct identities for repeated allocations reached through
  recursion by combining the static ID with a dynamic address frame; and
- expose the binder to completion, hover, references, rename, and semantic
  highlighting only within its lexical range.

## Sequenced work packages

The identifiers below are stable handoff labels. A commit may cover more than
one small task, but no work package is complete until all of its acceptance
criteria pass.

### QVR-000 — Preserve and complete the interrupted declaration cleanup

**Depends on:** nothing.

**Primary files:**

- `grammars/qvr/grammar.js`
- `grammars/qvr/test/corpus/qiec.txt`
- `src/quivers/dsl/ast_nodes/qiec.py`
- `src/quivers/dsl/ast_nodes/__init__.py`
- `src/quivers/dsl/parser/qiec.py`
- `src/quivers/dsl/emit.py`
- `src/quivers/dsl/qiec_lowering.py`
- `src/quivers/qiec/types.py`
- `src/quivers/qiec/effects.py`
- `src/quivers/qiec/checking.py`
- `src/quivers/qiec/coverage.py`
- `src/quivers/qiec/substitution.py`
- `src/quivers/qiec/builtins.py`
- `src/quivers/qiec/serialization.py`
- `src/quivers/qiec/__init__.py`
- `src/quivers/transpile/qiec_ir.py`
- `src/quivers/transpile/runtime_qiec.py`
- QIEC-focused tests, editor grammars, docs, and both changelog copies.

**Tasks:**

- Review every existing uncommitted hunk and finish the same structural
  deletion in files not yet touched.
- Remove the discarded declaration-option grammar nodes rather than accepting
  and ignoring them.
- Derive an effect ID from module and nominal effect name. Derive operation IDs
  from the effect ID and operation name. Static interface arguments distinguish
  applications but do not change the declaration identity.
- Ensure `EffectDef.apply`, equality, hashing, row unification, operation
  ownership, substitution, serialization, and IR conversion use the nominal
  identity consistently.
- Keep handler-level partial coverage and unknown-operation passing behavior;
  these are separate handler semantics.
- Replace tests that used the deleted syntax to trigger unrelated diagnostics
  with direct tests of the relevant error, such as duplicate operation names.
- Remove obsolete highlighter vocabulary and comments from TextMate,
  tree-sitter queries, Zed, Pygments registration, docs hooks, and release
  fixtures.
- Rewrite the changelog as final behavior. Do not add a `Fixed` item about an
  implementation that never shipped.

**Acceptance:**

- A plain effect header round-trips through parse and emit.
- Effect, operation, row, handler, module, and IR serialization tests pass.
- A repository-wide audit finds no field, rule, or example for the discarded
  declaration qualifiers.
- No historical Panproto object or released grammar snapshot changed.

### QVR-010 — Freeze executable surface syntax with golden examples

**Depends on:** QVR-000.

**Primary files:**

- `grammars/qvr/grammar.js`
- `grammars/qvr/test/corpus/qiec.txt`
- `docs/semantics/grammar.md`
- a new `tests/fixtures/qiec/` source corpus.

**Tasks:**

- Fix one syntax for computation application, pure binding, handler return and
  operation clauses, resumption invocation, scoped instances, and any explicit
  foreign-handler declaration.
- Resolve precedence between type application, constructor application,
  computation application, effect requests, and ordinary morphism expressions.
- Specify indentation and newline behavior for nested call, handle, case,
  local-instance, and marginalization bodies.
- Add positive tree-sitter corpus cases for every nesting combination.
- Add negative corpus cases for missing rows, malformed telescopes, ambiguous
  applications, duplicate clauses, and out-of-scope instance use.
- Create four golden source modules before implementing downstream passes:
  recursive indexed traversal, authored state handler, an effectful
  probabilistic helper called by a model, and a grouped-marginal model using a
  GADT-coded observation state.

**Acceptance:**

- `tree-sitter test` passes.
- Every golden source has one unambiguous parse tree.
- `parse(emit(parse(source)))` equals `parse(source)` for all golden modules.
- The grammar reference contains no syntax that lacks an AST task below.

### QVR-020 — Extend the source AST and tree walker

**Depends on:** QVR-010.

**Primary files:**

- `src/quivers/dsl/ast_nodes/qiec.py`
- `src/quivers/dsl/ast_nodes/program_steps.py`
- `src/quivers/dsl/ast_nodes/declarations.py`
- `src/quivers/dsl/ast_nodes/__init__.py`
- `src/quivers/dsl/parser/qiec.py`
- `src/quivers/dsl/parser/program_steps.py`
- `src/quivers/dsl/parser/statements.py`
- `src/quivers/dsl/emit.py`
- `src/quivers/dsl/qiec_tooling.py`

**Tasks:**

- Add explicit tagged-union variants for computation calls, pure expressions,
  resumption calls, local instances, authored handler bodies, handler return
  clauses, and operation patterns.
- Do not encode new variants with optional fields on unrelated nodes.
- Carry 1-based source line and column on each new node and binder.
- Reuse one stable pure-expression AST. If `LetExprNode` is generalized, remove
  redundant QIEC literal/variable wrappers once all callers migrate. Do not
  retain two isomorphic value trees.
- Make `program` steps elaborate-friendly: preserve site name, family,
  arguments, result constraint, plate axes, observation source, `over`, `via`,
  and reduction without consulting compiled Torch objects.
- Extend binder/reference traversal in `qiec_tooling.py` for call targets,
  handler parameters, resumptions, scoped instances, and case-bound names.
- Add model serialization/round-trip tests for every AST variant.

**Acceptance:**

- Every grammar field is consumed exactly once by a typed walker.
- Every AST variant emits canonical QVR.
- No parser code manufactures runtime callbacks, Torch values, or backend
  names.

### QVR-030 — Add stable call, expression, handler-body, and local-instance terms

**Depends on:** QVR-020.

**Primary files:**

- `src/quivers/qiec/identifiers.py`
- `src/quivers/qiec/terms.py`
- `src/quivers/qiec/effects.py`
- `src/quivers/qiec/module.py`
- `src/quivers/qiec/declarations.py`
- `src/quivers/qiec/substitution.py`
- `src/quivers/qiec/serialization.py`
- `src/quivers/qiec/__init__.py`
- `src/quivers/transpile/qiec_ir.py`

**Tasks:**

- Add `ComputationId` and stable IDs for pure primitives where required.
- Give each `NamedComputation` an ID and make calls point to it.
- Add core/IR nodes for computation call, pure primitive application,
  distribution construction, data reference, resumption invocation, local
  instance allocation, and authored handler clauses.
- Make all recursive structures encodable as finite trees with references to
  declarations by stable ID. Do not inline callee bodies into call nodes.
- Extend capture-avoiding substitution through every new type, row, value,
  handler, and computation field.
- Extend the serialization allowlist and strict decoder validation. Reject
  unknown tags, missing fields, extra fields, malformed IDs, non-finite
  literals, and illegal host objects.
- Update canonical hash/ID derivation tests so source locations do not change
  nominal declaration IDs but do distinguish request/call sites.
- Update the Didactic models in `qiec_ir.py` one-for-one. There must be no opaque
  JSON field for a new core node.

**Acceptance:**

- Every new record survives deterministic dump/load and QIEC-to-IR conversion.
- Recursive call graphs serialize because calls are references, not nested
  declaration objects.
- The stable boundary still rejects Python callables and mutable resources.

### QVR-040 — Implement whole-module name resolution and recursive checking

**Depends on:** QVR-030.

**Primary files:**

- `src/quivers/dsl/qiec_lowering.py`
- `src/quivers/qiec/checking.py`
- `src/quivers/qiec/module.py`
- `src/quivers/qiec/coverage.py`
- `src/quivers/qiec/kinds.py`
- `src/quivers/qiec/types.py`

**Tasks:**

- Split lowering into declaration collection, signature elaboration, and body
  elaboration. Predeclare all effect, family, constructor, handler, instance,
  and computation signatures before lowering bodies.
- Resolve computation names nominally and diagnose duplicate or ambiguous
  names at their authored locations.
- Instantiate computation telescopes and parameter types at each call.
- Infer the call's instantiated row and result; union/unify it with the caller's
  row under existing open-row/lacks rules.
- Construct the computation call graph and SCCs. Recheck each SCC against its
  declared signatures after all bodies are available.
- Check pure primitives against a central typed registry shared with compiler
  and renderers.
- Type handler return clauses, operation patterns, clause bodies, and
  resumptions. Validate coverage against the exact applied interface.
- Add grade-use analysis over returns, binds, calls, cases, nested handlers, and
  resumptions. Be conservative across recursive calls and open branches.
- Check local-instance scope and non-escape properties.
- Preserve GADT branch refinement through calls and pure expressions. Add tests
  where an impossible branch contains an effect and confirm it is excluded only
  when equality evidence proves impossibility.
- Add stable diagnostic codes for call resolution, call arity, recursion/SCC
  mismatch, handler-body typing, resumption misuse, and local-instance escape.

**Acceptance:**

- Direct recursion, mutual recursion, polymorphic calls, calls under case
  refinement, and calls under handlers all type-check in valid fixtures.
- Wrong static arity, wrong value arity, kind mismatch, result mismatch, missing
  row entry, skolem escape, instance escape, and grade violation all fail with
  source-located stable diagnostics.
- Module validation independently rechecks the result without trusting the QVR
  lowerer.

### QVR-050 — Implement CEK evaluation for calls and authored handlers

**Depends on:** QVR-040.

**Primary files:**

- `src/quivers/qiec/evaluator.py`
- `src/quivers/qiec/execution.py`
- `src/quivers/qiec/builtins.py`
- `src/quivers/transpile/runtime_qiec.py`
- `src/quivers/transpile/runtime_qiec.js`
- `src/quivers/transpile/runtime_qiec.jl`
- `src/quivers/transpile/runtime_qiec.scm`

**Tasks:**

- Add explicit call and return frames to the evaluator machine.
- Bind specialized static arguments and value parameters without mutating the
  stored declaration.
- Support recursive and tail-recursive calls without consuming Python stack per
  QIEC step.
- Add a configurable step budget and `qiec-run-fuel` diagnostic.
- Evaluate stable pure primitives through a closed registry.
- Evaluate authored handler clauses in the same machine as ordinary
  computations; bind operation arguments and the resumption local.
- Preserve deep-handler semantics when an authored clause resumes.
- Keep foreign-handler dispatch as an explicit alternate implementation.
- Allocate local instances with static identity plus dynamic address.
- Extend trace events with call, handler-clause, resumption, and local-instance
  frames. Verify deterministic addresses under replay and recursion.
- Port every new runtime operation to Python, JavaScript, Julia, and Scheme
  support libraries, or arrange target lowering so a support library does not
  receive that form.

**Acceptance:**

- Reference execution tests cover terminating recursion, mutual recursion,
  tail recursion beyond Python's recursion limit, fuel exhaustion, nested local
  instances, authored handlers, foreign handlers, and mixed authored/foreign
  nesting.
- Linear/affine/unrestricted runtime guards still catch invalid foreign
  handlers.
- Trace addresses are stable across identical invocations and distinct across
  recursive iterations and unrestricted shots.

### QVR-060 — Build the structural probabilistic value layer

**Depends on:** QVR-030, QVR-040.

**Primary files:**

- `src/quivers/qiec/builtins.py`
- `src/quivers/qiec/types.py`
- `src/quivers/qiec/terms.py`
- `src/quivers/continuous/family_spec.py`
- `src/quivers/transpile/family_meta.py`
- `src/quivers/transpile/ir.py`
- `src/quivers/transpile/lower.py`

**Tasks:**

- Define canonical type constructors for sites, sampleables/distributions,
  log weights, tensors, shapes, and any structured result used by standard
  handlers.
- Create one distribution-family registry that supplies family ID, parameter
  names, parameter types/constraints, sample/event type, support, reparameter-
  izability, finite-support data, and target spellings.
- Remove duplicate family metadata presently split between the compiler,
  continuous runtime, and transpiler where feasible. If target spellings remain
  backend-owned, the semantic family record must still be shared.
- Add stable expression nodes for distribution construction and log-density
  evaluation.
- Represent plate axes and `via` fibrations with typed shape/index data. Avoid
  storing only display strings when a stable object/index ID is available.
- Implement core validators and reference-evaluator constructors without
  serializing a Torch distribution.
- Add round-trip and type tests across scalar, vector, matrix, constrained,
  discrete, continuous, mixture, transformed, and compositional families.

**Acceptance:**

- A complete `Normal`, `Categorical`, `Dirichlet`, mixture, and transformed
  sample request is stable data before target lowering.
- A reference runtime can materialize and sample/score it through an attachment
  provider.
- All existing family-registry consistency tests consume the new semantic
  registry or a mechanically derived view of it.

### QVR-070 — Elaborate `program` through QIEC

**Depends on:** QVR-040, QVR-060.

**Primary files:**

- `src/quivers/dsl/compiler/programs.py`
- `src/quivers/dsl/compiler/core.py`
- `src/quivers/dsl/program_theory.py`
- `src/quivers/dsl/qiec_lowering.py`
- `src/quivers/dsl/ast_nodes/program_steps.py`
- `src/quivers/transpile/lower.py`
- `src/quivers/transpile/ir.py`

**Tasks:**

- Add a program elaborator that translates source `ProgramDecl` directly to a
  `NamedComputation` plus typed entry-point metadata.
- Map domain data, observations, scalar parameters, object parameters, and
  morphism parameters to typed computation parameters or explicit attachments.
- Generate canonical lexical instances for `Random` and `Score` when a program
  uses them. Their identities must be deterministic and visible in the checked
  row.
- Translate every `SampleStep`, `ObserveStep`, `ScoreStep`, `LetStep`,
  `MarginalizeStep`, grouped marginalization helper, and `ReturnStep` according
  to the probabilistic meaning above.
- Replace the internal `BindStep`, `DrawStep`, `PlateDrawStep`,
  `VectorisedObserveStep`, `GroupedLatentInitStep`, and
  `GroupedBodyObserveStep` source-compilation path with QIEC terms or a target
  plan mechanically derived from QIEC.
- Permit a program body to bind the result of a named computation call. Permit
  a named computation to call a program entry point using the same call node.
- Preserve existing shape, plate, observation, grouped marginalization, return
  label, and source-location behavior.
- Make compiler constraint/effect diagnostics point to the originating program
  step even after desugaring.
- Update `QVR_PROGRAM_PROTOCOL` so the compiled Panproto schema represents the
  unified computation/effect graph, not merely object/morphism/output metadata.
- Remove `Compiler.qiec_module` as a second product if `Compiler` already owns a
  unified checked module. Expose one clearly named checked-module property.

**Acceptance:**

- Compiling a module containing only `program` produces a nonempty QIEC module.
- Compiling a mixed module produces one call graph, not two executable payloads.
- A program can call an effectful helper whose row is reflected in the program
  type.
- An ordinary program executed through the reference runtime agrees with the
  existing `MonadicProgram.log_joint` oracle on representative models.
- No source path lowers a probabilistic step directly to `IRSample`,
  `IRObserve`, `IRScore`, or `IRMarginalize` without first producing checked
  QIEC.

**Transition state:** `src/quivers/dsl/program_elaboration.py` elaborates
every entry-point program (a program without object or morphism template
parameters) before `Lower` builds its plan, and `Lower.forward` and
`transpile` refuse a program the elaboration rejects. Two program constructs
have no elaboration yet and are reported under the diagnostic code
`qiec-program-gap` rather than approximated: the schema chart parser
(`chart_fold`, a chart method call on a parser bundle), and a call of a
program template. For such a program the transpile boundary and `Compiler`
lower the module again without its programs and record the gap on the module
(`QiecModule.gap`), so the program reaches the renderer through its plan
alone until those constructs land.
`Lower` still builds `IRProgram.body` from
the source after the elaboration has checked it; deriving the plan from the
checked module is QVR-110. `Compiler.qiec_module` is the compiler's one
checked-module product, and `Program.qiec` carries the same object on the
compiled container; `QVR_PROGRAM_PROTOCOL` records that module's effects,
instances, handlers, computations, and entry points as vertices of the
program schema. Program templates (programs with object or morphism
parameters) still compile through the classic route only; elaborating them as
polymorphic computations waits on the template instantiation story of
QVR-110.

### QVR-080 — Rebase the existing effect library on QIEC

**Depends on:** QVR-050, QVR-060, QVR-070.

**Primary files:**

- `src/quivers/effects/base.py`
- `src/quivers/effects/interpreter.py`
- `src/quivers/effects/condition.py`
- `src/quivers/effects/replay.py`
- `src/quivers/effects/trace_handler.py`
- `src/quivers/effects/block.py`
- `src/quivers/effects/mask.py`
- `src/quivers/effects/scale.py`
- `src/quivers/effects/do.py`
- `src/quivers/effects/collapse.py`
- `src/quivers/effects/lift.py`
- `src/quivers/effects/reparam/`
- corresponding public exports and tests.

**Tasks:**

- Inventory the current effect-message and handler protocols and map each to a
  canonical QIEC request or handler.
- Replace parallel message classes and dispatch loops with adapters over QIEC
  execution, then remove the obsolete representations in the same branch.
- Implement conditioning as a `Random` handler that emits `Score`.
- Implement intervention/replay as `Random` handlers with distinct missing and
  extra-site policies.
- Implement tracing as request observation keyed by stable site provenance and
  dynamic address.
- Implement scale and mask as `Score` handlers or transformations.
- Implement reparameterization as a `Random` handler that replaces one
  sampleable/request with another while preserving site identity relations.
- Ensure nested handler order matches the existing public effect-composition
  contract.
- Preserve public user-facing behavior where it is still intentional, but do
  not retain an old internal engine or “legacy” aliases.

**Acceptance:**

- Existing effect-library tests pass through QIEC execution.
- There is one runtime request identity and one trace-site identity.
- Conditioning, replay, intervention, trace, mask, scale, and reparameterization
  compose with authored handlers and effectful computation calls.

**Transition state:** `src/quivers/effects/program_module.py` encodes a
`MonadicProgram` to a kernel module whose draws are `Random.sample` requests
and whose densities are `Score.add` requests, and
`src/quivers/effects/interpreter.py` runs it on the reference `Evaluator`
with torch tensors as host values. Every `EffectHandler` is a description
of prelude handlers installed on the program's `random`, `score`, or
`param` instance, in stack order, so composition is the kernel's; the
message classes and the dispatch loop of the previous engine are gone.
`TraceHandler` reads densities from the contributions that reached the
accumulator under each site's provenance. The elaboration of programs
covers reductions, product groups and fibrations, nested grouped blocks,
hoisted draws, and the measure algebra families, and the classic compiler's
own diagnostics are reported before the elaboration's. What remains open is
the deduction and network gaps recorded on the module (QVR-090, QVR-100).

### QVR-090 — Integrate logic-programming components

**Depends on:** QVR-040, QVR-050, QVR-080.

**Primary files:**

- deduction, parser, chart, semiring, and search modules under `src/quivers/`;
- `src/quivers/qiec/builtins.py`;
- DSL compiler files for `rule`, `deduction`, and parser declarations;
- their tests and examples.

**Tasks:**

- Express branching/search through the canonical `Choose` interface.
- Express proof/chart weights through `Weight[K]` or `Score` according to the
  declared algebra; do not equate arbitrary semiring weights with log weights.
- Express mutable agenda/chart state through scoped `State` instances where
  state is semantically observable.
- Express failed derivations through `Abort` or an explicit empty choice.
- Make parser/deduction entry points named computations that can be called by
  model computations.
- Permit probabilistic models to call a logic computation and score or sample
  based on its typed result.
- Add an authored multi-shot search handler and verify unrestricted resumption
  addresses and context cloning.
- Remove any standalone free-monad/effect engine made redundant by QIEC.

**Acceptance:**

- A weighted ambiguous parse uses the same handler/checker/evaluator machinery
  as a probabilistic model.
- A model can call the parser, case-analyze an indexed result, and add its
  weight without crossing an opaque Python callback boundary.
- Existing chart/parser correctness tests remain green.

**Transition state:** `src/quivers/dsl/deduction_elaboration.py` elaborates
every `deduction` to a closed item family, the `Search` effect and its
instance, a `Weight[K]` instance, the module's `params` instance, and the
`eq`, `show`, `goal`, `axiom`, `derive`, and `run` computations; the
derivation chooses rules and split positions through `Search`, adds weights
through `Weight[K]`, fails by an empty choice, and reads learned weights
through `Param`. `run_deduction` and a program's `parse(D, sentence)` run it
on the reference machine, where the search handler resumes once per
alternative over a forkable collecting handler that resumes in tail
position (`TailResume`). Mutable agenda or chart state is not needed: the
computation enumerates derivations rather than tabulating them, to the
declared depth, and the agenda engine remains the tabulating torch runtime
the two are checked against. The schema chart parser (`parser(...)`,
`chart_fold`) is a differentiable tensor program of the structural
package and stays with QVR-100.

### QVR-100 — Integrate deep-learning components

**Depends on:** QVR-060, QVR-070, QVR-080.

**Primary files:**

- encoder/decoder/loss compiler modules;
- neural and inference modules under `src/quivers/`;
- pure primitive registry;
- runtime-provider interfaces;
- their tests and examples.

**Tasks:**

- Represent encoder/decoder invocation as typed computation or pure primitive
  calls, depending on whether the component has effects.
- Model parameter lookup, mutable training state, random dropout, and loss
  accumulation explicitly. Use pure calls for fixed deterministic transforms;
  use effects only when interpretation may vary.
- Keep learned tensors at the attachment/provider boundary while retaining a
  stable typed parameter identity in QIEC.
- Make neural outputs usable as distribution parameters in the structural
  sampleable representation.
- Make logic-derived indexed values usable as neural inputs only through an
  explicit typed encoding computation.
- Preserve autograd in Python-family reference execution and supported dynamic
  backends.
- Add capability diagnostics for targets that cannot carry a neural attachment
  or training-state effect.
- Remove direct compiler shortcuts that bypass the unified call/effect graph.

**Acceptance:**

- A neural conditional distribution used inside a program is represented by
  one checked call/effect graph.
- Gradients reach attached parameters in PyTorch-backed execution.
- A target without the required neural facility rejects the exact call or
  effect with a source-located diagnostic.

**Transition state:** a kernel morphism's parameter map is part of the
program's computation: `src/quivers/dsl/program_elaboration.py` reads the
map's learned tensors as typed inputs (`weight`, `bias`, `table` roles) and
composes affine maps, `tanh` layers for `[param_source=mlp]`, table lookups
for kernels over finite domains and `[role=embed]` embeddings, and per-row
comprehensions under a plate, so a neural conditional distribution is one
checked term whose inputs match the torch runtime's `MLPSource` and `Embed`
parameters shape for shape. `src/quivers/dsl/composite_lets.py` expands a
draw from a composite into one site per factor, declaring the replicas of a
`[replicate=k]` morphism and the copies a `stack` makes as morphisms of the
expanded module, threading the step's row into the chain's head and a tensor
product's factors into its branches. The transpile targets keep refusing
network and embedding kernels at the boundary (`param-source:<kind>`,
`embed:<name>`) since their tensors have no spelling in the wire form.
A `scan` expands to a step program and a let the elaboration turns into a
recursive helper over the positions of an open extent, with `run_program`
replaying a site's occurrences by `"<site>@<n>"`. Gradients reach the
parameters through the effects route, where the torch runtime's morphisms
are the host values of the kernel computation. Open: the schema chart
parser, and the encoder, decoder, and loss declarations of the structural
package.

### QVR-110 — Replace the split transpiler IR

**Depends on:** QVR-070.

**Primary files:**

- `src/quivers/transpile/ir.py`
- `src/quivers/transpile/qiec_ir.py`
- `src/quivers/transpile/lower.py`
- `src/quivers/transpile/_qiec_boundary.py`
- `src/quivers/transpile/_api.py`
- `src/quivers/transpile/_diagnostics.py`
- `src/quivers/transpile/renderer_registry.py`

**Tasks:**

- Replace `IRProgram(name, inputs, body, cards, qiec)` with one module root whose
  executable entry points are QIEC computations.
- Move program metadata such as data declarations, observation declarations,
  return labels, and cardinalities into typed declaration/entry-point records,
  not a parallel body.
- Make `Lower.forward` always invoke the QVR-to-QIEC elaborator first.
- Add a target-plan lowering pass that recognizes canonical probabilistic
  effects and produces optimized PPL nodes where sound.
- Delete `_qiec_boundary.py` if it exists only to validate a sidecar. Fold the
  check into the single lowering boundary.
- Replace `graft_qiec_dynamic` and `graft_qiec_static` with rendering of one
  target plan. Shared runtime helpers may remain, but “grafting” an independent
  module must disappear.
- Expand capability analysis to calls, recursion, authored handlers, local
  instances, pure primitives, distributions, plates, and entry-point ABI.
- Ensure structured diagnostics carry the source declaration and path through
  any target-plan rewrite.

**Acceptance:**

- Repository search finds no `IRProgram.qiec` access and no QIEC graft step.
- A renderer receives one lowered root.
- Target-plan optimization can be disabled in a test, and reference execution
  of the unoptimized QIEC graph yields the same result/score on deterministic
  fixtures.

**Transition state:** `src/quivers/transpile/plan.py` holds `Lower` and the
derivation: `Lower.forward` elaborates and checks the module
(`checked_module`) and derives the program's plan from its computation, so
`IRProgram(name, inputs, body, module, cards)` is one lowered root whose
`module` is the checked `IRQiecModule` and whose `body` is the plan
recognized from the computation's requests, bindings, and helper calls;
`src/quivers/transpile/lower.py` keeps the support tables (object shapes and
bounds, family sentinels, argument constraints, wire forms). The renderers
render the module's other computations with `render_computations_dynamic`
and `render_computations_static`, analyzing capabilities first;
`_qiec_boundary.py` is gone. The plan is the one target-plan pass: it
recognizes the canonical requests and emits the PPL nodes the renderers
consume, and it refuses, under a structured kind, every request or call the
target vocabulary has no statement for. Open: the plan is not yet an
optimization pass that can be switched off, since it has no rewrites beyond
recognition, and the capability analysis of calls, recursion, and authored
handlers is the analyzer's as before.

### QVR-120 — Update all eleven transpilers

**Depends on:** QVR-050, QVR-060, QVR-110.

**Primary files:**

- all modules in `src/quivers/transpile/renderers/`;
- `src/quivers/transpile/runtime_qiec.*`;
- target grammars and probe scripts;
- `tests/transpile/` and generated support documentation.

**Shared tasks:**

- Render named computation calls with correct static/value argument order.
- Render recursion or reject it before emission. Never emit an unresolved
  function name.
- Lower canonical `Random.sample`, observation-score, and `Score.add` patterns
  to native target constructs.
- Render residual user-defined effects and authored handlers on targets with a
  host runtime.
- Preserve site names and dynamic addresses needed by trace comparison.
- Preserve GADT case dispatch or reject unsupported runtime representations.
- Check all new target output with the real external parser when available.
- Add feature tags for each independently rejectable construct.

**Python-family targets — Pyro, NumPyro, PyMC, Edward2:**

- Integrate generated functions with each target's model-building entry point.
- Preserve native sample/observe/factor primitives and autodiff arrays.
- Ensure handler runtime state is invocation-local and safe under concurrent
  model calls.
- Execute at least one called, recursive, handled statistical model per target
  where the target runtime permits it.

**Julia targets — Turing and Gen:**

- Preserve Julia task-local handler state.
- Render recursion and authored clauses with valid Julia scoping.
- Ensure model macros/functions receive data and observations through their
  native ABI rather than global attachment maps when possible.

**JavaScript targets — WebPPL:**

- Preserve continuation behavior in WebPPL's execution model.
- Rewrite `runtime_qiec.js` in WebPPL's functional subset. The present
  runtime runs under Node but not under the `webppl` compiler, which rejects
  assignment, loops, `try`, `throw`, and native callbacks: the trampoline must
  become recursion under WebPPL's own CPS, call and instance serials must be
  threaded through the driver rather than counted globally, and the handler
  attachment protocol (contexts, forks, lifecycles) must pass and return
  state explicitly instead of mutating controller cells. Until then the
  WebPPL target's QIEC entry points and its distribution bridge
  (`runtime_qiec_webppl.js`) are exercised under Node only.
- Test recursion and multi-shot handlers for stack/address correctness.
- Integrate generated calls into the exported model rather than detached
  helper entry points.

**Scheme target — Church:**

- Preserve dynamic handler scope and optional entry arguments in valid Scheme.
- Test recursive calls and unrestricted resumptions with Chez Scheme.

**Static targets — Stan, BUGS, JAGS:**

- Monomorphize finite static arguments and inline acyclic pure calls where
  semantics are preserved.
- Permit canonical sample, observation, score, deterministic, return, and
  supported finite marginalization patterns.
- Reject arbitrary user handlers, open effect rows, dynamic local instances,
  non-eliminable GADT values, unrestricted resumptions, and recursion that
  cannot be structurally eliminated.
- Stan may emit user-defined pure functions when its type/recursion rules admit
  them. BUGS/JAGS should inline or reject according to their graph languages.
- Replace the current “pure scalar QIEC fragment” capability story with a
  feature matrix derived from the unified model plan.

**Acceptance:**

- Every target transpiles the same ordinary model fixtures it supported in
  `0.18.0` through the new core.
- Each dynamic target transpiles a model that calls an effectful helper.
- Every unsupported cell has an exact diagnostic fixture and documentation.
- External syntax checks and available runtime executions pass for emitted
  output.

### QVR-130 — Unify compiler and Python execution APIs

**Depends on:** QVR-070, QVR-080, QVR-110.

**Primary files:**

- `src/quivers/dsl/compiler/core.py`
- `src/quivers/program.py`
- `src/quivers/continuous/programs.py`
- `src/quivers/qiec/execution.py`
- inference, guide, trace, diagnostics, and formula integration modules.

**Tasks:**

- Give `Compiler` one checked executable module and one entry-point lookup API.
- Make direct Python execution of `program` and `define` entry points share
  argument validation, provider selection, tracing, and error codes.
- Rebuild `MonadicProgram` as a facade over the unified core or replace it at
  internal call sites. Do not keep two evaluators authoritative.
- Preserve fitting, guide construction, posterior predictive checks,
  diagnostics, and formula compilation by routing them through standard
  handlers/providers.
- Define ownership of observations, parameters, RNG state, score accumulation,
  and trace collection explicitly per invocation.
- Ensure no global mutable handler stack is introduced.

**Acceptance:**

- Existing Python API tests for compile, log joint, fit, guide, trace,
  predictive checks, and formulas remain green.
- The same entry computation may be invoked by CLI, REPL, tests, and a Python
  caller with the same validation and trace semantics.

### QVR-140 — Complete CLI, REPL, and TUI behavior

**Depends on:** QVR-040, QVR-050, QVR-130.

**Primary files:**

- `src/quivers/cli/check.py`
- `src/quivers/cli/run.py`
- `src/quivers/cli/repl_session.py`
- `src/quivers/cli/repl_complete.py`
- `src/quivers/cli/repl_highlight.py`
- `src/quivers/cli/repl_tui.py`
- CLI/TUI tests.

**Tasks:**

- Make `qvr check` run one elaboration/check path and report call graph, row,
  handler-body, recursion, and target-capability errors.
- Make `qvr run` select any executable entry point, including a `program`, with
  typed data, observations, statics, providers, and optional fuel.
- Remove UI language that treats QIEC computations as an auxiliary module.
- Extend `:type`, `:kind`, `:effects`, `:show`, `:run`, completion, and history
  to calls, scoped instances, handler clauses, and unified programs.
- Show inferred and declared rows after call expansion without printing
  unstable dataclass representations.
- Add TUI status/error displays for active entry point, provider, fuel
  exhaustion, and target capability.
- Keep JSON output stable and fully serializable.

**Acceptance:**

- CLI, REPL, and TUI execute the same nontrivial model fixture.
- Diagnostics have identical codes and source positions across all three.
- Completion never suggests a scoped instance or branch binder outside its
  lexical range.

### QVR-150 — Complete language-server and editor support

**Depends on:** QVR-010, QVR-020, QVR-040.

**Primary files:**

- `src/quivers/lsp/document.py`
- `src/quivers/lsp/server.py`
- `src/quivers/dsl/qiec_tooling.py`
- `editors/vscode-qvr/`
- `editors/zed-extension-qvr/`
- `grammars/qvr/queries/highlights.scm`
- `src/quivers/dsl/pygments_lexer.py`
- `docs/hooks/register_qvr_lexer.py`
- LSP/editor tests.

**Tasks:**

- Index computation declarations, call sites, recursion edges, handler
  parameters, operation parameters, resumptions, local instances, pure locals,
  and GADT branch binders.
- Implement hover for instantiated call signatures and inferred rows.
- Implement definition/references/rename for call targets and every new binder.
- Include calls and handler clauses in document symbols and outline nesting.
- Produce semantic tokens/highlights for declarations, effects, operations,
  handlers, resumptions, constructors, types, statics, and locals.
- Update formatter behavior for every new indented body and ensure formatting
  refuses invalid checked modules.
- Update target capability diagnostics after configuration changes without
  reparsing with a different language grammar.
- Regenerate VS Code compiled output and archive; keep source/archive parity.
- Update Zed queries and immutable grammar pin as required by its release
  process.

**Acceptance:**

- Position-aware tests cover shadowing and same-spelled operation names.
- Go-to-definition from a recursive call reaches its declaration.
- Rename does not cross lexical instance, branch-binder, or handler-clause
  scopes.
- Pygments, tree-sitter, TextMate, Zed, REPL, and TUI classify the same token
  corpus consistently.

### QVR-160 — Regenerate grammar, parser, package, and Panproto artifacts

**Depends on:** QVR-010, QVR-020, QVR-150.

**Primary files:**

- `grammars/qvr/src/`
- `src/quivers/dsl/_grammar_data/`
- `grammars/qvr/vcs/parsers/HEAD/`
- `grammars/qvr/vcs/.panproto/`
- packaged migration assets under `src/quivers/cli/migrations/`;
- wheel build hook and smoke tests.

**Tasks:**

- Run `tree-sitter generate` in `grammars/qvr` and inspect changes to
  `grammar.json`, `node-types.json`, and `parser.c`.
- Run `tree-sitter test`.
- Run `python tools/sync_grammar_data.py` and then its `--check` mode.
- Rebuild only the live `HEAD` parser snapshot with
  `python grammars/qvr/vcs/build_parsers.py --revision HEAD --force`.
- Update the `HEAD` grammar schema with
  `python grammars/qvr/vcs/build_schemas.py` without `--reset`.
- Derive/update the `v0.18.0 -> HEAD` migration. Since the QIEC surface did not
  exist in `0.18.0`, existing released source should remain byte-identical
  unless another final grammar change requires a real rewrite.
- Validate every historical object with its original stored hash. Treat any
  failure among the historical fixtures as a release blocker.
- Rebuild package-native parser libraries and source-binding manifests for
  Linux, macOS, and Windows through CI. Do not commit a local platform binary
  as a substitute for the matrix.
- Verify an installed wheel parses, highlights, migrates, checks, executes, and
  transpiles with compiler discovery disabled.

**Acceptance:**

- Generated sources and vendored package data are byte-synchronized.
- All historical tags and object IDs are unchanged.
- Migration tests pass from every supported source revision to `HEAD`.
- Platform wheel smoke jobs pass on all three operating systems.

### QVR-170 — Add end-to-end semantic and statistical tests

**Depends on:** QVR-050 through QVR-160.

**Primary files:**

- `tests/qiec/`
- `tests/dsl/`
- `tests/transpile/`
- `tests/cli/`
- `tests/lsp/`
- a new `tests/fixtures/qiec/` corpus;
- representative documentation examples.

**Required fixture A — recursive indexed computation:**

- Define `Nat` and `Vec[A](n)`.
- Define a recursive `fold` or `map` computation over `Vec`.
- Refine the tail length in the constructor branch.
- Call the recursive helper from another computation.
- Check, serialize, execute, and transpile it on each admitting target.

**Required fixture B — authored state handler:**

- Define `State[S]`, allocate a local instance, and handle it with authored
  clauses and an explicit return clause.
- Exercise get, put, nested handlers, and a computation call inside a clause.
- Verify instance non-escape and resumption grades.

**Required fixture C — nontrivial statistical model:**

- Define an index sort for observation status and a GADT whose constructors
  distinguish observed from missing measurements.
- Define an effectful helper that case-analyzes the GADT: observed values add a
  likelihood score; missing values request a posterior-predictive draw.
- Call the helper from an ordinary hierarchical `program` with at least one
  group-level latent, one observation plate, and one nonlinear deterministic
  transform.
- Use a standard condition/replay handler to switch between fitting and
  posterior prediction without changing the model definition.
- Verify that removing the indexed case makes the source ill-typed or loses the
  required branch refinement, and that removing the effectful call leaves an
  unhandled row. This establishes that the new features are necessary rather
  than decorative.
- Compare reference log joint and emitted-target score at multiple perturbed
  parameter/data points, including both GADT constructors.

**Required fixture D — exact grouped marginalization:**

- Use a finite indexed latent and a grouped/fibred observation structure.
- Elaborate `marginalize` to the standard enumeration/score handler.
- Compare the result with an independently computed `logsumexp` oracle.
- Exercise Stan's static lowering and at least one dynamic target.

**Required fixture E — logic/model composition:**

- Run an ambiguous weighted deduction under an unrestricted `Choose` handler.
- Call it from a probabilistic model and add the returned semiring/log weight
  through an explicit conversion.
- Verify multi-shot trace addresses and context cloning.

**Required fixture F — neural/model composition:**

- Call an encoder or decoder to parameterize a distribution.
- Sample/observe through canonical effects and backpropagate to an attached
  parameter in Python execution.
- Verify a target-specific capability refusal where the neural component cannot
  be represented.

**Cross-cutting test rules:**

- Test source parse, emit, lowering, independent module validation,
  serialization, IR conversion, reference evaluation, target lowering,
  external syntax, and available target runtime separately.
- Mutation tests must prove that target probes notice dropped calls, scores,
  observations, handler applications, and recursion edges.
- No test may declare success because a feature is silently absent from both
  expected and actual output.
- Preserve exact diagnostics for unsupported targets.
- Keep slow/probe markers accurate so CI does not accidentally skip required
  release evidence.

### QVR-180 — Rewrite documentation and changelog to the final architecture

**Depends on:** all implementation packages.

**Primary files:**

- `docs/developer/qiec.md`
- `docs/semantics/effects.md`
- `docs/semantics/typing.md`
- `docs/semantics/grammar.md`
- `docs/semantics/transpile-architecture.md`
- `docs/guides/dsl-overview.md`
- `docs/guides/dsl-declarations.md`
- `docs/guides/dsl-programs-and-lets.md`
- `docs/guides/effects.md`
- `docs/guides/repl-and-lsp.md`
- `docs/getting-started/architecture.md`
- `docs/getting-started/highlighting.md`
- `docs/transpile-support.md`
- `docs/api/cli.md`
- `docs/index.md`
- `CHANGELOG.md`
- `docs/developer/changelog.md`

**Tasks:**

- Remove every statement that describes QIEC and `program` as independent
  routes, mixed sidecars, grafted modules, or parallel IR components.
- Replace the current limitation statements about absent calls, recursion,
  scoped instances, and authored handler bodies with the implemented rules and
  their actual limits.
- Document standard probabilistic elaboration and why backend inference remains
  outside the source program.
- Document call typing, SCC checking, divergence, handler typing, grade
  enforcement, scoped-instance escape checks, and target capability rules.
- Include all required fixtures above as tested examples or link to their
  tested source files.
- Regenerate the transpiler support matrix from measured behavior.
- Keep `CHANGELOG.md` and `docs/developer/changelog.md` byte-identical.
- Describe `0.19.0` under `Added` and `Changed` as one final feature. Do not
  narrate abandoned branch designs or their cleanup.
- Run documentation fence tests and `mkdocs build --strict`.

**Acceptance:**

- Every QVR code fence parses and, where marked, checks/executes/transpiles.
- Documentation claims match test-backed target behavior.
- The changelog mirror test passes byte-for-byte.
- A repository search finds no claim that the two executable routes remain
  separate.

### QVR-190 — Final release gates and PR hygiene

**Depends on:** all preceding work packages.

**Tasks:**

- Organize commits using repository conventions and scoped Conventional Commit
  subjects. Do not squash unrelated user work into these commits.
- Suggested commit sequence:

  1. `refactor(qiec): simplify effect declarations`
  2. `feat(qiec): add recursive calls and authored handlers`
  3. `feat(qvr): elaborate programs through indexed effects`
  4. `refactor(effects): use the qiec runtime`
  5. `feat(transpile): lower unified computations on all targets`
  6. `feat(tooling): support unified computations across editors`
  7. `feat(migrations): update the qvr head grammar`
  8. `docs(qvr): document the unified model language`

- Run formatting before tests; inspect every formatter-generated diff.
- Run focused suites after each package and the complete gates at the end.
- Push only after the local tree is coherent.
- Verify every PR check against the final commit SHA.
- Do not tag, merge, publish, or create a release unless separately authorized.

**Required local commands:**

```bash
ruff check src/ tests/ hatch_build.py tools/release_wheel_smoke.py tools/check_editor_release.py
ruff format --check src/ tests/ hatch_build.py tools/release_wheel_smoke.py tools/check_editor_release.py
pyright src/quivers
python -m pytest tests/qiec tests/dsl tests/cli tests/lsp -q
python -m pytest tests/transpile -q -m "not probe and not slow"
python -m pytest tests/ -q -m "not probe" -n auto --dist loadfile
python -m pytest tests/ -m slow -q
mkdocs build --strict
python tools/check_editor_release.py --full
python tools/sync_grammar_data.py --check
python -m build --sdist --wheel
python -m twine check dist/*
```

Run the probe tier with real containers and without treating an absent runtime
as a skip. PR probes may use the repository's measurement cache; the release-
certifying run on `main` must execute cold according to `.github/workflows/ci.yml`.

**Acceptance:**

- Lint, format, typecheck, fast, slow, four probe shards, migration assets on
  Linux/macOS/Windows, editor parity, documentation, and distribution smoke all
  pass on the final SHA.
- `git diff --check` is clean.
- The worktree contains no unintended generated files or local binaries.
- The PR description and checklist state the single-core architecture and link
  to the nontrivial model evidence.

## Dependency and parallelization map

The critical path is:

```text
QVR-000
  -> QVR-010
  -> QVR-020
  -> QVR-030
  -> QVR-040
  -> QVR-050
  -> QVR-060
  -> QVR-070
  -> QVR-110
  -> QVR-120
  -> QVR-130
  -> QVR-140/QVR-150/QVR-160/QVR-170
  -> QVR-180
  -> QVR-190
```

After QVR-040 stabilizes the core contracts, a swarm can work with the
following non-overlapping ownership:

| Worker | Ownership | Must coordinate with |
|---|---|---|
| Core semantics | `src/quivers/qiec/`, `tests/qiec/` | surface AST and runtime ABI owners |
| Surface/compiler | grammar, `src/quivers/dsl/`, DSL tests | core term constructors and Panproto owner |
| Probabilistic/effects | `src/quivers/effects/`, continuous program adapters, effect tests | core evaluator and compiler owners |
| IR/transpiler foundation | `transpile/ir.py`, `qiec_ir.py`, `lower.py`, shared renderer helpers | all target owners |
| Python targets | Pyro, NumPyro, PyMC, Edward2 renderers/tests | transpiler foundation |
| Julia/JS/Scheme targets | Turing, Gen, WebPPL, Church renderers/runtimes/tests | transpiler foundation and core runtime |
| Static targets | Stan, BUGS, JAGS renderers/tests | capability analyzer and target-plan owner |
| Tooling | CLI, REPL, TUI, LSP, highlighters, editor packages | finalized surface AST |
| Migration/package | generated grammar, Panproto VCS, wheels | finalized grammar only |
| Documentation/evidence | docs, examples, changelog, support matrix | all behavior owners; work last on claims |

Every worker must be told that other workers share the tree, must not revert
their edits, and owns only the listed files. Shared foundation files require a
single owner until their public shapes stabilize.

## Release-blocker checklist

Any checked item may be completed independently; release requires all of them.

- [ ] Interrupted effect-declaration cleanup is complete and audited.
- [ ] Surface syntax is frozen and represented in golden corpus tests.
- [ ] Stable computation IDs and call terms exist.
- [ ] Direct and mutual recursion check, serialize, evaluate, and lower.
- [ ] Pure expressions and structural distributions exist in QIEC.
- [ ] Authored handler clauses and return clauses execute.
- [ ] Static and dynamic resumption-grade checks pass.
- [ ] Scoped instance allocation and non-escape checks pass.
- [ ] `program` elaborates to QIEC before any PPL IR.
- [ ] All probabilistic step forms elaborate through standard effects/handlers.
- [ ] Existing effect combinators use the QIEC request/handler engine.
- [ ] Logic-programming entry points use `Choose`/`Weight`/`State` as required.
- [ ] Deep-learning calls and parameter attachments use the unified core.
- [ ] `IRProgram.qiec` and graft-based split rendering are removed.
- [ ] All eleven target implementations and capability refusals are tested.
- [ ] Python compile/run/inference APIs have one authoritative evaluator path.
- [ ] CLI, REPL, and TUI support all new constructs.
- [ ] LSP navigation, completion, rename, symbols, formatting, and diagnostics
      support all new constructs.
- [ ] Pygments, tree-sitter, TextMate, VS Code, and Zed are synchronized.
- [ ] Current and `HEAD` parser artifacts are regenerated and packaged.
- [ ] Panproto migration succeeds without changing historical hashes.
- [ ] Nontrivial statistical, grouped marginal, logic, and neural fixtures pass.
- [ ] Documentation describes the implemented single-core system.
- [ ] Changelog copies are identical and describe final behavior only.
- [ ] Local lint, type, fast, slow, docs, build, and probe gates pass.
- [ ] PR checks are green on the final pushed SHA.

## Conditions that do not count as completion

- A typed kernel that only runs standalone `define` declarations.
- A QIEC module attached to an otherwise independently compiled `Program`.
- A renderer that emits `qiec_*` helpers which the generated model never calls.
- Examples that parse but do not lower, execute, or enter a backend model.
- Handler signatures whose bodies exist only as process-local callbacks.
- A computation graph with no call node presented as support for composition.
- Recursion simulated by host-language callbacks outside the stable core.
- Treating existing model steps as “conceptually effects” while leaving their
  compiler/IR path unchanged.
- Claiming all-target support because unsupported constructs were omitted.
- Passing only focused unit tests while generated grammar, editor, migration,
  packaging, slow, or probe gates are stale.
- Citing green checks for an earlier PR revision.
- Updating historical fixture hashes to make migration tests pass.
- Adding compatibility aliases for unreleased intermediate syntax.

The branch is done only when the nontrivial statistical fixture uses indexed
case refinement and an effectful computation call inside an ordinary model,
executes through the reference boundary, and reaches every target that claims
the necessary capabilities through the same checked core.
