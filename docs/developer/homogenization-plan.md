# QVR language homogenization — implementation plan

This document tracks the 12 homogenization moves from the design pass.
Every change is a clean breaking change; no backward-compat shims. The
target release is **0.11.0**, cut from this branch.

## Sequencing principle

The grammar is the foundation. Every later layer (AST, parser walkers,
compiler dispatch, emitter, highlighter, examples, docs, tests) reads
from the grammar's node-type set. We rewrite top-down once:

```
grammar.js  →  parser.c (regenerated)  →  AST nodes  →  parser walkers
            →  compiler dispatch  →  emitter  →  constraints
            →  pygments_lexer + repl_highlight (shared keyword tables)
            →  example .qvr files  →  docs/ code blocks
            →  test fixtures  →  full pytest sweep
```

We use `QVR_USE_LOCAL_GRAMMAR=1` throughout development so we don't
have to wait for a panproto-grammars-all upstream release; the bundled
grammar gets pinned to a vendor floor of "ships v0 grammar" with a
clear minimum panproto version once the upstream lands.

## The 12 moves

1. **Indented blocks everywhere.** Drop braces from `deduction`,
   `program`, `encoder`, `decoder`, `signature`, `loss`, `algebra`,
   `composition_rule`, `semigroupoid`, `bilinear_form`. Python-style
   indented bodies. TUI auto-indent already exists; verify it covers
   every block-opener token.
2. **`KEYWORD NAME(params) : SIGNATURE [options] BODY` on every
   typed declaration.** Param parens for `contraction`, `deduction`,
   `program`, `encoder`, `decoder`; everything else gets `()` when
   parameterless or just `NAME` if we keep the no-paren shorthand.
3. **One `composition NAME at LEVEL { ... }` for the composition
   tower.** Replaces `algebra`, `semigroupoid`, `bilinear_form`,
   `composition_rule` keywords. `LEVEL ∈ {algebra, semigroupoid,
   bilinear_form, composition_rule}`.
4. **One `morphism NAME : DOM -> COD [attrs]` for the morphism
   tower.** Replaces `latent`, `observed`, `kernel`, `embed`,
   `discretize`, `let`, `program`. Attributes: `[latent]`,
   `[observed]`, `[kernel]`, `[embed]`, `[discretize]`. `program`
   stays as its own keyword because it carries a body, not just a
   signature — but we rename it `program NAME(params) : DOM -> COD
   ! EFFECTS BODY` and drop the `let` form for deterministic
   pipelines (replaced by `morphism NAME : ... = EXPR`).
5. **One `[k=v, ...]` option syntax.** Move `depth N`, `start S`,
   `semiring R`, `! Sample, Score` into `[k=v, ...]`. Drop the inline
   `! EFFECTS` clause on programs in favour of `[effects=Sample,
   Score]`.
6. **One initializer mechanism.** `~ EXPR` covers all three of:
   constant value, distribution prior, named init recipe (`~ auto`,
   `~ xavier`, …). Drop `= EXPR` and `[init=NAME]`.
7. **Universal `## doc` attachment.** Every declaration carries
   `docs: tuple[str, ...]`. `_attach_docs` recognises them all.
8. **`type NAME : EXPR` replaces both `object` and `space`.** The
   RHS picks discrete vs continuous (integer literal → FinSet;
   `Euclidean(...)`, `Simplex`, etc. → ContinuousSpace; `{a, b, c}`
   → EnumSet; `FreeResiduated(...)` → FreeResiduated).
9. **One `over (axes) [iid] [via PATH]` clause.** Replaces `over`,
   `iid over`, `via`. `iid` and `via` are flags inside the option
   block.
10. **Every program step gets a leading keyword + indented body.**
    `sample x <- Family(...)`, `observe x <- ...`,
    `marginalize z : K <- ...`, `let x = ...`, `return x`. Drop the
    zero-keyword sample form. Drop `in { ... }` in favour of an
    indented `BODY:` block.
11. **`!` effects on every effectful declaration** (or fully derived
    and removed from the surface). Decision: move to `[effects=...]`
    inside the option block on every declaration kind (move 5
    subsumes this).
12. **Constructor calls in parens for every sized type.**
    `type X : FinSet(3)` replaces `object X : 3`;
    `type R : Real(low=0, high=1)` replaces `Real` with bounds in
    options; `kernel f : ... ~ Family(rank=4)` replaces `kernel
    f[4]`.

## Concrete file impact

| Layer | Files | Approximate size |
| --- | --- | --- |
| Tree-sitter grammar | `grammars/qvr/grammar.js` | 1900 lines (full rewrite) |
| Generated parser | `grammars/qvr/src/parser.c` | regenerate via `tree-sitter generate` |
| AST nodes | `src/quivers/dsl/ast_nodes.py` | ~30 declarations collapse to ~10 |
| Parser walkers | `src/quivers/dsl/parser.py` | 2300 lines, every `_walk_*_decl` rewritten |
| Compiler | `src/quivers/dsl/compiler/*.py` | every `_compile_*` dispatch + isinstance |
| Emitter | `src/quivers/dsl/emit.py` | every `_emit_*` |
| Constraint solver | `src/quivers/dsl/constraints.py` | every checker |
| Highlighter / lexer | `src/quivers/dsl/pygments_lexer.py`, `src/quivers/cli/repl_highlight.py` | keyword/builtin tables |
| Examples (source) | `docs/examples/source/*.qvr` | 36 files, hand-rewrite each |
| Example doc pages | `docs/examples/*.md` | 37 pages, each refers to its source |
| Tutorials | `docs/tutorials/qvr/*.md` | every QVR block in every tutorial |
| Guides | `docs/guides/*.md` | ~25 guides with QVR examples |
| Quickstart / README | `README.md`, `docs/index.md`, `docs/getting-started/quickstart.md` | hero example + every snippet |
| Regression suite | `regression.qvr` | rewrite |
| Tests | `tests/*.py`, `tests/data/*.qvr` | every inline QVR source |
| TUI auto-indent | `src/quivers/cli/repl_tui.py` | extend trigger list |

## Sequencing — one PR

Per user direction, the entire homogenization ships as one PR off
this branch. Internal commit ordering follows the layer cascade
below so each step can be reviewed in `git log`:

### Step 1 — surface change foundation (the grammar)

- Rewrite `grammars/qvr/grammar.js` end-to-end against moves 1–12.
- Regenerate `parser.c`, `grammar.json`, `node-types.json`.
- Update `grammars/qvr/queries/highlights.scm` for the new node kinds.
- Update `pygments_lexer.py` and `repl_highlight.py` keyword tables.
- Update the in-tree `_dev_grammar.py` build/cache invalidation.
- Re-enable `QVR_USE_LOCAL_GRAMMAR=1` in CI and in dev workflow.
- All other code still references the old AST; tests stay broken
  until PR 2. Land PR 1 first so the grammar work is reviewable in
  isolation.

### Step 2 — AST + parser + compiler + emitter

- Collapse `ast_nodes.py` to the new declaration set.
- Rewrite `parser.py` walkers against the new tree-sitter node types.
- Rewrite `compiler/_compile_statement` dispatch and every
  per-statement handler.
- Rewrite `emit.py` to round-trip the new AST.
- Update `constraints.py`.
- The Python tests will be massively broken; the goal of PR 2 is to
  get the library + library tests (`tests/test_compose.py`,
  `tests/test_inference.py`, …) green against the new surface, using
  hand-converted minimal QVR fragments. Doc-block tests stay broken.

### Step 3 — examples + docs

- Hand-rewrite every `docs/examples/source/*.qvr` (36 files).
- Update every `docs/examples/*.md` page accordingly.
- Walk every `docs/guides/*.md` and `docs/tutorials/**/*.md` for
  ` ```qvr ` blocks; rewrite each.
- Update `README.md`, `docs/index.md`,
  `docs/getting-started/quickstart.md` hero examples.
- Update `regression.qvr`.
- `tests/test_doc_blocks.py` becomes the acceptance gate for this
  PR.

### Step 4 — REPL/LSP/Jupyter polish + final cleanup

- Extend the TUI's auto-indent trigger list and bracket-pairing for
  the new keywords.
- Verify LSP semantic tokens use the new keyword set.
- Update `editors/vscode-qvr/syntaxes/qvr.tmLanguage.json` and
  `editors/zed-extension-qvr` highlight queries.
- Bump version to **0.11.0**, write the changelog entry, cut the
  release.

## What I'm asking before I start

This plan represents a 30+ hour, four-PR effort. Before I rewrite
grammar.js end-to-end:

1. **Sign-off on the move list.** Anything you want dropped or added?
   The biggest debatable items are #4 (collapsing seven morphism
   keywords into `morphism [attrs]` — some users may prefer the
   keyword tower for grep-ability) and #10 (forcing a `sample`
   keyword on every stochastic step instead of bare `<-`).
2. **Confirm the four-PR sequencing.** Or do you want it as one
   merged drop?
3. Version: **0.11.0** (pre-1.0 breaking changes go into minor).

Once sign-off lands, I start with PR 1 (grammar rewrite). The grammar
work is the longest single chunk — probably 4–6 hours by itself —
and everything else cascades.
