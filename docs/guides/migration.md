# Migrating `.qvr` source between grammar revisions

QVR's surface grammar evolves between releases. `qvr migrate` lowers
`.qvr` source written for one tagged grammar revision into source
shaped for a later revision. The transformation is grammar-bound at
every step: every per-declaration output is parse-validated against
the target revision's grammar, and the assembled output is
parse-validated as a whole file before being written.

This page covers what the migrator does today, how to use it, and
what's still pending.

## What `qvr migrate` does

The pipeline runs per source file:

```mermaid
flowchart LR
    A["source bytes<br/>(written against revision X)"]
    B["parse with X's<br/>tree-sitter grammar"]
    C["walk the parse-tree<br/>schema"]
    D["per-declaration<br/>converter (X → Y)"]
    E["per-declaration<br/>parse-validate (Y)"]
    F["concatenate"]
    G["whole-file<br/>parse-validate (Y)"]
    H["source bytes<br/>(shaped for revision Y)"]
    A --> B --> C --> D --> E --> F --> G --> H
```

Each adjacent revision pair `(X, Y)` on the migration
[`CHAIN`](#the-migration-chain) has its own converter module under
`src/quivers/cli/migrations/vX_Y_Z_to_vA_B_C.py`. Migrating across
multiple revisions composes the intermediate hops.

The per-revision tree-sitter parsers live at
`grammars/qvr/vcs/parsers/<rev>/qvr.{dylib,so,dll}`; the migration
schemas live in the [panproto VCS](#the-panproto-vcs-chain) at
`grammars/qvr/vcs/.panproto/`.

## Common invocations

Migrate one file in place from the default source revision to the
current grammar:

```bash
qvr migrate docs/examples/source/lda.qvr
```

Migrate every `.qvr` under a directory:

```bash
qvr migrate docs/examples/source/
```

Pick specific revisions explicitly:

```bash
qvr migrate --from v0.10.0 --to v0.11.0 docs/examples/source/lda.qvr
```

Dry-run (report what would change, write nothing):

```bash
qvr migrate --dry-run docs/examples/source/
```

Write migrated copies to a separate directory:

```bash
qvr migrate --output /tmp/migrated docs/examples/source/
```

Run the [coverage check](#-check-mode-coverage-against-the-vcs) against
the migration chain without migrating any files:

```bash
qvr migrate --check
```

`--from` defaults to the penultimate entry of `CHAIN` (currently
`v0.14.0`). `--to` defaults to `HEAD`, an alias for the chain's final
entry (currently `v0.15.0`).

## What survives migration

What's preserved by the migrator today:

- **Declarations handled by a full converter.** Each supported source
  declaration becomes the semantically equivalent target declaration,
  even when the surface changed (e.g. `latent f : A -> B` becomes
  `morphism f : A -> B [role=latent]`). Identity-scaffold hops pass
  bytes through and may fail final target validation when an older
  construct needs a converter.
- **Top-level comments.** Header comments (file preamble,
  between-decl explanations) pass through verbatim.
- **In-body comments.** Comments inside `program`, `deduction`,
  `marginalize`, `signature`, `encoder`, `decoder`, `loss`, and
  composition rule bodies pass through as their raw source text,
  interleaved with the (translated) structural body entries in
  document order.
- **Lexicon block comments.** Comments between lexicon entries
  survive.
- **Doc comments.** `#!` doc comment lines attached to a
  declaration migrate with that declaration.
- **Multi-line bracketed forms.** Where the source revision allows
  multi-line `[...]` / `(...)` / `{...}` and the source has
  interior comments, the migrator's
  `emit_bracketed_list` helper preserves them.

What's intentionally dropped or transformed:

- **Single-line interior comments.** A comment inside an
  inline-form bracketed list (e.g. `[role=latent, # comment\n
  over=cod]` written without a leading newline) cannot exist in
  the grammar: the inline form forbids newlines. The user must
  switch to the multi-line bracket form to retain such a comment.
- **Body keywords that became option entries.** v0.10.0's
  `deduction` body carried `semiring LogProb`, `start S`, `depth 6`
  on their own lines; these hoist into the header option block as
  `[semiring=LogProb, start=S, depth=6]`. Same for program effects
  (`! Score, Sample` → `[effects=[Score, Sample]]`) and
  marginalize plates (`over G` → `[over=G]`).

## The migration chain

The chain is declared in
[`src/quivers/cli/migrations/__init__.py`](https://github.com/FACTSlab/quivers/blob/main/src/quivers/cli/migrations/__init__.py)
as the tuple `CHAIN`. Each adjacent pair has a module:

| Pair | Status |
| ---- | ------ |
| `v0.2.0 → v0.3.0` | identity scaffold (no converters yet) |
| `v0.3.0 → v0.4.0` | identity scaffold |
| `v0.4.0 → v0.5.0` | identity scaffold |
| `v0.5.0 → v0.6.0` | identity scaffold |
| `v0.6.0 → v0.7.0` | identity scaffold |
| `v0.7.0 → v0.9.0` | identity scaffold |
| `v0.9.0 → v0.10.0` | identity (grammar byte-identical) |
| `v0.10.0 → v0.11.0` | full homogenization hop (all in-tree examples) |
| `v0.11.0 → v0.14.0` | identity (target grammar is an extension) |
| `v0.14.0 → v0.15.0` | full token-local converter; removed compose operators raise `MigrationError` |

The 0.10.0 → 0.11.0 and 0.14.0 → 0.15.0 hops have full converters.
The earlier scaffold hops parse and pass their source through
unchanged. They become non-trivial when older source needs lowering;
the
[`SOURCE_RULE_COVERAGE`](#-check-mode-coverage-against-the-vcs)
machinery makes the missing converters discoverable.

## `--check` mode: coverage against the VCS

The panproto VCS at `grammars/qvr/vcs/.panproto/` holds one commit
per distinct grammar revision. `qvr migrate --check` walks every
adjacent pair on `CHAIN`, computes
`panproto.diff_schemas(src_schema, tgt_schema)`, and reports:

- **added**: rules that appear in the target's grammar but not the
  source's.
- **removed**: rules that appear in the source's grammar but not
  the target's.
- **UNCOVERED removed rules**: rules removed at the target whose
  corresponding hop migrator has no entry in its
  `SOURCE_RULE_COVERAGE` set. Each one is a missing converter that
  will silently let source bytes through; the resulting target
  source will be invalid.

The command exits non-zero when any pair has uncovered removals,
which makes it CI-suitable.

Sample output:

```
v0.6.0 -> v0.7.0:
    removed: quantale_decl
    added:   algebra_decl
    UNCOVERED removed rules (no converter): quantale_decl

v0.10.0 -> v0.11.0:
    removed: (the homogenization-removed kinds)
    added:   (object_decl, morphism_decl, composition_decl, ...)
    all removed rules have converters [OK]
```

To clear an "uncovered" entry: write a converter for the rule in
the corresponding hop module and add the rule name to that
module's `SOURCE_RULE_COVERAGE` frozenset.

## VCS blame on migration failure

When a migrator encounters a top-level declaration whose `kind` it
has no converter for, it queries the panproto VCS for the rule's
history and writes a diagnostic to stderr alongside the
pass-through:

```
qvr migrate [v0.5.0 -> v0.6.0]: no converter for 'continuous_decl'.
VCS blame: introduced at v0.4.0; last present at v0.4.0.
```

This points the user at the precise release that needs a converter
written. The migration continues with the source bytes passed
through verbatim, which usually surfaces as a final-stage parse
error against the target grammar.

## Adding a new release

When a new QVR release ships:

1. Tag the release in git: `git tag v0.X.Y`.
2. Rebuild the VCS schema chain:
   ```bash
   python grammars/qvr/vcs/build_schemas.py
   ```
   Adds a new commit to `grammars/qvr/vcs/.panproto/` only if the
   tagged `grammars/qvr/grammar.js` differs in bytes from the
   previous tag's. Releases with identical grammars share commits
   (see `v0.10.0` / `v0.9.0` today).
3. Rebuild the per-revision parser:
   ```bash
   python grammars/qvr/vcs/build_parsers.py
   ```
   Produces `grammars/qvr/vcs/parsers/v0.X.Y/qvr.{dylib,so,dll}`.
4. Append the new revision to `CHAIN` in
   `src/quivers/cli/migrations/__init__.py`.
5. If the new grammar differs structurally: write
   `vP_Q_R_to_v0_X_Y.py` with per-decl converters and a
   `SOURCE_RULE_COVERAGE` frozenset listing every source rule it
   handles. Register it in `MIGRATORS`.
6. If the new grammar is byte-identical to the previous release:
   add a tiny identity module like `v0_9_0_to_v0_10_0.py` (a
   `migrate` function that returns its argument unchanged) and
   register it in `MIGRATORS`.
7. Run `qvr migrate --check` to confirm the new hop's coverage is
   complete.

## The panproto VCS chain

`grammars/qvr/vcs/` holds a panproto repository whose commits track
grammar evolution.

- One commit per distinct grammar revision; each commit holds a
  panproto `Schema` whose vertices are the rule names in the
  grammar's `grammar.json` and whose edges are the structural
  fan-out between rules. Vertices keyed by rule name means
  panproto's auto-derivation recognizes unchanged rules in O(1).
- Each commit tagged with the matching git tag (`v0.X.Y`); the
  working-tree grammar commits un-tagged.
- Used by `qvr migrate --check` to compute schema diffs and by the
  blame diagnostic to identify when a rule was introduced or
  removed.

The Python migrators do NOT consult the VCS at runtime to PERFORM
the migration; they are hand-written walks over the parsed source
schema. The VCS provides authoritative grammar history and powers
the coverage / blame tooling layered on top.

## Limitations and planned work

- **Earlier hops are identity scaffolds.** Migrating source from
  v0.2.0–v0.8.0 lineage will pass the source through unchanged at
  every hop until those modules are filled in. `qvr migrate
  --check` lists the rules each hop still needs converters for.
- **Interior-bracket comments in inline forms.** A `#` comment
  inside a single-line `[...]` / `(...)` / `{...}` cannot exist:
  the grammar forbids newlines in inline forms. The user must
  switch to multi-line form (newline immediately after the
  opener) to retain interior comments.
- **No backward migration.** `qvr migrate` only composes forward
  along `CHAIN`. Backward migration (rendering newer source as
  older) is not implemented.
- **No Schema-construction emit.** The migrator currently emits
  target source via per-declaration text construction validated
  through `lens.parse`, not via `SchemaBuilder` +
  `emit_pretty`. The construction-by-Schema path depends on
  several panproto upstream issues to resolve before it's the
  default; see
  [the panproto issues filed by quivers](https://github.com/panproto/panproto/issues?q=quivers).

## Related

- [`grammars/qvr/vcs/README.md`](https://github.com/FACTSlab/quivers/blob/main/grammars/qvr/vcs/README.md):
  the VCS scaffolding for grammar authors.
- The
  [DSL overview](dsl-overview.md)
  for the current (`v0.11.0`-shaped) source-level surface.
