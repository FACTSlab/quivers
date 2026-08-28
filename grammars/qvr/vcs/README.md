# QVR grammar VCS

This directory holds a panproto VCS (`.panproto/`) that tracks the
QVR grammar's evolution as a chain of panproto schema objects. It
drives batch migration of `.qvr` files in the
repository (examples, doc code blocks, regression fixtures) whenever
the grammar or AST shape changes.

## Layout

```
grammars/qvr/vcs/
  .panproto/        # panproto repository (content-addressed store)
  schemas/          # one panproto schema JSON per grammar release
    qvr-prior.json     # the surface before homogenization (object/space/
                       # alias/morphism/kernel/embed/discretize)
    qvr-head.json      # the surface after homogenization (type/morphism
                       # with role, unified option block, leading-keyword
                       # program steps)
  README.md         # this file
```

## Workflow

Every grammar release follows three steps:

1. Generate the panproto schema for the new grammar. Two paths:
   * **Preferred** (once `panproto-grammars-all` vendors the
     head-of-tree QVR grammar): parse a representative ``.qvr``
     file through `panproto.AstParserRegistry().parse_with_protocol`
     and let panproto extract the schema, then `schema.to_json()`
     into `schemas/qvr-<release>.json`.
   * **Bootstrap** (until panproto vendors the grammar): hand-write
     a schema JSON whose vertices are the new node kinds and whose
     edges are the field relations the grammar declares. The two
     starter files (`qvr-prior.json`, `qvr-head.json`) follow this
     bootstrap shape; treat them as anchors, replace with native
     panproto output as soon as the parser is available.
2. `schema add schemas/<new>.json && schema commit -m "<surface
   description>"`. Tag the commit if the release ships externally.
3. Run the batch migration over `.qvr` sources:
   `qvr migrate --from <prior-tag> --to HEAD <paths>`. The CLI
   lives at `src/quivers/cli/migrate.py` and drives panproto's
   `migrate_model` machinery; on success every targeted file
   parses and compiles against the head grammar.

## Why a VCS instead of hand-editing files

A grammar/AST change typically touches every `.qvr` source in the
tree. Hand-editing each one re-introduces drift, misses fenced doc
blocks, and forks the migration logic across files. A single
panproto migration centralizes the schema delta and reduces drift.
The migration command still parses and compiles its outputs; schema
construction alone does not guarantee a valid program.

## Out of scope

This VCS migrates `.qvr` source files only. The TUI, CLI, LSP
server, editor extensions, and pygments lexer all need direct
human updates whenever the surface grammar changes (auto-indent
triggers, completion-keyword lists, semantic-token classifier,
TextMate / Zed highlight queries). The migration engine does not
touch those.
