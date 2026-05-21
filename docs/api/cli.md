# CLI: `qvr`

The ``qvr`` console script ships with the package as a thin wrapper
around the parser, constraint solver, and compiler. Subcommands:

## `qvr check FILES...`

Parse, run the constraint solver, and compile every supplied
``.qvr`` file. Emits structured diagnostics; exits 0 on full
success, 1 on any error.

Flags:

- ``--json``, emit a single JSON document on stdout containing the
  full diagnostic list. Suitable for CI / pre-commit hooks.

Diagnostic codes:

- ``parse``, tree-sitter rejected the source.
- ``compile``, the compiler raised :class:`CompileError`.
- ``residuated_constraint``, a `TypeSlash` pattern appears outside
  a residuated context.
- ``effect_constraint``, a `TypeEffectApply` references an effect
  whose name doesn't match the conventional pattern.
- ``bundle_unknown_member``, a ``bundle`` declaration references a
  member that isn't a declared rule, schema, bundle, or built-in
  schema.
- ``io``, file-system error (file not found, permission denied).

## `qvr migrate PATHS...`

Lower ``.qvr`` source forward along the QVR grammar release chain.
The composer chains the adjacent-pair migrators defined in
[`quivers.cli.migrations`](#quivers.cli.migrations) so users do not
have to know the intermediate versions; pinning the boundary with
``--from`` / ``--to`` selects a sub-chain when needed.

Flags:

- ``--from VERSION``, source revision in the chain (defaults to
  the most recent release).
- ``--to VERSION``, target revision (defaults to ``HEAD``).
- ``--dry-run``, report which files would change without writing.
- ``--output DIR``, write migrated copies under ``DIR`` instead
  of overwriting the originals.

Directory arguments are walked recursively; individual ``.qvr``
files may also be supplied. The migration tooling is built on the
in-tree panproto VCS at ``grammars/qvr/vcs/`` so adding a new
release is purely additive to the migrations package.

## `qvr repl [FILE]`

Start the interactive REPL. Without a file, opens an empty
session; with a file, loads and elaborates it before dropping to
the prompt. See [REPL and Language Server](../guides/repl-and-lsp.md).

## `qvr lsp`

Run the Language Server over stdio. Editor extensions (VS Code,
Zed, Neovim) invoke this; the protocol is LSP 3.17. See
[REPL and Language Server](../guides/repl-and-lsp.md).

## Module reference

::: quivers.cli
::: quivers.cli.check
::: quivers.cli.migrate
::: quivers.cli.migrations
