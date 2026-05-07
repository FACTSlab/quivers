# CLI: `qvr`

The ``qvr`` console script ships with the package as a thin wrapper
around the parser, constraint solver, and compiler. Subcommands:

## `qvr check FILES...`

Parse, run the constraint solver, and compile every supplied
``.qvr`` file. Emits structured diagnostics; exits 0 on full
success, 1 on any error.

Flags:

- ``--json`` — emit a single JSON document on stdout containing the
  full diagnostic list. Suitable for CI / pre-commit hooks.

Diagnostic codes:

- ``parse`` — tree-sitter rejected the source.
- ``compile`` — the compiler raised :class:`CompileError`.
- ``residuated_constraint`` — a `TypeSlash` pattern appears outside
  a residuated context.
- ``effect_constraint`` — a `TypeEffectApply` references an effect
  whose name doesn't match the conventional pattern.
- ``bundle_unknown_member`` — a ``bundle`` declaration references a
  member that isn't a declared rule, schema, bundle, or built-in
  schema.
- ``io`` — file-system error (file not found, permission denied).

## Module reference

::: quivers.cli
::: quivers.cli.check
