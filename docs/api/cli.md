# CLI: `qvr`

The `qvr` console script ships with the package as a thin wrapper
around the parser, constraint solver, and compiler. Subcommands:

## `qvr check FILES...`

Parse, run the constraint solver, and compile every supplied
`.qvr` file. It emits structured diagnostics and exits 0 on full
success, 1 on any error.

Flags:

- `--json`: emit a single JSON document on stdout containing the
  full diagnostic list, as expected by CI and pre-commit hooks.
- `--target TARGET`: compare every checked QIEC computation with the selected
  transpiler's capabilities. Unsupported features are errors, so this mode is
  suitable for a target-specific release gate.

Diagnostic codes:

- `parse`: tree-sitter rejected the source.
- `compile`: the compiler raised `CompileError`.
- `residuated_constraint`: a `TypeSlash` pattern appears outside
  a residuated context.
- `effect_constraint`: a `TypeEffectApply` references an effect
  whose name doesn't match the conventional pattern.
- `bundle_unknown_member`: a `bundle` declaration references a
  member that isn't a declared rule, schema, bundle, or built-in
  schema.
- `family-arg-shape`: a distribution call has the wrong number of
  arguments or incompatible argument shapes.
- `implicit-family-defaults`: a distribution call relies on deprecated
  implicit defaults. This diagnostic is a warning.
- `qiec-route`: the exact `qvr-source/v0.19` to `qiec-core/v1alpha1`
  Didactic route is unavailable or mismatched.
- `qiec-kind`: a static argument or binder has the wrong kind.
- `qiec-index`: an index expression has the wrong sort or constructor shape.
- `qiec-row`: an effect row violates instance identity, unification, or a
  `lacks` constraint.
- `qiec-coverage`: indexed case branches are incomplete, duplicated, or not
  justified by refinement.
- `qiec-skolem-escape`: branch-local static evidence escapes its scope.
- `qiec-handler`: a handler signature violates coverage, interface evolution,
  or resumption requirements.
- `qiec-unhandled-effect`: a computation performs an instance absent from its
  declared row.
- `qiec-backend`: a requested backend boundary cannot preserve a QIEC form.
- `qiec:capability:FEATURE:COMPUTATION`: `--target` selected a backend that
  cannot preserve `FEATURE` in the named computation.
- `io`: a file-system error, such as a missing or unreadable file.

## `qvr migrate PATHS...`

Lower `.qvr` source forward along the QVR grammar release chain.
The composer chains the adjacent-pair migrators defined in
[`quivers.cli.migrations`](#quivers.cli.migrations) so users do not
have to know the intermediate versions; pinning the boundary with
`--from` / `--to` selects a sub-chain when needed.

Flags:

- `--from VERSION`: required source revision in the chain. QVR files do not
  encode an unambiguous grammar version.
- `--to VERSION`: target revision (defaults to `HEAD`).
- `--dry-run`: report which files would change without writing.
- `--output DIR`: write migrated copies under `DIR` instead
  of overwriting the originals.
- `--check`: compare each adjacent grammar pair with its migrator and
  fail if a removed rule has no converter. This mode does not migrate files.

Directory arguments are walked recursively; individual `.qvr`
files may also be supplied. The migration tooling is built on the
in-tree panproto VCS at `grammars/qvr/vcs/`. A new release adds a
migration without changing the earlier migration steps.

## `qvr repl [FILE]`

Start the interactive REPL. Without a file, opens an empty
session; with a file, loads and elaborates it before dropping to
the prompt. See [REPL and Language Server](../guides/repl-and-lsp.md).

## `qvr run FILE COMPUTATION [ARGUMENTS...]`

Execute one named computation from a checked QIEC module. Each positional
argument is a JSON literal and corresponds to one value parameter in
declaration order. The command validates the arguments and result against the
specialized QIEC types, gives the invocation a fresh runtime attachment table,
and exits 0 only when evaluation succeeds.

Flags:

- `--static NAME=TERM`: specialize one static telescope binder with a closed
  type, index, or effect term. Repeat the flag once for every binder.
- `--runtime FILE.json`: select runtime providers and their options through a
  non-executable JSON configuration.
- `--trace`: write stable execution events to stderr in human-readable mode.
- `--json`: emit the result, runtime label, and complete trace as JSON. A
  failure instead emits a JSON diagnostic and exits 1.

For instance, this command specializes `A` to `Int` and supplies the value
argument `7`:

```bash
qvr run identity.qvr identity 7 --static A=Int --json
```

The default runtime configuration selects the built-in `core` provider. A
configuration can select that provider with handler behavior or name an
installed `quivers.qiec_runtime` entry point:

```json
{
  "providers": [
    {
      "name": "core",
      "options": {
        "handlers": {
          "pass": {"kind": "passthrough"}
        }
      }
    }
  ]
}
```

The core provider accepts `passthrough`, `scripted`, and `state` handler
configurations. These options supply process-local clause behavior; the source
handler still fixes the checked effect, coverage, forwarding policy, clause
set, and resumption grades. Runtime plugins are discoverable by name but are
loaded only when the configuration explicitly selects them.

Execution failures use stable codes including `qiec-run-computation`,
`qiec-run-arity`, `qiec-run-static`, `qiec-run-provider`,
`qiec-run-validator`, `qiec-run-argument`, `qiec-run-evaluation`, and
`qiec-run-result`.

## `qvr lsp`

Run the Language Server over stdio, or pass `--tcp PORT` to bind a
TCP port. `--target TARGET` enables live `qiec:capability:*` diagnostics for
the selected transpiler. Editor extensions invoke this command; the protocol
is LSP 3.17. The server accepts later target changes through
`workspace/didChangeConfiguration`, using either
`{"qvr": {"transpileTarget": "TARGET"}}` or
`{"transpileTarget": "TARGET"}`; an empty target disables these diagnostics.
See [REPL and Language Server](../guides/repl-and-lsp.md).

## `qvr transpile FILE`

Transpile a `.qvr` file to a registered probabilistic-programming
backend. Use `--to BACKEND` for one target, `--to-all` for every
registered target, or `--list-targets` to inspect the registry. The
`--output` and `--out-dir` flags select destinations for single- and
multi-target runs, respectively.

## `qvr kernel`

Install or run the Quivers Jupyter kernel. `qvr kernel install`
registers the kernelspec; `--user` selects the user kernel directory,
and `--prefix PREFIX` selects an explicit Jupyter prefix.

## Module reference

::: quivers.cli
::: quivers.cli.check
::: quivers.cli.run
::: quivers.cli.migrate
::: quivers.cli.migrations
