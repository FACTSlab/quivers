# Docstring completeness backlog

## Purpose

This is a handoff for finishing docstring coverage across `src/quivers/`. It
is a description of work outstanding, not of released behavior.

The rule it enforces:

> Every callable documents its contract. A callable taking arguments carries a
> `Parameters` section, one returning a value carries `Returns`, a generator
> carries `Yields`, and one that can raise carries `Raises`. A callable doing
> none of those needs only a summary line.

Constructor parameters are documented on the **class** docstring, not on
`__init__`. `__init__` carries no separate parameter list.

## Why this matters more than it looks

A summary-only docstring makes a reader reconstruct the contract from the
body, and the contract is the part that matters at a call site. Undocumented
raises are the worst case: a caller cannot write correct error handling
against an exception nobody wrote down. Several of the sites found here raise
`ValueError` or `KeyError` from deep inside a validation path, and the only
existing record of that is the source.

## The audit

`tools/docstring_audit.py` reports any callable whose docstring omits a
section its own signature and body require. It judges code rather than
matching a template, so it does not flag a parameterless helper for lacking a
`Parameters` section.

```bash
python tools/docstring_audit.py                        # whole package
python tools/docstring_audit.py src/quivers/continuous # one subpackage
python tools/docstring_audit.py --list <path>          # name each callable
python tools/docstring_audit.py --check <path>         # exit 1 if incomplete
```

The tool passes its own `--check`, so it is also a worked example of the
target style.

## Measured state

At the time of writing: **4638 incomplete docstrings across 243 files**, by
subpackage.

| Subpackage | Incomplete |
| --- | ---: |
| `src/quivers/transpile` | 1537 |
| `src/quivers/dsl` | 546 |
| `src/quivers/continuous` | 497 |
| `src/quivers/cli` | 376 |
| `src/quivers/core` | 314 |
| `src/quivers/qiec` | 273 |
| `src/quivers/stochastic` | 244 |
| `src/quivers/inference` | 209 |
| `src/quivers/monadic` | 198 |
| `src/quivers/structural` | 106 |
| `src/quivers/analysis` | 57 |
| `src/quivers/effects` | 56 |

The heaviest single files are `transpile/lower.py` (135),
`continuous/families.py` (133), `cli/repl_session.py` (128), and
`core/algebras.py` (127).

`src/quivers/qiec/effects.py` is already complete and is the reference for
the intended depth: it documents what each argument is *for*, what a `None`
return distinguishes, and why each raise is an error rather than a silent
fallback.

## Suggested order

Public surface before internals, since that is where an undocumented contract
costs a user rather than a maintainer.

1. `continuous/families.py` and `continuous/spaces.py`. These are the
   family constructors users write in models.
2. `core/algebras.py` and `monadic/instances.py`. The categorical surface the
   documentation already links to from prose.
3. `dsl/compiler/` and `dsl/emit.py`. The compile path, where raises carry
   source locations a caller may want to catch.
4. `transpile/renderers/*`. Largest by count and most repetitive: the eleven
   renderers share a shape, so one careful pass sets the pattern for the rest.
   Resist copying a boilerplate sentence into all eleven; a renderer's
   `Raises` differs by what its target cannot express.
5. `cli/`, `inference/`, `stochastic/`, `structural/`, `analysis/`.

## Working rules

- **Do not change behavior.** This is a documentation pass. If a docstring
  cannot be written without discovering the function is wrong, stop and raise
  that separately rather than fixing it silently in a docs commit.
- **Document the raises that escape.** An exception caught internally is not
  part of the contract; one that reaches a caller is. Check the call graph
  where a helper re-raises.
- **Say what a `None` means.** Most `X | None` returns here distinguish two
  real cases, and which one is the whole content of the section.
- **Run the tests for the module you touched.** Docstrings are inert, but a
  bad edit inside a string literal is not, and several modules keep doctest
  style examples.
- **Keep the house voice.** No em-dashes. Match the surrounding file.

## Gate

Once a subpackage reads zero, add it to CI so the gap cannot reopen:

```yaml
- run: python tools/docstring_audit.py --check src/quivers/qiec
```

Extend the path list as subpackages are finished, rather than gating the whole
package at once and leaving the check red for the duration.

## Out of scope

`tests/` is not audited. Test functions carry a docstring stating what the
test pins, which the suite already does consistently, and they take fixtures
rather than a contractual parameter list.
