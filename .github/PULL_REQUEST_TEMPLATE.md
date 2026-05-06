<!--
Thanks for contributing to quivers! Please fill in each section below.
Delete sections that are clearly not applicable to this change.
-->

## Summary

<!--
One or two sentences describing what this PR does and why. Focus on
the *why* — the *what* is in the diff.
-->

## Motivation

<!--
What problem does this solve, or what capability does it add?
Link issues with `Closes #N` / `Refs #N`.
-->

## Changes

<!--
A bulleted list of the substantive changes. Group by package
(core/, dsl/, stochastic/, ...) when the PR touches several.
-->

-

## API impact

<!--
Tick the box that describes the impact on the public API surface.
Quivers is pre-1.0 and prefers clean breaking changes over compat shims.
-->

- [ ] No API change.
- [ ] Additive only (new symbols, no existing behaviour changed).
- [ ] Breaking change. Migration notes:

## Tests

<!--
What did you add or run? Link to specific test files when relevant.
-->

- [ ] `pytest -x` passes.
- [ ] `pyright src/quivers` is clean.
- [ ] `ruff check` and `ruff format --check` pass.
- [ ] New tests added for new behaviour.
- [ ] Tree-sitter fixtures updated (if `grammars/qvr/` changed).

## Documentation

- [ ] Docstrings updated for changed public API.
- [ ] User-facing pages under `docs/` updated.
- [ ] `docs/developer/changelog.md` has a `Unreleased` entry for this change.
- [ ] Denotational semantics under `docs/semantics/` updated (if the change affects the meaning of any DSL construct).

## Checklist

- [ ] Commits are focused and have descriptive messages.
- [ ] No backward-compatibility shims added (we are pre-1.0; prefer clean breaking changes).
- [ ] Comments describe the code as it stands; none reference prior states or removed code.
- [ ] No secrets, credentials, or large binary artefacts in the diff.
