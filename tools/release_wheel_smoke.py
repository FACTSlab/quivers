"""Exercise an installed Quivers wheel without parser compiler fallbacks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

import quivers
from quivers.cli.migrations import CHAIN, compose_migration, vcs_coverage_report
from quivers.cli.migrations import _grammar as migration_grammar
from quivers.dsl import parse
from quivers.dsl import _grammar_build as grammar_build
from quivers.dsl.pygments_lexer import QvrLexer
from quivers.transpile import available_targets, transpile


_EXPECTED_DEPENDENCIES = {
    "didactic": "0.15.0",
    "panproto": "0.74.2",
    "panproto-grammars-all": "0.74.2",
}


def _check_dependency_versions() -> None:
    """Require the release smoke's intentionally pinned dependency set."""

    for distribution, expected in _EXPECTED_DEPENDENCIES.items():
        try:
            actual = version(distribution)
        except PackageNotFoundError as error:
            raise SystemExit(
                f"installed-wheel smoke is missing {distribution}=={expected}"
            ) from error
        if actual != expected:
            raise SystemExit(
                "installed-wheel smoke requires "
                f"{distribution}=={expected}, found {actual}"
            )


def main() -> None:
    project = Path(sys.argv[1]).resolve()
    package = Path(quivers.__file__).resolve()
    if package.is_relative_to(project / "src"):
        raise SystemExit(f"smoke test imported the source checkout: {package}")
    _check_dependency_versions()

    def no_current_compiler() -> str:
        raise AssertionError("current parser attempted a source build")

    def no_migration_compiler(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("migration parser attempted a source build")

    grammar_build._compiler = no_current_compiler
    migration_grammar._compile_parser = no_migration_compiler

    module = parse("index Nat = Z | S(Nat)\n")
    if len(module.statements) != 1:
        raise SystemExit("installed-wheel v0.19 parser returned the wrong module")
    if not list(QvrLexer().get_tokens("index Nat = Z | S(Nat)\n")):
        raise SystemExit("installed-wheel highlighter returned no tokens")

    qiec_source = parse("define answer() : Int !{} =\n    return 42\n")
    rendered: dict[str, bytes] = {}
    for target in available_targets():
        rendered[target] = transpile(qiec_source, target=target)
        if b"qiec_answer" not in rendered[target]:
            raise SystemExit(
                f"installed-wheel {target} transpiler omitted QIEC computation"
            )
    namespace: dict[str, object] = {}
    exec(rendered["pyro"], namespace)
    if namespace["qiec_answer"]() != 42:  # type: ignore[operator]
        raise SystemExit("installed-wheel Python QIEC runtime returned wrong value")

    available = set(migration_grammar.available_revisions())
    missing = set(CHAIN) - available
    if missing:
        raise SystemExit(f"wheel is missing parser assets for: {sorted(missing)}")
    for revision in CHAIN:
        migration_grammar.registry_for(revision)

    source = b"bundle installed_wheel = [first, second]\n"
    expected = b"bundle installed_wheel : [first, second]\n"
    migrated = compose_migration("v0.14.0", "HEAD")(source)
    if migrated != expected:
        raise SystemExit(f"installed-wheel migration mismatch: {migrated!r}")

    uncovered = [report for report in vcs_coverage_report() if report.uncovered_removed]
    if uncovered:
        raise SystemExit(f"migration coverage has uncovered rules: {uncovered!r}")


if __name__ == "__main__":
    main()
