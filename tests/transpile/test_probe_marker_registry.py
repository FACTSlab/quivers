"""The `probe` marker registry, checked against the source.

[`PROBE_MODULES`][tests.transpile.conftest.PROBE_MODULES] decides which
tests carry the `probe` marker, and the fast CI tier deselects that
marker to buy turnaround. A hand-maintained list of that kind rots
quietly: a module added later launches containers, nobody adds it, and
the tier that was supposed to skip only the expensive tests silently
starts running them, or worse, the tier meant to gate a release stops
covering them.

So the list is derived from the source here and compared, rather than
trusted.
"""

from __future__ import annotations

import ast
import pathlib

from tests.transpile.conftest import PROBE_MODULES


_TRANSPILE_DIR = pathlib.Path(__file__).parent

#: Modules that call `run_probe` without being probe tiers themselves.
#: Only the driver's own unit tests qualify: they monkeypatch the
#: launch to prove a cache hit skips it, so no container is involved
#: and the fast tier can run them.
_NOT_PROBE_TIERS = frozenset({"test_probe_cache"})


def _modules_calling_run_probe() -> set[str]:
    """Stems of the modules under `tests/transpile/` whose source
    contains a call to `run_probe`.

    Parsed rather than grepped, so a mention inside a docstring or a
    comment does not count as a call.
    """
    found: set[str] = set()
    for path in sorted(_TRANSPILE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id
                if isinstance(func, ast.Name)
                else None
            )
            if name == "run_probe":
                found.add(path.stem)
                break
    return found


def test_every_module_that_launches_a_probe_is_registered() -> None:
    """A module calling `run_probe` must carry the marker."""
    callers = _modules_calling_run_probe() - _NOT_PROBE_TIERS
    unregistered = sorted(callers - PROBE_MODULES)
    assert not unregistered, (
        f"{unregistered!r} call `run_probe` but are absent from "
        f"`PROBE_MODULES`, so their tests carry no `probe` marker and "
        f"the fast tier would run them. Add them to the registry in "
        f"`tests/transpile/conftest.py`, or, if the call is not a real "
        f"container launch, to `_NOT_PROBE_TIERS` here with the reason."
    )


def test_the_registry_names_no_module_that_stopped_launching_probes() -> None:
    """And the converse, so the release tier's marker keeps meaning
    what it says. A registered module that no longer launches anything
    is a cell the fast tier skips for nothing."""
    callers = _modules_calling_run_probe()
    stale = sorted(PROBE_MODULES - callers)
    assert not stale, (
        f"{stale!r} are registered in `PROBE_MODULES` but no longer "
        f"call `run_probe`. The fast tier is skipping them for no "
        f"reason; drop them from the registry."
    )


def test_every_registered_module_exists() -> None:
    """A renamed file must not leave a dangling registry entry, which
    would silently register nothing at all."""
    missing = sorted(
        stem for stem in PROBE_MODULES if not (_TRANSPILE_DIR / f"{stem}.py").exists()
    )
    assert not missing, (
        f"`PROBE_MODULES` names {missing!r}, which do not exist under "
        f"{_TRANSPILE_DIR}. A renamed module leaves the marker applied "
        f"to nothing."
    )
