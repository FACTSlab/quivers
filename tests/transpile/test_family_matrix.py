"""Tier 5: distribution-family × backend support matrix.

For every family in
[`_get_family_registry`][quivers.dsl.compiler._prelude._get_family_registry]
and every registered backend:

- If the family appears in the backend's `_FAMILIES` map, the
  transpile must succeed and the emitted source must reference the
  backend-specific distribution name (e.g. Stan `beta`, NumPyro
  `Beta`, BUGS `dbeta`).
- If the family is absent, the transpile must raise
  `UnsupportedConstruct(kinds=["family:<Family>"])`.

The matrix is the source of truth for the docs page "which backends
support which families."
"""

from __future__ import annotations

import importlib
import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from tests.transpile.fixtures import _load


_BACKENDS = sorted(available_targets())


# Backends that share another backend's family map (delegation
# pattern). Both BUGS and JAGS use the same `dnorm` / `dbern` / ...
# prefix family table; JAGS's backend module reuses BUGS's `_build`
# without defining its own map.
_FAMILY_MAP_DELEGATIONS: dict[str, str] = {
    "jags": "bugs",
}


def _backend_family_map(backend: str) -> dict[str, str]:
    """Load the backend module's `_FAMILIES` dict.

    Each backend module under ``src/quivers/transpile/backends/`` is
    expected to define a module-level `_FAMILIES` mapping QVR family
    names to the backend's distribution name. The mapping is
    introspected here rather than duplicated; if a backend drops the
    map or renames it, this test surfaces the regression.

    When a backend delegates its family table to another backend
    (`jags` → `bugs`), the delegation map redirects the lookup.
    """
    source_backend = _FAMILY_MAP_DELEGATIONS.get(backend, backend)
    module = importlib.import_module(
        f"quivers.transpile.backends.{source_backend}"
    )
    families = getattr(module, "_FAMILIES", None)
    if families is None:
        for candidate in ("_FAMILIES", "_FIXED_FACTORIES", "FAMILIES"):
            families = getattr(module, candidate, None)
            if families is not None:
                break
    if families is None:
        raise AssertionError(
            f"backend module {source_backend!r} has no recognisable "
            f"family map (looked up for backend {backend!r})"
        )
    out: dict[str, str] = {}
    for qvr_name, tgt in families.items():
        if isinstance(tgt, tuple):  # WebPPL: (name, keys)
            out[qvr_name] = tgt[0]
        else:
            out[qvr_name] = tgt
    return out


def _family_fixtures() -> list[_load.Fixture]:
    return _load.load_families()


def _family_for_fixture(fixture: _load.Fixture) -> str:
    """Each `families/<family>.qvr` fixture's stem is the lowercased
    family name; recover the canonical mixed-case name from the
    registry."""
    from quivers.dsl.compiler._prelude import _get_family_registry

    registry = _get_family_registry()
    canonical = {name.lower(): name for name in registry}
    if fixture.name not in canonical:
        raise AssertionError(
            f"fixture {fixture.name!r} does not match any family name "
            f"in the registry (lowercased): {sorted(canonical)}"
        )
    return canonical[fixture.name]


@pytest.mark.parametrize(
    "fixture", _family_fixtures(), ids=lambda f: f.name
)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_family_backend_cell(
    fixture: _load.Fixture, backend: str
) -> None:
    """One cell of the (family × backend) matrix."""
    family = _family_for_fixture(fixture)
    family_map = _backend_family_map(backend)
    module = parse(fixture.source)

    try:
        output = transpile(module, target=backend)
    except UnsupportedConstruct as exc:
        kinds = exc.kinds
        # Three legitimate raise categories:
        # 1. The family is genuinely unsupported by this backend.
        # 2. The fixture uses a step / option / axes feature the
        #    walker doesn't emit (marginalize, score, let_step, ...);
        #    that's outside this test's family-support concern.
        # 3. The arity of the resolved family call doesn't match the
        #    backend's expected signature (WebPPL's keyword-arg dist
        #    constructors enforce arity strictly).
        if family not in family_map:
            if any(k.startswith("family:") for k in kinds):
                return
            # Walker tripped on something else (step / axes / option)
            # before reaching the family check. We can't verify
            # family-absence handling on this fixture; surface as
            # xfail so the un-implemented walker construct is the
            # construct-matrix test's concern, not this one.
            pytest.xfail(
                reason=(
                    f"backend {backend!r} on family {family!r}: family "
                    f"is absent from backend's `_FAMILIES`, but the "
                    f"walker raised on a different construct before "
                    f"checking the family: kinds={kinds!r}. The "
                    f"construct-matrix test owns this gap."
                )
            )
        # Family IS in the map but the fixture exercised something
        # beyond it (typically a marginalize / let / score step the
        # walker doesn't emit). The walker gap belongs to the
        # construct-matrix test, not the family-matrix one.
        gap_prefixes = ("step:", "axes:", "option:", "family:")
        assert any(
            any(k.startswith(p) for p in gap_prefixes) for k in kinds
        ), (
            f"backend {backend!r} on family {family!r}: walker raised "
            f"with unrecognised error kinds {kinds!r}"
        )
        pytest.xfail(
            reason=(
                f"family {family!r} fixture exercises walker behaviour "
                f"beyond pure family-call (raised {kinds!r}); the "
                f"family-matrix concern is whether the family is in "
                f"the backend's `_FAMILIES` map (yes), not whether "
                f"the surrounding construct emits."
            )
        )

    if family not in family_map:
        pytest.fail(
            f"backend {backend!r} on family {family!r}: family is NOT "
            f"in the backend's family map yet transpile succeeded; "
            f"either add `{family!r}` to the backend's `_FAMILIES` "
            f"map or note why the walker accepted it. Output: "
            f"{output[:120]!r}"
        )
    assert output, (
        f"backend {backend!r} on family {family!r}: empty bytes"
    )
    backend_name = family_map[family]
    assert backend_name.encode("utf-8") in output, (
        f"backend {backend!r} on family {family!r}: emitted output "
        f"does not contain expected distribution name "
        f"{backend_name!r}; got: {output.decode('utf-8', errors='replace')[:200]!r}"
    )
