"""Tier 5: distribution-family × backend support matrix.

For every family in
[`_get_family_registry`][quivers.dsl.compiler._prelude._get_family_registry]
and every registered backend:

- If `FAMILY_META[family].target_names` contains `backend`, the
  transpile must succeed and the emitted source must reference the
  backend-specific distribution name (e.g. Stan `beta`, NumPyro
  `Beta`, BUGS `dbeta`).
- If the family/backend pair is absent from `target_names`, the
  transpile must raise
  `UnsupportedConstruct(kinds=["family:<Family>"])`.

The matrix is the source of truth for the docs page "which backends
support which families."
"""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.transpile.family_meta import FAMILY_META
from tests.transpile.fixtures import _load


_BACKENDS = sorted(available_targets())


def _backend_target_name(family: str, backend: str) -> str | None:
    """Return the per-backend distribution name for ``family`` from
    [`FAMILY_META`][quivers.transpile.family_meta.FAMILY_META], or
    ``None`` if the backend is not registered for the family.

    The new Lower + Renderer pipeline derives every backend-specific
    distribution name from this central table; renderer modules no
    longer carry per-backend `_FAMILIES` dicts.
    """
    meta = FAMILY_META.get(family)
    if meta is None:
        return None
    return meta.target_names.get(backend)


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
    """One cell of the (family × backend) matrix.

    Two pre-declared outcomes:

    - `backend_name is None` (family absent from
      `FAMILY_META[family].target_names` for this backend): the
      pipeline MUST raise `UnsupportedConstruct` with at least one
      kind whose tag begins with ``"family:"``.
    - `backend_name is not None`: the pipeline MUST emit non-empty
      bytes that contain the backend-specific distribution name
      (e.g. Stan ``beta``, NumPyro ``Beta``).
    """
    family = _family_for_fixture(fixture)
    backend_name = _backend_target_name(family, backend)
    module = parse(fixture.source)

    if backend_name is None:
        with pytest.raises(UnsupportedConstruct) as exc_info:
            transpile(module, target=backend)
        kinds = exc_info.value.kinds
        assert any(k.startswith("family:") for k in kinds), (
            f"backend {backend!r} on family {family!r}: expected "
            f"`UnsupportedConstruct` with a `family:`-prefixed kind, "
            f"got kinds={kinds!r}. Either add {backend!r} to "
            f"`FAMILY_META[{family!r}].target_names` (if the renderer "
            f"now supports the family) or fix the renderer to raise "
            f"a family-tagged kind."
        )
        return

    output = transpile(module, target=backend)
    assert output, (
        f"backend {backend!r} on family {family!r}: empty bytes"
    )
    assert backend_name.encode("utf-8") in output, (
        f"backend {backend!r} on family {family!r}: emitted output "
        f"does not contain expected distribution name "
        f"{backend_name!r}; got: "
        f"{output.decode('utf-8', errors='replace')[:200]!r}"
    )
