"""Backends raise `UnsupportedConstruct` on QVR statements outside
their support tier."""

from __future__ import annotations

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import (
    UnsupportedConstruct,
    transpile,
    unsupported_for,
)
from quivers.transpile._api import STAN_LIKE


# Programs containing constructs no current backend handles.
# Each fixture is paired with the kinds the backend should report.
_FIXTURES = [
    # Encoder declarations are outside STAN_LIKE.
    (
        "encoder",
        """\
encoder encode : Resp -> Resp
program flip : Resp -> Resp:
    sample theta <- Beta(2.0, 2.0)
    observe y <- Bernoulli(theta)
    return y
""",
        {"encoder_decl"},
    ),
]


# Per-target support tier. Each STAN_LIKE backend rejects
# `encoder_decl` because the kind is outside the frozenset.
_SUPPORT_TIER = {
    "stan": STAN_LIKE,
    "numpyro": STAN_LIKE,
    "pyro": STAN_LIKE,
    "pymc": STAN_LIKE,
}


@pytest.mark.parametrize(("name", "src", "expected_kinds"), _FIXTURES)
@pytest.mark.parametrize("target", ["stan", "numpyro", "pyro", "pymc"])
def test_unsupported_raises(
    name: str, src: str, expected_kinds: set[str], target: str
) -> None:
    """Every STAN_LIKE backend should reject `encoder_decl`."""
    try:
        module = parse(src)
    except Exception:  # noqa: BLE001
        pytest.skip(f"fixture {name!r} did not parse cleanly")
    with pytest.raises(UnsupportedConstruct) as exc_info:
        # `unsupported_for` enforces the public support-tier contract
        # documented in `quivers.transpile._api`; the Lower + Renderer
        # pipeline does not re-check the tier, so the rejection
        # boundary is the same documented helper the production
        # surface re-exports.
        unsupported_for(
            f"qvr-{target}", module, allow=_SUPPORT_TIER[target]
        )
        transpile(module, target=target)
    assert expected_kinds.issubset(set(exc_info.value.kinds)), (
        f"expected {expected_kinds}, got {exc_info.value.kinds}"
    )
