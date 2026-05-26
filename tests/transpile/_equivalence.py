"""Measure-equivalence check for two log-density evaluators.

The transpile correctness contract is:

> For a probabilistic program P defining a joint density `p_P(θ, y)`
> on a fixed (parameter, data) space with a fixed base measure, and
> the QVR-source program Q on the same spaces, **P is equivalent to Q
> iff there exists a constant `c ∈ ℝ` such that
> `log p_P(θ, y) − log p_Q(θ, y) = c` for every `(θ, y)` in the
> parameter / data support**.

This module implements that contract as a single function: given two
sequences of log-density values at the *same* sequence of
parameter/data points, assert the maximum deviation of the
pointwise difference from its mean is below a floating-point
tolerance. The mean absorbs the additive constant (Jacobian terms,
`log Z` differences from different framings).

The same helper applies symmetrically for backend-vs-backend
comparisons; transitivity is enforced separately by
[`assert_transitive`][tests.transpile._equivalence.assert_transitive].
"""

from __future__ import annotations

import math


_DEFAULT_ATOL = 1e-6
"""Per-point absolute tolerance on the constant-spread check. Set
slightly above the float64 round-off floor (~1e-15 per arithmetic
op, scaled by the number of ops in a typical log-density)."""


def assert_log_density_match(
    qvr_lps: list[float],
    target_lps: list[float],
    *,
    atol: float = _DEFAULT_ATOL,
    context: str = "",
) -> float:
    """Assert two log-density sequences differ by a constant.

    Parameters
    ----------
    qvr_lps
        Log-density values produced by evaluating the QVR program at a
        sequence of (θ, y) test points.
    target_lps
        Log-density values produced by the transpiled backend at the
        *same* sequence of test points (same order).
    atol
        Absolute tolerance on the maximum spread of the per-point
        differences ``target_lps[i] - qvr_lps[i]``. The mean of these
        differences is subtracted out before the max-abs is taken,
        absorbing any genuine additive constant.
    context
        Free-form string included in the failure message
        (e.g. ``"stan@beta_bernoulli"``).

    Returns
    -------
    float
        The constant ``c`` (mean of pointwise differences). Callers
        can pass this constant on to
        [`assert_transitive`][tests.transpile._equivalence.assert_transitive].

    Raises
    ------
    AssertionError
        If the spread exceeds ``atol`` or if the two sequences have
        different lengths.
    """
    if len(qvr_lps) != len(target_lps):
        raise AssertionError(
            f"{context + ': ' if context else ''}"
            f"length mismatch: {len(qvr_lps)} vs {len(target_lps)}"
        )
    n = len(qvr_lps)
    if n == 0:
        raise AssertionError(
            f"{context + ': ' if context else ''}"
            "empty point set; equivalence is vacuous"
        )
    diffs = [t - q for q, t in zip(qvr_lps, target_lps)]
    for i, d in enumerate(diffs):
        if not math.isfinite(d):
            raise AssertionError(
                f"{context + ': ' if context else ''}"
                f"non-finite difference at index {i}: "
                f"qvr={qvr_lps[i]!r}, target={target_lps[i]!r}"
            )
    mean = sum(diffs) / n
    spread = max(abs(d - mean) for d in diffs)
    if spread > atol:
        worst = max(range(n), key=lambda i: abs(diffs[i] - mean))
        raise AssertionError(
            f"{context + ': ' if context else ''}"
            f"log-density spread {spread:.6e} exceeds atol {atol:.6e}; "
            f"constant c = {mean:.6e}; worst point index {worst} "
            f"(qvr={qvr_lps[worst]!r}, target={target_lps[worst]!r}, "
            f"diff={diffs[worst]!r})"
        )
    return mean


def assert_transitive(
    c_target_a_minus_qvr: float,
    c_target_b_minus_qvr: float,
    target_a_lps: list[float],
    target_b_lps: list[float],
    *,
    atol: float = _DEFAULT_ATOL,
    context: str = "",
) -> None:
    """Assert pairwise transitivity across two backends.

    If `(QVR, A)` differ by constant `c_a` and `(QVR, B)` differ by
    constant `c_b`, then `(A, B)` must differ by ``c_b - c_a``
    everywhere. This is the third leg of the equivalence triangle.
    """
    expected = c_target_b_minus_qvr - c_target_a_minus_qvr
    assert_log_density_match(
        target_a_lps,
        [b - expected for b in target_b_lps],
        atol=atol,
        context=f"{context} (transitivity)" if context else "transitivity",
    )


# ---------------------------------------------------------------------------
# Deterministic point sets per fixture.
# ---------------------------------------------------------------------------


def deterministic_grid(
    boundaries: dict[str, tuple[float, float]],
    *,
    points_per_axis: int = 5,
    cap: int = 256,
) -> list[dict[str, float]]:
    """Build a deterministic parameter grid.

    Each entry in ``boundaries`` maps a parameter name to ``(lo,
    hi)``. The grid is the tensor product of ``points_per_axis``
    equally-spaced points per axis (inclusive of endpoints, with a
    small inward shift so boundary-of-support is sampled without
    sitting exactly on the singularity). Capped at ``cap`` total
    points; if the tensor product exceeds the cap, a deterministic
    quasi-random sub-sample is taken via a Halton sequence over the
    remaining axes.
    """
    if not boundaries:
        return [{}]
    axes = sorted(boundaries)
    points_per: dict[str, list[float]] = {}
    for axis in axes:
        lo, hi = boundaries[axis]
        # Inward shift so boundary points don't sit on a Jacobian
        # singularity (e.g. log(0) at a Beta-prior boundary).
        shift = 0.01 * (hi - lo)
        lo2 = lo + shift
        hi2 = hi - shift
        if points_per_axis == 1:
            points_per[axis] = [(lo2 + hi2) / 2]
        else:
            step = (hi2 - lo2) / (points_per_axis - 1)
            points_per[axis] = [lo2 + i * step for i in range(points_per_axis)]

    grid: list[dict[str, float]] = [{}]
    for axis in axes:
        next_grid: list[dict[str, float]] = []
        for partial in grid:
            for v in points_per[axis]:
                child = dict(partial)
                child[axis] = v
                next_grid.append(child)
        grid = next_grid
        if len(grid) > cap:
            grid = _halton_subsample(grid, cap, axes)
    return grid[:cap]


def _halton_subsample(
    grid: list[dict[str, float]], target: int, axes: list[str]
) -> list[dict[str, float]]:
    """Quasi-random subsample down to ``target`` items via Halton
    sequence indices."""
    del axes
    if len(grid) <= target:
        return grid
    indices: list[int] = []
    for i in range(target):
        h = _halton(i, base=2)
        indices.append(int(h * len(grid)) % len(grid))
    seen: set[int] = set()
    out: list[dict[str, float]] = []
    for idx in indices:
        while idx in seen:
            idx = (idx + 1) % len(grid)
        seen.add(idx)
        out.append(grid[idx])
    return out


def _halton(index: int, base: int = 2) -> float:
    """Halton sequence value at ``index`` (base ``base``)."""
    f = 1.0
    r = 0.0
    i = index + 1  # Halton is 1-indexed
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


__all__ = [
    "assert_log_density_match",
    "assert_transitive",
    "deterministic_grid",
]
