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


_DEFAULT_ATOL = 5e-4
"""Floor tolerance for the constant-spread check when no fixture-shape
signal is available.

Empirical cross-backend agreement at this scale:

* beta_bernoulli (50 Bernoulli obs): Stan vs QVR ~4.4e-6,
  NumPyro vs QVR ~5e-10 (with float64 enabled).
* bayes_linear_regression (60 Normal obs with let-derived mu):
  NumPyro vs QVR ~2.1e-4.

5e-4 sits about an order above the empirical 60-obs floor and far
below the smallest semantic discrepancy a real bug would produce
(parameter swap = at least 0.01 nat per point on a non-trivial
range; family swap = orders of magnitude more). Prefer the
adaptive estimator
[`adaptive_atol`][tests.transpile._equivalence.adaptive_atol]
over this floor when the observed-data count is known."""


_PER_OBS_ROUNDOFF_ESTIMATE = 5e-16
"""Float64 per-call round-off contributed by one observation-site
``Distribution.log_prob`` evaluation, measured empirically across
torch / cmdstanpy / NumPyro on the gallery fixtures. The spread
grows roughly linearly with the observed-data count because each
log-prob sum accumulates this round-off and the outer sum
compounds it."""


_TOLERANCE_HEADROOM = 100.0
"""Multiplier applied above the round-off estimate to give the
adaptive tolerance enough headroom to absorb genuine cross-backend
constant differences (e.g. one backend computes
``Normal.log_prob`` through `(y-mu)/sigma` vs `(y-mu) * (1/sigma)`,
which disagree at the last 1-2 ULPs but accumulate). Two orders of
magnitude above the per-point floor is empirically the sweet spot:
high enough to never trip on benign round-off, low enough to catch
a parameter swap (which is at least 1e-2 per point on the gallery
fixtures' parameter ranges)."""


def adaptive_atol(
    *, n_obs: int, condition_number: float = 1.0
) -> float:
    """Per-fixture adaptive tolerance for
    [`assert_log_density_match`][tests.transpile._equivalence.assert_log_density_match].

    Parameters
    ----------
    n_obs
        Number of observed-data sites the log-density evaluation
        sums over. The constant-spread tolerance grows linearly in
        ``n_obs`` because each per-site `Distribution.log_prob`
        evaluation contributes
        [`_PER_OBS_ROUNDOFF_ESTIMATE`][tests.transpile._equivalence._PER_OBS_ROUNDOFF_ESTIMATE]
        of float64 round-off and the outer sum accumulates it.
    condition_number
        Multiplier reflecting numerical conditioning of the fixture
        (e.g. the ratio of largest to smallest eigenvalue of a
        covariance matrix). Defaults to 1.0 for fixtures with no
        matrix ops. The
        [`ill_conditioned_mvn`][tests.transpile.test_numeric_equivalence._NUMERICALLY_FRAGILE]
        case sits around 1e10 conditioning; the adaptive estimator
        scales tolerance with the condition number for those.

    Returns
    -------
    float
        Maximum of the
        [`_DEFAULT_ATOL`][tests.transpile._equivalence._DEFAULT_ATOL]
        floor and ``n_obs * condition_number *
        _PER_OBS_ROUNDOFF_ESTIMATE * _TOLERANCE_HEADROOM``. The
        max-with-floor keeps single-observation fixtures from
        getting a tolerance tighter than the floor (where benign
        cross-backend constant offsets already eat the headroom).

    Notes
    -----
    The adaptive estimator is monotone in ``n_obs`` and
    ``condition_number`` -- larger fixtures get larger tolerances,
    so the spread-bug detection threshold doesn't shrink as
    coverage grows. The estimator never returns a value smaller
    than the floor; passing ``condition_number=0`` (degenerate)
    falls through to the floor.

    A real bug (parameter swap, family swap, factor structure
    error) produces a non-constant spread of at least 1e-2 per
    point on the gallery's parameter ranges, three orders of
    magnitude above any tolerance this estimator can return for
    real-world fixtures (a 1000-obs ill-conditioned MVN at
    condition_number=1e10 still returns 5e-1 nats, which is two
    orders below the 1e-2 per-point bug-detection threshold).
    """
    if n_obs <= 0:
        return _DEFAULT_ATOL
    adaptive = (
        n_obs * max(condition_number, 1.0)
        * _PER_OBS_ROUNDOFF_ESTIMATE * _TOLERANCE_HEADROOM
    )
    return max(_DEFAULT_ATOL, adaptive)


def assert_log_density_match(
    qvr_lps: list[float],
    target_lps: list[float],
    *,
    atol: float = _DEFAULT_ATOL,
    context: str = "",
    labels: list[str] | None = None,
    min_points: int = 1,
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
    labels
        Optional per-point description, same length as the point set
        (e.g. the
        [`perturbation_labels`][tests.transpile._gallery_data.perturbation_labels]
        of a gallery point list). When supplied, the failure message
        names the perturbation carried by the worst point and the full
        per-point difference table, so a broken constancy localises to
        the section that moved -- latents, data, or both -- instead of
        reporting only that some point disagreed.
    min_points
        Smallest point count at which the caller considers this check
        meaningful. The constant-spread contract is a statement about
        *variation* of the difference across points, so a single point
        satisfies it identically: ``max_i |d_i − mean(d)|`` is exactly
        0 when ``n == 1``, whatever the two evaluators computed. A
        caller whose contract needs real variation passes
        ``min_points=2`` (or higher) and gets a loud failure instead of
        a vacuous pass if its point set ever collapses. Defaults to 1
        for callers that legitimately compare a fixed single point.

    Returns
    -------
    float
        The constant ``c`` (mean of pointwise differences). Callers
        can pass this constant on to
        [`assert_transitive`][tests.transpile._equivalence.assert_transitive].

    Raises
    ------
    AssertionError
        If the spread exceeds ``atol``, if the two sequences have
        different lengths, or if the point count is below
        ``min_points``.

    Notes
    -----
    The tolerance is deliberately *not* scaled by the point count. The
    quantity bounded is a deviation from a mean, not a sum over points,
    so it does not grow with ``n``; the per-point round-off that
    [`adaptive_atol`][tests.transpile._equivalence.adaptive_atol]
    models scales with the *observation* count inside one evaluation,
    which is unchanged by adding more evaluation points. Widening
    ``atol`` because a larger point set started failing would convert a
    detected measure-inequivalence back into a pass.
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
    if n < min_points:
        raise AssertionError(
            f"{context + ': ' if context else ''}"
            f"{n} evaluation point(s) but the caller requires at "
            f"least {min_points}: the constant-spread contract is a "
            f"statement about how the difference varies across "
            f"points, so it passes unconditionally on a point set "
            f"this small"
        )
    if labels is not None and len(labels) != n:
        raise AssertionError(
            f"{context + ': ' if context else ''}"
            f"labels length {len(labels)} does not match the "
            f"{n}-point set"
        )
    diffs = [t - q for q, t in zip(qvr_lps, target_lps)]
    for i, d in enumerate(diffs):
        if not math.isfinite(d):
            raise AssertionError(
                f"{context + ': ' if context else ''}"
                f"non-finite difference at index {i}"
                f"{_label_suffix(labels, i)}: "
                f"qvr={qvr_lps[i]!r}, target={target_lps[i]!r}"
            )
    mean = sum(diffs) / n
    spread = max(abs(d - mean) for d in diffs)
    if spread > atol:
        worst = max(range(n), key=lambda i: abs(diffs[i] - mean))
        table = "; ".join(
            f"[{i}]{_label_suffix(labels, i)} qvr={qvr_lps[i]:.6f} "
            f"target={target_lps[i]:.6f} diff={diffs[i]:.6f}"
            for i in range(n)
        )
        raise AssertionError(
            f"{context + ': ' if context else ''}"
            f"log-density spread {spread:.6e} exceeds atol {atol:.6e}; "
            f"constant c = {mean:.6e}; worst point index {worst}"
            f"{_label_suffix(labels, worst)} "
            f"(qvr={qvr_lps[worst]!r}, target={target_lps[worst]!r}, "
            f"diff={diffs[worst]!r}). Per-point table: {table}"
        )
    return mean


def _label_suffix(labels: list[str] | None, index: int) -> str:
    """`" (<label>)"` when the caller supplied labels, else `""`."""
    if labels is None:
        return ""
    return f" ({labels[index]})"


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
    "adaptive_atol",
    "assert_log_density_match",
    "assert_transitive",
    "deterministic_grid",
]
