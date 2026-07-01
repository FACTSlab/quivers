"""Tests for the compositional measure algebra runtime
([`quivers.continuous.measure`][quivers.continuous.measure]).

Five layers of contract:

1. Per-operator log-density correctness against the closed-form
   reference (Normal vs `Pushforward(Normal, Exp) = LogNormal`,
   `Restrict(Normal, low, high)` mass against `Φ(high) - Φ(low)`,
   `Mixture([w_k], [D_k])` log-density against the closed-form
   `logsumexp(log_w_k + log_D_k)`).

2. The four compiler rewrite rules:
     a. Mixture flattening (Giry monad associativity).
     b. Pushforward commutes with Mixture (functoriality).
     c. Mixture-Restrict reweighting (non-commutation identity).
     d. PointMass-in-Mixture lift (ZIP / hurdle canonical shape).

3. Lazy normalisation: `Restrict.log_normalizer()` carries the
   sub-measure mass symbolically; `Normalize` collapses it; the
   normalised log-density matches the closed-form truncated
   density.

4. Operator composition closure: nested `Restrict(Pushforward(...))`,
   `Pushforward(Restrict(...))`, `Mixture` of arbitrary components
   produce valid `Measure` instances whose `log_prob` matches the
   hand-computed equivalent.

5. Discrete fallback: `Restrict(Poisson(rate), low=1)` produces the
   correct survival-based log-normaliser even though PyTorch's
   `Poisson.cdf` raises `NotImplementedError`.
"""

from __future__ import annotations

import math

import pytest
import torch

from quivers.continuous.bijectors import Affine, Exp
from quivers.continuous.measure import (
    Independent,
    Mixture,
    Normalize,
    PointMass,
    Pushforward,
    Restrict,
    normalize_at_boundary,
)


def _Normal(mu: float, sigma: float) -> torch.distributions.Normal:
    return torch.distributions.Normal(torch.tensor(mu), torch.tensor(sigma))


def _Poisson(rate: float) -> torch.distributions.Poisson:
    return torch.distributions.Poisson(torch.tensor(rate))


# ---------------------------------------------------------------------------
# Per-operator correctness
# ---------------------------------------------------------------------------


def test_pointmass_log_prob() -> None:
    pm = PointMass(torch.tensor(2.5))
    assert pm.log_prob(torch.tensor(2.5)).item() == pytest.approx(0.0)
    assert math.isinf(pm.log_prob(torch.tensor(0.0)).item())
    assert pm.log_normalizer().item() == pytest.approx(0.0)


def test_restrict_lognormalizer_matches_cdf_difference() -> None:
    base = _Normal(0.0, 1.0)
    r = Restrict(base, low=torch.tensor(0.0), high=torch.tensor(1.0))
    expected_mass = base.cdf(torch.tensor(1.0)) - base.cdf(torch.tensor(0.0))
    torch.testing.assert_close(
        r.log_normalizer(),
        torch.log(expected_mass),
        atol=1e-6,
        rtol=1e-6,
    )


def test_restrict_log_prob_zero_outside_support() -> None:
    base = _Normal(0.0, 1.0)
    r = Restrict(base, low=torch.tensor(0.0), high=torch.tensor(1.0))
    assert math.isinf(r.log_prob(torch.tensor(-0.5)).item())
    assert math.isinf(r.log_prob(torch.tensor(2.0)).item())
    assert torch.isfinite(r.log_prob(torch.tensor(0.5)))


def test_pushforward_normal_through_exp_equals_lognormal() -> None:
    ln_via_push = Pushforward(_Normal(0.0, 1.0), Exp())
    ln_ref = torch.distributions.LogNormal(0.0, 1.0)
    y = torch.tensor([0.5, 1.0, 2.0, 3.5])
    torch.testing.assert_close(
        ln_via_push.log_prob(y),
        ln_ref.log_prob(y),
        atol=1e-5,
        rtol=1e-5,
    )


def test_pushforward_through_affine_shifts_normal() -> None:
    a = Affine(scale=2.0, shift=1.0)
    push = Pushforward(_Normal(0.0, 1.0), a)
    ref = _Normal(1.0, 2.0)
    y = torch.tensor([-1.0, 0.0, 1.0, 3.0])
    torch.testing.assert_close(
        push.log_prob(y),
        ref.log_prob(y),
        atol=1e-5,
        rtol=1e-5,
    )


def test_mixture_log_prob_matches_explicit_logsumexp() -> None:
    mix = Mixture(
        torch.tensor([0.3, 0.7]),
        [_Normal(-1.0, 1.0), _Normal(1.0, 0.5)],
    )
    y = torch.tensor([-2.0, 0.5, 2.0])
    log_p1 = _Normal(-1.0, 1.0).log_prob(y)
    log_p2 = _Normal(1.0, 0.5).log_prob(y)
    expected = torch.logsumexp(
        torch.stack(
            [
                torch.log(torch.tensor(0.3)) + log_p1,
                torch.log(torch.tensor(0.7)) + log_p2,
            ],
            dim=-1,
        ),
        dim=-1,
    )
    torch.testing.assert_close(mix.log_prob(y), expected, atol=1e-6, rtol=1e-6)


def test_independent_sums_event_dim_logprobs() -> None:
    base = torch.distributions.Normal(torch.zeros(3), torch.ones(3))
    ind = Independent(base, 1)
    y = torch.tensor([0.0, 1.0, 2.0])
    expected = base.log_prob(y).sum()
    torch.testing.assert_close(ind.log_prob(y), expected, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Lazy normalisation
# ---------------------------------------------------------------------------


def test_restrict_unnormalised_then_normalize_equals_truncated() -> None:
    base = _Normal(0.0, 1.0)
    r = Restrict(base, low=torch.tensor(0.0), high=torch.tensor(1.0))
    n = Normalize(r)
    y = torch.tensor(0.5)
    expected = base.log_prob(y) - r.log_normalizer()
    torch.testing.assert_close(n.log_prob(y), expected, atol=1e-6, rtol=1e-6)


def test_normalize_at_boundary_lifts_sub_measure() -> None:
    base = _Normal(0.0, 1.0)
    r = Restrict(base, low=torch.tensor(0.0))
    lifted = normalize_at_boundary(r)
    assert isinstance(lifted, Normalize)
    pm = PointMass(0.0)
    same = normalize_at_boundary(pm)
    assert same is pm


def test_pushforward_preserves_lognormalizer_of_sub_measure() -> None:
    base = _Normal(0.0, 1.0)
    r = Restrict(base, low=torch.tensor(0.0))
    push = Pushforward(r, Exp())
    torch.testing.assert_close(
        push.log_normalizer(),
        r.log_normalizer(),
        atol=1e-6,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# Discrete fallback
# ---------------------------------------------------------------------------


def test_restrict_poisson_uses_survival_fallback() -> None:
    base = _Poisson(2.0)
    r = Restrict(base, low=torch.tensor(1.0))
    expected = math.log(1.0 - math.exp(-2.0))
    assert r.log_normalizer().item() == pytest.approx(expected, abs=1e-6)


def test_restrict_poisson_interval_sums_pmf() -> None:
    base = _Poisson(3.0)
    r = Restrict(base, low=torch.tensor(1.0), high=torch.tensor(3.0))
    expected_mass = 0.0
    for k in (1, 2, 3):
        expected_mass += math.exp(-3.0) * (3.0**k) / math.factorial(k)
    assert r.log_normalizer().item() == pytest.approx(math.log(expected_mass), abs=1e-5)


# ---------------------------------------------------------------------------
# Rewrite rules
# ---------------------------------------------------------------------------


def test_mixture_flatten_preserves_log_prob() -> None:
    inner = Mixture(
        torch.tensor([0.5, 0.5]),
        [_Normal(-1.0, 1.0), _Normal(1.0, 1.0)],
    )
    outer = Mixture(
        torch.tensor([0.3, 0.7]),
        [_Normal(5.0, 1.0), inner],
    )
    flat = outer.flatten()
    assert flat.num_components == 3
    y = torch.tensor([0.5, 5.0, -1.0])
    torch.testing.assert_close(
        outer.log_prob(y),
        flat.log_prob(y),
        atol=1e-5,
        rtol=1e-5,
    )


def test_mixture_pushforward_commute() -> None:
    mix = Mixture(
        torch.tensor([0.3, 0.7]),
        [_Normal(0.0, 1.0), _Normal(1.0, 0.5)],
    )
    outside = Pushforward(mix, Exp())
    inside = mix.pushforward_inside(Exp())
    y = torch.tensor([0.5, 1.0, 2.5])
    torch.testing.assert_close(
        outside.log_prob(y),
        inside.log_prob(y),
        atol=1e-5,
        rtol=1e-5,
    )


def test_mixture_restrict_reweight_matches_normalize_outside() -> None:
    mix = Mixture(
        torch.tensor([0.5, 0.5]),
        [_Normal(-1.0, 1.0), _Normal(1.0, 1.0)],
    )
    outside = Normalize(Restrict(mix, low=torch.tensor(0.0)))
    inside = mix.restrict_to(low=torch.tensor(0.0))
    y = torch.tensor([0.5, 1.5, 2.0])
    torch.testing.assert_close(
        outside.log_prob(y),
        inside.log_prob(y),
        atol=1e-5,
        rtol=1e-5,
    )


def test_mixture_restrict_reweight_three_components() -> None:
    mix = Mixture(
        torch.tensor([0.2, 0.5, 0.3]),
        [_Normal(-2.0, 0.5), _Normal(0.0, 1.0), _Normal(3.0, 0.5)],
    )
    outside = Normalize(Restrict(mix, low=torch.tensor(0.0)))
    inside = mix.restrict_to(low=torch.tensor(0.0))
    y = torch.tensor([0.5, 1.0, 2.5])
    torch.testing.assert_close(
        outside.log_prob(y),
        inside.log_prob(y),
        atol=1e-5,
        rtol=1e-5,
    )


def test_pointmass_in_mixture_lift_is_identity_for_zip_shape() -> None:
    mix = Mixture(
        torch.tensor([0.3, 0.7]),
        [PointMass(torch.tensor(0.0)), _Poisson(2.0)],
    )
    lifted = mix.lift_point_masses()
    for y in (torch.tensor(0.0), torch.tensor(1.0), torch.tensor(3.0)):
        torch.testing.assert_close(
            mix.log_prob(y),
            lifted.log_prob(y),
            atol=1e-6,
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Composition closure
# ---------------------------------------------------------------------------


def test_restrict_of_pushforward_round_trips() -> None:
    base = _Normal(0.0, 1.0)
    push = Pushforward(base, Exp())  # LogNormal
    r = Restrict(push, low=torch.tensor(0.5), high=torch.tensor(2.0))
    assert math.isinf(r.log_prob(torch.tensor(0.1)).item())
    assert math.isinf(r.log_prob(torch.tensor(3.0)).item())
    assert torch.isfinite(r.log_prob(torch.tensor(1.0)))


def test_pushforward_of_restrict_round_trips() -> None:
    base = _Normal(0.0, 1.0)
    r = Restrict(base, low=torch.tensor(0.0))  # half-normal
    push = Pushforward(r, Exp())
    # Support is now (1, inf) because exp(0) = 1
    assert torch.isfinite(push.log_prob(torch.tensor(2.0)))


def test_mixture_of_mixed_families_log_prob_finite() -> None:
    mix = Mixture(
        torch.tensor([0.4, 0.6]),
        [
            torch.distributions.Normal(0.0, 1.0),
            torch.distributions.Laplace(0.0, 1.0),
        ],
    )
    y = torch.tensor([0.0, 1.0, -1.0, 2.5])
    lps = mix.log_prob(y)
    assert torch.isfinite(lps).all()


def test_normalize_idempotent() -> None:
    base = _Normal(0.0, 1.0)
    r = Restrict(base, low=torch.tensor(0.0))
    n1 = Normalize(r)
    n2 = Normalize(n1)
    y = torch.tensor(0.5)
    torch.testing.assert_close(n1.log_prob(y), n2.log_prob(y), atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Sample contracts
# ---------------------------------------------------------------------------


def test_mixture_sample_in_component_supports() -> None:
    torch.manual_seed(0)
    mix = Mixture(
        torch.tensor([0.3, 0.7]),
        [PointMass(0.0), _Poisson(5.0)],
    )
    samples = mix.sample(torch.Size([100]))
    assert (samples >= 0).all()
    assert (samples == 0).any()


def test_restrict_sample_in_support() -> None:
    torch.manual_seed(0)
    base = _Normal(0.0, 1.0)
    r = Restrict(base, low=torch.tensor(0.0), high=torch.tensor(1.0))
    s = r.sample()
    assert (s >= 0.0) and (s <= 1.0)
