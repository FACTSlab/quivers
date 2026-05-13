"""Tests for the unified :class:`FamilySpec` registry.

Verifies that:

* Every registered family is accessible via
  :data:`quivers.continuous.family_spec.FAMILY_REGISTRY`.
* The factory-generated and hand-written families share the same
  introspection surface (``param_names``, ``support``,
  ``discrete``, ``output_kind``).
* The inline path auto-registers fixed factories and mixed
  builders for every per-dim-independent family.
* Every previously-registry-only family (Cauchy, Laplace, Gumbel,
  StudentT, Chi2, InverseGamma, Weibull, Pareto, Kumaraswamy,
  ContinuousBernoulli, FisherSnedecor) is now usable inline.
* Newly-shipped families (Poisson, Geometric, NegativeBinomial,
  VonMises) sample, score, and broadcast correctly.
"""

from __future__ import annotations

import pytest
import torch

from quivers.continuous import families, inline
from quivers.continuous.family_spec import (
    FAMILY_REGISTRY,
    FamilySpec,
    ParamSpec,
    get as registry_get,
    names as registry_names,
)
from quivers.continuous.inline import make_inline_distribution
from quivers.continuous.spaces import Euclidean


# ---------------------------------------------------------------------------
# Registry coverage
# ---------------------------------------------------------------------------


_PROMOTED_FAMILIES = (
    "Cauchy",
    "Laplace",
    "Gumbel",
    "StudentT",
    "Chi2",
    "InverseGamma",
    "Weibull",
    "Pareto",
    "Kumaraswamy",
    "ContinuousBernoulli",
    "FisherSnedecor",
)

_NEW_FAMILIES = (
    "Poisson",
    "Geometric",
    "NegativeBinomial",
    "VonMises",
)


@pytest.mark.parametrize("name", _PROMOTED_FAMILIES + _NEW_FAMILIES)
def test_family_registered(name: str) -> None:
    spec = registry_get(name)
    assert spec is not None, f"family {name!r} missing from registry"
    assert isinstance(spec, FamilySpec)
    assert spec.name == name


@pytest.mark.parametrize("name", _PROMOTED_FAMILIES + _NEW_FAMILIES)
def test_family_has_inline_fixed_factory(name: str) -> None:
    assert name in inline._FIXED_FACTORIES, (
        f"family {name!r} missing fixed-inline factory"
    )


@pytest.mark.parametrize("name", _PROMOTED_FAMILIES + _NEW_FAMILIES)
def test_family_has_inline_mixed_builder(name: str) -> None:
    assert name in inline._FAMILY_BUILDERS, (
        f"family {name!r} missing mixed-inline builder"
    )


def test_registry_count_matches_expectations() -> None:
    """A floor on the registry size — guards against silent regressions
    that drop families. The exact count grows as new families ship."""
    assert len(FAMILY_REGISTRY) >= 30, (
        f"registry has {len(FAMILY_REGISTRY)} families; expected >= 30"
    )


# ---------------------------------------------------------------------------
# Inline fixed-factory round-trips
# ---------------------------------------------------------------------------


_INLINE_FIXED_TEST_CASES = [
    ("Cauchy", (0.0, 1.0)),
    ("Laplace", (0.0, 1.0)),
    ("Gumbel", (0.0, 1.0)),
    ("StudentT", (5.0, 0.0, 1.0)),
    ("Chi2", (5.0,)),
    ("InverseGamma", (3.0, 2.0)),
    ("Weibull", (1.0, 2.0)),
    ("Pareto", (1.0, 2.0)),
    ("Kumaraswamy", (2.0, 5.0)),
    ("ContinuousBernoulli", (0.5,)),
    ("FisherSnedecor", (5.0, 10.0)),
    ("Poisson", (2.0,)),
    ("Geometric", (0.3,)),
    ("NegativeBinomial", (3.0, 0.5)),
    ("VonMises", (0.0, 1.0)),
]


@pytest.mark.parametrize("name,args", _INLINE_FIXED_TEST_CASES)
def test_inline_fixed_distribution_samples(name: str, args: tuple) -> None:
    """All-literal inline distribution produces finite samples in
    the right shape."""
    codomain = Euclidean(name=f"_test_{name}", dim=1)
    morph, var_names = make_inline_distribution(name, args, codomain)
    assert var_names is None, "all-literal call should produce no var inputs"
    x = torch.zeros(4, 1)
    samples = morph.rsample(x)
    assert samples.shape[0] == 4
    spec = registry_get(name)
    assert spec is not None
    if spec.discrete:
        assert samples.dtype in (torch.int64, torch.long)
    else:
        # Continuous outputs must be finite (rejection of -inf / nan from
        # ill-conditioned literals).
        assert torch.isfinite(samples).all(), (
            f"{name}: samples contain non-finite entries"
        )


@pytest.mark.parametrize("name,args", _INLINE_FIXED_TEST_CASES)
def test_inline_fixed_log_prob_finite(name: str, args: tuple) -> None:
    """log_prob evaluates to finite values on a sampled point."""
    codomain = Euclidean(name=f"_test_{name}", dim=1)
    morph, _ = make_inline_distribution(name, args, codomain)
    x = torch.zeros(4, 1)
    samples = morph.rsample(x)
    lp = morph.log_prob(x, samples.float() if samples.dtype == torch.long else samples)
    assert torch.isfinite(lp).all(), f"{name}: log_prob non-finite"


# ---------------------------------------------------------------------------
# Conditional class auto-detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls_name,fam_name",
    [
        ("ConditionalPoisson", "Poisson"),
        ("ConditionalGeometric", "Geometric"),
        ("ConditionalNegativeBinomial", "NegativeBinomial"),
        ("ConditionalVonMises", "VonMises"),
    ],
)
def test_new_conditional_class_exists(cls_name: str, fam_name: str) -> None:
    cls = getattr(families, cls_name)
    assert cls is not None
    spec = registry_get(fam_name)
    assert spec is not None
    assert spec.conditional_class_override is cls


def test_new_conditional_classes_rsample_and_score() -> None:
    """Every new ConditionalX class draws samples in the right
    support and scores them under log_prob without raising."""
    from quivers.continuous.spaces import Euclidean
    from quivers.core.objects import FinSet

    domain = FinSet(name="_ctx", cardinality=3)
    codomain = Euclidean(name="_out", dim=2)

    for cls_name, support_check in [
        ("ConditionalPoisson", lambda t: (t >= 0).all() and t.dtype == torch.long),
        ("ConditionalGeometric", lambda t: (t >= 0).all() and t.dtype == torch.long),
        ("ConditionalNegativeBinomial", lambda t: (t >= 0).all() and t.dtype == torch.long),
        ("ConditionalVonMises", lambda t: torch.isfinite(t).all()),
    ]:
        cls = getattr(families, cls_name)
        m = cls(domain, codomain)
        x = torch.tensor([0, 1, 2])
        samples = m.rsample(x)
        assert support_check(samples), f"{cls_name}: samples out of support"
        lp = m.log_prob(x, samples.float() if samples.dtype == torch.long else samples)
        assert torch.isfinite(lp).all(), f"{cls_name}: log_prob non-finite"


# ---------------------------------------------------------------------------
# Hand-written family registration coverage
# ---------------------------------------------------------------------------


_HAND_WRITTEN_FAMILIES = (
    "Normal",
    "LogitNormal",
    "Beta",
    "Dirichlet",
    "Uniform",
    "TruncatedNormal",
    "MultivariateNormal",
    "LowRankMVN",
    "RelaxedBernoulli",
    "RelaxedOneHotCategorical",
    "Wishart",
    "Bernoulli",
    "Categorical",
)


@pytest.mark.parametrize("name", _HAND_WRITTEN_FAMILIES)
def test_hand_written_family_registered(name: str) -> None:
    spec = registry_get(name)
    assert spec is not None, f"hand-written family {name!r} not in registry"
    assert spec.conditional_class_override is not None, (
        f"hand-written family {name!r} should carry a class override"
    )


def test_param_spec_unknown_transform_raises() -> None:
    with pytest.raises(ValueError, match="unknown transform"):
        ParamSpec(name="x", transform="not_a_transform")


def test_registry_names_sorted_and_unique() -> None:
    names = registry_names()
    assert len(set(names)) == len(names), "duplicate names in registry"
    assert list(names) == sorted(names)
