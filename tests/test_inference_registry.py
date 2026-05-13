"""Tests for :class:`quivers.inference.registry.LatentRegistry`.

The registry is the single source of truth for per-site
introspection (support, dims, plate-vs-scalar, bijector
composition); every variational guide and MCMC kernel consumes
it. Bugs in the registry propagate everywhere, so the test
matrix is broad: flat-vector round-trip, bijector forward /
inverse Jacobian round-trip, plate-vs-scalar shape contract,
observed-site filtering, support-driven dim handling for
``simplex`` (stick-breaking d ↔ d-1) and ``corr_cholesky``.
"""

from __future__ import annotations

import torch

from quivers.dsl import loads
from quivers.inference.registry import LatentRegistry, LatentSite


def _hierarchical_model():
    return loads(
        "object Subj : 4\n"
        "object Resp : 12\n"
        "program p : Resp -> Resp\n"
        "    sigma <- HalfNormal(1.0)\n"
        "    by_subj : Subj <- Normal(0.0, sigma)\n"
        "    let mu = sigmoid(by_subj[subj_idx])\n"
        "    observe r : Resp <- Bernoulli(mu)\n"
        "    return mu\n"
        "export p\n"
    ).morphism


def _dirichlet_model():
    return loads(
        "object Cat : 4\n"
        "program p : Cat -> Cat\n"
        "    pc <- Dirichlet(1.0)\n"
        "    return pc\n"
        "export p\n"
    ).morphism


def test_registry_skips_observed_sites() -> None:
    reg = LatentRegistry.from_model(_hierarchical_model(), {"r"})
    assert set(reg.sites.keys()) == {"sigma", "by_subj"}
    assert "r" not in reg.sites


def test_registry_classifies_plate_vs_scalar() -> None:
    reg = LatentRegistry.from_model(_hierarchical_model(), {"r"})
    assert reg.sites["sigma"].is_plate is False
    assert reg.sites["sigma"].plate_index_size == 0
    assert reg.sites["by_subj"].is_plate is True
    assert reg.sites["by_subj"].plate_index_size == 4


def test_registry_dim_layout_for_scalar_and_plate() -> None:
    reg = LatentRegistry.from_model(_hierarchical_model(), {"r"})
    sigma: LatentSite = reg.sites["sigma"]
    by_subj: LatentSite = reg.sites["by_subj"]
    assert sigma.constrained_dim == 1
    assert sigma.unconstrained_dim == 1
    assert sigma.flat_offset == 0
    assert sigma.flat_length == 1
    assert by_subj.constrained_dim == 1
    assert by_subj.unconstrained_dim == 1
    assert by_subj.flat_offset == 1
    assert by_subj.flat_length == 4
    assert reg.total_unconstrained_dim == 5


def test_registry_dirichlet_simplex_dim_reduction() -> None:
    """``simplex`` is the canonical non-dim-preserving support:
    constrained side has dim d, unconstrained side has dim d-1."""
    reg = LatentRegistry.from_model(_dirichlet_model(), set())
    pc: LatentSite = reg.sites["pc"]
    assert pc.constrained_dim == 4
    assert pc.unconstrained_dim == 3


def test_registry_flatten_unflatten_roundtrip() -> None:
    reg = LatentRegistry.from_model(_hierarchical_model(), {"r"})
    z = reg.randn_unconstrained()
    flat = reg.flatten_unconstrained(z)
    assert flat.shape == (reg.total_unconstrained_dim,)
    z2 = reg.unflatten_unconstrained(flat)
    for name in reg.names:
        assert torch.allclose(z[name], z2[name])


def test_registry_flatten_preserves_leading_batch_axes() -> None:
    reg = LatentRegistry.from_model(_hierarchical_model(), {"r"})
    z = reg.randn_unconstrained(leading_shape=(3,))
    assert z["sigma"].shape == (3, 1)
    assert z["by_subj"].shape == (3, 4, 1)
    flat = reg.flatten_unconstrained(z)
    assert flat.shape == (3, reg.total_unconstrained_dim)
    z2 = reg.unflatten_unconstrained(flat)
    for name in reg.names:
        assert torch.allclose(z[name], z2[name])


def test_registry_to_constrained_squeezes_scalar_event() -> None:
    """Scalar-event sites have their trailing length-1 axis
    squeezed to match the trace-side shape convention."""
    reg = LatentRegistry.from_model(_hierarchical_model(), {"r"})
    z = reg.zero_unconstrained()
    v, _log_dets = reg.to_constrained(z)
    # `sigma` becomes a 0-d scalar; `by_subj` becomes (|Subj|,).
    assert v["sigma"].shape == ()
    assert v["by_subj"].shape == (4,)


def test_registry_constrained_unconstrained_inverse_roundtrip() -> None:
    """Going u -> v -> u must recover the original unconstrained
    values up to floating-point precision."""
    reg = LatentRegistry.from_model(_hierarchical_model(), {"r"})
    z = reg.randn_unconstrained()
    v, _ = reg.to_constrained(z)
    z2, _ = reg.to_unconstrained(v)
    for name in reg.names:
        assert torch.allclose(z[name], z2[name], atol=1e-5), (
            f"{name}: {(z[name] - z2[name]).abs().max().item()}"
        )


def test_registry_dirichlet_constrained_lies_on_simplex() -> None:
    reg = LatentRegistry.from_model(_dirichlet_model(), set())
    z = reg.randn_unconstrained()
    v, _ = reg.to_constrained(z)
    assert v["pc"].shape == (4,)
    assert torch.allclose(v["pc"].sum(), torch.tensor(1.0), atol=1e-5)
    assert (v["pc"] >= 0).all()


def test_registry_supports_halfnormal_positive_constraint() -> None:
    """``sigma <- HalfNormal(...)`` should produce a positive
    constrained value for any unconstrained input."""
    reg = LatentRegistry.from_model(_hierarchical_model(), {"r"})
    z = reg.randn_unconstrained(leading_shape=(20,), scale=5.0)
    v, _ = reg.to_constrained(z)
    assert (v["sigma"] > 0).all()


def test_registry_aggregate_log_abs_det_broadcasts() -> None:
    reg = LatentRegistry.from_model(_hierarchical_model(), {"r"})
    z = reg.randn_unconstrained(leading_shape=(7,))
    _, log_dets = reg.to_constrained(z)
    total = LatentRegistry.aggregate_log_abs_det(log_dets, leading_shape=(7,))
    assert total.shape == (7,)
    assert torch.isfinite(total).all()


def test_registry_zero_unconstrained_shapes() -> None:
    reg = LatentRegistry.from_model(_hierarchical_model(), {"r"})
    z = reg.zero_unconstrained(leading_shape=(2, 3))
    assert z["sigma"].shape == (2, 3, 1)
    assert z["by_subj"].shape == (2, 3, 4, 1)
    assert (z["sigma"] == 0).all()


# ---------------------------------------------------------------------------
# Additional registry surface
# ---------------------------------------------------------------------------


def test_registry_iter_yields_sites_in_declaration_order() -> None:
    """Iterating the registry yields LatentSite objects in the
    order they appear in the model's _step_specs."""
    from quivers.dsl import loads
    from quivers.inference.registry import LatentRegistry, LatentSite

    src = (
        "object Subj : 3\n"
        "object Resp : 6\n"
        "program p : Resp -> Resp\n"
        "    sigma <- HalfNormal(1.0)\n"
        "    by_subj : Subj <- Normal(0.0, sigma)\n"
        "    let mu = sigmoid(by_subj[subj_idx])\n"
        "    observe r : Resp <- Bernoulli(mu)\n"
        "    return mu\n"
        "export p\n"
    )
    model = loads(src).morphism
    reg = LatentRegistry.from_model(model, observed_names={"r"})
    sites = list(reg)
    names = [s.name for s in sites]
    assert names == ["sigma", "by_subj"]
    assert all(isinstance(s, LatentSite) for s in sites)


def test_registry_contains_and_getitem() -> None:
    from quivers.dsl import loads
    from quivers.inference.registry import LatentRegistry

    src = (
        "object Subj : 3\n"
        "object Resp : 6\n"
        "program p : Resp -> Resp\n"
        "    sigma <- HalfNormal(1.0)\n"
        "    by_subj : Subj <- Normal(0.0, sigma)\n"
        "    let mu = sigmoid(by_subj[subj_idx])\n"
        "    observe r : Resp <- Bernoulli(mu)\n"
        "    return mu\n"
        "export p\n"
    )
    model = loads(src).morphism
    reg = LatentRegistry.from_model(model, observed_names={"r"})
    assert "sigma" in reg
    assert "by_subj" in reg
    assert "r" not in reg  # observed
    site = reg["sigma"]
    assert site.name == "sigma"


def test_registry_len_matches_total_latents() -> None:
    from quivers.dsl import loads
    from quivers.inference.registry import LatentRegistry

    src = (
        "object Resp : 4\n"
        "program p : Resp -> Resp\n"
        "    a <- Normal(0.0, 1.0)\n"
        "    b <- Normal(0.0, 1.0)\n"
        "    observe r : Resp <- Normal(a + b, 1.0)\n"
        "    return a\n"
        "export p\n"
    )
    model = loads(src).morphism
    reg = LatentRegistry.from_model(model, observed_names={"r"})
    assert len(reg) == 2


def test_registry_observed_names_returns_frozenset() -> None:
    from quivers.dsl import loads
    from quivers.inference.registry import LatentRegistry

    src = (
        "object Resp : 4\n"
        "program p : Resp -> Resp\n"
        "    mu <- Normal(0.0, 1.0)\n"
        "    observe r : Resp <- Normal(mu, 1.0)\n"
        "    return mu\n"
        "export p\n"
    )
    model = loads(src).morphism
    reg = LatentRegistry.from_model(model, observed_names={"r"})
    assert reg.observed_names == frozenset({"r"})


def test_registry_model_property_returns_construction_model() -> None:
    from quivers.dsl import loads
    from quivers.inference.registry import LatentRegistry

    src = (
        "object Resp : 4\n"
        "program p : Resp -> Resp\n"
        "    mu <- Normal(0.0, 1.0)\n"
        "    observe r : Resp <- Normal(mu, 1.0)\n"
        "    return mu\n"
        "export p\n"
    )
    model = loads(src).morphism
    reg = LatentRegistry.from_model(model, observed_names={"r"})
    assert reg.model is model
