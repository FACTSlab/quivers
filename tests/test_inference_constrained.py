"""Tests for constrained-support variational guides, data-dict host
passing, and inline Dirichlet priors.

* :class:`quivers.inference.AutoNormalGuide` and
  :class:`quivers.inference.AutoDeltaGuide` respect each latent's
  prior support: samples are drawn in unconstrained Normal space and
  pushed through :func:`biject_to(support)` so they land inside the
  prior's support (``[0, ∞)`` for ``HalfNormal`` / ``Gamma`` /
  ``Exponential``, ``(0, 1)`` for ``Beta`` / ``LogitNormal``, the
  simplex for ``Dirichlet``, an arbitrary interval for ``Uniform``).
* :func:`quivers.inference.condition` exposes every key in its
  ``data`` dict that does not match a declared sample / observe site
  as a deterministic value in the program's runtime environment,
  visible to ``let``-expression gather (the canonical per-row
  hierarchical-regression idiom).
* Inline ``Dirichlet(concentration)`` is supported as a prior; scalar
  concentration broadcasts to a symmetric Dirichlet on the codomain's
  simplex.
"""

from __future__ import annotations

import pytest
import torch

from quivers.dsl import loads
from quivers.inference import AutoDeltaGuide, AutoNormalGuide
from quivers.inference.conditioning import condition
from quivers.inference.trace import trace


# ---------------------------------------------------------------------------
# Constrained-support guides.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dist_spec", "predicate"),
    [
        ("HalfNormal(1.0)", lambda v: (v >= 0).all()),
        ("HalfCauchy(1.0)", lambda v: (v >= 0).all()),
        ("Exponential(1.0)", lambda v: (v > 0).all()),
        ("Gamma(2.0, 1.0)", lambda v: (v > 0).all()),
        ("LogNormal(0.0, 1.0)", lambda v: (v > 0).all()),
        ("Beta(2.0, 5.0)", lambda v: ((v > 0) & (v < 1)).all()),
        ("Uniform(0.0, 1.0)", lambda v: ((v > 0) & (v < 1)).all()),
        ("Uniform(-3.0, 5.0)", lambda v: ((v > -3.0) & (v < 5.0)).all()),
        ("LogitNormal(0.0, 1.0)", lambda v: ((v > 0) & (v < 1)).all()),
    ],
)
def test_autonormal_guide_samples_in_support(dist_spec: str, predicate) -> None:
    """The Normal-then-bijector pipeline produces samples in every
    family's declared support, so the prior's ``log_prob`` evaluates
    without ``ValueError: Expected value … to be within the support``.
    """
    src = (
        "object N : 4\n"
        "program p : N -> N\n"
        f"    v <- {dist_spec}\n"
        "    return v\n"
        "export p\n"
    )
    prog = loads(src)
    guide = AutoNormalGuide(prog.morphism, observed_names=set())
    samples = guide.rsample(torch.zeros(8, 1))
    assert predicate(samples["v"]), (
        f"samples for {dist_spec} not in support: {samples['v']}"
    )

    lp = guide.log_prob(torch.zeros(8, 1), samples)
    assert lp.shape == (8,)
    assert torch.isfinite(lp).all()


@pytest.mark.parametrize(
    ("dist_spec", "predicate"),
    [
        ("HalfNormal(1.0)", lambda v: (v >= 0).all()),
        ("Beta(2.0, 5.0)", lambda v: ((v > 0) & (v < 1)).all()),
        ("Uniform(0.0, 1.0)", lambda v: ((v > 0) & (v < 1)).all()),
    ],
)
def test_autodelta_guide_samples_in_support(dist_spec: str, predicate) -> None:
    """``AutoDeltaGuide`` likewise transforms its point estimate
    through the support bijector so it lies inside the prior's
    support at all times during optimisation."""
    src = (
        "object N : 4\n"
        "program p : N -> N\n"
        f"    v <- {dist_spec}\n"
        "    return v\n"
        "export p\n"
    )
    prog = loads(src)
    guide = AutoDeltaGuide(prog.morphism, observed_names=set())
    samples = guide.rsample(torch.zeros(8, 1))
    assert predicate(samples["v"])


def test_autonormal_gradient_flows_through_bijector() -> None:
    """The constrained ELBO contribution remains differentiable end-
    to-end: log_prob at constrained values produces gradients that
    flow back to the guide's loc/log_scale parameters via the
    Jacobian correction."""
    prog = loads(
        "object N : 1\n"
        "program p : N -> N\n"
        "    sigma <- HalfNormal(1.0)\n"
        "    return sigma\n"
        "export p\n"
    )
    guide = AutoNormalGuide(prog.morphism, observed_names=set())
    samples = guide.rsample(torch.zeros(4, 1))
    lp = guide.log_prob(torch.zeros(4, 1), samples).sum()
    lp.backward()
    loc = guide.loc_sigma
    assert loc.grad is not None
    assert torch.isfinite(loc.grad).all()


# ---------------------------------------------------------------------------
# Host-data passing through ``condition``.
# ---------------------------------------------------------------------------


def test_condition_data_dict_visible_to_let_expression() -> None:
    """A key in the ``condition`` data dict that does not match a
    declared sample / observe site is exposed as a deterministic
    value to ``let``-expression evaluation. This unlocks per-row
    covariate / index passing for hierarchical regression."""
    prog = loads(
        "object Subj : 4\n"
        "object Resp : 12\n"
        "\n"
        "program p : Resp -> Resp\n"
        "    by_subj : Subj <- Normal(0.0, 1.0)\n"
        "    let mu = by_subj[subj_idx]\n"
        "    return mu\n"
        "export p\n"
    )
    model = prog.morphism
    subj_idx = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    cond = condition(model, {"subj_idx": subj_idx})
    tr = cond.trace(torch.zeros(12, 1))

    # The point of this test: the let-expression reference to
    # ``subj_idx`` resolves at runtime to the value supplied through
    # ``condition``; it no longer raises ``undefined variable
    # 'subj_idx' in let expression`` at compile time.
    by_subj = tr.sites["by_subj"].value
    mu = tr.sites["mu"].value
    # by_subj advance-indexed along its first axis with subj_idx gives
    # mu the row-axis of subj_idx prepended to whatever by_subj's
    # trailing shape is.
    expected = by_subj[subj_idx]
    assert torch.allclose(mu, expected)


def test_condition_data_dict_alongside_observations() -> None:
    """Host data and observations share the same ``data`` channel;
    the trace clamps observed sample sites and exposes the rest as
    pre-populated environment entries."""
    prog = loads(
        "object Subj : 3\n"
        "object Resp : 6\n"
        "\n"
        "program p : Resp -> Resp\n"
        "    by_subj : Subj <- Normal(0.0, 1.0)\n"
        "    let mu = by_subj[subj_idx]\n"
        "    observe r : Resp <- Normal(mu, 1.0)\n"
        "    return r\n"
        "export p\n"
    )
    model = prog.morphism
    subj_idx = torch.tensor([0, 1, 2, 0, 1, 2])
    r = torch.zeros(6)
    cond = condition(model, {"subj_idx": subj_idx, "r": r})
    tr = cond.trace(torch.zeros(6, 1))

    # `r` is clamped as an observed sample site.
    assert torch.allclose(tr.sites["r"].value, r)
    # `subj_idx` populated env, and the gather fired without error.
    assert "mu" in tr.sites


# ---------------------------------------------------------------------------
# Inline Dirichlet.
# ---------------------------------------------------------------------------


def test_inline_dirichlet_scalar_concentration() -> None:
    """``Dirichlet(alpha)`` with a scalar concentration is a symmetric
    Dirichlet on the simplex of the declared codomain's dimension."""
    prog = loads(
        "object Cat : 3\n"
        "program p : Cat -> Cat\n"
        "    pc <- Dirichlet(1.0)\n"
        "    return pc\n"
        "export p\n"
    )
    tr = trace(prog.morphism, torch.zeros(5, 1))
    pc = tr.sites["pc"].value
    assert pc.shape == (5, 3)
    assert torch.allclose(pc.sum(dim=-1), torch.ones(5), atol=1e-5)
    assert (pc >= 0).all()


def test_inline_dirichlet_under_autonormal_guide() -> None:
    """Dirichlet's simplex support is realised in the guide via
    ``biject_to(simplex) = StickBreakingTransform()``; guide samples
    lie on the simplex and ``log_prob`` evaluates."""
    prog = loads(
        "object Cat : 4\n"
        "program p : Cat -> Cat\n"
        "    pc <- Dirichlet(2.0)\n"
        "    return pc\n"
        "export p\n"
    )
    guide = AutoNormalGuide(prog.morphism, observed_names=set())
    samples = guide.rsample(torch.zeros(6, 1))
    assert samples["pc"].shape == (6, 4)
    assert torch.allclose(samples["pc"].sum(dim=-1), torch.ones(6), atol=1e-4)
    assert (samples["pc"] >= 0).all()

    lp = guide.log_prob(torch.zeros(6, 1), samples)
    assert lp.shape == (6,)
    assert torch.isfinite(lp).all()


def test_inline_dirichlet_score_under_prior() -> None:
    """The end-to-end claim: a Dirichlet sample drawn by the guide
    can be scored under the model's prior without the ``Expected
    value to be within the support`` failure that motivated the
    constrained-support guide fix."""
    prog = loads(
        "object Cat : 3\n"
        "program p : Cat -> Cat\n"
        "    pc <- Dirichlet(1.0)\n"
        "    return pc\n"
        "export p\n"
    )
    guide = AutoNormalGuide(prog.morphism, observed_names=set())
    z = guide.rsample(torch.zeros(4, 1))
    tr = trace(prog.morphism, torch.zeros(4, 1), observations={"pc": z["pc"]})
    site = tr.sites["pc"]
    assert site.is_observed
    assert torch.isfinite(site.log_prob).all()
