"""Tests for the inference lifts in :mod:`quivers.inference.lifts`.

Three lifts are exercised:

* :func:`bayesian_lift_parameters` lifts a module's parameters into
  Normal-prior sample sites and returns a :class:`MonadicProgram`
  the inference layer drives directly.
* :func:`lift_to_bayesian_program` lifts a deterministic morphism
  plus an arbitrary :class:`torch.distributions.Distribution`
  observation family into the same shape.
* :func:`monte_carlo_log_joint` wraps a program so its
  ``log_joint`` MC-draws named intermediate sample sites instead
  of requiring them in the observations dict.
"""

from __future__ import annotations

import os

os.environ.setdefault("QVR_USE_LOCAL_GRAMMAR", "1")

import torch
import torch.distributions as D
import torch.nn as nn

from quivers.continuous.inline import FixedDistribution
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import Unit
from quivers.inference import (
    bayesian_lift_parameters,
    lift_to_bayesian_program,
    monte_carlo_log_joint,
    LatentRegistry,
)


# ---------------------------------------------------------------------------
# bayesian_lift_parameters
# ---------------------------------------------------------------------------


class _LinearLikelihood(nn.Module):
    """Tiny ``y = a x + b`` model with a Normal likelihood."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    def log_joint(self, x: torch.Tensor, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        pred = self.a * x.squeeze(-1) + self.b
        return D.Normal(pred, 0.1).log_prob(obs["y"]).sum().unsqueeze(0)


def test_bayesian_lift_parameters_yields_two_site_program():
    torch.manual_seed(0)
    model = _LinearLikelihood()
    x = torch.linspace(0.0, 1.0, 8).unsqueeze(-1)
    obs = {"y": 2.0 * x.squeeze(-1) + 0.5}
    lifted, _, _ = bayesian_lift_parameters(model, x, obs, prior_scale=1.0)
    reg = LatentRegistry.from_model(lifted, observed_names=set())
    assert reg.total_unconstrained_dim == 2


def test_bayesian_lift_log_joint_is_finite():
    torch.manual_seed(0)
    model = _LinearLikelihood()
    x = torch.linspace(0.0, 1.0, 8).unsqueeze(-1)
    obs = {"y": 2.0 * x.squeeze(-1) + 0.5}
    lifted, lx, lobs = bayesian_lift_parameters(model, x, obs, prior_scale=1.0)
    lj = lifted.log_joint(
        lx, {"theta__a": torch.zeros(1, 1), "theta__b": torch.zeros(1, 1), **lobs}
    )
    assert torch.isfinite(lj).all()


# ---------------------------------------------------------------------------
# lift_to_bayesian_program
# ---------------------------------------------------------------------------


class _MeanOnly(nn.Module):
    """Deterministic morphism whose ``rsample(x)`` is a learnable mean."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))

    def rsample(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu.unsqueeze(0).expand(x.shape[0], -1)


def test_lift_with_normal_family_runs_and_walks_parameters():
    torch.manual_seed(0)
    inner = _MeanOnly(dim=3)
    y = torch.tensor([[1.1, 1.9, 3.2]])
    lifted, _, _ = lift_to_bayesian_program(
        inner,
        location_fn=inner.rsample,
        parameter_prior_scale=1.0,
        observation_family=D.Normal,
        observation_kwargs={"scale": 0.5},
        target_key="Y",
        x=torch.zeros(1, 1),
        observations={"Y": y},
    )
    reg = LatentRegistry.from_model(lifted, observed_names=set())
    assert reg.total_unconstrained_dim == 3


def test_lift_with_bernoulli_family():
    torch.manual_seed(0)
    inner = _MeanOnly(dim=2)
    y = torch.tensor([[1.0, 0.0]])
    lifted, _, _ = lift_to_bayesian_program(
        inner,
        location_fn=inner.rsample,
        parameter_prior_scale=1.0,
        observation_family=D.Bernoulli,
        observation_kwargs={},
        target_key="Y",
        x=torch.zeros(1, 1),
        observations={"Y": y},
    )
    reg = LatentRegistry.from_model(lifted, observed_names=set())
    assert reg.total_unconstrained_dim == 2


def test_lift_with_categorical_family():
    torch.manual_seed(0)
    inner = _MeanOnly(dim=4)
    y = torch.tensor([1])
    lifted, _, _ = lift_to_bayesian_program(
        inner,
        location_fn=inner.rsample,
        parameter_prior_scale=1.0,
        observation_family=D.Categorical,
        observation_kwargs={},
        target_key="cls",
        x=torch.zeros(1, 1),
        observations={"cls": y},
    )
    reg = LatentRegistry.from_model(lifted, observed_names=set())
    assert reg.total_unconstrained_dim == 4


# ---------------------------------------------------------------------------
# monte_carlo_log_joint
# ---------------------------------------------------------------------------


def test_bayesian_lift_with_additional_latents_cancels_placeholder():
    """The placeholder prior on a lifted latent must cancel exactly,
    so the lifted log-density equals
    ``log p(theta) + log p_inner(z, y | x, theta)``.
    """
    torch.manual_seed(0)
    model = _LinearLikelihood()
    x = torch.tensor([[1.0]])
    # Treat y as a "latent" of shape (1,) lifted into NUTS, and put
    # nothing in observations. Manually inject a latent value through
    # the env and verify the lifted log-density equals the inner's
    # log_joint with that same latent value, irrespective of
    # placeholder scale.
    lifted, lx, _ = bayesian_lift_parameters(
        model,
        x,
        {},
        prior_scale=1.0,
        additional_latents={"y": (1,)},
        latent_placeholder_scale=10.0,
    )
    # Two evaluations with different placeholder scales should give
    # the same score after cancellation, given the same theta and z.
    lifted2, lx2, _ = bayesian_lift_parameters(
        model,
        x,
        {},
        prior_scale=1.0,
        additional_latents={"y": (1,)},
        latent_placeholder_scale=1.0,
    )
    # Build matching envs: zero theta, latent y = 2.0
    env_keys_1 = [
        spec.morphism_name
        for spec in lifted._step_specs
        if hasattr(spec, "vars") and spec.vars[0].startswith(("theta", "latent"))
    ]
    env_keys_2 = [
        spec.morphism_name
        for spec in lifted2._step_specs
        if hasattr(spec, "vars") and spec.vars[0].startswith(("theta", "latent"))
    ]
    assert env_keys_1 == env_keys_2  # same step layout
    # Direct log_joint comparison: same args, different placeholder
    # scales → identical scores (within float tolerance).
    env = {}
    for spec in lifted._step_specs:
        if hasattr(spec, "vars") and spec.vars[0].startswith(("theta", "latent")):
            morph = lifted._modules[spec.morphism_name]
            d = morph.codomain.dim if hasattr(morph.codomain, "dim") else 1
            v = (
                torch.full((1, d), 2.0)
                if spec.vars[0].startswith("latent")
                else torch.zeros(1, d)
            )
            env[spec.vars[0]] = v
    s1 = float(lifted.log_joint(lx, env)[0])
    s2 = float(lifted2.log_joint(lx2, env)[0])
    # Score differs only by the prior on theta (same in both), so
    # cancellation of the latent placeholder makes them equal.
    assert abs(s1 - s2) < 1e-4, f"placeholders did not cancel: {s1} vs {s2}"


def test_monte_carlo_log_joint_draws_step_site():
    """A two-step program with a Normal sample site ``h`` and an
    observed ``y``: the wrapper MC-draws ``h`` and the resulting
    log_joint accepts only ``{"y": ...}``."""

    def _prior(b, device):
        return D.Normal(
            torch.zeros(b, 1, device=device),
            torch.ones(b, 1, device=device),
        )

    def _likelihood(b, device):
        return D.Normal(
            torch.zeros(b, 1, device=device),
            torch.ones(b, 1, device=device),
        )

    h_morph = FixedDistribution(Euclidean(name="R", dim=1), _prior, discrete=False)
    y_morph = FixedDistribution(Euclidean(name="R", dim=1), _likelihood, discrete=False)
    inner = MonadicProgram(
        domain=Unit,
        codomain=Unit,
        steps=[
            (("h",), h_morph, None),
            (("y",), y_morph, None, True),
        ],
        return_vars=("y",),
    )
    wrapped = monte_carlo_log_joint(inner, sample_sites=["h"])
    out = wrapped.log_joint(torch.zeros(1, 1), {"y": torch.zeros(1, 1)})
    assert out.shape == (1,) and torch.isfinite(out).all()
