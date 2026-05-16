"""Shape-contract tests for ``MonadicProgram.log_joint`` rank promotion.

Regression coverage for the bug where the single-arg branch of
``_resolve_input`` returned a rank-1 continuous tensor verbatim, which
``_NeuralSource``'s ``nn.Linear`` then mis-interpreted as a row vector
and crashed batched SVI on chained continuous steps.
"""

from __future__ import annotations

import torch

from quivers.continuous.families import ConditionalNormal
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet


def _chained_continuous_model() -> MonadicProgram:
    unit = FinSet(name="Unit", cardinality=1)
    r = Euclidean(name="real", dim=1)
    prior = ConditionalNormal(unit, r)
    likelihood = ConditionalNormal(r, r)
    return MonadicProgram(
        unit,
        r,
        steps=[
            (("z",), prior, None),
            (("y",), likelihood, ("z",)),
        ],
        return_vars=("y",),
    )


def test_log_joint_rank1_continuous_input_batched() -> None:
    """A 1-D continuous draw feeding a downstream continuous step does not
    crash and returns a finite scalar."""
    torch.manual_seed(0)
    model = _chained_continuous_model()
    batch = 10
    x = torch.zeros(batch, dtype=torch.long)
    z = torch.randn(batch)
    y = torch.randn(batch, 1)
    out = model.log_joint(x, {"z": z, "y": y})
    assert torch.isfinite(out).all()


def test_log_joint_rsample_round_trip() -> None:
    """Values produced by ``rsample`` (the natural upstream of log_joint
    in SVI) feed straight into a downstream continuous step's
    ``log_prob`` without a rank-mismatch crash."""
    torch.manual_seed(0)
    model = _chained_continuous_model()
    batch = 10
    x = torch.zeros(batch, dtype=torch.long)
    prior_name = model._step_specs[0].morphism_name
    prior = getattr(model, prior_name)
    z = prior.rsample(x)
    y = torch.randn(batch, 1)
    out = model.log_joint(x, {"z": z, "y": y})
    assert torch.isfinite(out).all()


def test_log_joint_svi_loop_runs() -> None:
    """A short SVI-style loop on the chained continuous model produces
    finite gradients every step."""
    from quivers.inference import ELBO, AutoNormalGuide

    torch.manual_seed(0)
    model = _chained_continuous_model()
    guide = AutoNormalGuide(model, observed_names={"y"})
    elbo = ELBO(num_particles=1)

    batch = 10
    x = torch.zeros(batch, dtype=torch.long)
    y = torch.randn(batch, 1)

    params = list(guide.parameters()) + list(model.parameters())
    opt = torch.optim.Adam(params, lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        loss = elbo(model, guide, x, {"y": y})
        assert torch.isfinite(loss).all()
        loss.backward()
        opt.step()
