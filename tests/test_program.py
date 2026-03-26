"""Tests for Program (nn.Module wrapper) and end-to-end training."""

import torch
import torch.nn.functional as F
import pytest

from quivers.core.objects import FinSet
from quivers.core.morphisms import morphism, observed
from quivers.program import Program


class TestProgram:
    def test_basic_forward(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        f = morphism(x, y)
        prog = Program(f)

        out = prog()
        assert out.shape == (3, 4)
        assert (out > 0).all()
        assert (out < 1).all()

    def test_parameters_collected(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)

        f = morphism(x, y)
        g = morphism(y, z)
        h = f >> g
        prog = Program(h)

        params = list(prog.parameters())
        assert len(params) == 2

    def test_nll_loss(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        f = morphism(x, y)
        prog = Program(f)

        domain_idx = torch.tensor([0, 1, 2])
        codomain_idx = torch.tensor([0, 1, 2])

        loss = prog.nll_loss(domain_idx, codomain_idx)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_bce_loss(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        f = morphism(x, y)
        prog = Program(f)

        target = torch.rand(3, 4)
        loss = prog.bce_loss(target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_log_membership(self):
        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        f = morphism(x, y)
        prog = Program(f)

        log_m = prog.log_membership()
        assert log_m.shape == (3, 4)
        assert (log_m <= 0).all()


class TestTrainingLoop:
    def test_loss_decreases(self):
        """End-to-end: train a composed model, verify loss decreases."""
        torch.manual_seed(42)

        x = FinSet("X", 3)
        y = FinSet("Y", 4)
        z = FinSet("Z", 2)

        f = morphism(x, y)
        g = morphism(y, z)
        h = f >> g

        prog = Program(h)
        optimizer = torch.optim.Adam(prog.parameters(), lr=0.05)

        # target: a specific boolean relation
        target = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        losses = []

        for _ in range(100):
            optimizer.zero_grad()
            out = prog()
            loss = F.binary_cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # loss should decrease substantially
        assert losses[-1] < losses[0]
        # final loss should be reasonably small
        assert losses[-1] < 0.5

    def test_single_morphism_training(self):
        """Train a single latent morphism to match a target."""
        torch.manual_seed(42)

        x = FinSet("X", 4)
        y = FinSet("Y", 3)

        f = morphism(x, y)
        prog = Program(f)
        optimizer = torch.optim.Adam(prog.parameters(), lr=0.1)

        # target: identity-like (one-hot rows)
        target = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])

        initial_loss = None

        for step in range(200):
            optimizer.zero_grad()
            out = prog()
            loss = F.binary_cross_entropy(out, target)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()

        final_loss = loss.item()
        assert final_loss < initial_loss
        assert final_loss < 0.2

    def test_train_latent_to_satisfy_inequality(self):
        """Train a latent morphism so its tensor is dominated by a target.

        This is the abstract version of learning predicate denotations
        that satisfy entailment constraints: find f such that
        f.tensor <= target pointwise, penalizing violations via a
        hinge-like loss.

        Analogous to learning an interpretation where 'dog' <= 'animal'
        must hold, without encoding any particular predicates.
        """
        torch.manual_seed(99)

        x = FinSet("X", 4)
        y = FinSet("Y", 5)

        f = morphism(x, y)
        prog = Program(f)

        # target upper bound: random values in [0.3, 0.8]
        target = torch.rand(4, 5) * 0.5 + 0.3

        optimizer = torch.optim.Adam(prog.parameters(), lr=0.05)

        for _ in range(300):
            optimizer.zero_grad()
            out = prog()

            # penalize wherever f exceeds the target
            violation = torch.relu(out - target)
            # also push f toward the target (not just below it)
            gap = (target - out).clamp(min=0)
            loss = violation.sum() + 0.1 * gap.sum()

            loss.backward()
            optimizer.step()

        final = prog()

        # f should be <= target everywhere (within tolerance)
        assert (final <= target + 0.05).all()
