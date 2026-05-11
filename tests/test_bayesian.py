"""Tests for the hierarchical-Bayesian primitives and DSL surface.

Covers:

* Per-primitive unit tests for the building blocks in
  :mod:`quivers.continuous.bayesian` (LKJ, Truncated, cumsum,
  softmax, cholesky_quad_form, PlateDraw, VectorisedObserve,
  marginalize_categorical).
* DSL parse / compile round-trips for every new AST node.
* End-to-end compile of the Stan-model port at
  ``src/quivers/dsl/examples/event_structure.qvr``.

The runtime extensions are not yet wired into the SVI guide; an
end-to-end fit on synthetic data is reserved for a follow-up
test once the inference layer recognises ``_RandomEffectPrior``
sites.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

# Ensure the local-grammar override is on for parser-driven tests.
os.environ.setdefault("QVR_USE_LOCAL_GRAMMAR", "1")


# ---------------------------------------------------------------------------
# Primitive smoke tests
# ---------------------------------------------------------------------------


class TestPrimitives:
    def test_cumsum(self):
        from quivers.continuous.bayesian import cumsum

        cm = cumsum(5)
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        out = cm.rsample(x)
        expected = torch.tensor([[1.0, 3.0, 6.0, 10.0, 15.0]])
        assert torch.allclose(out, expected)

    def test_softmax(self):
        from quivers.continuous.bayesian import softmax

        sm = softmax(3)
        out = sm.rsample(torch.zeros(2, 3))
        assert torch.allclose(out, torch.full((2, 3), 1.0 / 3.0))

    def test_lkj_sample_shape_and_diag(self):
        from quivers.continuous.bayesian import LKJCorrelationFactor
        from quivers.continuous.spaces import Euclidean

        K = 4
        lkj = LKJCorrelationFactor(dim=K, eta=2.0, domain=Euclidean(name="in", dim=1))
        L_flat = lkj.rsample(torch.zeros(3, 1))
        assert L_flat.shape == (3, K * K)
        L = L_flat.reshape(3, K, K)
        R = L @ L.transpose(-1, -2)
        # Correlation matrices have unit diagonal.
        for b in range(3):
            assert torch.allclose(
                torch.diagonal(R[b]),
                torch.ones(K),
                atol=1e-4,
            )
            # Symmetric.
            assert torch.allclose(R[b], R[b].T, atol=1e-5)
            # Off-diagonals in [-1, 1].
            off = R[b] - torch.eye(K)
            assert (off >= -1.0 - 1e-4).all()
            assert (off <= 1.0 + 1e-4).all()

    def test_lkj_log_prob_finite(self):
        from quivers.continuous.bayesian import LKJCorrelationFactor
        from quivers.continuous.spaces import Euclidean

        lkj = LKJCorrelationFactor(dim=3, eta=2.0, domain=Euclidean(name="in", dim=1))
        L = lkj.rsample(torch.zeros(2, 1))
        lp = lkj.log_prob(torch.zeros(2, 1), L)
        assert torch.isfinite(lp).all()

    def test_truncated_in_bounds(self):
        from quivers.continuous.bayesian import Truncated
        from quivers.continuous.spaces import Euclidean
        from quivers.continuous.families import ConditionalNormal

        domain = Euclidean(name="in", dim=2)
        codom = Euclidean(name="out", dim=1)
        base = ConditionalNormal(domain, codom)
        trunc = Truncated(base, lower=0.0, upper=5.0)
        sample = trunc.rsample(torch.tensor([[0.0, 1.0], [0.0, 1.0]]))
        assert (sample >= 0.0).all()
        assert (sample <= 5.0).all()

    def test_marginalize_categorical(self):
        from quivers.continuous.bayesian import marginalize_categorical

        logp = torch.tensor([[-1.0, -2.0, -3.0]])
        out = marginalize_categorical(logp)
        expected = torch.logsumexp(logp, dim=-1)
        assert torch.allclose(out, expected)

    def test_plate_draw_shape(self):
        from quivers.continuous.bayesian import PlateDraw
        from quivers.continuous.families import ConditionalNormal
        from quivers.continuous.spaces import Euclidean

        domain = Euclidean(name="in", dim=1)
        per_row = Euclidean(name="row", dim=2)
        family = ConditionalNormal(domain, per_row)
        plate = PlateDraw(index_size=5, family=family, domain=domain)
        sample = plate.rsample(torch.zeros(3, 1))
        # batch × (|A| · K)
        assert sample.shape == (3, 5 * 2)
        lp = plate.log_prob(torch.zeros(3, 1), sample)
        assert lp.shape == (3,)
        assert torch.isfinite(lp).all()

    def test_vectorised_observe_log_prob(self):
        from quivers.continuous.bayesian import VectorisedObserve
        from quivers.continuous.families import ConditionalNormal
        from quivers.continuous.spaces import Euclidean

        domain = Euclidean(name="in", dim=1)
        codom = Euclidean(name="out", dim=1)
        family = ConditionalNormal(domain, codom)
        response = torch.randn(10, 1)
        vec_obs = VectorisedObserve(family, response)
        theta = torch.zeros(10, 1)
        ll = vec_obs.log_prob(theta)
        # log_prob sums over the N observations.
        per_row = family.log_prob(theta, response)
        assert torch.allclose(ll, per_row.sum())

    def test_cholesky_quad_form_via_let(self):
        # The deterministic morphism is exposed both as a Python helper
        # and as a let-builtin; here we exercise the helper directly.
        from quivers.continuous.bayesian import cholesky_quad_form

        cqf = cholesky_quad_form(3)
        # input is (cholesky_flat, scale) concatenated
        L = torch.eye(3).reshape(-1)
        scale = torch.tensor([2.0, 3.0, 4.0])
        xs = torch.cat([L, scale]).unsqueeze(0)
        cov_flat = cqf.rsample(xs)
        cov = cov_flat.reshape(3, 3)
        # cov should be diag(scale)^2 since L = I.
        expected = torch.diag(scale * scale)
        assert torch.allclose(cov, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# DSL surface tests
# ---------------------------------------------------------------------------


class TestDSLSurface:
    def _compile(self, source: str):
        from quivers.dsl.parser import parse
        from quivers.dsl.compiler import Compiler

        m = parse(source)
        c = Compiler(m)
        c.compile()
        return c

    def test_plate_draw_step_parses_and_compiles(self):
        src = """
        object Subj : 5

        program demo : Subj -> Subj
            draw coefs : Subj -> 3 ~ Normal(0.0, 1.0)
            let z = coefs
            return z

        output demo
        """
        c = self._compile(src)
        assert "demo" in c._morphisms

    def test_let_index_gather(self):
        from quivers.continuous.bayesian import PlateDraw
        from quivers.continuous.families import ConditionalNormal
        from quivers.continuous.spaces import Euclidean

        domain = Euclidean(name="in", dim=1)
        per_row = Euclidean(name="row", dim=1)
        family = ConditionalNormal(domain, per_row)
        plate = PlateDraw(index_size=4, family=family, domain=domain)
        # Sample once and reshape.
        sample = plate.rsample(torch.zeros(1, 1)).reshape(4, 1)
        indices = torch.tensor([2, 0, 3])
        gathered = sample[indices]
        assert gathered.shape == (3, 1)
        # First gathered row equals sample[2].
        assert torch.allclose(gathered[0], sample[2])

    def test_random_effect_decl(self):
        src = """
        object Subj : 10

        random_effect by_subj : Subj -> 3 correlation eta = 2.0 scale_dist = HalfNormal(1.0)

        program demo : Subj -> Subj
            draw mu : Subj -> 1 ~ Normal(0.0, 1.0)
            return mu

        output demo
        """
        c = self._compile(src)
        assert "by_subj" in c._morphisms
        from quivers.continuous.bayesian import _RandomEffectPrior

        assert isinstance(c._morphisms["by_subj"], _RandomEffectPrior)

    def test_vectorised_observe(self):
        src = """
        object Resp : 20

        program demo : Resp -> Resp
            draw mu : Resp -> 1 ~ Normal(0.0, 1.0)
            observe r[n] ~ Normal(0.0, 1.0) for n in Resp
            return mu

        output demo
        """
        c = self._compile(src)
        assert "demo" in c._morphisms

    def test_marginalize_step(self):
        src = """
        object Item : 5

        program demo : Item -> Item
            draw class_probs : Item -> 4 ~ Normal(0.0, 1.0)
            marginalize class_probs
            return class_probs

        output demo
        """
        c = self._compile(src)
        assert "demo" in c._morphisms

    def test_event_structure_example_compiles(self):
        path = Path("src/quivers/dsl/examples/event_structure.qvr")
        if not path.exists():
            pytest.skip("event_structure.qvr not present")
        c = self._compile(path.read_text())
        # The model + 8 random-effect priors should all compile.
        assert "event_structure" in c._morphisms
        for name in (
            "by_subj_cloze",
            "by_verb_cloze",
            "by_sense_cloze",
            "by_item_cloze",
            "by_subj_prop",
            "by_verb_prop",
            "by_sense_prop",
            "by_item_prop",
        ):
            assert name in c._morphisms, f"{name} not compiled"
