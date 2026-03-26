"""Tests for noisy-OR tensor operations."""

import torch
import pytest
from quivers.core.tensor_ops import (
    noisy_or_contract,
    noisy_or_reduce,
    noisy_and_reduce,
    componentwise_lift,
)


class TestNoisyOrContract:
    def test_boolean_matrices(self):
        """Noisy-OR on {0,1} matrices should match boolean matrix multiply."""
        m = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        n = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        result = noisy_or_contract(m, n, n_contract=1)

        # boolean matmul: OR over j of (m[i,j] AND n[j,k])
        expected = torch.tensor([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
        ])

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_identity_composition(self):
        """Composing with a one-hot identity should be approximately identity."""
        # identity on a 3-element set
        eye = torch.eye(3)
        m = torch.rand(4, 3)

        result = noisy_or_contract(m, eye, n_contract=1)

        # noisy-OR with identity: for each (i,k),
        # 1 - prod_j(1 - m[i,j]*eye[j,k]) = 1 - (1 - m[i,k]) = m[i,k]
        # (since eye[j,k]=0 for j!=k, those terms contribute (1-0)=1)
        torch.testing.assert_close(result, m, atol=1e-5, rtol=1e-5)

    def test_shape_2d(self):
        m = torch.rand(3, 4)
        n = torch.rand(4, 5)
        result = noisy_or_contract(m, n, n_contract=1)
        assert result.shape == (3, 5)

    def test_shape_higher_order(self):
        """Contract over multi-dimensional shared set."""
        m = torch.rand(3, 4, 5)  # domain (3,), shared (4, 5)
        n = torch.rand(4, 5, 2)  # shared (4, 5), codomain (2,)
        result = noisy_or_contract(m, n, n_contract=2)
        assert result.shape == (3, 2)

    def test_shape_mismatch_raises(self):
        m = torch.rand(3, 4)
        n = torch.rand(5, 2)

        with pytest.raises(ValueError, match="shared dimensions"):
            noisy_or_contract(m, n, n_contract=1)

    def test_values_in_unit_interval(self):
        m = torch.rand(5, 6)
        n = torch.rand(6, 3)
        result = noisy_or_contract(m, n, n_contract=1)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_gradient_flows(self):
        """Verify gradients are non-None and finite."""
        m = torch.rand(3, 4, requires_grad=True)
        n = torch.rand(4, 5, requires_grad=True)

        result = noisy_or_contract(m, n, n_contract=1)
        loss = result.sum()
        loss.backward()

        assert m.grad is not None
        assert n.grad is not None
        assert torch.isfinite(m.grad).all()
        assert torch.isfinite(n.grad).all()

    def test_zero_matrix(self):
        """Composing with zeros should give zeros."""
        m = torch.zeros(3, 4)
        n = torch.rand(4, 5)
        result = noisy_or_contract(m, n, n_contract=1)
        torch.testing.assert_close(
            result, torch.zeros(3, 5), atol=1e-5, rtol=1e-5
        )

    def test_monotonicity(self):
        """Higher input values should produce higher (or equal) outputs."""
        m_low = torch.rand(3, 4) * 0.3
        m_high = m_low + 0.3
        n = torch.rand(4, 5)

        result_low = noisy_or_contract(m_low, n, n_contract=1)
        result_high = noisy_or_contract(m_high, n, n_contract=1)

        # noisy-OR is monotone in its inputs
        assert (result_high >= result_low - 1e-6).all()


class TestNoisyOrReduce:
    def test_single_dim(self):
        t = torch.tensor([[0.5, 0.3], [0.8, 0.1]])
        result = noisy_or_reduce(t, dim=1)

        # 1 - (1-0.5)(1-0.3) = 1 - 0.35 = 0.65
        # 1 - (1-0.8)(1-0.1) = 1 - 0.18 = 0.82
        expected = torch.tensor([0.65, 0.82])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_reduces_shape(self):
        t = torch.rand(3, 4, 5)
        result = noisy_or_reduce(t, dim=1)
        assert result.shape == (3, 5)

    def test_multi_dim_reduce(self):
        t = torch.rand(3, 4, 5)
        result = noisy_or_reduce(t, dim=(1, 2))
        assert result.shape == (3,)

    def test_boolean_reduce(self):
        """On {0,1} inputs, noisy-OR reduce = logical OR."""
        t = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        result = noisy_or_reduce(t, dim=1)
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestNoisyAndReduce:
    def test_basic(self):
        t = torch.tensor([0.5, 0.3])
        result = noisy_and_reduce(t, dim=0)
        expected = torch.tensor(0.15)
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_reduces_shape(self):
        t = torch.rand(3, 4, 5)
        result = noisy_and_reduce(t, dim=1)
        assert result.shape == (3, 5)


class TestMonotonicityProperties:
    """Abstract property tests verifying the primitives support
    patterns needed for semantic-like systems (e.g., entailment,
    quantification, predicate decomposition) without encoding any
    particular model.
    """

    def test_noisy_or_reduce_monotonicity(self):
        """Existential weakening: every slice is <= the noisy-OR reduction.

        For any tensor R and dimension d: R[..., i, ...] <= noisy_or_reduce(R, d)
        for all i.  This is the abstract property that makes existential
        quantification (noisy-OR) an upper bound on any witness.
        """
        torch.manual_seed(0)
        r = torch.rand(5, 6, 4)
        reduced = noisy_or_reduce(r, dim=1)  # (5, 4)

        for i in range(6):
            assert (r[:, i, :] <= reduced + 1e-6).all()

    def test_noisy_and_reduce_monotonicity(self):
        """Universal strengthening: product reduction <= every slice.

        For any tensor R and dimension d: noisy_and_reduce(R, d) <= R[..., i, ...]
        for all i.  The universal quantifier (product) is a lower bound.
        """
        torch.manual_seed(1)
        r = torch.rand(5, 6, 4)
        reduced = noisy_and_reduce(r, dim=1)  # (5, 4)

        for i in range(6):
            assert (reduced <= r[:, i, :] + 1e-6).all()

    def test_marginalization_monotonicity(self):
        """Predicate decomposition pattern: marginalizing a relation
        over some dimensions gives an upper bound on every slice.

        If R(a, b, c) is a ternary relation and we marginalize out dim 0:
        R[i, :, :] <= noisy_or_reduce(R, dim=0) for all i.

        This is the abstract version of: give(a, r, t) <= has_after(r, t)
        where has_after is the marginalization of give over the agent dim.
        """
        torch.manual_seed(2)
        r = torch.rand(4, 5, 3)
        marginalized = noisy_or_reduce(r, dim=0)  # (5, 3)

        for i in range(4):
            assert (r[i] <= marginalized + 1e-6).all()

    def test_product_implication(self):
        """Product implication pattern: when P <= Q pointwise, the
        implication degree P -> Q = 1 - P*(1-Q) is high.

        This is the abstract entailment property used for NLI:
        if the premise tensor is dominated by the hypothesis tensor,
        their implication degree approaches 1.
        """
        torch.manual_seed(3)
        p = torch.rand(5, 4) * 0.5  # values in [0, 0.5]
        q = p + torch.rand(5, 4) * 0.5  # q >= p everywhere

        implication = 1.0 - p * (1.0 - q)

        # since p <= q, each factor 1-q <= 1-p, so p*(1-q) <= p*(1-p) <= 0.25
        # therefore implication >= 0.75
        assert (implication >= 0.75 - 1e-6).all()

    def test_subsumption_monotonicity(self):
        """Upward monotonicity of existential quantification under subsumption.

        If P <= Q pointwise (P is subsumed by Q), and R is any predicate,
        then noisy_or(P * R) <= noisy_or(Q * R).

        This is the abstract version of: 'some dog runs' entails 'some animal runs'
        when dog <= animal pointwise.
        """
        torch.manual_seed(4)
        n = 8
        p = torch.rand(n) * 0.5
        q = p + torch.rand(n) * 0.5  # q >= p
        r = torch.rand(n)

        pr_exists = noisy_or_reduce(p * r, dim=0)
        qr_exists = noisy_or_reduce(q * r, dim=0)

        assert pr_exists.item() <= qr_exists.item() + 1e-6

    def test_negation_complement_involution(self):
        """Double negation is identity: 1 - (1 - x) = x.

        Verifies the complement operation composes with itself to give identity.
        """
        torch.manual_seed(5)
        x = torch.rand(5, 4)
        double_neg = 1.0 - (1.0 - x)
        torch.testing.assert_close(double_neg, x, atol=1e-6, rtol=1e-6)

    def test_de_morgan_quantifiers(self):
        """Quantifier duality: ~exists x. P(x) = forall x. ~P(x).

        In product fuzzy logic:
        1 - noisy_or_reduce(P) = noisy_and_reduce(1 - P)

        Both should evaluate identically.
        """
        torch.manual_seed(6)
        p = torch.rand(8)

        lhs = 1.0 - noisy_or_reduce(p, dim=0)
        rhs = noisy_and_reduce(1.0 - p, dim=0)

        torch.testing.assert_close(lhs, rhs, atol=1e-5, rtol=1e-5)
