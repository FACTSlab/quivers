"""Combination tests: marginalize blocks mixing depth, fibration
shape, and reduction across the same model.

Each test stresses a different combination of the features issue
#9 mentions: nesting depth × fibration arity × reduction mode ×
continuous-latent dependence. The goal is to catch regressions
that only appear when features interact (one of the canonical
failure modes during the refactor).
"""

from __future__ import annotations

import os
from textwrap import dedent

import pytest
import torch

from quivers.continuous.bayesian import marginalize_grouped


_LOCAL_GRAMMAR = pytest.mark.skipif(
    os.environ.get("QVR_USE_LOCAL_GRAMMAR", "") not in ("1", "true", "True"),
    reason="needs QVR_USE_LOCAL_GRAMMAR=1 to pick up the in-tree grammar",
)


# ---------------------------------------------------------------------------
# Surface: combinations of features in one DSL program
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
class TestSurfaceCombinations:
    """Each test compiles a program that combines several issue-#9
    features in one program."""

    def test_nested_with_product_fibration_at_inner_level(self) -> None:
        """Outer single-fibration block + inner product-fibration
        block."""
        from quivers.dsl import loads

        src = dedent("""
        object Item : 2
        object Subj : 2
        object Resp : 4
        object K_outer : 2
        object K_inner : 2

        program nested_prod : Resp -> Resp
            probs_outer : K_outer <- HalfNormal(1.0)
            probs_inner : K_inner <- HalfNormal(1.0)
            outer_idx : Resp <- HalfNormal(1.0)
            item_idx : Resp <- HalfNormal(1.0)
            subj_idx : Resp <- HalfNormal(1.0)
            marginalize outer : K_outer <- Dirichlet(probs_outer)
                over Item via outer_idx
                in {
                    marginalize inner : K_inner <- Dirichlet(probs_inner)
                        over Item * Subj via product(item_idx, subj_idx)
                        in {
                            observe r : Resp <- HalfNormal(1.0)
                        }
                }
            return probs_outer
        export nested_prod
        """)
        m = loads(src)
        assert m.morphism is not None

    def test_nested_with_mixed_reductions(self) -> None:
        """Outer logsumexp + inner sum reduction in the same
        program."""
        from quivers.dsl import loads

        src = dedent("""
        object Item : 2
        object Subj : 2
        object Resp : 4
        object K_outer : 2
        object K_inner : 2

        program mixed_reds : Resp -> Resp
            probs_outer : K_outer <- HalfNormal(1.0)
            probs_inner : K_inner <- HalfNormal(1.0)
            outer_idx : Resp <- HalfNormal(1.0)
            inner_idx : Resp <- HalfNormal(1.0)
            marginalize outer : K_outer <- Dirichlet(probs_outer)
                over Item via outer_idx reduction = logsumexp
                in {
                    marginalize inner : K_inner <- Dirichlet(probs_inner)
                        over Subj via inner_idx reduction = sum
                        in {
                            observe r : Resp <- HalfNormal(1.0)
                        }
                }
            return probs_outer
        export mixed_reds
        """)
        m = loads(src)
        assert m.morphism is not None

    def test_three_level_with_continuous_latents_in_scope(self) -> None:
        """A 3-level nested marginalize with a continuous latent
        declared at the program scope (in scope inside every
        block's body)."""
        from quivers.dsl import loads

        src = dedent("""
        object G_a : 2
        object G_b : 2
        object G_c : 2
        object Resp : 6
        object K_a : 2
        object K_b : 2
        object K_c : 2

        program three_with_cont : Resp -> Resp
            mu_shift <- Normal(0.0, 1.0)
            probs_a : K_a <- HalfNormal(1.0)
            probs_b : K_b <- HalfNormal(1.0)
            probs_c : K_c <- HalfNormal(1.0)
            idx_a : Resp <- HalfNormal(1.0)
            idx_b : Resp <- HalfNormal(1.0)
            idx_c : Resp <- HalfNormal(1.0)
            marginalize a : K_a <- Dirichlet(probs_a)
                over G_a via idx_a
                in {
                    marginalize b : K_b <- Dirichlet(probs_b)
                        over G_b via idx_b
                        in {
                            marginalize c : K_c <- Dirichlet(probs_c)
                                over G_c via idx_c
                                in {
                                    observe r : Resp <- Normal(mu_shift, 1.0)
                                }
                        }
                }
            return mu_shift
        export three_with_cont
        """)
        m = loads(src)
        assert m.morphism is not None


# ---------------------------------------------------------------------------
# Runtime: end-to-end log_joint for the surface combinations
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
class TestRuntimeCombinations:
    """The same surface combinations run end-to-end through
    log_joint, producing a finite scalar."""

    def test_three_level_with_continuous_latent_runs(self) -> None:
        from quivers.dsl import loads

        src = dedent("""
        object G_a : 2
        object G_b : 2
        object G_c : 2
        object Resp : 6
        object K_a : 2
        object K_b : 2
        object K_c : 2

        program three_with_cont : Resp -> Resp
            mu_shift <- Normal(0.0, 1.0)
            probs_a : K_a <- HalfNormal(1.0)
            probs_b : K_b <- HalfNormal(1.0)
            probs_c : K_c <- HalfNormal(1.0)
            idx_a : Resp <- HalfNormal(1.0)
            idx_b : Resp <- HalfNormal(1.0)
            idx_c : Resp <- HalfNormal(1.0)
            marginalize a : K_a <- Dirichlet(probs_a)
                over G_a via idx_a
                in {
                    marginalize b : K_b <- Dirichlet(probs_b)
                        over G_b via idx_b
                        in {
                            marginalize c : K_c <- Dirichlet(probs_c)
                                over G_c via idx_c
                                in {
                                    observe r : Resp <- Normal(mu_shift, 1.0)
                                }
                        }
                }
            return mu_shift
        export three_with_cont
        """)
        model = loads(src).morphism
        obs = {
            "mu_shift": torch.tensor([0.5]),
            "probs_a": torch.tensor([0.5, 0.5]),
            "probs_b": torch.tensor([0.5, 0.5]),
            "probs_c": torch.tensor([0.5, 0.5]),
            "idx_a": torch.tensor([0, 0, 1, 1, 0, 1]),
            "idx_b": torch.tensor([0, 1, 0, 1, 1, 0]),
            "idx_c": torch.tensor([0, 0, 0, 1, 1, 1]),
            "r": torch.zeros(6),
        }
        out = model.log_joint(torch.zeros(1, 1), obs)
        assert torch.isfinite(out).all()

    def test_nested_product_fibration_runs(self) -> None:
        from quivers.dsl import loads

        src = dedent("""
        object Item : 2
        object Subj : 2
        object Resp : 4
        object K_outer : 2
        object K_inner : 2

        program nested_prod : Resp -> Resp
            probs_outer : K_outer <- HalfNormal(1.0)
            probs_inner : K_inner <- HalfNormal(1.0)
            outer_idx : Resp <- HalfNormal(1.0)
            item_idx : Resp <- HalfNormal(1.0)
            subj_idx : Resp <- HalfNormal(1.0)
            marginalize outer : K_outer <- Dirichlet(probs_outer)
                over Item via outer_idx
                in {
                    marginalize inner : K_inner <- Dirichlet(probs_inner)
                        over Item * Subj via product(item_idx, subj_idx)
                        in {
                            observe r : Resp <- HalfNormal(1.0)
                        }
                }
            return probs_outer
        export nested_prod
        """)
        model = loads(src).morphism
        obs = {
            "probs_outer": torch.tensor([0.5, 0.5]),
            "probs_inner": torch.tensor([0.5, 0.5]),
            "outer_idx": torch.tensor([0, 0, 1, 1]),
            "item_idx": torch.tensor([0, 0, 1, 1]),
            "subj_idx": torch.tensor([0, 1, 0, 1]),
            "r": torch.zeros(4),
        }
        out = model.log_joint(torch.zeros(1, 1), obs)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Primitive: combinations of fibration arity, reduction mode, and
# extra-axes (multi-level outer broadcast).
# ---------------------------------------------------------------------------


class TestPrimitiveCombinations:
    """The runtime primitive composing extra axes × fibration arity ×
    reduction mode."""

    def test_extra_axes_with_product_fibration(self) -> None:
        """Combine outer-broadcast axes (from upstream nested
        levels) with a product fibration at the current level."""
        torch.manual_seed(0)
        N, K = 6, 3
        K_extra1, K_extra2 = 2, 2
        G1, G2 = 2, 3
        # ll shape: (N, K_extra1, K_extra2, K)
        ll = torch.randn(N, K_extra1, K_extra2, K)
        idx_a = torch.tensor([0, 1, 0, 1, 0, 1])
        idx_b = torch.tensor([0, 0, 1, 1, 2, 2])
        prior = torch.log(torch.ones(K) / K)
        out = marginalize_grouped(
            ll, (idx_a, idx_b), prior, (G1, G2)
        )
        # Hand-rolled reference: scatter along product idx, then
        # logsumexp over K, sum over flat group axis. Extra axes
        # pass through.
        flat_idx = idx_a * G2 + idx_b
        grouped = torch.zeros(G1 * G2, K_extra1, K_extra2, K)
        grouped = grouped.index_add(0, flat_idx, ll)
        per_group = torch.logsumexp(
            prior + grouped, dim=-1
        )  # (G1*G2, K_extra1, K_extra2)
        expected = per_group.sum(dim=0)  # (K_extra1, K_extra2)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_extra_axes_with_sum_reduction(self) -> None:
        torch.manual_seed(0)
        N, K = 5, 3
        K_extra = 2
        G = 2
        ll = torch.randn(N, K_extra, K)
        idx = torch.tensor([0, 0, 1, 1, 0])
        prior = torch.log(torch.ones(K) / K)
        out = marginalize_grouped(
            ll, idx, prior, G, reduction="sum"
        )
        grouped = torch.zeros(G, K_extra, K)
        grouped = grouped.index_add(0, idx, ll)
        per_group = (prior + grouped).sum(dim=-1)  # (G, K_extra)
        expected = per_group.sum(dim=0)  # (K_extra,)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_no_row_axis_with_product_per_cell_prior(self) -> None:
        """The intermediate-level path (no row axis) with a
        per-cell prior broadcasting over a multi-axis class
        broadcast."""
        torch.manual_seed(0)
        K_outer, K = 2, 3
        # Intermediate contribution: shape (K_outer, K).
        ll = torch.randn(K_outer, K)
        prior = torch.log(
            torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])
        )
        out = marginalize_grouped(
            ll,
            torch.zeros(0, dtype=torch.long),
            prior,
            1,
        )
        expected = torch.logsumexp(prior + ll, dim=-1)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_three_level_with_mixed_reductions_composition(self) -> None:
        """Three nested primitive calls with three different
        reductions (logsumexp, sum, mean) compose to a hand-rolled
        reference."""
        torch.manual_seed(0)
        N = 4
        K_a, K_b, K_c = 2, 2, 3
        G_c = 2
        ll = torch.randn(N, K_a, K_b, K_c)
        idx_c = torch.tensor([0, 1, 0, 1])
        prior_c = torch.log(torch.ones(K_c) / K_c)
        prior_b = torch.log(torch.ones(K_b) / K_b)
        prior_a = torch.log(torch.ones(K_a) / K_a)
        # Innermost level uses logsumexp.
        c_out = marginalize_grouped(
            ll, idx_c, prior_c, G_c, reduction="logsumexp"
        )
        assert c_out.shape == (K_a, K_b)
        # Middle level uses sum.
        b_out = marginalize_grouped(
            c_out,
            torch.zeros(0, dtype=torch.long),
            prior_b,
            1,
            reduction="sum",
        )
        assert b_out.shape == (K_a,)
        # Outer level uses mean.
        a_out = marginalize_grouped(
            b_out,
            torch.zeros(0, dtype=torch.long),
            prior_a,
            1,
            reduction="mean",
        )
        # Hand-rolled reference.
        grouped = torch.zeros(G_c, K_a, K_b, K_c)
        grouped = grouped.index_add(0, idx_c, ll)
        per_group_c = torch.logsumexp(
            prior_c + grouped, dim=-1
        )  # (G_c, K_a, K_b)
        c_ref = per_group_c.sum(dim=0)  # (K_a, K_b)
        b_ref = (prior_b + c_ref).sum(dim=-1)  # (K_a,)
        a_ref = (prior_a + b_ref).mean(dim=-1)  # scalar
        assert torch.allclose(a_out, a_ref, atol=1e-6)
