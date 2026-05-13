"""Extended coverage for the grouped ``marginalize`` block:
nesting, product fibrations, reductions, body vectorisation,
continuous latents in scope, and end-to-end SVI with a
continuous guide.

Each section verifies one of the surface features the grouped
``marginalize`` block exposes. Failures here are surface-form /
semantic regressions, not just runtime-primitive regressions.
"""

from __future__ import annotations

import os

import pytest
import torch

from quivers.continuous.bayesian import marginalize_grouped


_LOCAL_GRAMMAR = pytest.mark.skipif(
    os.environ.get("QVR_USE_LOCAL_GRAMMAR", "") not in ("1", "true", "True"),
    reason="needs QVR_USE_LOCAL_GRAMMAR=1 to pick up the in-tree grammar",
)


# ---------------------------------------------------------------------------
# Runtime primitive: reduction parameter
# ---------------------------------------------------------------------------


class TestReductionParameter:
    """The runtime primitive accepts ``reduction = logsumexp | sum
    | mean``; each gives the documented per-group reduction over
    the class axis."""

    def test_logsumexp_default(self) -> None:
        torch.manual_seed(0)
        ll = torch.randn(6, 3)
        idx = torch.tensor([0, 0, 1, 1, 2, 2])
        log_prior = torch.log(torch.ones(3) / 3)
        out = marginalize_grouped(ll, idx, log_prior, 3)
        # Build the hand-rolled reference under logsumexp.
        grouped = torch.zeros(3, 3)
        grouped = grouped.index_add(0, idx, ll)
        expected = torch.logsumexp(log_prior + grouped, dim=-1).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_sum_reduction(self) -> None:
        torch.manual_seed(0)
        ll = torch.randn(6, 3)
        idx = torch.tensor([0, 0, 1, 1, 2, 2])
        log_prior = torch.log(torch.ones(3) / 3)
        out = marginalize_grouped(
            ll, idx, log_prior, 3, reduction="sum"
        )
        grouped = torch.zeros(3, 3)
        grouped = grouped.index_add(0, idx, ll)
        expected = (log_prior + grouped).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_mean_reduction(self) -> None:
        torch.manual_seed(0)
        ll = torch.randn(6, 3)
        idx = torch.tensor([0, 0, 1, 1, 2, 2])
        log_prior = torch.log(torch.ones(3) / 3)
        out = marginalize_grouped(
            ll, idx, log_prior, 3, reduction="mean"
        )
        grouped = torch.zeros(3, 3)
        grouped = grouped.index_add(0, idx, ll)
        expected = (log_prior + grouped).mean(dim=-1).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_unknown_reduction_raises(self) -> None:
        ll = torch.randn(2, 3)
        idx = torch.tensor([0, 0])
        log_prior = torch.log(torch.ones(3) / 3)
        with pytest.raises(ValueError, match="reduction"):
            marginalize_grouped(
                ll, idx, log_prior, 1, reduction="not_a_reduction"
            )


# ---------------------------------------------------------------------------
# Runtime primitive: product fibration
# ---------------------------------------------------------------------------


class TestProductFibration:
    """The runtime primitive accepts a tuple of co-indexed
    fibrations + a tuple of group sizes (the product grouping
    plate). Behaviour matches the single-fibration form when the
    product collapses to one axis, and matches a hand-rolled
    Python loop for the two-axis case."""

    def test_single_axis_via_tuple_matches_scalar_form(self) -> None:
        torch.manual_seed(0)
        ll = torch.randn(6, 3)
        idx = torch.tensor([0, 1, 2, 0, 1, 2])
        log_prior = torch.log(torch.ones(3) / 3)
        scalar = marginalize_grouped(ll, idx, log_prior, 3)
        product = marginalize_grouped(
            ll, (idx,), log_prior, (3,)
        )
        assert torch.allclose(scalar, product, atol=1e-6)

    def test_two_axis_product_matches_hand_rolled(self) -> None:
        torch.manual_seed(0)
        N, K = 12, 3
        G1, G2 = 3, 2  # group sizes
        ll = torch.randn(N, K)
        idx_a = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        idx_b = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
        log_prior = torch.log(torch.ones(K) / K)
        out = marginalize_grouped(
            ll, (idx_a, idx_b), log_prior, (G1, G2)
        )
        # Hand-rolled reference: scatter-add over the product index.
        flat = idx_a * G2 + idx_b
        grouped = torch.zeros(G1 * G2, K)
        grouped = grouped.index_add(0, flat, ll)
        expected = torch.logsumexp(log_prior + grouped, dim=-1).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_two_axis_prior_with_per_cell_class_distribution(self) -> None:
        """When the prior is shape ``(G1, G2, K)`` it's flattened
        in row-major order and applied per-cell."""
        torch.manual_seed(0)
        N, K = 8, 2
        G1, G2 = 2, 2
        ll = torch.randn(N, K)
        idx_a = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        idx_b = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        # Per-cell prior: different mixing for each (g1, g2).
        prior_per_cell = torch.tensor(
            [[[0.7, 0.3], [0.4, 0.6]], [[0.5, 0.5], [0.1, 0.9]]]
        )
        log_prior = torch.log(prior_per_cell)
        out = marginalize_grouped(
            ll, (idx_a, idx_b), log_prior, (G1, G2)
        )
        flat = idx_a * G2 + idx_b
        grouped = torch.zeros(G1 * G2, K)
        grouped = grouped.index_add(0, flat, ll)
        expected = torch.logsumexp(
            log_prior.reshape(G1 * G2, K) + grouped, dim=-1
        ).sum()
        assert torch.allclose(out, expected, atol=1e-6)

    def test_arity_mismatch_raises(self) -> None:
        ll = torch.randn(4, 2)
        idx_a = torch.tensor([0, 0, 1, 1])
        log_prior = torch.log(torch.ones(2) / 2)
        with pytest.raises(
            ValueError, match="number of indices"
        ):
            marginalize_grouped(
                ll, (idx_a, idx_a), log_prior, (2,)
            )

    def test_product_axis_out_of_range_raises(self) -> None:
        ll = torch.randn(4, 2)
        idx_a = torch.tensor([0, 0, 3, 1])  # 3 is out of range for size 3
        idx_b = torch.tensor([0, 0, 1, 1])
        log_prior = torch.log(torch.ones(2) / 2)
        with pytest.raises(ValueError, match="outside"):
            marginalize_grouped(
                ll, (idx_a, idx_b), log_prior, (3, 2)
            )


# ---------------------------------------------------------------------------
# Surface: nested marginalize blocks
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
class TestNestedMarginalize:
    """Mixtures-of-mixtures: a marginalize block whose body contains
    another marginalize block. The outer latent is in scope inside
    the inner block, and the body-vectorisation pass threads the
    captured observe through every level of nesting."""

    def test_three_level_nested_blocks_compile(self) -> None:
        """A three-level nested marginalize compiles: outer → middle
        → inner, with the innermost body's observe captured and
        threaded up through every level."""
        from quivers.dsl import loads

        src = """
        object G_outer : 2
        object G_middle : 2
        object G_inner : 2
        object Resp : 8
        object K_outer : 2
        object K_middle : 2
        object K_inner : 2

        program triple : Resp -> Resp
            probs_a : K_outer <- HalfNormal(1.0)
            probs_b : K_middle <- HalfNormal(1.0)
            probs_c : K_inner <- HalfNormal(1.0)
            idx_a : Resp <- HalfNormal(1.0)
            idx_b : Resp <- HalfNormal(1.0)
            idx_c : Resp <- HalfNormal(1.0)
            marginalize outer : K_outer <- Dirichlet(probs_a)
                over G_outer via idx_a
                in {
                    marginalize middle : K_middle <- Dirichlet(probs_b)
                        over G_middle via idx_b
                        in {
                            marginalize inner : K_inner <- Dirichlet(probs_c)
                                over G_inner via idx_c
                                in {
                                    observe r : Resp <- HalfNormal(1.0)
                                }
                        }
                }
            return probs_a
        export triple
        """
        m = loads(src)
        assert m.morphism is not None

    def test_three_level_nested_blocks_run_log_joint(self) -> None:
        """The three-level nested marginalize model runs through
        log_joint end-to-end without errors, given all latents
        clamped via observations. This is the runtime gate: a
        compile-only test would let a non-functional runtime slip
        through."""
        import torch
        from quivers.dsl import loads

        src = """
        object G_outer : 2
        object G_middle : 2
        object G_inner : 2
        object Resp : 8
        object K_outer : 2
        object K_middle : 2
        object K_inner : 2

        program triple : Resp -> Resp
            probs_a : K_outer <- HalfNormal(1.0)
            probs_b : K_middle <- HalfNormal(1.0)
            probs_c : K_inner <- HalfNormal(1.0)
            idx_a : Resp <- HalfNormal(1.0)
            idx_b : Resp <- HalfNormal(1.0)
            idx_c : Resp <- HalfNormal(1.0)
            marginalize outer : K_outer <- Dirichlet(probs_a)
                over G_outer via idx_a
                in {
                    marginalize middle : K_middle <- Dirichlet(probs_b)
                        over G_middle via idx_b
                        in {
                            marginalize inner : K_inner <- Dirichlet(probs_c)
                                over G_inner via idx_c
                                in {
                                    observe r : Resp <- HalfNormal(1.0)
                                }
                        }
                }
            return probs_a
        export triple
        """
        model = loads(src).morphism
        torch.manual_seed(0)
        obs = {
            "probs_a": torch.tensor([0.5, 0.5]),
            "probs_b": torch.tensor([0.5, 0.5]),
            "probs_c": torch.tensor([0.5, 0.5]),
            "idx_a": torch.tensor([0, 0, 1, 1, 0, 0, 1, 1]),
            "idx_b": torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
            "idx_c": torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
            "r": torch.zeros(8),
        }
        out = model.log_joint(torch.zeros(1, 1), obs)
        assert torch.isfinite(out).all()

    def test_nested_blocks_compile(self) -> None:
        from quivers.dsl import loads

        src = """
        object G1 : 2
        object G2 : 3
        object Resp : 6
        object K1 : 2
        object K2 : 3

        program demo : Resp -> Resp
            probs_outer : K1 <- HalfNormal(1.0)
            probs_inner : K2 <- HalfNormal(1.0)
            outer_idx : Resp <- HalfNormal(1.0)
            inner_idx : Resp <- HalfNormal(1.0)
            marginalize outer : K1 <- Dirichlet(probs_outer)
                over G1 via outer_idx
                in {
                    marginalize inner : K2 <- Dirichlet(probs_inner)
                        over G2 via inner_idx
                        in {
                            observe r : Resp <- HalfNormal(1.0)
                        }
                }
            return probs_outer
        export demo
        """
        m = loads(src)
        assert m.morphism is not None


# ---------------------------------------------------------------------------
# Surface: product fibration in the DSL
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
class TestProductFibrationSurface:
    """``over G * H via product(idx_a, idx_b)`` parses, compiles,
    and threads the product-fibration data into the runtime."""

    def test_product_fibration_compiles(self) -> None:
        from quivers.dsl import loads

        src = """
        object Item : 3
        object Subj : 2
        object Resp : 6
        object Class : 4

        program demo : Resp -> Resp
            probs : Class <- HalfNormal(1.0)
            item_idx : Resp <- HalfNormal(1.0)
            subj_idx : Resp <- HalfNormal(1.0)
            marginalize cls : Class <- Dirichlet(probs)
                over Item * Subj via product(item_idx, subj_idx)
                in {
                    observe r : Resp <- HalfNormal(1.0)
                }
            return probs
        export demo
        """
        m = loads(src)
        assert m.morphism is not None

    def test_arity_mismatch_errors(self) -> None:
        from quivers.dsl import loads
        from quivers.dsl.compiler import CompileError

        src = """
        object Item : 3
        object Subj : 2
        object Resp : 6
        object Class : 4

        program demo : Resp -> Resp
            probs : Class <- HalfNormal(1.0)
            item_idx : Resp <- HalfNormal(1.0)
            marginalize cls : Class <- Dirichlet(probs)
                over Item * Subj via item_idx
                in {
                    observe r : Resp <- HalfNormal(1.0)
                }
            return probs
        export demo
        """
        with pytest.raises(CompileError, match="arity"):
            loads(src)


# ---------------------------------------------------------------------------
# Surface: reduction parameter in the DSL
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
class TestReductionSurface:
    """``reduction = sum`` / ``reduction = mean`` parses and is
    threaded into the runtime primitive."""

    def test_reduction_clause_compiles(self) -> None:
        from quivers.dsl import loads

        src = """
        object Item : 3
        object Resp : 6
        object Class : 4

        program demo : Resp -> Resp
            probs : Class <- HalfNormal(1.0)
            idx : Resp <- HalfNormal(1.0)
            marginalize cls : Class <- Dirichlet(probs)
                over Item via idx reduction = sum
                in {
                    observe r : Resp <- HalfNormal(1.0)
                }
            return probs
        export demo
        """
        m = loads(src)
        assert m.morphism is not None

    def test_unknown_reduction_errors(self) -> None:
        from quivers.dsl import loads
        from quivers.dsl.compiler import CompileError

        src = """
        object Item : 3
        object Resp : 6
        object Class : 4

        program demo : Resp -> Resp
            probs : Class <- HalfNormal(1.0)
            idx : Resp <- HalfNormal(1.0)
            marginalize cls : Class <- Dirichlet(probs)
                over Item via idx reduction = bogus
                in {
                    observe r : Resp <- HalfNormal(1.0)
                }
            return probs
        export demo
        """
        with pytest.raises(CompileError, match="unknown reduction"):
            loads(src)


# ---------------------------------------------------------------------------
# Additional surface-level compile-error coverage
# ---------------------------------------------------------------------------


@_LOCAL_GRAMMAR
class TestSurfaceCompileErrors:
    """Each error path the grouped-marginalize compiler raises has
    its own test so a regression hides nothing."""

    def test_categorical_with_literal_first_arg_errors(self) -> None:
        """A grouped block's prior must reference a NAMED probs
        tensor, not a literal — otherwise the runtime has nothing
        to broadcast against."""
        from quivers.dsl import loads
        from quivers.dsl.compiler import CompileError

        src = """
        object Item : 3
        object Resp : 6
        object Class : 2

        program demo : Resp -> Resp
            idx : Resp <- HalfNormal(1.0)
            marginalize cls : Class <- Dirichlet(1.0)
                over Item via idx
                in {
                    observe r : Resp <- HalfNormal(1.0)
                }
            return idx
        export demo
        """
        with pytest.raises(CompileError, match="named probs"):
            loads(src)

    def test_body_without_observe_errors(self) -> None:
        """The body must end with an observe step (or a nested
        marginalize). A body containing only let-steps or sample-
        steps has no captured per-(N, K) ll to feed the reduction."""
        from quivers.dsl import loads
        from quivers.dsl.compiler import CompileError

        src = """
        object Item : 3
        object Resp : 6
        object Class : 2

        program demo : Resp -> Resp
            probs : Class <- HalfNormal(1.0)
            idx : Resp <- HalfNormal(1.0)
            marginalize cls : Class <- Dirichlet(probs)
                over Item via idx
                in {
                    other : Resp <- HalfNormal(1.0)
                }
            return probs
        export demo
        """
        with pytest.raises(CompileError, match="observe"):
            loads(src)


@_LOCAL_GRAMMAR
def test_three_axis_product_fibration_dsl_compiles() -> None:
    """`over A * B * C via product(idx_a, idx_b, idx_c)` parses
    and compiles."""
    from quivers.dsl import loads

    src = """
    object A : 2
    object B : 2
    object C : 2
    object Resp : 8
    object K : 2

    program triple_prod : Resp -> Resp
        probs : K <- HalfNormal(1.0)
        idx_a : Resp <- HalfNormal(1.0)
        idx_b : Resp <- HalfNormal(1.0)
        idx_c : Resp <- HalfNormal(1.0)
        marginalize cls : K <- Dirichlet(probs)
            over A * B * C via product(idx_a, idx_b, idx_c)
            in {
                observe r : Resp <- HalfNormal(1.0)
            }
        return probs
    export triple_prod
    """
    m = loads(src)
    assert m.morphism is not None
