"""Tests for the algebraic-effect-handler machinery.

Covers `TraceHandler`, `clamp`, `do`, `mask`, `scale`, `block`,
`replay`, and their composition on the handler stack.
"""

from __future__ import annotations

import pytest
import torch

from quivers.continuous.families import ConditionalNormal
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import FinSet
from quivers.effects import (
    BlockHandler,
    ClampHandler,
    DoHandler,
    LiftHandler,
    MaskHandler,
    Message,
    ReplayHandler,
    ScaleHandler,
    TraceHandler,
    apply_stack,
    block,
    clamp,
    do,
    lift,
    mask,
    replay,
    run_program,
    scale,
)
from quivers.effects.base import _handler_stack
from quivers.inference.trace import Trace, trace


@pytest.fixture(autouse=True)
def _reset_handler_stack():
    """Ensure a clean thread-local handler stack around every test.

    A test that intentionally leaves handlers on the stack (or fails
    mid-`with` before its __exit__ can pop) would otherwise leak
    state into subsequent tests.
    """
    _handler_stack().clear()
    yield
    _handler_stack().clear()


def _simple_program() -> MonadicProgram:
    """A minimal ``z ~ prior; y ~ likelihood(z)`` model."""
    Unit = FinSet(name="Unit", cardinality=1)
    R1 = Euclidean(name="R1", dim=1)
    prior = ConditionalNormal(Unit, R1)
    likelihood = ConditionalNormal(R1, R1)
    return MonadicProgram(
        Unit,
        R1,
        steps=[
            (("z",), prior, None),
            (("y",), likelihood, ("z",)),
        ],
        return_vars=("y",),
    )


def _batch(n: int = 4) -> torch.Tensor:
    # The prior morphism domain is `FinSet(1)`: an integer index
    # tensor of shape ``(n,)``.
    return torch.zeros(n, dtype=torch.long)


class TestHandlerStack:
    """Stack lifecycle: push on __enter__, pop on __exit__."""

    def test_empty_by_default(self) -> None:
        assert _handler_stack() == []

    def test_push_and_pop(self) -> None:
        h = TraceHandler()
        with h:
            assert _handler_stack()[-1] is h
        assert _handler_stack() == []

    def test_nested_stack_order(self) -> None:
        outer = TraceHandler()
        inner = TraceHandler()
        with outer:
            with inner:
                stack = _handler_stack()
                assert stack[0] is outer
                assert stack[1] is inner
        assert _handler_stack() == []

    def test_stack_corruption_raises(self) -> None:
        outer = TraceHandler()
        inner = TraceHandler()
        outer.__enter__()
        inner.__enter__()
        try:
            # Popping in the wrong order should raise.
            with pytest.raises(RuntimeError, match="handler stack corruption"):
                outer.__exit__(None, None, None)
        finally:
            # Always clean up so a test failure does not leak handler
            # state into subsequent tests via the thread-local stack.
            inner.__exit__(None, None, None)
            outer.__exit__(None, None, None)


class TestTraceHandler:
    """`TraceHandler` records every site the program visits."""

    def test_trace_records_every_site(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        tr = trace(prog, x)
        assert isinstance(tr, Trace)
        assert set(tr.sites.keys()) == {"z", "y"}
        assert tr.log_joint is not None
        assert tr.log_joint.shape == (4,)

    def test_output_is_populated(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        tr = trace(prog, x)
        assert tr.output is not None
        assert isinstance(tr.output, torch.Tensor)
        assert tr.output.shape == (4, 1)

    def test_stochastic_and_latent_partitions(self) -> None:
        prog = _simple_program()
        tr = trace(prog, _batch(4))
        # No sites are observed, so latent == stochastic == all sites.
        assert set(tr.latent_sites.keys()) == {"z", "y"}
        assert set(tr.stochastic_sites.keys()) == {"z", "y"}
        assert tr.observed_sites == {}

    def test_log_joint_is_sum_of_site_log_probs(self) -> None:
        torch.manual_seed(0)
        prog = _simple_program()
        x = _batch(4)
        tr = trace(prog, x)
        expected = tr.sites["z"].log_prob + tr.sites["y"].log_prob
        assert tr.log_joint is not None
        torch.testing.assert_close(tr.log_joint, expected)


class TestClampHandler:
    """`clamp` clamps named sample sites."""

    def test_clamp_sets_value(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        z_val = torch.full((4, 1), 3.7)
        with clamp({"z": z_val}):
            tr = trace(prog, x)
        torch.testing.assert_close(tr.sites["z"].value, z_val)
        assert tr.sites["z"].is_observed

    def test_clamp_still_scores_log_prob(self) -> None:
        """The clamped site's log_prob is scored under the prior."""
        torch.manual_seed(0)
        prog = _simple_program()
        x = _batch(4)
        z_val = torch.zeros(4, 1)
        with clamp({"z": z_val}):
            tr = trace(prog, x)
        # log N(0; mu, sigma) under the prior parameters should equal
        # what the morphism's log_prob would compute directly.
        prior = tr.sites["z"].morphism
        assert prior is not None
        expected = prior.log_prob(x, z_val)
        torch.testing.assert_close(tr.sites["z"].log_prob, expected)


class TestDoHandler:
    """`do` intervenes without contributing log-density."""

    def test_do_zeros_log_prob(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        z_val = torch.full((4, 1), 3.7)
        with do({"z": z_val}):
            tr = trace(prog, x)
        # Value is clamped; log_prob is zero (no density contribution).
        torch.testing.assert_close(tr.sites["z"].value, z_val)
        torch.testing.assert_close(
            tr.sites["z"].log_prob,
            torch.zeros(4),
        )

    def test_do_only_contributes_child_density(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        z_val = torch.zeros(4, 1)
        with do({"z": z_val}):
            tr = trace(prog, x)
        # log_joint = 0 (from z) + log p(y | z=z_val).
        assert tr.log_joint is not None
        torch.testing.assert_close(tr.log_joint, tr.sites["y"].log_prob)


class TestMaskHandler:
    """`mask` gates log-density per element."""

    def test_mask_zeroes_out_selected_rows(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        # Keep first two rows; zero out the last two.
        m = torch.tensor([1.0, 1.0, 0.0, 0.0])
        with mask(m):
            tr = trace(prog, x)
        for name in ("z", "y"):
            site = tr.sites[name]
            assert torch.allclose(site.log_prob[2:], torch.zeros(2))
            assert not torch.allclose(site.log_prob[:2], torch.zeros(2))


class TestScaleHandler:
    """`scale` multiplies log-density by a scalar."""

    def test_scale_multiplies(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        torch.manual_seed(0)
        tr_ref = trace(prog, x)
        torch.manual_seed(0)
        with scale(2.5):
            tr_sc = trace(prog, x)
        for name in ("z", "y"):
            torch.testing.assert_close(
                tr_sc.sites[name].log_prob,
                tr_ref.sites[name].log_prob * 2.5,
            )


class TestComposition:
    """Handlers compose on the stack."""

    def test_clamp_scale_mask_stack(self) -> None:
        """A stack of ``clamp + scale + mask`` produces the
        expected per-site log-density."""
        torch.manual_seed(0)
        prog = _simple_program()
        x = _batch(4)
        z_val = torch.zeros(4, 1)
        m = torch.tensor([1.0, 1.0, 0.0, 0.0])
        factor = 2.5

        # Baseline: clamp alone.
        torch.manual_seed(0)
        with clamp({"z": z_val}):
            tr_ref = trace(prog, x)

        # Stacked: clamp + scale + mask (outer-to-inner order).
        torch.manual_seed(0)
        with clamp({"z": z_val}):
            with scale(factor):
                with mask(m):
                    tr_stack = trace(prog, x)

        # The innermost handler (mask) rewrites log_prob last, so
        # each site's log-density is factor * mask * baseline. Post-
        # hooks run inner-to-outer: mask fires, then scale — but the
        # postprocess order in `apply_stack` walks the `seen` list
        # in reverse, so mask (pushed last) runs first, then scale.
        # The composed operator is thus `scale * mask`.
        for name in ("z", "y"):
            expected = tr_ref.sites[name].log_prob * factor * m
            torch.testing.assert_close(tr_stack.sites[name].log_prob, expected)

    def test_do_then_trace(self) -> None:
        """Intervention followed by recording gives a trace whose
        joint density excludes the intervened site."""
        prog = _simple_program()
        x = _batch(4)
        z_val = torch.zeros(4, 1)
        with do({"z": z_val}):
            tr = trace(prog, x)
        assert tr.log_joint is not None
        torch.testing.assert_close(tr.log_joint, tr.sites["y"].log_prob)


class TestBlockHandler:
    """`block` hides sites from outer handlers."""

    def test_block_hides_from_outer_trace(self) -> None:
        """Outer TraceHandler wrapped around block should not see
        hidden sites."""
        prog = _simple_program()
        x = _batch(4)
        outer = TraceHandler()
        with outer:
            with block(hide=["z"]):
                run_program(prog, x)
        assert "z" not in outer.trace.sites
        assert "y" in outer.trace.sites

    def test_expose_only_names_selected_sites(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        outer = TraceHandler()
        with outer:
            with block(expose=["y"]):
                run_program(prog, x)
        assert "z" not in outer.trace.sites
        assert "y" in outer.trace.sites

    def test_hide_and_expose_are_exclusive(self) -> None:
        with pytest.raises(ValueError, match="at most one"):
            BlockHandler(hide=["z"], expose=["y"])


class TestReplayHandler:
    """`replay` reinstalls values from a captured trace."""

    def test_replay_reinstalls_values(self) -> None:
        prog = _simple_program()
        x = _batch(4)
        torch.manual_seed(0)
        tr1 = trace(prog, x)
        # Rerun under a different seed but replay tr1's sample.
        torch.manual_seed(42)
        with replay(tr1):
            tr2 = trace(prog, x)
        torch.testing.assert_close(tr1.sites["z"].value, tr2.sites["z"].value)


class TestLiftHandler:
    """`lift` samples one prior draw per parameter."""

    def test_lift_populates_sampled_params(self) -> None:
        torch.manual_seed(0)
        prog = _simple_program()
        x = _batch(4)
        h = LiftHandler(prior_scale=0.5)
        with h:
            run_program(prog, x)
        # The ConditionalNormal parameter source contains at least one
        # nn.Parameter (bias / weight); every one should be sampled.
        assert len(h.sampled_params) > 0


class TestApplyStackDefault:
    """The `apply_stack(msg, default=...)` contract runs default
    between the pre-pass and post-pass, and only when the pre-pass
    left the site unresolved."""

    def test_default_fires_when_value_missing(self) -> None:
        called = {"ran": False}

        def default(m: Message) -> None:
            m.value = torch.tensor([1.0])
            m.log_prob = torch.tensor([0.0])
            called["ran"] = True

        msg = Message(kind="sample", name="x")
        apply_stack(msg, default=default)
        assert called["ran"]
        assert msg.value is not None

    def test_default_can_see_prepass_state(self) -> None:
        """A handler-supplied value is visible to `default`."""

        class Preset(TraceHandler):
            def _pyro_sample(self, msg: Message) -> None:
                msg.value = torch.tensor([7.0])
                msg.log_prob = torch.tensor([0.0])

        seen: dict[str, torch.Tensor | None] = {"value": None}

        def default(m: Message) -> None:
            seen["value"] = m.value

        with Preset():
            apply_stack(Message(kind="sample", name="x"), default=default)
        assert seen["value"] is not None
        assert seen["value"].item() == 7.0


class TestFactoriesReturnHandlers:
    """The short-name factories should return the corresponding handler."""

    def test_clamp_factory(self) -> None:
        assert isinstance(clamp({}), ClampHandler)

    def test_do_factory(self) -> None:
        assert isinstance(do({}), DoHandler)

    def test_scale_factory(self) -> None:
        assert isinstance(scale(1.0), ScaleHandler)

    def test_mask_factory(self) -> None:
        assert isinstance(mask(torch.tensor(1.0)), MaskHandler)

    def test_block_factory(self) -> None:
        assert isinstance(block(hide=["x"]), BlockHandler)

    def test_replay_factory(self) -> None:
        assert isinstance(replay(Trace()), ReplayHandler)

    def test_lift_factory(self) -> None:
        assert isinstance(lift(1.0), LiftHandler)
