"""Handler-aware interpreter for `MonadicProgram`.

`run_program` walks a `MonadicProgram` step by step, and every
observable action goes through
`quivers.effects.base.apply_stack` before the site's value is
finalised. Handlers see the message; if none supplies a value the
`apply_stack` default callback samples from the underlying morphism
(or evaluates the let / score callable) and installs the fallback.

The public entry point is `run_program(program, x, observations=None)`,
which returns the program's output. Handlers accumulate state on
the side. A caller that wants a `Trace` stacks a `TraceHandler`
before `run_program` and reads its `trace` attribute afterwards;
the thin `quivers.inference.trace.trace` wrapper does exactly this.
"""

from __future__ import annotations

from typing import cast

import torch

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.programs import (
    MonadicProgram,
    _LetSpec,
    _ScoreSpec,
    _StepSpec,
)
from quivers.effects.base import Message, apply_stack


def _prepopulate_env(
    program: MonadicProgram,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Bind program parameters and host-data covariates into the env.

    Splits the input tensor along declared parameters, then surfaces
    any ``observations`` key that does not match a declared site as
    a plain env binding (the host-data channel for integer index
    arrays consumed inside let bindings).
    """
    env: dict[str, torch.Tensor] = {"_x_input": x}

    if program._params is not None and program._param_dims is not None:
        splits = torch.split(x, program._param_dims, dim=-1)
        assert program._param_is_continuous is not None
        for pname, chunk, is_cont in zip(
            program._params,
            splits,
            program._param_is_continuous,
        ):
            if not is_cont and chunk.shape[-1] == 1:
                env[pname] = chunk.squeeze(-1)
            else:
                env[pname] = chunk

    declared: set[str] = set()
    for spec in program._step_specs:
        if isinstance(spec, (_LetSpec, _ScoreSpec)):
            declared.add(spec.var)
        else:
            declared.update(spec.vars)
    for key, val in observations.items():
        if key not in declared:
            env[key] = val

    return env


def _emit_let(
    spec: _LetSpec,
    env: dict[str, torch.Tensor],
    x: torch.Tensor,
) -> None:
    """Compute a `let` binding through the handler stack."""
    msg = Message(kind="let", name=spec.var, is_deterministic=True)

    def default(m: Message) -> None:
        if m.value is None:
            if isinstance(spec.value, str):
                m.value = env[spec.value]
            elif callable(spec.value):
                m.value = cast(torch.Tensor, spec.value(env))
            else:
                m.value = torch.full(
                    (x.shape[0],),
                    float(spec.value),
                    device=x.device,
                )
        if m.log_prob is None:
            m.log_prob = torch.zeros((), device=x.device)

    apply_stack(msg, default=default)
    assert msg.value is not None
    env[spec.var] = msg.value


def _emit_score(
    spec: _ScoreSpec,
    env: dict[str, torch.Tensor],
) -> None:
    """Compute a `score` step through the handler stack."""
    msg = Message(kind="score", name=spec.var, is_deterministic=True)

    def default(m: Message) -> None:
        if m.value is None:
            m.value = cast(torch.Tensor, spec.score(env))
        if m.log_prob is None:
            m.log_prob = m.value

    apply_stack(msg, default=default)
    assert msg.value is not None
    env[spec.var] = msg.value


def _emit_sample(
    program: MonadicProgram,
    spec: _StepSpec,
    x: torch.Tensor,
    env: dict[str, torch.Tensor],
    observations: dict[str, torch.Tensor],
) -> None:
    """Compute a single draw step and emit one message per bound var.

    Multi-variable destructuring steps still fire the underlying
    morphism once (its samples are joint), but each destructured
    variable becomes its own message so downstream handlers can
    address them by name.
    """
    assert program._modules[spec.morphism_name] is not None
    morph = cast(ContinuousMorphism, program._modules[spec.morphism_name])
    inp = program._resolve_input(spec, x, env)

    if len(spec.vars) == 1:
        var_name = spec.vars[0]
        is_obs = var_name in observations
        msg = Message(
            kind="observe" if is_obs else "sample",
            name=var_name,
            morphism=morph,
            input=inp,
            is_observed=is_obs,
        )
        if is_obs:
            clamped = observations[var_name]
            # A clamped plate latent may arrive flattened to
            # ``(|A| * prod(event),)`` (the shape a host-data / test
            # point produces for a matrix-valued latent). Restore its
            # structured ``(|A|, *event)`` shape so a downstream gather
            # over the plate axis preserves the per-row event
            # coordinates and the family scores each row over its full
            # event. Non-plate sites carry no such method and pass
            # through unchanged.
            canonical = getattr(morph, "canonical_latent", None)
            if canonical is not None:
                clamped = canonical(clamped)
            msg.value = clamped

        def default(m: Message) -> None:
            if m.value is None:
                m.value = morph.rsample(inp)
            if m.log_prob is None:
                assert m.value is not None
                if m.is_deterministic:
                    m.log_prob = torch.zeros((), device=x.device)
                else:
                    m.log_prob = morph.log_prob(inp, m.value)

        apply_stack(msg, default=default)
        assert msg.value is not None
        env[var_name] = msg.value
        return

    # Destructuring step: sample once, split, then emit one message
    # per bound variable so handlers can address each by name.
    any_observed = any(v in observations for v in spec.vars)
    if any_observed:
        for v in spec.vars:
            if v in observations:
                env[v] = observations[v]
            else:
                result = morph.rsample(inp)
                program._bind_result(spec, result, env)
                break
    else:
        result = morph.rsample(inp)
        program._bind_result(spec, result, env)

    if hasattr(morph, "log_joint") and hasattr(morph, "_return_vars"):
        sub_morph = cast(MonadicProgram, morph)
        sub_intermediates: dict[str, torch.Tensor] = {}
        for sub_spec in sub_morph._step_specs:
            if isinstance(sub_spec, (_LetSpec, _ScoreSpec)):
                continue
            for sv in sub_spec.vars:
                if sv in env:
                    sub_intermediates[sv] = env[sv]
        joint_lp = sub_morph.log_joint(inp, sub_intermediates)
    else:
        parts = [env[v] for v in spec.vars]
        stacked = program._stack_tensors(parts)
        joint_lp = morph.log_prob(inp, stacked)

    per_var_lp = joint_lp / len(spec.vars)
    for v in spec.vars:
        is_obs = v in observations
        msg = Message(
            kind="observe" if is_obs else "sample",
            name=v,
            morphism=morph,
            input=inp,
            value=env[v],
            log_prob=per_var_lp,
            is_observed=is_obs,
        )
        # No default: value and log_prob already set from the joint
        # computation above.
        apply_stack(msg)
        assert msg.value is not None
        env[v] = msg.value


def run_program(
    program: MonadicProgram,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Execute a program through the active handler stack.

    Walks `program._step_specs` in order. Each step emits a
    `Message` (or one per destructured var for tuple-returning
    steps); the stack sees the message via
    `quivers.effects.base.apply_stack`, which runs the pre-pass,
    then the default site computation, then the post-pass.

    Parameters
    ----------
    program : MonadicProgram
        Program to execute.
    x : torch.Tensor
        Program input. Shape ``(batch, ...)``.
    observations : dict[str, torch.Tensor] or None
        Values to clamp observed sites to. Undeclared keys become
        env bindings (the host-data channel).

    Returns
    -------
    torch.Tensor or dict[str, torch.Tensor]
        The program's return value(s).
    """
    if observations is None:
        observations = {}

    env = _prepopulate_env(program, x, observations)

    for spec in program._step_specs:
        if isinstance(spec, _ScoreSpec):
            _emit_score(spec, env)
        elif isinstance(spec, _LetSpec):
            _emit_let(spec, env, x)
        else:
            _emit_sample(program, spec, x, env, observations)

    if program._return_is_single:
        return env[program._return_vars[0]]
    keys = program._return_labels if program._return_labels else program._return_vars
    return {k: env[v] for k, v in zip(keys, program._return_vars)}
