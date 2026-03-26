"""Execution trace for monadic programs.

A trace records every sample site visited during program execution,
capturing the morphism, sampled or observed value, and log-density
at each site. This is the foundation for all inference algorithms:
SVI uses traces to compute the ELBO, and conditioning operates by
clamping trace sites to observed data.

The ``trace`` function is a free function that operates on any
MonadicProgram without modifying it — it walks the program's step
specs and resolves inputs using the program's existing infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from typing import cast

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.programs import MonadicProgram, _LetSpec


@dataclass
class SampleSite:
    """Record of a single sample site in a program trace.

    Parameters
    ----------
    name : str
        Variable name bound at this site.
    morphism : ContinuousMorphism or None
        The distribution morphism (None for let bindings).
    value : torch.Tensor
        The sampled or observed value.
    log_prob : torch.Tensor
        Log-density of the value under the morphism. Shape (batch,).
        Zero for let bindings.
    is_observed : bool
        Whether this site was clamped to an observed value.
    is_deterministic : bool
        Whether this is a deterministic let binding.
    """

    name: str
    morphism: ContinuousMorphism | None
    value: torch.Tensor
    log_prob: torch.Tensor
    is_observed: bool = False
    is_deterministic: bool = False


@dataclass
class Trace:
    """Complete execution trace of a monadic program.

    Parameters
    ----------
    sites : dict[str, SampleSite]
        All sample sites keyed by variable name.
    output : torch.Tensor or dict[str, torch.Tensor]
        The program's return value.
    log_joint : torch.Tensor
        Sum of log-densities across all stochastic sites. Shape (batch,).
    """

    sites: dict[str, SampleSite] = field(default_factory=dict)
    output: torch.Tensor | dict[str, torch.Tensor] | None = None
    log_joint: torch.Tensor | None = None

    @property
    def stochastic_sites(self) -> dict[str, SampleSite]:
        """Return only stochastic (non-deterministic) sites."""
        return {k: v for k, v in self.sites.items() if not v.is_deterministic}

    @property
    def latent_sites(self) -> dict[str, SampleSite]:
        """Return only latent (non-observed, non-deterministic) sites."""
        return {
            k: v
            for k, v in self.sites.items()
            if not v.is_observed and not v.is_deterministic
        }

    @property
    def observed_sites(self) -> dict[str, SampleSite]:
        """Return only observed sites."""
        return {k: v for k, v in self.sites.items() if v.is_observed}


def trace(
    program: MonadicProgram,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor] | None = None,
) -> Trace:
    """Execute a program and record all sample sites.

    Walks the program's step specs in order, sampling from each
    morphism (or clamping to observed values) and recording the
    value and log-density at each site.

    Parameters
    ----------
    program : MonadicProgram
        The program to trace.
    x : torch.Tensor
        Program input. Shape (batch, ...).
    observations : dict[str, torch.Tensor] or None
        Values to clamp observed variables to. Keys are variable
        names, values are tensors of the appropriate shape.

    Returns
    -------
    Trace
        Complete execution trace with all sites, output, and log-joint.
    """
    if observations is None:
        observations = {}

    env: dict[str, torch.Tensor] = {}
    tr = Trace()
    total_lp = torch.zeros(x.shape[0], device=x.device)

    # pre-populate env with named params
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

    for spec in program._step_specs:
        if isinstance(spec, _LetSpec):
            # deterministic binding
            if isinstance(spec.value, str):
                env[spec.var] = env[spec.value]

            elif callable(spec.value):
                env[spec.var] = cast(torch.Tensor, spec.value(env))

            else:
                env[spec.var] = torch.full(
                    (x.shape[0],),
                    spec.value,
                    device=x.device,
                )

            tr.sites[spec.var] = SampleSite(
                name=spec.var,
                morphism=None,
                value=env[spec.var],
                log_prob=torch.zeros(x.shape[0], device=x.device),
                is_deterministic=True,
            )
            continue

        # stochastic draw step
        assert program._modules[spec.morphism_name] is not None
        morph = cast(ContinuousMorphism, program._modules[spec.morphism_name])
        inp = program._resolve_input(spec, x, env)

        if len(spec.vars) == 1:
            var_name = spec.vars[0]
            is_obs = var_name in observations

            if is_obs:
                # clamp to observed value
                val = observations[var_name]
                env[var_name] = val

            else:
                # sample from the morphism
                val = morph.rsample(inp)
                env[var_name] = val

            lp = morph.log_prob(inp, val)
            total_lp = total_lp + lp

            tr.sites[var_name] = SampleSite(
                name=var_name,
                morphism=morph,
                value=val,
                log_prob=lp,
                is_observed=is_obs,
            )

        else:
            # destructuring step
            # check if any destructured vars are observed
            any_observed = any(v in observations for v in spec.vars)

            if any_observed:
                # clamp all destructured vars from observations
                for v in spec.vars:
                    if v in observations:
                        env[v] = observations[v]

                    else:
                        # if only some are observed, we need to sample the rest
                        # for now, treat as fully observed or fully latent
                        result = morph.rsample(inp)
                        program._bind_result(spec, result, env)
                        break

            else:
                result = morph.rsample(inp)
                program._bind_result(spec, result, env)

            # compute log-prob for the full step
            if hasattr(morph, "log_joint") and hasattr(morph, "_return_vars"):
                # sub-program: reconstruct intermediates
                sub_morph = cast(MonadicProgram, morph)
                sub_intermediates = {}

                for sub_spec in sub_morph._step_specs:
                    if isinstance(sub_spec, _LetSpec):
                        continue

                    for sv in sub_spec.vars:
                        if sv in env:
                            sub_intermediates[sv] = env[sv]

                lp = sub_morph.log_joint(inp, sub_intermediates)

            else:
                # product morphism: stack and evaluate
                parts = [env[v] for v in spec.vars]
                stacked = program._stack_tensors(parts)
                lp = morph.log_prob(inp, stacked)

            total_lp = total_lp + lp

            # record each destructured variable as a site
            for v in spec.vars:
                tr.sites[v] = SampleSite(
                    name=v,
                    morphism=morph,
                    value=env[v],
                    log_prob=lp / len(spec.vars),  # split log-prob evenly
                    is_observed=v in observations,
                )

    # compute output
    if program._return_is_single:
        tr.output = env[program._return_vars[0]]

    else:
        keys = (
            program._return_labels if program._return_labels else program._return_vars
        )
        tr.output = {k: env[v] for k, v in zip(keys, program._return_vars)}

    tr.log_joint = total_lp
    return tr
