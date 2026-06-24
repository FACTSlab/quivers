"""Execution trace for monadic programs.

A trace records every sample site visited during program execution,
capturing the morphism, sampled or observed value, and log-density
at each site. This is the foundation for all inference algorithms:
SVI uses traces to compute the ELBO, and conditioning operates by
clamping trace sites to observed data.

The ``trace`` function is a free function that operates on any
MonadicProgram without modifying it: it walks the program's step
specs and resolves inputs using the program's existing infrastructure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from typing import cast

from quivers.continuous.morphisms import ContinuousMorphism
from quivers.continuous.programs import (
    MonadicProgram,
    _LetSpec,
    _ScoreSpec,
    _StepSpec,
)


_BRACKET_ARG_RE = re.compile(r"^([A-Za-z_]\w*)\[([A-Za-z_]\w*)\]$")


def _bracket_index_name(arg: str | None) -> str | None:
    """Return the index name of a ``name[idx]`` arg, else None.

    The bracket form is the surface notation used inside marginalize
    scopes to index into a per-class array via the marginalized
    latent: ``Categorical(emission_rows[state])`` resolves to
    ``env["emission_rows"][env["state"]]`` at trace time, so the
    enumerator must substitute every ``k`` for ``state`` before
    re-resolving the observe's input.
    """
    if arg is None:
        return None
    m = _BRACKET_ARG_RE.match(arg)
    if m is None:
        return None
    return m.group(2)


@dataclass(frozen=True)
class _MarginalizeBlock:
    """One ungrouped marginalize block detected in a program's step
    sequence.

    The compiler emits ``[latent_step, ...body steps, _ScoreSpec(
    "_marg_<latent>")]`` for every ``marginalize <latent> ~
    Categorical(probs)`` block without an ``[over=...]`` plate. The
    trace enumerates the latent across its discrete support, scoring
    every body observe at each candidate value, and replaces the
    triple (latent sample + obs log-prob + score callable) with a
    single ``log_sum_exp_k(log p(k) + sum_o log p(o | k))``
    accumulation. The result is the exact marginal log-likelihood
    Stan computes via the same ``log_sum_exp`` idiom.
    """

    latent_name: str
    latent_idx: int
    body_indices: tuple[int, ...]
    score_idx: int


def _find_ungrouped_marginalize_blocks(
    step_specs: tuple[object, ...],
) -> tuple[_MarginalizeBlock, ...]:
    """Detect every ungrouped marginalize block in `step_specs`.

    A block is signalled by a ``_ScoreSpec`` whose var begins with
    ``_marg_`` and whose suffix is the name of a recent ``_StepSpec``
    that binds a single variable. The block's body is the run of
    steps between the latent step (exclusive) and the score step
    (exclusive); at least one body step must reference the latent
    via a ``name[latent]`` bracket arg, signifying the
    marginalize-then-index pattern the enumeration trace handles.

    Score sites whose latent has no matching ``_StepSpec`` are the
    grouped-marginalize path: that path's runtime callable already
    consumes per-class log-likelihood tensors populated by
    ``GroupedBodyObserveStep`` runtime callables, so the trace runs
    them through the standard ``_ScoreSpec`` branch.
    """
    blocks: list[_MarginalizeBlock] = []
    for s_idx, spec in enumerate(step_specs):
        if not isinstance(spec, _ScoreSpec):
            continue
        if not spec.var.startswith("_marg_"):
            continue
        latent_name = spec.var[len("_marg_"):]
        latent_idx: int | None = None
        for j in range(s_idx - 1, -1, -1):
            prev = step_specs[j]
            if isinstance(prev, _StepSpec) and prev.vars == (latent_name,):
                latent_idx = j
                break
        if latent_idx is None:
            continue
        body_indices = tuple(range(latent_idx + 1, s_idx))
        references_latent = False
        for j in body_indices:
            entry = step_specs[j]
            if not isinstance(entry, _StepSpec) or entry.args is None:
                continue
            for arg in entry.args:
                if _bracket_index_name(arg) == latent_name:
                    references_latent = True
                    break
            if references_latent:
                break
        if not references_latent:
            continue
        blocks.append(
            _MarginalizeBlock(
                latent_name=latent_name,
                latent_idx=latent_idx,
                body_indices=body_indices,
                score_idx=s_idx,
            )
        )
    return tuple(blocks)


def _categorical_support_size(
    morph: ContinuousMorphism,
    args: tuple[str, ...] | None,
    env: dict[str, torch.Tensor],
) -> int:
    """Discrete support size K of a marginalized categorical latent.

    Source of truth: the trailing dimension of the latent's
    ``probs`` argument at runtime. The categorical family's
    codomain is one integer per draw (``dim == 1``); the K-axis lives
    on the parameter tensor the user supplies (``initial_row``,
    ``theta_d``, ``mixing_weights``, ...), so the enumerator inspects
    ``env[probs_arg].shape[-1]`` directly.

    Bernoulli-shaped latents whose family takes no probs argument
    fall back to ``K=2``: the family's codomain is still scalar but
    the support has exactly two integer atoms.
    """
    if args:
        for raw in args:
            base = raw.split("[", 1)[0]
            if base in env:
                return int(env[base].shape[-1])
    fam = getattr(morph, "family", None)
    if fam is None:
        fam = getattr(morph, "_family", None)
    if fam is not None:
        cod = getattr(fam, "codomain", None)
        if cod is not None:
            dim = getattr(cod, "dim", None)
            if isinstance(dim, int) and dim > 1:
                return dim
    raise RuntimeError(
        f"trace: cannot determine discrete support size for marginalized "
        f"latent: morphism {type(morph).__name__} has no probs argument "
        f"and no codomain dim > 1"
    )


@dataclass
class SampleSite:
    """Record of a single sample site in a program trace.

    Holds a ``torch.Tensor`` per site; not a value type.

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

    Mutable accumulator: ``sites`` grows as the program executes; not a
    value type.

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


def _per_class_log_prob(
    morph: ContinuousMorphism,
    arg_names: tuple[str, ...] | None,
    env: dict[str, torch.Tensor],
    value: torch.Tensor,
) -> torch.Tensor:
    """Log-prob of `value` under `morph` re-built from `env` args.

    Bypasses ``MixedInlineDistribution._resolve_params``: that helper
    splits a stacked-input vector by per-arg dim widths recorded at
    compile time, which assumes the marginalized latent contributed
    its own per-row sample shape. Inside the enumeration each arg is
    instead resolved against ``env`` directly (lifting bracket-index
    references through the latent's clamped value) and handed to the
    family's underlying ``_dist_builder`` unchanged, so a Categorical
    over emission_rows[k] sees the full ``(...,16)`` probability
    vector rather than a shape-1 placeholder.

    Returns the broadcast log-prob; callers reduce / sum as needed.
    """
    from quivers.continuous.programs import _lookup_arg  # noqa: PLC0415

    # The marginalized latent's family lives either directly on
    # ``morph`` (a `MixedInlineDistribution` constructed inline) or
    # one level deeper at ``morph._family`` (a plate / observe wrapper
    # around the inline family). Walk one level to find the dist
    # builder; an absent builder falls back to the morphism's own
    # ``log_prob`` (which handles non-inline families correctly).
    builder = getattr(morph, "_dist_builder", None)
    discrete_attr = getattr(morph, "_discrete", None)
    if builder is None:
        family = getattr(morph, "_family", None)
        if family is not None:
            builder = getattr(family, "_dist_builder", None)
            discrete_attr = getattr(family, "_discrete", discrete_attr)
    if builder is None or not arg_names:
        return morph.log_prob(env.get("_x_input", value), value)
    parts = [_lookup_arg(env, a) for a in arg_names]
    dist = builder(parts)
    discrete = bool(discrete_attr)
    if discrete and value.dtype.is_floating_point:
        value = value.long()
    # Broadcast `value` against the distribution's batch+event shape.
    # The categorical over a (..., K) probs has batch_shape (...) and
    # event_shape (). When `value` is shorter, expand to match;
    # when `value` has its own per-row axis but the distribution's
    # batch_shape is unbatched (a single shared parameter), let the
    # underlying ``dist.log_prob`` broadcast naturally.
    batch_shape = tuple(dist.batch_shape)
    event_shape = tuple(dist.event_shape)
    target_dim = len(batch_shape) + len(event_shape)
    while value.dim() < target_dim:
        value = value.unsqueeze(0)
    return dist.log_prob(value)


def _reduce_to_per_k(
    lp: torch.Tensor,
    K: int,
    batch_size: int,
) -> torch.Tensor:
    """Reduce a body-step log-prob to a ``(batch,)`` per-k contribution.

    The enumeration computes one contribution per per-k iteration; the
    body-observe's log-prob may carry per-row axes (a ``(plate_rows,)``
    or ``(N,)``-shaped tensor) that the marginal sums out, since the
    plate rows are conditionally independent given the latent. The
    routine collapses every non-batch axis into a single ``(batch,)``
    contribution, broadcasting scalars to the batch dim and summing
    over per-row axes that exceed the batch dim.
    """
    if lp.dim() == 0:
        return lp.expand(batch_size)
    flat = lp.reshape(-1)
    if flat.shape[0] == batch_size:
        return flat
    if flat.shape[0] == 1:
        return flat.expand(batch_size)
    # The body observe contributed a per-row scalar (the i-th plate's
    # contribution); sum across rows to get the joint log-likelihood
    # at this k, then broadcast to the batch.
    return flat.sum().expand(batch_size)


def _accumulate_marginalize_block(
    program: MonadicProgram,
    block: _MarginalizeBlock,
    env: dict[str, torch.Tensor],
    observations: dict[str, torch.Tensor],
    tr: "Trace",
    x: torch.Tensor,
) -> None:
    """Enumerate the discrete latent and accumulate the marginal.

    For each candidate value ``k`` in the latent's discrete support:

    1. Bind ``env[latent] = full((batch,), k)`` as a long tensor so
       any downstream bracket-gather (``emission_rows[state]``)
       resolves to a concrete per-class slice.
    2. Evaluate every body step in scope: deterministic let bindings
       run unchanged; observe steps compute their log-likelihood
       contribution against the clamped observation and accumulate
       into ``per_k_log_lik[k]``.
    3. Score the latent's prior at ``k`` via the latent's family
       and accumulate into ``per_k_log_lik[k]``.

    After the per-k pass, the marginal is
    ``log p_QVR(obs) = log_sum_exp_k(per_k_log_lik[k])``. The result
    is bound to ``env[_marg_<latent>]`` so the outer trace loop
    contributes it to ``total_lp`` exactly once; the latent itself
    is bound to its posterior expectation so downstream let bindings
    that consume the latent see a coherent value.
    """
    latent_spec = program._step_specs[block.latent_idx]
    assert isinstance(latent_spec, _StepSpec)
    latent_morph = cast(
        ContinuousMorphism,
        program._modules[latent_spec.morphism_name],
    )
    K = _categorical_support_size(latent_morph, latent_spec.args, env)

    batch_size = x.shape[0]
    per_k_log_liks: list[torch.Tensor] = []

    for k in range(K):
        per_k = torch.zeros(batch_size, device=x.device)
        # Bind the latent to k as a long tensor; the bracket-index
        # resolver consumes integer indices.
        env[block.latent_name] = torch.full(
            (batch_size,),
            k,
            dtype=torch.long,
            device=x.device,
        )
        # Latent prior log-pmf at k. Use the family-aware helper so
        # the categorical sees its full probs vector even when the
        # compiled family's ``_param_spec`` would otherwise truncate
        # the input.
        prior_lp = _per_class_log_prob(
            latent_morph,
            latent_spec.args,
            env,
            env[block.latent_name],
        )
        per_k = per_k + _reduce_to_per_k(prior_lp, K, batch_size)
        # Walk the body in order; let bindings and observes both run
        # against the env with the latent clamped to k.
        for j in block.body_indices:
            entry = program._step_specs[j]
            if isinstance(entry, _LetSpec):
                if isinstance(entry.value, str):
                    env[entry.var] = env[entry.value]
                elif callable(entry.value):
                    env[entry.var] = cast(torch.Tensor, entry.value(env))
                else:
                    env[entry.var] = torch.full(
                        (batch_size,),
                        entry.value,
                        device=x.device,
                    )
                continue
            if not isinstance(entry, _StepSpec):
                continue
            assert program._modules[entry.morphism_name] is not None
            obs_morph = cast(
                ContinuousMorphism,
                program._modules[entry.morphism_name],
            )
            obs_var = entry.vars[0]
            if obs_var in observations:
                val = observations[obs_var]
                env[obs_var] = val
            else:
                # Re-resolve the input for the family-bypass path:
                # `_resolve_input` stacks args into a single tensor
                # for `morph.rsample`, but for enumeration we hand the
                # rebuilt distribution the raw args and let it sample
                # natively (Categorical / Bernoulli / Poisson).
                from quivers.continuous.programs import _lookup_arg  # noqa: PLC0415
                family = getattr(obs_morph, "_family", None)
                builder = getattr(family, "_dist_builder", None) if family else None
                if builder is not None and entry.args:
                    parts = [_lookup_arg(env, a) for a in entry.args]
                    dist = builder(parts)
                    val = dist.sample()
                else:
                    val = obs_morph.rsample(
                        program._resolve_input(entry, x, env)
                    )
                env[obs_var] = val
            lp = _per_class_log_prob(obs_morph, entry.args, env, val)
            per_k = per_k + _reduce_to_per_k(lp, K, batch_size)
        per_k_log_liks.append(per_k)

    stacked = torch.stack(per_k_log_liks, dim=-1)  # (batch, K)
    log_marginal = torch.logsumexp(stacked, dim=-1)  # (batch,)
    env[f"_marg_{block.latent_name}"] = log_marginal

    log_post = stacked - log_marginal.unsqueeze(-1)
    post = log_post.exp()
    k_grid = torch.arange(
        K, dtype=x.dtype, device=x.device
    ).expand_as(post)
    latent_expectation = (post * k_grid).sum(dim=-1)
    env[block.latent_name] = latent_expectation
    tr.sites[block.latent_name] = SampleSite(
        name=block.latent_name,
        morphism=latent_morph,
        value=latent_expectation,
        log_prob=torch.zeros(batch_size, device=x.device),
        is_deterministic=True,
    )

    for j in block.body_indices:
        entry = program._step_specs[j]
        if isinstance(entry, _StepSpec):
            for name in entry.vars:
                value = env.get(name, torch.zeros(batch_size, device=x.device))
                tr.sites[name] = SampleSite(
                    name=name,
                    morphism=cast(
                        ContinuousMorphism,
                        program._modules[entry.morphism_name],
                    ),
                    value=value,
                    log_prob=torch.zeros(batch_size, device=x.device),
                    is_observed=name in observations,
                )
        elif isinstance(entry, _LetSpec):
            tr.sites[entry.var] = SampleSite(
                name=entry.var,
                morphism=None,
                value=env.get(entry.var, torch.zeros(batch_size, device=x.device)),
                log_prob=torch.zeros(batch_size, device=x.device),
                is_deterministic=True,
            )


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
    # Reserved synthetic key: compiler-emitted let-callables (e.g.
    # captured observes inside a grouped marginalize block when the
    # family takes the program input directly) read ``env["_x_input"]``.
    env["_x_input"] = x
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

    # Pre-populate env with any keys in ``observations`` that don't
    # match a declared sample/observe site. This is the host-data
    # channel: `condition(model, {"resp": y, "subj_idx": idx})` makes
    # ``subj_idx`` visible to ``let mu = by_subj[subj_idx]`` inside
    # the program body without forcing the user to redeclare every
    # per-row covariate as a sample site. Without this, integer index
    # arrays would have nowhere to live — observations are clamped on
    # sample sites, and program inputs are a single tensor with a
    # fixed factoring.
    _declared: set[str] = set()
    for _spec in program._step_specs:
        if isinstance(_spec, (_LetSpec, _ScoreSpec)):
            _declared.add(_spec.var)
        else:
            _declared.update(_spec.vars)
    for _key, _val in observations.items():
        if _key not in _declared:
            env[_key] = _val

    # Detect ungrouped marginalize blocks (the discrete-latent
    # forward-algorithm pattern: ``marginalize state ~ Categorical(p)``
    # followed by observe steps that bracket-index into a per-class
    # tensor via the latent). The enumeration replaces the latent
    # sample, the body observe scores, and the trailing
    # ``_marg_<latent>`` score callable with a single ``log_sum_exp_k(
    # log p(k) + sum_obs log p(obs | state=k))`` accumulation, which
    # is the exact closed-form marginal Stan computes via the
    # parallel ``log_sum_exp`` idiom.
    _blocks = _find_ungrouped_marginalize_blocks(program._step_specs)
    _block_by_latent_idx: dict[int, _MarginalizeBlock] = {
        b.latent_idx: b for b in _blocks
    }
    _block_by_body_idx: dict[int, _MarginalizeBlock] = {}
    _block_by_score_idx: dict[int, _MarginalizeBlock] = {
        b.score_idx: b for b in _blocks
    }
    for b in _blocks:
        for body_idx in b.body_indices:
            _block_by_body_idx[body_idx] = b
    _block_first_body_idx: dict[int, _MarginalizeBlock] = {
        b.body_indices[0]: b for b in _blocks if b.body_indices
    }

    for step_idx, spec in enumerate(program._step_specs):
        # Latent of an ungrouped marginalize: defer the draw; the
        # enumeration at the first body step handles every per-k
        # contribution and the trailing score step finalises the
        # accumulator.
        if step_idx in _block_by_latent_idx:
            continue
        # Final score step of an ungrouped marginalize: already
        # accumulated at the first-body trigger; skip.
        if step_idx in _block_by_score_idx:
            block = _block_by_score_idx[step_idx]
            score_name = spec.var if isinstance(spec, _ScoreSpec) else f"_marg_{block.latent_name}"
            if score_name in env:
                tr.sites[score_name] = SampleSite(
                    name=score_name,
                    morphism=None,
                    value=env[score_name],
                    log_prob=env[score_name],
                    is_deterministic=True,
                )
            continue
        # First body step of an ungrouped marginalize: drive the
        # enumeration over the latent's discrete support and skip
        # the body's per-step scoring (every body step contributes
        # its log-likelihood to the per-k accumulator).
        if step_idx in _block_first_body_idx:
            block = _block_first_body_idx[step_idx]
            _accumulate_marginalize_block(
                program,
                block,
                env,
                observations,
                tr,
                x,
            )
            total_lp = total_lp + env[f"_marg_{block.latent_name}"]
            continue
        # Subsequent body steps of an ungrouped marginalize: already
        # accumulated at the first-body trigger.
        if step_idx in _block_by_body_idx:
            continue

        if isinstance(spec, _ScoreSpec):
            # Score step (compiled marginalize): the callable returns
            # a (batch,)-shaped log-density contribution that is both
            # bound to env and accumulated into the trace's joint.
            val = cast(torch.Tensor, spec.score(env))
            env[spec.var] = val
            total_lp = total_lp + val
            tr.sites[spec.var] = SampleSite(
                name=spec.var,
                morphism=None,
                value=val,
                log_prob=val,
                is_deterministic=True,
            )
            continue

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
                    if isinstance(sub_spec, (_LetSpec, _ScoreSpec)):
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
