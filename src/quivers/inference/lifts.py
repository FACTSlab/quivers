"""Lift non-Bayesian morphisms into Bayesian `MonadicProgram`s.

The inference layer (SVI, NUTS, `LatentRegistry`) operates on
`MonadicProgram`s with explicit `sample` priors and `observe`
likelihood steps. Two patterns require a lift before that contract
applies:

1. A composed deterministic morphism (e.g. a chain of `[role=kernel]`
   morphisms whose composition has no `~ Family` prior) carries
   learnable `nn.Parameter`s but no priors and no observation
   family. `lift_to_bayesian_program` produces a proper
   `MonadicProgram` by attaching a Normal prior to every parameter
   and an observation family of the user's choice on the morphism's
   output.

2. A `MonadicProgram` declares intermediate latents via `sample`
   steps that have no externally observed value (an LM's hidden
   state `h`, a state-space model's per-step latent vector). The
   inference layer expects the caller to supply every latent in
   the observations dict. `monte_carlo_log_joint` forward-samples
   the named latents from their declared family and merges the
   draws into the observations dict before calling the inner's
   `log_joint`.

Both functions return artefacts the inference layer consumes
directly; no adapter classes, no per-family helpers.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from collections.abc import Mapping

# Distribution constructors accept these scalar / tensor argument
# kinds. Listed here so callers don't need to know the union shape.
type DistributionArg = torch.Tensor | float | int

import torch
import torch.distributions as D
import torch.nn as nn
from torch.distributions import constraints as _constraints

from quivers.continuous.inline import FixedDistribution
from quivers.continuous.programs import MonadicProgram
from quivers.continuous.spaces import Euclidean
from quivers.core.objects import Unit


__all__ = [
    "bayesian_lift_parameters",
    "lift_to_bayesian_program",
    "lift_from_log_prob",
    "monte_carlo_log_joint",
]


# ---------------------------------------------------------------------------
# Parameter-override context manager.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _swap_named_parameters(
    locator: Callable[[str], tuple[nn.Module, str]],
    overrides: dict[str, torch.Tensor],
) -> Iterator[None]:
    """Temporarily replace each named parameter with an override tensor.

    The replacement tensors live in the parent module's
    ``_parameters`` dict for the duration of the context. Because
    the overrides are plain (non-Parameter) tensors with their own
    autograd graphs, downstream reads through the standard module
    attribute path flow gradients back to the overrides.
    """
    if not overrides:
        yield
        return
    saved: dict[str, nn.Parameter] = {}
    placed: dict[str, tuple[nn.Module, str]] = {}
    try:
        for path, override in overrides.items():
            parent, local = locator(path)
            original = parent._parameters.get(local)
            if original is None:
                raise KeyError(
                    f"_swap_named_parameters: parameter {path!r} not "
                    f"found on {parent!r}"
                )
            if override.shape != original.shape:
                raise ValueError(
                    f"_swap_named_parameters: override for {path!r} "
                    f"has shape {tuple(override.shape)}; expected "
                    f"{tuple(original.shape)}"
                )
            saved[path] = original
            del parent._parameters[local]
            object.__setattr__(parent, local, override)
            placed[path] = (parent, local)
        yield
    finally:
        for path, (parent, local) in placed.items():
            try:
                object.__delattr__(parent, local)
            except AttributeError:
                pass
            parent._parameters[local] = saved[path]


def _build_locator(
    inner_model: nn.Module,
) -> tuple[Callable[[str], tuple[nn.Module, str]], list[str], list[nn.Parameter]]:
    """Return ``(locator, paths, params)`` over ``inner_model``."""
    paths: list[str] = []
    params: list[nn.Parameter] = []
    locations: dict[str, tuple[nn.Module, str]] = {}
    for path, p in inner_model.named_parameters():
        paths.append(path)
        params.append(p)
        parts = path.split(".")
        parent: nn.Module = inner_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        locations[path] = (parent, parts[-1])

    def locator(path: str) -> tuple[nn.Module, str]:
        return locations[path]

    return locator, paths, params


# ---------------------------------------------------------------------------
# Pure-parameter Bayesian lift.
# ---------------------------------------------------------------------------


def _make_normal_prior_morphism(scale: float, dim: int = 1) -> "FixedDistribution":
    cod = Euclidean(name=f"R{dim}" if dim > 1 else "R", dim=dim)

    def _make(batch: int, device: torch.device, _scale: float = scale, _dim: int = dim) -> D.Distribution:
        loc = torch.zeros(batch, _dim, device=device)
        sc = torch.full((batch, _dim), float(_scale), device=device)
        return D.Normal(loc, sc)

    return FixedDistribution(
        codomain=cod,
        make_dist=_make,
        discrete=False,
        support=_constraints.real,
    )


def bayesian_lift_parameters(
    inner_model: nn.Module,
    x: torch.Tensor,
    observations: dict[str, torch.Tensor],
    *,
    prior_scale: float = 1.0,
    site_prefix: str = "theta",
    additional_latents: dict[str, tuple[int, ...]] | None = None,
    latent_placeholder_scale: float = 10.0,
) -> tuple[MonadicProgram, torch.Tensor, dict[str, torch.Tensor]]:
    """Lift every learnable parameter of ``inner_model`` into a
    Normal-prior sample site, and optionally lift named latent
    sites of the inner program into NUTS-sampleable variables.

    Mathematics
    -----------
    Let :math:`\\theta` denote the inner model's learnable
    parameters and :math:`\\mathbf{z}` an optional collection of
    intermediate latents named in ``additional_latents``. The
    target joint posterior is

    .. math::
        p(\\theta, \\mathbf{z} \\mid x, y) \\;\\propto\\;
        p(\\theta) \\, p_{\\mathrm{inner}}(\\mathbf{z}, y \\mid x, \\theta).

    The lifted program declares

    * one Normal sample site
      :math:`\\theta_i \\sim \\mathcal{N}(0, \\sigma_\\theta^2)` per
      parameter (``prior_scale``);
    * one Normal sample site per latent in
      ``additional_latents`` with a *placeholder* scale
      :math:`\\sigma_z` (``latent_placeholder_scale``); and
    * one score step that, after substituting :math:`\\theta`
      into the inner's parameter slots, computes

      .. math::
          \\log p_{\\mathrm{inner}}(\\mathbf{z}, y \\mid x, \\theta)
          \\;-\\; \\sum_{z \\in \\text{latents}} \\log \\mathcal{N}(z; 0, \\sigma_z^2).

    The placeholder priors on :math:`\\mathbf{z}` cancel exactly,
    so the lifted log-density equals
    :math:`\\log p(\\theta) + \\log p_{\\mathrm{inner}}(\\mathbf{z}, y \\mid x, \\theta)`
    pointwise. NUTS therefore samples
    :math:`(\\theta, \\mathbf{z})` from the exact joint posterior,
    and the log-density is deterministic given the full state
    (no MC noise across leapfrog steps).

    Methodological notes
    --------------------
    *Why a Normal prior on parameters?*

    1. **Maximum entropy.** Among all distributions on
       :math:`\\mathbb{R}^n` with finite variance,
       :math:`\\mathcal{N}(0, \\sigma_\\theta^2)` is the
       maximum-entropy choice. Among priors that admit any second
       moment at all, it is the least informative.
    2. **Equivalence to weight decay.** A
       :math:`\\mathcal{N}(0, \\sigma_\\theta^2)` prior is the MAP
       equivalent of L2 regularization with coefficient
       :math:`1/(2\\sigma_\\theta^2)`. Standard frequentist weight
       decay and gradient-descent training inherit a direct
       Bayesian reading under this prior.
    3. **Computational fit with NUTS.** The unconstrained support
       :math:`\\mathbb{R}^n` matches NUTS's native state space, so
       no bijector is needed between latent and prior support.
       The log-density is smooth everywhere, so leapfrog dynamics
       are well-behaved.

    *Assumptions the user must respect.*

    1. **Parameters must be unconstrained reals.** If a learnable
       represents a variance, rate, probability, or simplex
       component, a Normal prior is mathematically invalid (it
       places mass on the forbidden region). Models must use the
       unconstrained parameterization (log-scale, logit-p,
       log-rate, soft-max logits, etc.). In QVR's standard model
       definitions, distribution families are parameterized in
       this way and `torch.nn.Parameter`\\ s are
       unconstrained reals.
    2. **The default ``prior_scale=1.0`` assumes O(1) parameter
       magnitude.** This is consistent with typical neural-network
       initialization schemes (Xavier, Kaiming). Override
       ``prior_scale`` for models with very different expected
       parameter magnitudes.
    3. **A Normal prior is generic, not informed.** The lift is a
       one-size-fits-all wrapper. Users with substantive domain
       knowledge about :math:`\\theta` should write a ``program``
       block with explicit ``sample`` priors instead of relying on
       the lift.

    *Why a placeholder Normal on lifted latents?*

    The placeholder prior on each
    :math:`\\mathbf{z}_i \\sim \\mathcal{N}(0, \\sigma_z^2)` is
    *algebraically irrelevant* by construction: the lifted
    sample-site prior and the placeholder cancellation in the
    score step sum to zero pointwise. The target distribution NUTS
    samples is the true joint posterior regardless of
    :math:`\\sigma_z`. A placeholder exists at all because NUTS's
    `LatentRegistry` enumerates dimensions from declared
    sample sites; each site needs a base measure to define the
    unconstrained support and to seed mass-matrix and step-size
    adaptation during warmup. Normal is the standard choice for an
    unconstrained latent.

    :math:`\\sigma_z` affects *mixing efficiency*, not
    *correctness*. A placeholder scale mismatched to the true
    posterior scale of :math:`\\mathbf{z}` lengthens warmup (the
    mass matrix has to adapt further) without biasing the chain.
    The default ``latent_placeholder_scale=10.0`` is large enough
    that initial NUTS proposals span a meaningful neighbourhood of
    zero, small enough that they do not immediately diverge.

    Parameters
    ----------
    inner_model : nn.Module
        Module exposing ``log_joint(x, observations) -> Tensor``.
    x, observations : tensor, dict
        Passed straight through to the inner's ``log_joint``. When
        ``additional_latents`` is supplied, ``observations`` must
        not contain those keys; they are supplied per NUTS
        evaluation from the env.
    prior_scale : float
        :math:`\\sigma_\\theta`. Standard deviation of the Normal
        prior on each parameter site.
    site_prefix : str
        Stem of each parameter sample-site's name.
    additional_latents : dict[str, tuple[int, ...]] | None
        Mapping from intermediate-latent site name (a key the
        inner's ``log_joint`` expects in its observations dict)
        to the latent's tensor shape (without the batch dim).
        When ``None`` (the default), the lift is parameter-only
        and ``inner.log_joint`` must accept ``observations`` as
        passed in.
    latent_placeholder_scale : float
        :math:`\\sigma_z`. Standard deviation of the placeholder
        Normal prior on each lifted latent. Any positive value
        works (it cancels exactly in the score step); a moderate
        value keeps NUTS's initial proposal magnitudes in a
        reasonable range.

    Returns
    -------
    (model, x_, observations_)
        The lifted ``MonadicProgram`` and the placeholder
        input / empty observation dict the inference layer feeds
        it.
    """
    if not hasattr(inner_model, "log_joint"):
        raise ValueError(
            "bayesian_lift_parameters: inner_model must expose "
            "``log_joint(x, observations)``"
        )
    if additional_latents and any(k in observations for k in additional_latents):
        bad = sorted(k for k in additional_latents if k in observations)
        raise ValueError(
            "bayesian_lift_parameters: keys "
            f"{bad!r} appear in both ``observations`` and "
            "``additional_latents``; a latent cannot be both "
            "observed and sampled"
        )

    inner_log_joint = inner_model.log_joint

    params = list(inner_model.named_parameters())
    if not params and not additional_latents:
        raise ValueError(
            "bayesian_lift_parameters: inner_model has no parameters "
            "and no additional_latents were declared"
        )

    safe_paths = [n.replace(".", "_") for n, _ in params]
    param_sites = [f"{site_prefix}__{p}" for p in safe_paths]
    locator, _paths, _ = _build_locator(inner_model)
    flat_dims = [int(p.numel()) for _, p in params]
    param_shapes = [tuple(p.shape) for _, p in params]
    param_paths = [n for n, _ in params]

    latent_names: list[str] = []
    latent_shapes: list[tuple[int, ...]] = []
    latent_flat_dims: list[int] = []
    latent_sites: list[str] = []
    if additional_latents:
        for name, shape in additional_latents.items():
            latent_names.append(name)
            latent_shapes.append(tuple(int(s) for s in shape))
            flat = 1
            for s in shape:
                flat *= int(s)
            latent_flat_dims.append(flat)
            latent_sites.append(f"latent__{name}")

    steps: list[tuple] = []
    for site, dim in zip(param_sites, flat_dims):
        steps.append((
            (site,),
            _make_normal_prior_morphism(prior_scale, dim=dim),
            None,
        ))
    for site, dim in zip(latent_sites, latent_flat_dims):
        steps.append((
            (site,),
            _make_normal_prior_morphism(latent_placeholder_scale, dim=dim),
            None,
        ))

    placeholder_scale_f = float(latent_placeholder_scale)

    def _score_fn(
        env: dict[str, torch.Tensor],
        _param_paths: list[str] = list(param_paths),
        _param_shapes: list[tuple[int, ...]] = list(param_shapes),
        _param_sites: list[str] = list(param_sites),
        _latent_names: list[str] = list(latent_names),
        _latent_shapes: list[tuple[int, ...]] = list(latent_shapes),
        _latent_sites: list[str] = list(latent_sites),
        _locator: Callable[[str], tuple[nn.Module, str]] = locator,
        _inner_x: torch.Tensor = x,
        _inner_obs: dict[str, torch.Tensor] = dict(observations),
        _placeholder_scale: float = placeholder_scale_f,
    ) -> torch.Tensor:
        ref = env[_param_sites[0]] if _param_sites else env[_latent_sites[0]]
        batch = ref.shape[0]
        out = torch.zeros(batch, dtype=torch.get_default_dtype())
        for b in range(batch):
            overrides: dict[str, torch.Tensor] = {}
            for path, shape, name in zip(_param_paths, _param_shapes, _param_sites):
                v = env[name][b]
                overrides[path] = v.reshape(shape) if shape else v.reshape(())
            latent_dict: dict[str, torch.Tensor] = {}
            placeholder_log_prior = torch.zeros((), dtype=torch.get_default_dtype())
            for name, shape, site in zip(_latent_names, _latent_shapes, _latent_sites):
                flat_b = env[site][b]
                latent_dict[name] = (
                    flat_b.reshape(shape) if shape else flat_b.reshape(())
                )
                placeholder_log_prior = placeholder_log_prior + D.Normal(
                    torch.zeros_like(flat_b),
                    torch.full_like(flat_b, _placeholder_scale),
                ).log_prob(flat_b).sum()
            with _swap_named_parameters(_locator, overrides):
                merged_obs = {**_inner_obs, **latent_dict}
                ll = inner_log_joint(_inner_x, merged_obs)
                ll_b = ll.sum() if ll.dim() > 0 else ll
                # Cancel the placeholder priors on lifted latents
                # so the net log-density is exactly
                # ``log p(theta) + log p_inner(z, y | x, theta)``.
                out[b] = ll_b - placeholder_log_prior
        return out

    steps.append((("log_lik",), None, _score_fn, True))
    lifted = MonadicProgram(
        domain=Unit,
        codomain=Unit,
        steps=steps,
        return_vars=("log_lik",),
    )
    return lifted, torch.zeros(1, 1), {}


# ---------------------------------------------------------------------------
# Deterministic-morphism + observation-family lift.
# ---------------------------------------------------------------------------


def lift_to_bayesian_program(
    parameter_module: nn.Module,
    *,
    location_fn: Callable[[torch.Tensor], torch.Tensor],
    parameter_prior_scale: float = 1.0,
    observation_family: type[D.Distribution],
    observation_kwargs: Mapping[str, DistributionArg] | None = None,
    target_key: str = "Y",
    x: torch.Tensor | None = None,
    observations: dict[str, torch.Tensor] | None = None,
) -> tuple[MonadicProgram, torch.Tensor, dict[str, torch.Tensor]]:
    """Lift a deterministic parameter-only model into a Bayesian
    `MonadicProgram` under a chosen observation family.

    The returned program has:

    * one Normal prior sample site per learnable
      `torch.nn.Parameter` of ``parameter_module`` (the
      parameter lift, with standard deviation
      ``parameter_prior_scale``);
    * one score step that
      (i) substitutes the sampled values into ``parameter_module``'s
      parameter slots,
      (ii) calls ``location_fn(x)`` to obtain the family's
      location tensor (e.g. ``lambda x: morphism.rsample(x)`` for
      input-driven morphisms, ``lambda _: morphism.tensor`` for
      parameter-only morphisms whose output is exposed via the
      ``tensor`` attribute, or ``lambda x: prog(x)`` for a program's
      forward call),
      (iii) builds
      ``observation_family(location, **observation_kwargs)``, and
      (iv) returns its log-probability at
      ``observations[target_key]``, reduced over event axes.

    Any ``torch.distributions.Distribution`` subclass works as
    ``observation_family``. The first positional parameter of the
    family (``loc`` for Normal, ``probs`` / ``logits`` for
    Categorical, etc.) receives ``location_fn``'s output; the
    remaining parameters come from ``observation_kwargs``.

    Parameters
    ----------
    parameter_module : nn.Module
        The module whose learnable parameters get Normal priors.
        For an input-driven morphism this is typically the
        morphism itself. For a program whose morphism is a
        parameter-only `ComposedMorphism`, pass the program
        and use ``location_fn=lambda _: prog.morphism.tensor``.
    location_fn : callable
        ``x -> Tensor`` returning the family's location parameter
        for input ``x``. Called inside the score step, after the
        parameter substitution, so the location reflects the
        current sampled :math:`\\theta`.
    parameter_prior_scale : float
        Standard deviation of the Normal prior on every parameter.
    observation_family : type
        ``torch.distributions.Distribution`` subclass.
    observation_kwargs : dict, optional
        Keyword arguments forwarded to ``observation_family``
        alongside the location tensor. Use this for the family's
        scale / concentration / total_count, etc.
    target_key : str
        Key in the observations dict whose value is the observed
        data.
    x, observations : tensor, dict, optional
        The forward input and the surrounding observations dict.
        ``x`` defaults to ``torch.zeros(1, 1)``; ``observations``
        is required to contain the entry under ``target_key`` at
        fit time.

    Returns
    -------
    (model, x_, observations_)
        The lifted program plus the input + empty observation dict
        the inference layer feeds it. The original
        ``observations[target_key]`` is captured by the score
        closure.
    """
    if observation_kwargs is None:
        observation_kwargs = {}
    if x is None:
        x = torch.zeros(1, 1)
    if observations is None:
        observations = {}

    def _score_against_family(
        x_in: torch.Tensor,
        obs: dict[str, torch.Tensor],
        _location_fn: Callable[[torch.Tensor], torch.Tensor] = location_fn,
        _family: type[D.Distribution] = observation_family,
        _family_kwargs: Mapping[str, DistributionArg] = dict(observation_kwargs),
        _key: str = target_key,
    ) -> torch.Tensor:
        location = _location_fn(x_in)
        # Pyright cannot statically verify ``**_family_kwargs``
        # against an arbitrary ``Distribution`` subclass signature:
        # each subclass has a different ctor (Normal takes scale,
        # Multinomial takes total_count, ...). The runtime contract
        # is the user's responsibility, documented on
        # ``observation_kwargs``.
        dist = _family(location, **_family_kwargs)  # pyright: ignore[reportCallIssue, reportArgumentType]
        lp = dist.log_prob(obs[_key])
        # Reduce every event axis to a (batch,)-shaped score.
        while lp.dim() > 1:
            lp = lp.sum(dim=-1)
        return lp

    class _ParameterModuleWithObservation(nn.Module):
        def __init__(self, _m: nn.Module = parameter_module) -> None:
            super().__init__()
            self.add_module("_param_module", _m)
            for attr in ("domain", "codomain"):
                v = getattr(_m, attr, None)
                if v is not None:
                    object.__setattr__(self, attr, v)

        def log_joint(
            self, x_in: torch.Tensor, obs: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            return _score_against_family(x_in, obs)

    wrapped = _ParameterModuleWithObservation()
    return bayesian_lift_parameters(
        wrapped, x, observations, prior_scale=parameter_prior_scale,
    )


# ---------------------------------------------------------------------------
# Log-prob morphism lift.
# ---------------------------------------------------------------------------


def lift_from_log_prob(
    parameter_module: nn.Module,
    *,
    log_prob_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    parameter_prior_scale: float = 1.0,
    target_key: str = "Y",
    x: torch.Tensor | None = None,
    observations: dict[str, torch.Tensor] | None = None,
) -> tuple[MonadicProgram, torch.Tensor, dict[str, torch.Tensor]]:
    """Lift a parameter-only model whose forward is a
    ``log_prob(x, y)``-style function into a Bayesian
    `MonadicProgram` over its parameters.

    Use this when the wrapped morphism already exposes a method
    that returns :math:`\\log p(y \\mid x)` directly (e.g. a
    `SampledComposition` over a Normal kernel, a VAE's
    encoder-decoder composition). The lifted program puts Normal
    priors on every learnable parameter and uses the supplied
    ``log_prob_fn`` to score the observation.

    Parameters
    ----------
    parameter_module : nn.Module
        Module whose learnable parameters get Normal priors.
    log_prob_fn : callable
        ``(x, y) -> Tensor``. Reads ``y = observations[target_key]``
        and returns a :math:`(\\text{batch},)`-shaped log-density.
        Called inside the score step after parameter substitution,
        so ``log_prob_fn``'s output reflects the current sampled
        :math:`\\theta`.
    parameter_prior_scale : float
        Standard deviation of the Normal prior on every parameter.
    target_key : str
        Observation-dict key for the observed data ``y``.
    x, observations : tensor, dict, optional
        Forward input and observations dict; defaults are
        ``torch.zeros(1, 1)`` and ``{}``.

    Returns
    -------
    (model, x_, observations_)
        The lifted program plus the input + empty observation dict
        the inference layer feeds it.
    """
    if x is None:
        x = torch.zeros(1, 1)
    if observations is None:
        observations = {}

    class _LogProbWithBayesianPriors(nn.Module):
        def __init__(self, _m: nn.Module = parameter_module) -> None:
            super().__init__()
            self.add_module("_param_module", _m)
            for attr in ("domain", "codomain"):
                v = getattr(_m, attr, None)
                if v is not None:
                    object.__setattr__(self, attr, v)

        def log_joint(
            self, x_in: torch.Tensor, obs: dict[str, torch.Tensor],
            _log_prob_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = log_prob_fn,
            _key: str = target_key,
        ) -> torch.Tensor:
            lp = _log_prob_fn(x_in, obs[_key])
            while lp.dim() > 1:
                lp = lp.sum(dim=-1)
            return lp

    wrapped = _LogProbWithBayesianPriors()
    return bayesian_lift_parameters(
        wrapped, x, observations, prior_scale=parameter_prior_scale,
    )


# ---------------------------------------------------------------------------
# Monte-Carlo sample-prefix lift.
# ---------------------------------------------------------------------------


def monte_carlo_log_joint(
    inner_model: nn.Module,
    *,
    sample_sites: list[str],
    keep_inner_observations: bool = True,
) -> nn.Module:
    """Wrap a program so its ``log_joint`` MC-draws the named
    intermediate sample sites and returns the *conditional* data
    likelihood at the draw.

    Mathematics
    -----------
    Given an inner program with parameters :math:`\\theta`, named
    intermediate latents :math:`\\mathbf{z}` (in ``sample_sites``),
    and observed data :math:`y`, the wrapper returns

    .. math::
        \\log p_{\\mathrm{inner}}(y \\mid \\mathbf{z}_*, \\theta),
        \\qquad
        \\mathbf{z}_* \\sim p_{\\mathrm{inner}}(\\mathbf{z} \\mid x, \\theta).

    This is a *single-sample Monte-Carlo estimator* of
    :math:`\\log p_{\\mathrm{inner}}(y \\mid x, \\theta)`. By Jensen,
    its expectation lower-bounds the true marginal likelihood:

    .. math::
        \\mathbb{E}_{\\mathbf{z}}\\bigl[\\log p(y \\mid \\mathbf{z}, \\theta)\\bigr]
        \\;\\le\\; \\log p(y \\mid x, \\theta).

    Implementation: for each name in ``sample_sites`` the wrapper
    resolves the site's morphism (through the inner's
    ``_step_specs`` or, as a fallback, ``inner._modules`` under the
    conventional ``_step_<site>`` / ``<site>`` keys), draws
    :math:`\\mathbf{z}_* = \\mathrm{morphism.rsample}(x)`, merges
    the draws into the observation dict, calls
    ``inner_model.log_joint(x, merged_obs)``, and subtracts
    :math:`\\log p(\\mathbf{z}_* \\mid x, \\theta)` so the residual
    is the conditional likelihood above (not the joint, which
    would double-count the latent's prior).

    Intended use
    ------------
    * **SVI / SGD**: this is a valid stochastic gradient estimator
      of the parameters' marginal-likelihood gradient. The mean of
      :math:`\\nabla_\\theta \\log p(y \\mid \\mathbf{z}_*, \\theta)`
      over draws of :math:`\\mathbf{z}_*` equals the corresponding
      ELBO-style descent direction, and SVI converges to a
      stationary point of that bound.
    * **NUTS / HMC**: *do not* use this wrapper for NUTS over a
      model whose log-density depends on :math:`\\mathbf{z}`.
      Re-drawing :math:`\\mathbf{z}_*` on every leapfrog evaluation
      makes the energy stochastic, which breaks the Hamiltonian
      symplectic invariant and biases the chain. The rigorous
      route is to lift :math:`\\mathbf{z}` as an additional NUTS
      latent via
      `bayesian_lift_parameters` with
      ``additional_latents={'<name>': <shape>}`` and let NUTS
      sample :math:`(\\theta, \\mathbf{z})` from the exact joint
      posterior. The lifted log-density is then deterministic
      given the full state.

    Gradient flow back to the inner's parameters is preserved
    when the underlying morphisms are reparameterised
    (``Normal``, ``MultivariateNormal``, etc.).

    Parameters
    ----------
    inner_model : nn.Module
        Typically a `MonadicProgram` with one or more
        ``sample`` steps whose values are not externally observed.
    sample_sites : list[str]
        Names of ``sample`` steps to MC-draw at log_joint time.
    keep_inner_observations : bool
        When True, the wrapper merges its caller's observations
        dict with the MC draws; when False, only the MC draws are
        forwarded.

    Returns
    -------
    nn.Module
        Exposes ``log_joint(x, observations) -> Tensor``.
    """
    if not all(isinstance(name, str) for name in sample_sites):
        raise TypeError(
            "monte_carlo_log_joint: sample_sites must be a list of strings"
        )

    class _MCLogJoint(nn.Module):
        def __init__(self, _inner: nn.Module = inner_model) -> None:
            super().__init__()
            self.add_module("_inner", _inner)
            for attr in ("domain", "codomain"):
                v = getattr(_inner, attr, None)
                if v is not None:
                    object.__setattr__(self, attr, v)

        def log_joint(
            self,
            x: torch.Tensor,
            observations: dict[str, torch.Tensor],
        ) -> torch.Tensor:
            merged: dict[str, torch.Tensor] = (
                dict(observations) if keep_inner_observations else {}
            )
            inner = self._inner
            modules = inner._modules
            step_specs = getattr(inner, "_step_specs", None)
            log_prior_correction = torch.zeros(
                x.shape[0],
                device=x.device,
                dtype=torch.get_default_dtype(),
            )
            for site in sample_sites:
                morph = None
                if step_specs is not None:
                    for spec in step_specs:
                        vars_ = getattr(spec, "vars", None)
                        if vars_ and site in vars_:
                            mname = getattr(spec, "morphism_name", None)
                            if mname is not None:
                                morph = modules.get(mname)
                                break
                if morph is None:
                    morph = modules.get(f"_step_{site}") or modules.get(site)
                if morph is None:
                    raise KeyError(
                        f"monte_carlo_log_joint: site {site!r} not "
                        f"resolvable via _step_specs or _modules"
                    )
                draw = morph.rsample(x)
                merged[site] = draw
                # ``inner.log_joint`` sums ``log_prob`` over every
                # named site, including the MC-drawn ones. To
                # produce the data likelihood given the draw (a
                # single-sample Monte-Carlo estimator of
                # ``log p(obs | theta)``), subtract the MC sites'
                # prior contributions so the residual is exactly
                # ``log p(observed | drawn, theta)``.
                if hasattr(morph, "log_prob"):
                    lp = morph.log_prob(x, draw)
                    while lp.dim() > 1:
                        lp = lp.sum(dim=-1)
                    log_prior_correction = log_prior_correction + lp
            return inner.log_joint(x, merged) - log_prior_correction

    return _MCLogJoint()
